#!/usr/bin/env python3
"""Evaluate a trained graph model or run baselines stored under ``models/``.

Trained model evaluations load graphs from ``data/processed/<variant>/`` using either
the monolithic graph artifact or matching shard files (``-shardNNN.pkl`` / ``-shardNNN.pt``)
based on the settings stored alongside the run directory.
Baseline evaluations operate on the lighter parquet splits in ``data/interim/<variant>/`` to
avoid materialising heavyweight graph artifacts in memory.

Usage
-----
python src/05_eval.py --run-directory models/<run>

Outputs
-------
Metrics are written to ``models/<run>/evaluations/model.json``.
"""

import argparse
import json
import logging
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.autonotebook import tqdm

from modules.baselines import evaluate_baselines
from modules.candidates import CandidateConfig, build_candidates, score_candidates_from_logits
from modules.config import ModelConfig, TrainingConfig
from modules.data_encoders import (
    GlobalIntEncoder,
    GraphStreamDataset,
    base_dataset_name,
    dataset_variant_name,
    discover_graph_artifacts,
    graph_dataset_filename,
)
from modules.model_store import baseline_dir, config_copy_path, evaluations_dir, get_checkpoint_path
from modules.models import BaseGraphModel, build_model
from modules.repair_eval import (
    ConstraintRepairHeuristics,
    RepairSample,
    ViolationContext,
    evaluate_global_repair_samples,
    evaluate_repair_samples,
    load_global_eval_rows,
    load_violation_contexts,
)
from modules.reranker_eval import CandidateConstraintEvaluator
from modules.training_utils import load_graph_dataset
from modules.policy import (
    POLICY_NAMES,
    PolicyDecision,
    filter_candidates_by_policy,
)

NONE_CLASS_INDEX = 0  # Bass-style datasets reserve class 0 for "no triple"
ACTIONS = ("add", "del")
PAPER_SUITE_TAGS: set[str] = {
    "b0_eswc_reproduction",
    "a1_factorized_imitation",
    "m1c_safe_factor_chooser",
    "m1d_safe_factor_direct",
    "g0_globalfix_reference",
}


def _torch_load_trusted(path: Path) -> object:
    """Load trusted local torch artifacts with PyTorch 2.6+ compatibility."""
    return torch.load(path, map_location="cpu", weights_only=False)


# --- EVALUATION METRICS DEFINITIONS --- #


@dataclass
class TripleCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, tp: int, fp: int, fn: int) -> None:
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)


@dataclass
class RepairSupport:
    contexts: list[ViolationContext]
    heuristics: ConstraintRepairHeuristics
    none_class: int = NONE_CLASS_INDEX

    def build_postprocess(self) -> tuple[Callable[[torch.Tensor, torch.Tensor, list[str]], None], dict[str, object]]:
        state: dict[str, object] = {}

        def _postprocess(predictions: torch.Tensor, targets: torch.Tensor, kinds: list[str]) -> None:
            samples = _repair_samples_from_predictions(predictions, targets, kinds, self.none_class)
            state["repair_metrics"] = evaluate_repair_samples(
                samples=samples,
                contexts=self.contexts,
                heuristics=self.heuristics,
                actions=ACTIONS,
            )

        return _postprocess, state


@dataclass
class GlobalMetricsSupport:
    rows: list[object]
    evaluator: CandidateConstraintEvaluator
    none_class: int = NONE_CLASS_INDEX

    def build_postprocess(
        self,
        test_data: Sequence[Data] | GraphStreamDataset | None = None,
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor, list[str]], None], dict[str, object]]:
        state: dict[str, object] = {}

        pre_vectors: list[dict[str, object]] | None = None
        # Streamed datasets intentionally do not implement truthiness via __len__.
        if test_data is not None:
            vectors: list[dict[str, object]] = []
            any_vectors = False
            for data in test_data:
                pre_satisfied = getattr(data, "factor_satisfied_pre", None)
                pre_checkable = getattr(data, "factor_checkable_pre", None)
                primary_index = getattr(data, "primary_factor_index", None)
                if pre_satisfied is None and pre_checkable is None and primary_index is None:
                    vectors.append({})
                    continue
                any_vectors = True
                vectors.append(
                    {
                        "pre_satisfied": pre_satisfied,
                        "pre_checkable": pre_checkable,
                        "primary_factor_index": primary_index,
                    }
                )
            if any_vectors:
                pre_vectors = vectors

        def _postprocess(predictions: torch.Tensor, targets: torch.Tensor, kinds: list[str]) -> None:
            samples = _repair_samples_from_predictions(predictions, targets, kinds, self.none_class)
            state["global_metrics"] = evaluate_global_repair_samples(
                samples=samples,
                rows=self.rows,
                evaluator=self.evaluator,
                none_class=self.none_class,
                pre_vectors=pre_vectors,
            )
            state["global_metrics_per_constraint_type"] = state["global_metrics"].get("per_constraint_type", {})

        return _postprocess, state


@dataclass
class ChooserSupport:
    contexts: Sequence[ViolationContext]
    heuristics: ConstraintRepairHeuristics
    candidate_cfg: CandidateConfig


@dataclass
class PolicySupport:
    contexts: Sequence[ViolationContext]
    heuristics: ConstraintRepairHeuristics
    candidate_cfg: CandidateConfig
    filter_strict: bool = True


@dataclass
class DirectSafetySupport:
    contexts: Sequence[ViolationContext]
    heuristics: ConstraintRepairHeuristics
    candidate_cfg: CandidateConfig


def _repair_samples_from_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    kinds: Sequence[str],
    none_class: int,
) -> list[RepairSample]:
    samples: list[RepairSample] = []
    for idx, kind in enumerate(kinds):
        pred_triples = _triples_from_indices(predictions[idx], none_class)
        gold_triples = _triples_from_indices(targets[idx], none_class)
        samples.append(
            RepairSample(
                constraint_type=kind,
                predicted=_single_triple_by_action(pred_triples),
                gold=_single_triple_by_action(gold_triples),
            )
        )
    return samples


def _single_triple_by_action(
    triples: Iterable[tuple[str, int, int, int]],
) -> dict[str, tuple[int, int, int] | None]:
    per_action: dict[str, tuple[int, int, int] | None] = {action: None for action in ACTIONS}
    for action, s, p, o in triples:
        per_action[action] = (s, p, o)
    return per_action


def _safe_div(num: int, denom: int) -> float:
    return float(num) / denom if denom else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _triples_from_indices(row: torch.Tensor, none_class: int) -> list[tuple[str, int, int, int]]:
    """Return predicted triples (action, s, p, o) ignoring NONE placeholders."""
    if row.ndim != 1:
        row = row.reshape(-1)
    if row.numel() != 6:
        raise ValueError(f"Expected prediction vector of length 6, got shape {tuple(row.shape)}")
    triples: list[tuple[str, int, int, int]] = []
    add_vals = row[:3].tolist()
    if all(int(v) != none_class for v in add_vals):
        triples.append(("add", int(add_vals[0]), int(add_vals[1]), int(add_vals[2])))
    del_vals = row[3:].tolist()
    if all(int(v) != none_class for v in del_vals):
        triples.append(("del", int(del_vals[0]), int(del_vals[1]), int(del_vals[2])))
    return triples


def _split_by_action(triples: Iterable[tuple[str, int, int, int]]) -> dict[str, set[tuple[int, int, int]]]:
    """Split triples into buckets by action type."""
    buckets = {action: set() for action in ACTIONS}
    for action, s, p, o in triples:
        if action in buckets:
            buckets[action].add((s, p, o))
    return buckets


def _normalize_precomputed_predictions(predictions: torch.Tensor | Iterable, none_class: int) -> torch.Tensor:
    if isinstance(predictions, torch.Tensor):
        tensor = predictions.detach().cpu()
    else:
        tensor = torch.tensor(list(predictions), dtype=torch.long)

    if tensor.dim() == 1 and tensor.numel() == 6:
        tensor = tensor.view(1, 6)
    if tensor.dim() == 3:
        tensor = tensor.argmax(dim=-1)
    if tensor.dim() != 2 or tensor.size(-1) != 6:
        raise ValueError(f"Precomputed predictions must have shape (N,6); got {tuple(tensor.shape)}")
    tensor = tensor.to(dtype=torch.long)
    if (tensor < 0).any():
        raise ValueError("Precomputed predictions contain negative indices.")
    if none_class < 0:
        raise ValueError("none_class must be non-negative.")
    return tensor


def _load_reranker_predictions(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Reranker predictions not found at {path}")

    suffix = path.suffix.lower()
    payload: object
    if suffix in {".pt", ".pth"}:
        payload = _torch_load_trusted(path)
    elif suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            payload = rows
        else:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
    else:
        raise ValueError("Unsupported reranker predictions format; expected .pt/.pth/.json/.jsonl")

    if isinstance(payload, dict):
        for key in ("predictions", "candidate_slots", "edits", "chosen_edits"):
            if key in payload:
                payload = payload[key]
                break

    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, list):
        if not payload:
            return torch.empty((0, 6), dtype=torch.long)
        if isinstance(payload[0], (list, tuple)):
            return torch.tensor(payload, dtype=torch.long)
        if isinstance(payload[0], dict):
            rows = []
            for entry in payload:
                add = entry.get("add")
                delete = entry.get("del") or entry.get("delete")
                if add is None:
                    add = [NONE_CLASS_INDEX, NONE_CLASS_INDEX, NONE_CLASS_INDEX]
                if delete is None:
                    delete = [NONE_CLASS_INDEX, NONE_CLASS_INDEX, NONE_CLASS_INDEX]
                rows.append(list(add) + list(delete))
            return torch.tensor(rows, dtype=torch.long)
    raise ValueError("Unsupported reranker predictions payload structure.")


def _maybe_prepare_repair_support(
    dataset_variant: str,
    min_occurrence: int,
    *,
    split: str,
    none_class: int,
) -> RepairSupport | None:
    variant_dir = dataset_variant
    suffix = f"_minocc{min_occurrence}"
    if not variant_dir.endswith(suffix):
        variant_dir = f"{variant_dir}{suffix}"

    interim_base = Path("data/interim") / variant_dir
    encoder_path = interim_base / "globalintencoder.txt"

    contexts = load_violation_contexts(interim_base, split, none_class=none_class)
    placeholder_ids = load_placeholder_ids(encoder_path)

    encoder = GlobalIntEncoder()
    encoder.load(encoder_path)
    encoder.freeze()

    heuristics = ConstraintRepairHeuristics(
        encoder=encoder,
        placeholder_ids=placeholder_ids,
        none_class=none_class,
    )

    return RepairSupport(contexts=contexts, heuristics=heuristics, none_class=none_class)


def _maybe_prepare_global_support(
    dataset_variant: str,
    min_occurrence: int,
    *,
    split: str,
    none_class: int,
    constraint_scope: str = "local",
    strict: bool = False,
    registry_dataset: str | None = None,
) -> GlobalMetricsSupport | None:
    variant_dir = dataset_variant
    suffix = f"_minocc{min_occurrence}"
    if not variant_dir.endswith(suffix):
        variant_dir = f"{variant_dir}{suffix}"

    interim_base = Path("data/interim") / variant_dir
    encoder_path = interim_base / "globalintencoder.txt"
    registry_candidates = []
    if registry_dataset:
        registry_candidates.append(registry_dataset)
    registry_candidates.append(dataset_variant)
    fallback_name = base_dataset_name(dataset_variant)
    registry_candidates.append(fallback_name)
    if "_strat" in fallback_name:
        registry_candidates.append(fallback_name.split("_strat", 1)[0])

    registry_path = None
    for candidate in dict.fromkeys(registry_candidates):
        candidate_path = Path("data/interim") / f"constraint_registry_{candidate}.parquet"
        if candidate_path.exists():
            registry_path = candidate_path
            break

    if registry_path is None:
        message = f"Constraint registry not found for candidates: {', '.join(dict.fromkeys(registry_candidates))}."
        if strict:
            raise RuntimeError(
                message + " Strict global metrics require the registry; rebuild interim data or disable strict mode."
            )
        logging.warning("%s Skipping global metrics.", message)
        return None
    if not encoder_path.exists():
        message = f"Global encoder not found at {encoder_path}."
        if strict:
            raise RuntimeError(
                message + " Strict global metrics require the encoder; rebuild interim data or disable strict mode."
            )
        logging.warning("%s Skipping global metrics.", message)
        return None

    try:
        rows = load_global_eval_rows(interim_base, split)
    except FileNotFoundError as exc:
        if strict:
            raise RuntimeError(
                f"Global eval rows unavailable at {interim_base}: {exc}. "
                "Strict global metrics require parquet rows with factor fields."
            ) from exc
        logging.warning("Global eval rows unavailable: %s", exc)
        return None

    encoder = GlobalIntEncoder()
    encoder.load(encoder_path)
    encoder.freeze()

    try:
        evaluator = CandidateConstraintEvaluator(
            str(registry_path),
            encoder=encoder,
            assume_complete=True,
            constraint_scope=constraint_scope,
            use_encoded_ids=True,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "Failed to construct constraint evaluator for strict global metrics. "
                "Verify registry format and encoder compatibility."
            ) from exc
        logging.warning("Constraint evaluator unavailable: %s", exc)
        return None

    return GlobalMetricsSupport(rows=rows, evaluator=evaluator, none_class=none_class)


def _aggregate_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    kinds: Iterable[str],
    none_class: int,
) -> tuple[TripleCounts, dict[str, dict[str, TripleCounts]]]:
    """Aggregate true/false positive/negative counts globally and per constraint type."""
    global_counts = TripleCounts()

    def _make_entry() -> dict[str, TripleCounts]:
        return {
            "micro": TripleCounts(),
            "add": TripleCounts(),
            "del": TripleCounts(),
        }

    per_type_counts: dict[str, dict[str, TripleCounts]] = defaultdict(_make_entry)

    for idx, kind in enumerate(kinds):
        pred_triples = _triples_from_indices(predictions[idx], none_class)
        gold_triples = _triples_from_indices(targets[idx], none_class)

        pred_by_action = _split_by_action(pred_triples)
        gold_by_action = _split_by_action(gold_triples)

        type_tp = type_fp = type_fn = 0

        for action in ACTIONS:
            pred_set = pred_by_action[action]
            gold_set = gold_by_action[action]

            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            per_type_counts[kind][action].update(tp, fp, fn)

            type_tp += tp
            type_fp += fp
            type_fn += fn

        per_type_counts[kind]["micro"].update(type_tp, type_fp, type_fn)
        global_counts.update(type_tp, type_fp, type_fn)

    return global_counts, per_type_counts


def _metrics_from_counts(counts: TripleCounts) -> dict[str, float]:
    precision = _safe_div(counts.tp, counts.tp + counts.fp)
    recall = _safe_div(counts.tp, counts.tp + counts.fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


# --- MAIN EVALUATION FUNCTION --- #


@torch.no_grad()
def eval(
    model: BaseGraphModel | None,
    test_data,
    batch_size: int = 16,
    device: torch.device | str = torch.device("cpu"),
    none_class: int = NONE_CLASS_INDEX,
    postprocess: Callable[[torch.Tensor, torch.Tensor, list[str]], None] | None = None,
    precomputed_predictions: torch.Tensor | None = None,
    chooser_support: ChooserSupport | None = None,
    direct_safety_support: DirectSafetySupport | None = None,
    policy_support: PolicySupport | None = None,
) -> dict[str, object]:
    """Evaluate a model and return Bass-style precision/recall/F1 metrics."""
    if isinstance(device, str):
        device = torch.device(device)

    test_loader = DataLoader(test_data, batch_size=batch_size)

    kinds: list[str] = []

    if precomputed_predictions is None and model is None:
        raise ValueError("Either a model or precomputed_predictions must be provided.")
    if precomputed_predictions is not None and chooser_support is not None:
        raise ValueError("Chooser support cannot be used with precomputed predictions.")
    if precomputed_predictions is not None and direct_safety_support is not None:
        raise ValueError("Direct safety support cannot be used with precomputed predictions.")
    if precomputed_predictions is not None and policy_support is not None:
        raise ValueError("Policy support cannot be used with precomputed predictions.")
    if chooser_support is not None and direct_safety_support is not None:
        raise ValueError("Chooser support and direct safety support cannot be used together.")
    if precomputed_predictions is None and model is not None:
        model.eval()
    predictions, targets = [], []
    output_logged = False
    for data in tqdm(test_loader, desc="Test Batches"):
        batch_graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
        kinds.extend((getattr(graph, "constraint_type", None) or "UNKNOWN") for graph in batch_graphs)
        if precomputed_predictions is None:
            data = data.to(device)
            out = model(data)  # raw logits expected
            if isinstance(out, dict):
                if not output_logged:
                    logging.info("Model forward returned dict outputs; using edit_logits.")
                    output_logged = True
                logits = out.get("edit_logits")
                graph_emb = out.get("graph_emb")
                if logits is None:
                    raise KeyError("Model output dict missing 'edit_logits'.")
            else:
                if not output_logged:
                    logging.info("Model forward returned tensor outputs.")
                    output_logged = True
                logits = out
                graph_emb = None
            if policy_support is not None:
                if graph_emb is None:
                    raise RuntimeError("Policy choice requires graph_emb from model outputs.")
                policy_logits = out.get("policy_logits") if isinstance(out, dict) else None
                if policy_logits is None:
                    raise RuntimeError("Policy choice enabled but model output missing policy_logits.")
                graphs = data.to_data_list()
                batch_preds = []
                for idx, graph in enumerate(graphs):
                    context_index = int(getattr(graph, "context_index", idx))
                    if context_index >= len(policy_support.contexts):
                        raise RuntimeError("Policy context index out of bounds.")
                    context = policy_support.contexts[context_index]
                    policy_id = int(torch.argmax(policy_logits[idx]).item())
                    candidates, _ = build_candidates(
                        graph=graph,
                        context=context,
                        heuristics=policy_support.heuristics,
                        proposal_logits=logits[idx].detach(),
                        cfg=policy_support.candidate_cfg,
                        placeholder_ids=set(policy_support.heuristics.placeholder_ids.values()),
                        num_target_ids=model.num_target_ids if model is not None else logits.size(-1),
                    )
                    candidates_filtered, match_mask = filter_candidates_by_policy(
                        candidates,
                        policy_id,
                        context,
                        strict=policy_support.filter_strict,
                        none_class=NONE_CLASS_INDEX,
                    )
                    if chooser_support is not None:
                        candidate_tensor = torch.tensor(candidates_filtered, dtype=torch.long, device=logits.device)
                        scores = model.score_candidates(graph_emb[idx], candidate_tensor)
                        if not policy_support.filter_strict:
                            mask_tensor = torch.tensor(
                                match_mask[: len(candidates_filtered)],
                                dtype=scores.dtype,
                                device=scores.device,
                            )
                            scores = scores + (mask_tensor - 1.0)
                        best_idx = int(torch.argmax(scores).item())
                        batch_preds.append(list(candidates_filtered[best_idx]))
                    else:
                        batch_preds.append(list(candidates_filtered[0]))
                predictions.append(torch.tensor(batch_preds, dtype=torch.long).cpu())
            elif chooser_support is not None:
                if graph_emb is None:
                    raise RuntimeError("Chooser mode requires graph_emb from model outputs.")
                graphs = data.to_data_list()
                batch_preds = []
                for idx, graph in enumerate(graphs):
                    context_index = int(getattr(graph, "context_index", idx))
                    if context_index >= len(chooser_support.contexts):
                        raise RuntimeError("Chooser context index out of bounds.")
                    context = chooser_support.contexts[context_index]
                    candidates, _ = build_candidates(
                        graph=graph,
                        context=context,
                        heuristics=chooser_support.heuristics,
                        proposal_logits=logits[idx].detach(),
                        cfg=chooser_support.candidate_cfg,
                        placeholder_ids=set(chooser_support.heuristics.placeholder_ids.values()),
                        num_target_ids=model.num_target_ids if model is not None else logits.size(-1),
                    )
                    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
                    scores = model.score_candidates(graph_emb[idx], candidate_tensor)
                    best_idx = int(torch.argmax(scores).item())
                    batch_preds.append(list(candidates[best_idx]))
                predictions.append(torch.tensor(batch_preds, dtype=torch.long).cpu())
            elif direct_safety_support is not None:
                graphs = data.to_data_list()
                batch_preds = []
                for idx, graph in enumerate(graphs):
                    context_index = int(getattr(graph, "context_index", idx))
                    if context_index >= len(direct_safety_support.contexts):
                        raise RuntimeError("Direct safety context index out of bounds.")
                    context = direct_safety_support.contexts[context_index]
                    candidates, _ = build_candidates(
                        graph=graph,
                        context=context,
                        heuristics=direct_safety_support.heuristics,
                        proposal_logits=logits[idx].detach(),
                        cfg=direct_safety_support.candidate_cfg,
                        placeholder_ids=set(direct_safety_support.heuristics.placeholder_ids.values()),
                        num_target_ids=model.num_target_ids if model is not None else logits.size(-1),
                    )
                    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
                    scores = score_candidates_from_logits(logits[idx], candidate_tensor)
                    best_idx = int(torch.argmax(scores).item())
                    batch_preds.append(list(candidates[best_idx]))
                predictions.append(torch.tensor(batch_preds, dtype=torch.long).cpu())
            else:
                out = logits.argmax(dim=-1)  # class predictions per action
                predictions.append(out.cpu())
        targets.append(data.y.cpu())

    if precomputed_predictions is None:
        predictions = torch.cat(predictions, dim=0)
    else:
        predictions = precomputed_predictions
    targets = torch.cat(targets, dim=0)

    if predictions.shape[0] != len(kinds):
        raise ValueError(
            f"Constraint type list has length {len(kinds)} but received {predictions.shape[0]} predictions."
        )

    if postprocess is not None:
        postprocess(predictions, targets, kinds)

    global_counts, per_type_counts = _aggregate_counts(predictions, targets, kinds, none_class)

    global_micro_metrics = _metrics_from_counts(global_counts)

    per_type_metrics: dict[str, dict[str, dict[str, float | int] | dict[str, dict[str, float | int]]]] = {}
    for kind, count_dict in per_type_counts.items():
        micro_counts = count_dict["micro"]
        type_micro_metrics = _metrics_from_counts(micro_counts)

        action_metrics: dict[str, dict[str, float | int]] = {}
        macro_components: list[dict[str, float]] = []
        for action in ACTIONS:
            action_counts = count_dict[action]
            action_metric = _metrics_from_counts(action_counts)
            action_metrics[action] = {
                "precision": action_metric["precision"],
                "recall": action_metric["recall"],
                "f1": action_metric["f1"],
                "tp": action_counts.tp,
                "fp": action_counts.fp,
                "fn": action_counts.fn,
            }
            if action_counts.tp + action_counts.fp + action_counts.fn > 0:
                macro_components.append(action_metric)

        if macro_components:
            macro_precision = sum(m["precision"] for m in macro_components) / len(macro_components)
            macro_recall = sum(m["recall"] for m in macro_components) / len(macro_components)
            macro_f1 = sum(m["f1"] for m in macro_components) / len(macro_components)
        else:
            macro_precision = macro_recall = macro_f1 = 0.0

        per_type_metrics[kind] = {
            "micro": {
                "precision": type_micro_metrics["precision"],
                "recall": type_micro_metrics["recall"],
                "f1": type_micro_metrics["f1"],
                "tp": micro_counts.tp,
                "fp": micro_counts.fp,
                "fn": micro_counts.fn,
            },
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "per_action": action_metrics,
        }

    if per_type_metrics:
        macro_precision = sum(pt["micro"]["precision"] for pt in per_type_metrics.values()) / len(per_type_metrics)
        macro_recall = sum(pt["micro"]["recall"] for pt in per_type_metrics.values()) / len(per_type_metrics)
        macro_f1 = sum(pt["micro"]["f1"] for pt in per_type_metrics.values()) / len(per_type_metrics)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    results: dict[str, object] = {
        "micro_precision": global_micro_metrics["precision"],
        "micro_recall": global_micro_metrics["recall"],
        "micro_f1": global_micro_metrics["f1"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "global_counts": {
            "tp": global_counts.tp,
            "fp": global_counts.fp,
            "fn": global_counts.fn,
        },
        "support_per_constraint_type": dict(Counter(kinds)),
        "per_constraint_type": per_type_metrics,
    }

    logging.info(
        "Micro Precision %.4f Recall %.4f F1 %.4f",
        results["micro_precision"],
        results["micro_recall"],
        results["micro_f1"],
    )
    logging.info(
        "Macro Precision %.4f Recall %.4f F1 %.4f",
        results["macro_precision"],
        results["macro_recall"],
        results["macro_f1"],
    )

    for kind, metrics in per_type_metrics.items():
        logging.info(
            "Type %s micro P %.4f R %.4f F %.4f | macro P %.4f R %.4f F %.4f",
            kind,
            metrics["micro"]["precision"],
            metrics["micro"]["recall"],
            metrics["micro"]["f1"],
            metrics["macro"]["precision"],
            metrics["macro"]["recall"],
            metrics["macro"]["f1"],
        )

        for action, action_metrics in metrics.get("per_action", {}).items():
            logging.debug(
                "  Action %s: P %.4f R %.4f F %.4f (TP=%d FP=%d FN=%d)",
                action,
                action_metrics["precision"],
                action_metrics["recall"],
                action_metrics["f1"],
                action_metrics["tp"],
                action_metrics["fp"],
                action_metrics["fn"],
            )

    return results


def _run_and_save(
    model,
    test_data,
    device,
    output_dir: Path,
    *,
    postprocess: Callable[[torch.Tensor, torch.Tensor, list[str]], None] | None = None,
    postprocess_state: dict[str, object] | None = None,
    precomputed_predictions: torch.Tensor | None = None,
    write_per_constraint_csv: bool = False,
    chooser_support: ChooserSupport | None = None,
    direct_safety_support: DirectSafetySupport | None = None,
    policy_support: PolicySupport | None = None,
    selection_weights: dict[str, float] | None = None,
    selection_disruption_field: str = "mean_disruption_total",
) -> dict[str, object]:
    normalized_predictions = None
    if precomputed_predictions is not None:
        normalized_predictions = _normalize_precomputed_predictions(precomputed_predictions, NONE_CLASS_INDEX)
    metrics = eval(
        model,
        test_data,
        device=device,
        postprocess=postprocess,
        precomputed_predictions=normalized_predictions,
        chooser_support=chooser_support,
        direct_safety_support=direct_safety_support,
        policy_support=policy_support,
    )
    metrics["global_metrics_computed"] = bool(
        postprocess_state and isinstance(postprocess_state, dict) and "global_metrics" in postprocess_state
    )
    if postprocess_state and "repair_metrics" in postprocess_state:
        metrics["repair_metrics"] = postprocess_state["repair_metrics"]
    if postprocess_state and "global_metrics" in postprocess_state:
        metrics["global_metrics"] = postprocess_state["global_metrics"]
    if postprocess_state and "global_metrics_per_constraint_type" in postprocess_state:
        metrics["global_metrics_per_constraint_type"] = postprocess_state["global_metrics_per_constraint_type"]
    if "global_metrics" in metrics and isinstance(metrics["global_metrics"], dict):
        overall = metrics["global_metrics"].get("overall")
        if isinstance(overall, dict):
            metrics["overall_gfr"] = overall.get("gfr", 0.0)
            metrics["overall_srr"] = overall.get("srr", 0.0)
            metrics["overall_sir"] = overall.get("sir", 0.0)
            disruption = overall.get("disruption", {}) if isinstance(overall.get("disruption"), dict) else {}
            metrics["mean_disruption_add"] = disruption.get("added_triples_mean", 0.0)
            metrics["mean_disruption_del"] = disruption.get("deleted_triples_mean", 0.0)
            metrics["mean_disruption_total"] = disruption.get("total_ops_mean", 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "model.json"
    with open(results_path, "w", encoding="utf-8") as f:
        weights = selection_weights or {}
        selection_block = compute_model_selection_block(
            metrics,
            weights=weights,
            disruption_field=selection_disruption_field,
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Model selection block: %s", selection_block)
        payload = {"model_selection": selection_block}
        payload.update(metrics)
        json.dump(payload, f, indent=4)
    if write_per_constraint_csv:
        _write_per_constraint_csv(metrics, output_dir)
    return metrics


# Small helpers to keep main() clean
def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {dev}")
    return dev


def load_split(
    base_path: Path,
    encoding: str,
    split: str,
    *,
    constraint_representation: str = "factorized",
) -> list[Data] | GraphStreamDataset:
    """Load a dataset split saved as a monolithic file or shard collection."""
    path = base_path / graph_dataset_filename(
        split,
        encoding,
        constraint_representation=constraint_representation,
    )
    return load_graph_dataset(path)


def _safe_constraint_type(value: object) -> str:
    """Normalize constraint type values coming from parquet rows."""
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float) and math.isnan(value):
        return "UNKNOWN"
    return str(value)


def _peek_graph(dataset) -> Data | None:
    if isinstance(dataset, list):
        return dataset[0] if dataset else None
    iterator = iter(dataset)
    try:
        return next(iterator)
    except StopIteration:
        return None
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _data_has_factor_fields(test_data) -> bool:
    sample = _peek_graph(test_data)
    if sample is None:
        return False
    has_ids = hasattr(sample, "factor_constraint_ids")
    has_labels = hasattr(sample, "factor_satisfied_pre") or hasattr(sample, "factor_checkable_pre")
    return has_ids and has_labels


def _dataset_graph_count(dataset, graph_path: Path) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        manifest_path = graph_path.with_suffix(graph_path.suffix + ".manifest.json")
        if not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            logging.exception("Failed to read graph manifest at %s", manifest_path)
            return None
        graph_count = payload.get("graph_count")
        return int(graph_count) if isinstance(graph_count, int) else None


def _infer_config_tag_from_run_dir(run_directory: Path) -> str:
    parts = run_directory.name.split("_")
    if len(parts) < 3:
        return run_directory.name
    idx = 1
    while idx < len(parts) and parts[idx].upper() == parts[idx]:
        idx += 1
    if idx >= len(parts):
        return run_directory.name
    return "_".join(parts[idx:])


def _is_paper_suite_config(model_cfg: ModelConfig, training_cfg: TrainingConfig | None) -> bool:
    if model_cfg.model == "RERANKER":
        return True
    if model_cfg.constraint_representation == "eswc_passive":
        return True
    if training_cfg is None:
        return False
    if training_cfg.chooser.enabled or training_cfg.direct_safety.enabled:
        return True
    return bool(model_cfg.pressure_enabled and model_cfg.constraint_representation == "factorized")


def _summarize_repair_per_type(repair_metrics: dict[str, object] | None) -> dict[str, dict[str, float | int]]:
    if not repair_metrics:
        return {}
    per_type = repair_metrics.get("per_constraint_type") or {}
    summary: dict[str, dict[str, float | int]] = {}
    for constraint_type, actions in per_type.items():
        if not isinstance(actions, dict):
            continue
        totals = {"total": 0, "exact": 0, "alternative": 0, "missing": 0, "failed": 0}
        for action_counts in actions.values():
            if not isinstance(action_counts, dict):
                continue
            for key in totals:
                totals[key] += int(action_counts.get(key, 0))
        total = totals["total"]
        exact = totals["exact"]
        alternative = totals["alternative"]
        summary[constraint_type] = {
            "total": total,
            "exact": exact,
            "alternative": alternative,
            "fix_rate": float(exact + alternative) / total if total else 0.0,
            "exact_rate": float(exact) / total if total else 0.0,
            "alternative_rate": float(alternative) / total if total else 0.0,
        }
    return summary


def _compute_primary_fix_rate(repair_metrics: dict[str, object] | None) -> float | None:
    if not repair_metrics:
        return None
    per_type = _summarize_repair_per_type(repair_metrics)
    total = 0
    exact = 0
    alternative = 0
    for metrics in per_type.values():
        total += int(metrics.get("total", 0))
        exact += int(metrics.get("exact", 0))
        alternative += int(metrics.get("alternative", 0))
    if total == 0:
        return 0.0
    return float(exact + alternative) / total


def compute_model_selection_block(
    metrics: dict[str, object],
    weights: dict[str, float],
    disruption_field: str,
) -> dict[str, object]:
    missing_fields: list[str] = []

    primary_fix_rate = _compute_primary_fix_rate(metrics.get("repair_metrics"))
    if primary_fix_rate is None:
        missing_fields.append("primary_fix_rate")

    fidelity_micro_f1 = float(metrics.get("micro_f1", 0.0))
    if "micro_f1" not in metrics:
        missing_fields.append("fidelity_micro_f1")

    srr_value = metrics.get("overall_srr")
    secondary_regression_rate_srr = float(srr_value) if srr_value is not None else None
    if secondary_regression_rate_srr is None:
        missing_fields.append("secondary_regression_rate_srr")

    disruption_total_ops_mean = metrics.get("mean_disruption_total")
    disruption_changed_triples_mean = metrics.get("disruption_changed_triples_mean")
    overall = metrics.get("global_metrics")
    if isinstance(overall, dict):
        overall = overall.get("overall")
    if isinstance(overall, dict):
        disruption = overall.get("disruption", {})
        if isinstance(disruption, dict):
            if disruption_total_ops_mean is None:
                disruption_total_ops_mean = disruption.get("total_ops_mean")
            if disruption_changed_triples_mean is None:
                disruption_changed_triples_mean = disruption.get("changed_triples_mean")

    disruption_value = None
    if disruption_field == "mean_disruption_total":
        disruption_value = disruption_total_ops_mean
    elif disruption_field == "disruption_changed_triples_mean":
        disruption_value = disruption_changed_triples_mean
    if disruption_value is None:
        missing_fields.append(disruption_field)
    else:
        disruption_value = float(disruption_value)

    score_raw = 0.0
    score_log = 0.0
    w_primary = float(weights.get("primary", 0.0))
    w_srr = float(weights.get("srr", 0.0))
    w_fidelity = float(weights.get("fidelity", 0.0))
    w_disrupt = float(weights.get("disrupt", 0.0))

    if primary_fix_rate is not None:
        score_raw += w_primary * primary_fix_rate
        score_log += w_primary * primary_fix_rate
    if secondary_regression_rate_srr is not None:
        score_raw += w_srr * (1.0 - secondary_regression_rate_srr)
        score_log += w_srr * (1.0 - secondary_regression_rate_srr)
    score_raw += w_fidelity * fidelity_micro_f1
    score_log += w_fidelity * fidelity_micro_f1
    if disruption_value is not None:
        score_raw -= w_disrupt * disruption_value
        score_log -= w_disrupt * math.log1p(disruption_value)

    selection_block = {
        "primary_fix_rate": primary_fix_rate,
        "secondary_regression_rate_srr": secondary_regression_rate_srr,
        "fidelity_micro_f1": fidelity_micro_f1,
        "disruption_total_ops_mean": float(disruption_total_ops_mean)
        if disruption_total_ops_mean is not None
        else None,
        "disruption_changed_triples_mean": float(disruption_changed_triples_mean)
        if disruption_changed_triples_mean is not None
        else None,
        "score_weighted_sum": score_raw,
        "score_weighted_sum_log_disruption": score_log,
        "weights": {
            "primary": w_primary,
            "srr": w_srr,
            "fidelity": w_fidelity,
            "disrupt": w_disrupt,
            "disruption_field": disruption_field,
        },
        "missing_fields": sorted(set(missing_fields)),
        "notes": "score = w_primary*primary_fix_rate + w_srr*(1 - srr) + w_fidelity*fidelity_micro_f1 - w_disrupt*disruption",
    }
    return selection_block


def _write_per_constraint_csv(metrics: dict[str, object], output_dir: Path) -> None:
    per_type_metrics = metrics.get("per_constraint_type") or {}
    support_counts = metrics.get("support_per_constraint_type") or {}
    repair_summary = _summarize_repair_per_type(metrics.get("repair_metrics"))
    global_per_type = metrics.get("global_metrics_per_constraint_type") or {}

    rows: list[dict[str, object]] = []
    for constraint_type, type_metrics in per_type_metrics.items():
        micro = type_metrics.get("micro", {})
        global_metrics = global_per_type.get(constraint_type, {})
        disruption = global_metrics.get("disruption", {}) if isinstance(global_metrics, dict) else {}
        repair = repair_summary.get(constraint_type, {})
        rows.append(
            {
                "constraint_type": constraint_type,
                "support": int(support_counts.get(constraint_type, 0)),
                "fidelity_micro_f1": float(micro.get("f1", 0.0)),
                "primary_fix_rate": float(repair.get("fix_rate", 0.0)),
                "primary_exact_rate": float(repair.get("exact_rate", 0.0)),
                "primary_alternative_rate": float(repair.get("alternative_rate", 0.0)),
                "primary_total": int(repair.get("total", 0)),
                "gfr": float(global_metrics.get("gfr", 0.0)) if isinstance(global_metrics, dict) else 0.0,
                "srr": float(global_metrics.get("srr", 0.0)) if isinstance(global_metrics, dict) else 0.0,
                "sir": float(global_metrics.get("sir", 0.0)) if isinstance(global_metrics, dict) else 0.0,
                "disruption_add_mean": float(disruption.get("added_triples_mean", 0.0)),
                "disruption_del_mean": float(disruption.get("deleted_triples_mean", 0.0)),
                "disruption_total_ops_mean": float(disruption.get("total_ops_mean", 0.0)),
                "disruption_changed_mean": float(disruption.get("changed_triples_mean", 0.0)),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "per_constraint.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _smoke_check_global_metrics(
    support: GlobalMetricsSupport,
    test_data,
    *,
    none_class: int,
) -> None:
    sample_graphs: list[Data] = []
    for idx, graph in enumerate(test_data):
        if idx >= 3 or idx >= len(support.rows):
            break
        sample_graphs.append(graph)
    sample_count = len(sample_graphs)
    if sample_count == 0:
        logging.info("Skipping global metrics smoke check (no samples).")
        return
    preds = torch.cat([graph.y for graph in sample_graphs], dim=0)
    kinds = [(getattr(graph, "constraint_type", None) or "UNKNOWN") for graph in sample_graphs]
    samples = _repair_samples_from_predictions(preds, preds, kinds, none_class)
    evaluate_global_repair_samples(
        samples=samples,
        rows=support.rows[:sample_count],
        evaluator=support.evaluator,
        none_class=none_class,
        pre_vectors=None,
    )
    logging.info("Global metrics smoke check passed for %d samples.", sample_count)


def load_baseline_split_from_parquet(base_path: Path, split: str) -> tuple[list[Data], int]:
    """Load baseline data from parquet, returning lightweight graphs and the max node index."""
    path = base_path / f"df_{split}.parquet"
    dataframe = pd.read_parquet(path)

    data_list: list[Data] = []
    max_index = NONE_CLASS_INDEX

    for row in dataframe.itertuples(index=False):
        y = torch.tensor(
            [
                int(row.add_subject),
                int(row.add_predicate),
                int(row.add_object),
                int(row.del_subject),
                int(row.del_predicate),
                int(row.del_object),
            ],
            dtype=torch.long,
        ).unsqueeze(0)

        focus_triple = torch.tensor(
            [int(row.subject), int(row.predicate), int(row.object)],
            dtype=torch.long,
        )

        graph = Data(
            x=torch.zeros((1, 1), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=y,
        )
        graph.focus_triple = focus_triple
        shape_id_value = getattr(row, "constraint_id", None)
        if isinstance(shape_id_value, float) and math.isnan(shape_id_value):
            shape_id_value = None
        elif shape_id_value is not None:
            shape_id_value = int(shape_id_value)
        if shape_id_value is not None:
            graph.shape_id = shape_id_value
        graph.constraint_type = _safe_constraint_type(row.constraint_type)

        factor_constraint_ids = getattr(row, "factor_constraint_ids", None)
        if factor_constraint_ids is not None:
            graph.factor_constraint_ids = torch.tensor(factor_constraint_ids, dtype=torch.long).view(-1)
        factor_types = getattr(row, "factor_types", None)
        if factor_types is not None:
            graph.factor_types = torch.tensor(factor_types, dtype=torch.long).view(-1)
        factor_checkable_pre = getattr(row, "factor_checkable_pre", None)
        if factor_checkable_pre is not None:
            graph.factor_checkable_pre = torch.tensor(factor_checkable_pre, dtype=torch.bool).view(-1)
        factor_satisfied_pre = getattr(row, "factor_satisfied_pre", None)
        if factor_satisfied_pre is not None:
            graph.factor_satisfied_pre = torch.tensor(factor_satisfied_pre, dtype=torch.long).view(-1)
        factor_checkable_post_gold = getattr(row, "factor_checkable_post_gold", None)
        if factor_checkable_post_gold is not None:
            graph.factor_checkable_post_gold = torch.tensor(
                factor_checkable_post_gold, dtype=torch.bool
            ).view(-1)
        factor_satisfied_post_gold = getattr(row, "factor_satisfied_post_gold", None)
        if factor_satisfied_post_gold is not None:
            graph.factor_satisfied_post_gold = torch.tensor(
                factor_satisfied_post_gold, dtype=torch.long
            ).view(-1)
        primary_factor_index = getattr(row, "primary_factor_index", None)
        if primary_factor_index is not None:
            if not (isinstance(primary_factor_index, float) and math.isnan(primary_factor_index)):
                graph.primary_factor_index = int(primary_factor_index)

        data_list.append(graph)

        y_max = int(y.max().item()) if y.numel() else NONE_CLASS_INDEX
        focus_max = int(focus_triple.max().item()) if focus_triple.numel() else NONE_CLASS_INDEX
        current_max = max(y_max, focus_max)
        if current_max > max_index:
            max_index = current_max

    del dataframe
    return data_list, max_index


def _resolve_baseline_interim_paths(dataset: str, min_occurrence: int) -> tuple[Path, Path]:
    variant = dataset_variant_name(dataset, min_occurrence)
    base_path = Path("data/interim") / variant
    labeled_path = Path("data/interim") / f"{variant}_labeled"
    data_path = labeled_path if (labeled_path / "df_train.parquet").exists() else base_path
    encoder_path = labeled_path / "globalintencoder.txt"
    if not encoder_path.exists():
        encoder_path = base_path / "globalintencoder.txt"
    return data_path, encoder_path


def load_placeholder_ids(encoder_path: Path) -> dict[str, int]:
    """Load placeholder token ids (e.g. 'subject') from the global encoder."""
    if not encoder_path.exists():
        logging.warning("Placeholder encoder not found at %s", encoder_path)
        return {}

    encoder = GlobalIntEncoder()
    encoder.load(encoder_path)
    encoder.freeze()

    placeholders: dict[str, int] = {}
    for token in ("subject", "predicate", "object", "other_subject", "other_predicate", "other_object"):
        idx = encoder.encode(token)
        if idx:
            placeholders[token] = idx

    return placeholders


def _checkpoint_has_chooser_state(state_dict: dict[str, object]) -> bool:
    return any(key.startswith(("_candidate_id_embeddings.", "_chooser_head.")) for key in state_dict)


def _enable_checkpoint_optional_heads(model: BaseGraphModel, state_dict: dict[str, object]) -> None:
    if _checkpoint_has_chooser_state(state_dict) and not model.chooser_enabled:
        model.enable_chooser()


# Load and run a trained torch model
def evaluate_trained_model(
    *,
    run_directory: Path,
    model_cfg: ModelConfig,
    device: torch.device,
    test_data,
    postprocess: Callable[[torch.Tensor, torch.Tensor, list[str]], None] | None = None,
    postprocess_state: dict[str, object] | None = None,
    precomputed_predictions: torch.Tensor | None = None,
    write_per_constraint_csv: bool = False,
    chooser_support: ChooserSupport | None = None,
    direct_safety_support: DirectSafetySupport | None = None,
    policy_support: PolicySupport | None = None,
    selection_weights: dict[str, float] | None = None,
    selection_disruption_field: str = "mean_disruption_total",
) -> None:
    checkpoint_path = get_checkpoint_path(run_directory)

    if precomputed_predictions is None and not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained model not found at {checkpoint_path}")

    model = None
    if precomputed_predictions is None:
        logging.info("Using model artifacts in %s", run_directory)
        logging.info("Loading checkpoint from %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, BaseGraphModel):
            model = checkpoint.to(device)
            if chooser_support is not None and not model.chooser_enabled:
                model.enable_chooser()
        elif isinstance(checkpoint, dict):
            model_name = checkpoint.get("model_name") or model_cfg.model
            num_graph_nodes = checkpoint.get("num_graph_nodes")
            state_dict = checkpoint.get("model_state")
            if num_graph_nodes is None or state_dict is None:
                raise ValueError("Checkpoint is missing required fields: 'num_graph_nodes' or 'model_state'.")

            checkpoint_model_cfg = checkpoint.get("model_cfg", {})
            if checkpoint_model_cfg:
                effective_model_cfg = ModelConfig.from_mapping(checkpoint_model_cfg)
            else:
                effective_model_cfg = model_cfg

            model = build_model(
                model_name,
                int(num_graph_nodes),
                effective_model_cfg,
            )
            _enable_checkpoint_optional_heads(model, state_dict)
            if chooser_support is not None and not model.chooser_enabled:
                model.enable_chooser()
            model.load_state_dict(state_dict)
            model.to(device)
            printable_config = {
                k: v
                for k, v in effective_model_cfg.to_dict().items()
                if k not in ["entity_class_ids", "predicate_class_ids"]
            }
            logging.info("Loaded checkpoint with config: %s", printable_config)
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    _run_and_save(
        model=model,
        test_data=test_data,
        device=device,
        output_dir=evaluations_dir(run_directory, create=True),
        postprocess=postprocess,
        postprocess_state=postprocess_state,
        precomputed_predictions=precomputed_predictions,
        write_per_constraint_csv=write_per_constraint_csv,
        chooser_support=chooser_support,
        direct_safety_support=direct_safety_support,
        policy_support=policy_support,
        selection_weights=selection_weights,
        selection_disruption_field=selection_disruption_field,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model or run baselines on the graphs dataset.")
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="Path to the stored run directory under ./models.",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run the three baselines (DFB, AMB, CSM).",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset variant (baselines mode), e.g. full or full_strat1m.",
    )
    parser.add_argument(
        "--registry-dataset",
        default=None,
        help="Raw dataset name for constraint_registry_<dataset>.parquet. Defaults to automatic fallback.",
    )
    parser.add_argument(
        "--min-occurrence",
        type=int,
        default=100,
        help="Min occurrence for processed path (default 100).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Verbosity level for logging output.",
    )
    parser.add_argument(
        "--global-metrics",
        action="store_true",
        help="Compute global constraint metrics when possible (deprecated; now auto-enabled when available).",
    )
    parser.add_argument(
        "--no-global-metrics",
        action="store_true",
        help="Disable global metrics computation (GFR/SRR/SIR/disruption).",
    )
    parser.add_argument(
        "--per-constraint-csv",
        action="store_true",
        help="Write per-constraint breakdown CSV in the evaluations directory.",
    )
    parser.add_argument(
        "--strict-global-metrics",
        action="store_true",
        help="Fail fast unless global metrics (GFR/SRR/SIR/disruption) can be computed.",
    )
    parser.add_argument(
        "--reranker-predictions",
        type=Path,
        help="Optional path to reranker predictions to evaluate without a model forward pass.",
    )
    parser.add_argument(
        "--use-chooser",
        action="store_true",
        help="Use proposal chooser head to select candidate edits instead of slot-wise argmax.",
    )
    parser.add_argument(
        "--use-policy-choice",
        action="store_true",
        help="Use policy choice head to filter candidates before selection.",
    )
    parser.add_argument(
        "--score-w-primary",
        type=float,
        default=1.0,
        help="Weight for primary fix rate in model selection score.",
    )
    parser.add_argument(
        "--score-w-srr",
        type=float,
        default=1.0,
        help="Weight for secondary regression rate term (applied to 1 - srr).",
    )
    parser.add_argument(
        "--score-w-fidelity",
        type=float,
        default=0.5,
        help="Weight for fidelity micro F1 in model selection score.",
    )
    parser.add_argument(
        "--score-w-disrupt",
        type=float,
        default=0.2,
        help="Weight for disruption penalty in model selection score.",
    )
    parser.add_argument(
        "--score-disruption-field",
        choices=["mean_disruption_total", "disruption_changed_triples_mean"],
        default="mean_disruption_total",
        help="Disruption field used in model selection score.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        force=True,
    )

    if args.run_baselines:
        if not args.dataset:
            raise ValueError("--run-baselines requires --dataset.")
        if args.reranker_predictions:
            raise ValueError("--reranker-predictions is only supported for trained model evaluation.")
        return args

    if args.run_directory is None:
        raise ValueError("Either --run_directory (trained model) or --run-baselines with --dataset must be provided.")

    run_directory = Path(args.run_directory).resolve()
    models_root = Path("models").resolve()
    try:
        run_directory.relative_to(models_root)
    except ValueError as exc:
        raise ValueError(f"Run directory must be inside {models_root}") from exc

    if not run_directory.exists():
        raise FileNotFoundError(f"Run directory not found at {run_directory}")
    if not run_directory.is_dir():
        raise NotADirectoryError(f"Run directory path is not a directory: {run_directory}")

    args.run_directory = run_directory
    return args


def main():
    args = parse_args()
    device = get_device()
    selection_weights = {
        "primary": args.score_w_primary,
        "srr": args.score_w_srr,
        "fidelity": args.score_w_fidelity,
        "disrupt": args.score_w_disrupt,
    }

    if not args.run_baselines:
        run_directory = args.run_directory
        config_path = config_copy_path(run_directory)
        if not config_path.exists():
            raise FileNotFoundError(f"Stored configuration file not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            experiment_config = json.load(f)

        model_cfg = ModelConfig.from_mapping(experiment_config["model_config"])
        training_payload = experiment_config.get("training_config", {})
        if "reranker_config" in experiment_config or str(model_cfg.model).upper() == "RERANKER":
            allowed = set(TrainingConfig.__dataclass_fields__.keys())
            training_payload = {k: v for k, v in dict(training_payload).items() if k in allowed}
        training_cfg = TrainingConfig.from_mapping(training_payload)
        config_tag = _infer_config_tag_from_run_dir(run_directory)
        strict_global = bool(
            args.strict_global_metrics
            or _is_paper_suite_config(model_cfg, training_cfg)
            or any(config_tag.startswith(tag) for tag in PAPER_SUITE_TAGS)
        )
        if strict_global and args.no_global_metrics:
            raise RuntimeError(
                "Strict global metrics cannot be combined with --no-global-metrics. "
                "Remove --no-global-metrics or disable strict mode."
            )
        if args.use_chooser and args.reranker_predictions:
            raise RuntimeError("--use-chooser cannot be combined with --reranker-predictions.")
        if args.use_policy_choice and args.reranker_predictions:
            raise RuntimeError("--use-policy-choice cannot be combined with --reranker-predictions.")

        logging.info(
            "Evaluating dataset=%s min_occurrence=%s | encoding=%s model=%s",
            model_cfg.dataset_variant,
            model_cfg.min_occurrence,
            model_cfg.encoding,
            model_cfg.model,
        )

        variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
        base_path = Path("data/processed") / variant
        test_graph_path = base_path / graph_dataset_filename(
            "test",
            model_cfg.encoding,
            constraint_representation=model_cfg.constraint_representation,
        )
        test_data = load_split(
            base_path,
            model_cfg.encoding,
            "test",
            constraint_representation=model_cfg.constraint_representation,
        )
        test_graph_count = _dataset_graph_count(test_data, test_graph_path)

        postprocess_fns: list[Callable[[torch.Tensor, torch.Tensor, list[str]], None]] = []
        postprocess_states: list[dict[str, object]] = []
        chooser_support: ChooserSupport | None = None
        direct_safety_support: DirectSafetySupport | None = None
        policy_support: PolicySupport | None = None

        repair_support = _maybe_prepare_repair_support(
            model_cfg.dataset_variant,
            model_cfg.min_occurrence,
            split="test",
            none_class=NONE_CLASS_INDEX,
        )
        if repair_support:
            repair_postprocess, repair_state = repair_support.build_postprocess()
            postprocess_fns.append(repair_postprocess)
            postprocess_states.append(repair_state)

        if args.use_chooser:
            if not training_cfg.chooser.enabled:
                raise RuntimeError("Chooser evaluation requested but chooser is disabled in training config.")
            interim_base = Path("data/interim") / dataset_variant_name(
                model_cfg.dataset_variant, model_cfg.min_occurrence
            )
            encoder_path = interim_base / "globalintencoder.txt"
            if not encoder_path.exists():
                raise FileNotFoundError(f"Global encoder not found at {encoder_path}")
            encoder = GlobalIntEncoder()
            encoder.load(encoder_path)
            encoder.freeze()
            placeholder_ids = load_placeholder_ids(encoder_path)
            heuristics = ConstraintRepairHeuristics(
                encoder=encoder,
                placeholder_ids=placeholder_ids,
                none_class=NONE_CLASS_INDEX,
            )
            contexts = load_violation_contexts(interim_base, "test", none_class=NONE_CLASS_INDEX)
            if test_graph_count is not None and len(contexts) != test_graph_count:
                raise RuntimeError("Mismatch between test graphs and violation contexts for chooser evaluation.")
            if isinstance(test_data, list):
                for idx, graph in enumerate(test_data):
                    setattr(graph, "context_index", idx)
            candidate_cfg = CandidateConfig(
                topk_candidates=training_cfg.chooser.topk_candidates,
                max_candidates_total=training_cfg.chooser.max_candidates_total,
                include_gold=False,
            )
            chooser_support = ChooserSupport(
                contexts=contexts,
                heuristics=heuristics,
                candidate_cfg=candidate_cfg,
            )

        if training_cfg.direct_safety.enabled and not args.reranker_predictions:
            if args.use_chooser:
                raise RuntimeError("Direct safety evaluation cannot be combined with chooser evaluation.")
            if args.use_policy_choice:
                raise RuntimeError("Direct safety evaluation cannot be combined with policy-choice evaluation.")
            interim_base = Path("data/interim") / dataset_variant_name(
                model_cfg.dataset_variant, model_cfg.min_occurrence
            )
            encoder_path = interim_base / "globalintencoder.txt"
            if not encoder_path.exists():
                raise FileNotFoundError(f"Global encoder not found at {encoder_path}")
            encoder = GlobalIntEncoder()
            encoder.load(encoder_path)
            encoder.freeze()
            placeholder_ids = load_placeholder_ids(encoder_path)
            heuristics = ConstraintRepairHeuristics(
                encoder=encoder,
                placeholder_ids=placeholder_ids,
                none_class=NONE_CLASS_INDEX,
            )
            contexts = load_violation_contexts(interim_base, "test", none_class=NONE_CLASS_INDEX)
            if test_graph_count is not None and len(contexts) != test_graph_count:
                raise RuntimeError("Mismatch between test graphs and violation contexts for direct safety evaluation.")
            if isinstance(test_data, list):
                for idx, graph in enumerate(test_data):
                    if not hasattr(graph, "context_index"):
                        setattr(graph, "context_index", idx)
            candidate_cfg = CandidateConfig(
                topk_candidates=training_cfg.direct_safety.topk_candidates,
                max_candidates_total=training_cfg.direct_safety.max_candidates_total,
                include_gold=False,
            )
            direct_safety_support = DirectSafetySupport(
                contexts=contexts,
                heuristics=heuristics,
                candidate_cfg=candidate_cfg,
            )

        if args.use_policy_choice:
            if not model_cfg.enable_policy_choice:
                raise RuntimeError(
                    "Policy choice evaluation requested but policy choice is disabled in model config."
                )
            interim_base = Path("data/interim") / dataset_variant_name(
                model_cfg.dataset_variant, model_cfg.min_occurrence
            )
            encoder_path = interim_base / "globalintencoder.txt"
            if not encoder_path.exists():
                raise FileNotFoundError(f"Global encoder not found at {encoder_path}")
            encoder = GlobalIntEncoder()
            encoder.load(encoder_path)
            encoder.freeze()
            placeholder_ids = load_placeholder_ids(encoder_path)
            heuristics = ConstraintRepairHeuristics(
                encoder=encoder,
                placeholder_ids=placeholder_ids,
                none_class=NONE_CLASS_INDEX,
            )
            contexts = load_violation_contexts(interim_base, "test", none_class=NONE_CLASS_INDEX)
            if test_graph_count is not None and len(contexts) != test_graph_count:
                raise RuntimeError("Mismatch between test graphs and violation contexts for policy choice.")
            if isinstance(test_data, list):
                for idx, graph in enumerate(test_data):
                    setattr(graph, "context_index", idx)
            candidate_cfg = CandidateConfig(include_gold=False)
            policy_support = PolicySupport(
                contexts=contexts,
                heuristics=heuristics,
                candidate_cfg=candidate_cfg,
                filter_strict=training_cfg.policy_filter_strict,
            )

        global_support = None
        if strict_global and not _data_has_factor_fields(test_data):
            raise RuntimeError(
                "Strict global metrics require factor fields on test graphs. "
                "Rebuild graphs with factor labels (factor_* fields) or disable strict mode."
            )
        global_metrics_enabled = (
            True
            if strict_global
            else (not args.no_global_metrics) and (args.global_metrics or _data_has_factor_fields(test_data))
        )
        if global_metrics_enabled:
            global_support = _maybe_prepare_global_support(
                model_cfg.dataset_variant,
                model_cfg.min_occurrence,
                split="test",
                none_class=NONE_CLASS_INDEX,
                strict=strict_global,
                registry_dataset=args.registry_dataset,
            )
        if global_support:
            global_postprocess, global_state = global_support.build_postprocess(test_data)
            postprocess_fns.append(global_postprocess)
            postprocess_states.append(global_state)
            _smoke_check_global_metrics(global_support, test_data, none_class=NONE_CLASS_INDEX)
        elif global_metrics_enabled:
            if strict_global:
                raise RuntimeError(
                    "Strict global metrics requested but could not be prepared. "
                    "Verify registry/encoder availability and factor fields."
                )
            logging.warning("Global metrics requested but could not be prepared; skipping.")

        postprocess = None
        postprocess_state: dict[str, object] | None = None
        if postprocess_fns:
            combined_state: dict[str, object] = {}

            def postprocess(predictions: torch.Tensor, targets: torch.Tensor, kinds: list[str]) -> None:
                for fn in postprocess_fns:
                    fn(predictions, targets, kinds)
                for state in postprocess_states:
                    combined_state.update(state)

            postprocess_state = combined_state

        precomputed_predictions = None
        if args.reranker_predictions:
            precomputed_predictions = _load_reranker_predictions(Path(args.reranker_predictions))
            logging.info("Loaded reranker predictions from %s", args.reranker_predictions)

        evaluate_trained_model(
            run_directory=run_directory,
            model_cfg=model_cfg,
            device=device,
            test_data=test_data,
            postprocess=postprocess,
            postprocess_state=postprocess_state,
            precomputed_predictions=precomputed_predictions,
            write_per_constraint_csv=True if strict_global else (args.per_constraint_csv or global_metrics_enabled),
            chooser_support=chooser_support,
            direct_safety_support=direct_safety_support,
            policy_support=policy_support,
            selection_weights=selection_weights,
            selection_disruption_field=args.score_disruption_field,
        )

    else:  # Evaluate baselines
        if args.use_chooser:
            raise RuntimeError("--use-chooser is only supported for trained model evaluation.")
        if args.use_policy_choice:
            raise RuntimeError("--use-policy-choice is only supported for trained model evaluation.")
        logging.info(
            "Evaluating baselines dataset=%s min_occurrence=%s using interim parquet data",
            args.dataset,
            args.min_occurrence,
        )
        variant = dataset_variant_name(args.dataset, args.min_occurrence)
        base_path, encoder_path = _resolve_baseline_interim_paths(args.dataset, args.min_occurrence)
        logging.info("Baseline parquet source: %s", base_path)

        train_data, train_max = load_baseline_split_from_parquet(base_path, "train")
        test_data, test_max = load_baseline_split_from_parquet(base_path, "test")
        num_graph_nodes = max(train_max, test_max, NONE_CLASS_INDEX) + 1

        placeholder_ids = load_placeholder_ids(encoder_path)

        repair_support = _maybe_prepare_repair_support(
            args.dataset,
            args.min_occurrence,
            split="test",
            none_class=NONE_CLASS_INDEX,
        )

        global_support = None
        strict_global = bool(args.strict_global_metrics)
        if strict_global and args.no_global_metrics:
            raise RuntimeError(
                "Strict global metrics cannot be combined with --no-global-metrics. "
                "Remove --no-global-metrics or disable strict mode."
            )
        global_metrics_enabled = True if strict_global else not args.no_global_metrics
        if global_metrics_enabled:
            global_support = _maybe_prepare_global_support(
                args.dataset,
                args.min_occurrence,
                split="test",
                none_class=NONE_CLASS_INDEX,
                strict=strict_global,
                registry_dataset=args.registry_dataset,
            )
            if not global_support:
                if strict_global:
                    raise RuntimeError(
                        "Strict global metrics requested but could not be prepared. "
                        "Verify registry/encoder availability and factor fields."
                    )
                logging.warning("Global metrics requested but could not be prepared; skipping.")

        repair_builder = None
        if repair_support or global_support:

            def repair_builder():
                postprocess_fns: list[Callable[[torch.Tensor, torch.Tensor, list[str]], None]] = []
                postprocess_states: list[dict[str, object]] = []
                if repair_support:
                    fn, state = repair_support.build_postprocess()
                    postprocess_fns.append(fn)
                    postprocess_states.append(state)
                if global_support:
                    fn, state = global_support.build_postprocess(test_data)
                    postprocess_fns.append(fn)
                    postprocess_states.append(state)
                combined_state: dict[str, object] = {}

                def postprocess(predictions: torch.Tensor, targets: torch.Tensor, kinds: list[str]) -> None:
                    for fn in postprocess_fns:
                        fn(predictions, targets, kinds)
                    for state in postprocess_states:
                        combined_state.update(state)

                return postprocess, combined_state

        def save_run(name: str, model: BaseGraphModel) -> dict[str, object]:
            postprocess = None
            state: dict[str, object] | None = None
            if repair_builder:
                postprocess, state = repair_builder()
            metrics = eval(model, test_data, device=device, postprocess=postprocess)
            if state and "repair_metrics" in state:
                metrics["repair_metrics"] = state["repair_metrics"]
            if state and "global_metrics" in state:
                metrics["global_metrics"] = state["global_metrics"]
            if state and "global_metrics_per_constraint_type" in state:
                metrics["global_metrics_per_constraint_type"] = state["global_metrics_per_constraint_type"]
            if "global_metrics" in metrics and isinstance(metrics["global_metrics"], dict):
                overall = metrics["global_metrics"].get("overall")
                if isinstance(overall, dict):
                    metrics["overall_gfr"] = overall.get("gfr", 0.0)
                    metrics["overall_srr"] = overall.get("srr", 0.0)
                    metrics["overall_sir"] = overall.get("sir", 0.0)
                    disruption = overall.get("disruption", {}) if isinstance(overall.get("disruption"), dict) else {}
                    metrics["mean_disruption_add"] = disruption.get("added_triples_mean", 0.0)
                    metrics["mean_disruption_del"] = disruption.get("deleted_triples_mean", 0.0)
                    metrics["mean_disruption_total"] = disruption.get("total_ops_mean", 0.0)
            output_root = baseline_dir(args.dataset, "parquet", create=True)
            output_root.mkdir(parents=True, exist_ok=True)
            output_path = output_root / f"{name}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                selection_block = compute_model_selection_block(
                    metrics,
                    weights=selection_weights,
                    disruption_field=args.score_disruption_field,
                )
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("Model selection block: %s", selection_block)
                payload = {"model_selection": selection_block}
                payload.update(metrics)
                json.dump(payload, handle, indent=4)
            if strict_global or args.per_constraint_csv or global_metrics_enabled:
                _write_per_constraint_csv(metrics, output_root / name)
            return metrics

        evaluate_baselines(
            baseline_choice="all",
            dataset=args.dataset,
            encoding="parquet",
            num_graph_nodes=num_graph_nodes,
            default_add_class=NONE_CLASS_INDEX,
            default_del_class=NONE_CLASS_INDEX,
            inverse_map_json=None,
            symmetric_set_json=None,
            fit_csm_on_train=True,
            train_data=train_data,
            device=device,
            save_run=save_run,
            results_dir=None,
            placeholders=placeholder_ids,
        )


if __name__ == "__main__":
    main()
