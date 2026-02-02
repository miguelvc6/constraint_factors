#!/usr/bin/env python3
"""Evaluate a trained graph model or run baselines stored under ``models/``.

Trained model evaluations load graphs from ``data/processed/<variant>/test_graph-<encoding>.pkl``
based on the settings stored alongside the run directory (same structure as ``04_train.py``).
Baseline evaluations operate on the lighter parquet splits in ``data/interim/<variant>/`` to
avoid materialising heavyweight pickle files in memory.

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.autonotebook import tqdm

from modules.baselines import evaluate_baselines
from modules.config import ModelConfig
from modules.data_encoders import GlobalIntEncoder
from modules.model_store import config_copy_path, evaluations_dir, get_checkpoint_path
from modules.models import BaseGraphModel, build_model
from modules.repair_eval import (
    ConstraintRepairHeuristics,
    RepairSample,
    ViolationContext,
    evaluate_repair_samples,
    load_violation_contexts,
)

NONE_CLASS_INDEX = 0  # Bass-style datasets reserve class 0 for "no triple"
ACTIONS = ("add", "del")

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
    model: BaseGraphModel,
    test_data: list[Data],
    batch_size: int = 16,
    device: torch.device | str = torch.device("cpu"),
    none_class: int = NONE_CLASS_INDEX,
    postprocess: Callable[[torch.Tensor, torch.Tensor, list[str]], None] | None = None,
) -> dict[str, object]:
    """Evaluate a model and return Bass-style precision/recall/F1 metrics."""
    if isinstance(device, str):
        device = torch.device(device)

    test_loader = DataLoader(test_data, batch_size=batch_size)

    kinds: list[str] = [(getattr(d, "constraint_type", None) or "UNKNOWN") for d in test_data]

    model.eval()
    predictions, targets = [], []
    for data in tqdm(test_loader, desc="Test Batches"):
        data = data.to(device)
        out = model(data)  # raw logits expected
        out = out.argmax(dim=-1)  # class predictions per action
        predictions.append(out.cpu())
        targets.append(data.y.cpu())

    predictions = torch.cat(predictions, dim=0)
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
) -> dict[str, object]:
    metrics = eval(model, test_data, device=device, postprocess=postprocess)
    if postprocess_state and "repair_metrics" in postprocess_state:
        metrics["repair_metrics"] = postprocess_state["repair_metrics"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "model.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    return metrics


# Small helpers to keep main() clean
def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {dev}")
    return dev


def load_split(base_path: Path, encoding: str, split: str) -> list[Data]:
    """Load a dataset split saved as either a list or a pickled stream of ``Data`` objects."""
    path = base_path / f"{split}_graph-{encoding}.pkl"

    with path.open("rb") as f:
        first_object = pickle.load(f)

        if isinstance(first_object, list):
            return first_object
        if isinstance(first_object, Data):
            graphs: list[Data] = [first_object]
            while True:
                try:
                    graphs.append(pickle.load(f))
                except EOFError:
                    break
            return graphs

    raise TypeError(
        "Unsupported dataset format encountered. Expected a list of Data objects or a pickled Data stream."
    )


def _safe_constraint_type(value: object) -> str:
    """Normalize constraint type values coming from parquet rows."""
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float) and math.isnan(value):
        return "UNKNOWN"
    return str(value)


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

        data_list.append(graph)

        y_max = int(y.max().item()) if y.numel() else NONE_CLASS_INDEX
        focus_max = int(focus_triple.max().item()) if focus_triple.numel() else NONE_CLASS_INDEX
        current_max = max(y_max, focus_max)
        if current_max > max_index:
            max_index = current_max

    del dataframe
    return data_list, max_index


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


# Load and run a trained torch model
def evaluate_trained_model(
    *,
    run_directory: Path,
    model_cfg: ModelConfig,
    device: torch.device,
    test_data: list[Data],
    postprocess: Callable[[torch.Tensor, torch.Tensor, list[str]], None] | None = None,
    postprocess_state: dict[str, object] | None = None,
) -> None:
    checkpoint_path = get_checkpoint_path(run_directory)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained model not found at {checkpoint_path}")

    logging.info("Using model artifacts in %s", run_directory)
    logging.info("Loading checkpoint from %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, BaseGraphModel):
        model = checkpoint.to(device)
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
        model.load_state_dict(state_dict)
        model.to(device)
        printable_config = {k: v for k, v in effective_model_cfg.to_dict().items() if k != "entity_class_ids"}
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
        choices=["sample", "full"],
        help="Dataset variant (baselines mode).",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        force=True,
    )

    if args.run_baselines:
        if not args.dataset:
            raise ValueError("--run-baselines requires --dataset.")
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

    if not args.run_baselines:
        run_directory = args.run_directory
        config_path = config_copy_path(run_directory)
        if not config_path.exists():
            raise FileNotFoundError(f"Stored configuration file not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            experiment_config = json.load(f)

        model_cfg = ModelConfig.from_mapping(experiment_config["model_config"])

        logging.info(
            "Evaluating dataset=%s min_occurrence=%s | encoding=%s model=%s",
            model_cfg.dataset_variant,
            model_cfg.min_occurrence,
            model_cfg.encoding,
            model_cfg.model,
        )

        base_path = Path("data/processed") / f"{model_cfg.dataset_variant}_minocc{model_cfg.min_occurrence}"
        test_data = load_split(base_path, model_cfg.encoding, "test")

        repair_postprocess = None
        repair_state: dict[str, object] | None = None
        repair_support = _maybe_prepare_repair_support(
            model_cfg.dataset_variant,
            model_cfg.min_occurrence,
            split="test",
            none_class=NONE_CLASS_INDEX,
        )
        if repair_support:
            repair_postprocess, repair_state = repair_support.build_postprocess()

        evaluate_trained_model(
            run_directory=run_directory,
            model_cfg=model_cfg,
            device=device,
            test_data=test_data,
            postprocess=repair_postprocess,
            postprocess_state=repair_state,
        )

    else:  # Evaluate baselines
        logging.info(
            "Evaluating baselines dataset=%s min_occurrence=%s using interim parquet data",
            args.dataset,
            args.min_occurrence,
        )
        base_path = Path("data/interim") / f"{args.dataset}_minocc{args.min_occurrence}"

        train_data, train_max = load_baseline_split_from_parquet(base_path, "train")
        test_data, test_max = load_baseline_split_from_parquet(base_path, "test")
        num_graph_nodes = max(train_max, test_max, NONE_CLASS_INDEX) + 1

        placeholder_ids = load_placeholder_ids(base_path / "globalintencoder.txt")

        repair_support = _maybe_prepare_repair_support(
            args.dataset,
            args.min_occurrence,
            split="test",
            none_class=NONE_CLASS_INDEX,
        )

        repair_builder = None
        if repair_support:

            def repair_builder():
                return repair_support.build_postprocess()

        def save_run(name: str, model: BaseGraphModel) -> dict[str, object]:
            postprocess = None
            state: dict[str, object] | None = None
            if repair_builder:
                postprocess, state = repair_builder()
            metrics = eval(model, test_data, device=device, postprocess=postprocess)
            if state and "repair_metrics" in state:
                metrics["repair_metrics"] = state["repair_metrics"]
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
