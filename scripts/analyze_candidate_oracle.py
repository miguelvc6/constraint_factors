#!/usr/bin/env python3
"""Analyze candidate-set oracle headroom for a trained proposal run.

The script compares the model-selected edit against the best candidate available
inside the same generated candidate set. It does not train or mutate model
artifacts; outputs are written under the requested oracle directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.autonotebook import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.candidates import CandidateConfig, build_candidates, score_candidates_from_logits
from modules.config import ModelConfig, TrainingConfig
from modules.data_encoders import dataset_variant_name, graph_dataset_filename
from modules.model_store import config_copy_path
from modules.training_utils import load_graph_dataset

NONE_CLASS_INDEX = 0


def _load_eval_module():
    module_path = ROOT / "src" / "09_eval.py"
    spec = importlib.util.spec_from_file_location("eval_09_for_candidate_oracle", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load eval helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate-oracle headroom for a trained run.")
    parser.add_argument("--run-directory", required=True, help="Trained run directory under models/.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-directory>/evaluations/oracle.",
    )
    parser.add_argument(
        "--strict-global-metrics",
        action="store_true",
        help="Require strict symbolic evaluator setup, matching paper-suite evaluation.",
    )
    parser.add_argument(
        "--registry-dataset",
        default="full",
        help="Registry dataset fallback for strict symbolic evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix length for smoke tests.")
    parser.add_argument(
        "--max-safe-disruption",
        type=int,
        default=2,
        help="Maximum add+delete operation count for the safe-oracle availability flag.",
    )
    parser.add_argument(
        "--examples-limit",
        type=int,
        default=200,
        help="Maximum number of oracle gap examples to write.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _load_config(run_directory: Path) -> tuple[ModelConfig, TrainingConfig]:
    config_path = config_copy_path(run_directory)
    if not config_path.exists():
        raise FileNotFoundError(f"Stored configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return (
        ModelConfig.from_mapping(payload["model_config"]),
        TrainingConfig.from_mapping(payload.get("training_config", {})),
    )


def _load_test_data(model_cfg: ModelConfig):
    variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    base_path = ROOT / "data" / "processed" / variant
    graph_path = base_path / graph_dataset_filename(
        "test",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )
    return load_graph_dataset(graph_path)


def _set_context_indices(test_data: Any) -> None:
    if isinstance(test_data, list):
        for idx, graph in enumerate(test_data):
            setattr(graph, "context_index", idx)


def _candidate_config(training_cfg: TrainingConfig) -> CandidateConfig:
    if training_cfg.chooser.enabled:
        return CandidateConfig(
            topk_candidates=training_cfg.chooser.topk_candidates,
            max_candidates_total=training_cfg.chooser.max_candidates_total,
            include_gold=False,
        )
    if training_cfg.direct_safety.enabled:
        return CandidateConfig(
            topk_candidates=training_cfg.direct_safety.topk_candidates,
            max_candidates_total=training_cfg.direct_safety.max_candidates_total,
            include_gold=False,
        )
    return CandidateConfig(include_gold=False)


def _density_bucket(size: int) -> str:
    if size <= 0:
        return "0"
    if size == 1:
        return "1"
    if size <= 4:
        return "2_4"
    if size <= 16:
        return "5_16"
    if size <= 64:
        return "17_64"
    return "65_plus"


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).detach().cpu().tolist()
    if isinstance(value, str):
        return [part for part in value.split("|") if part]
    try:
        if isinstance(value, float) and math.isnan(value):
            return []
    except Exception:
        pass
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _row_context(row: Any) -> tuple[str, str, int]:
    constraint_type = getattr(row, "constraint_type", None) or "UNKNOWN"
    local_ids = _as_sequence(getattr(row, "local_constraint_ids", None))
    if not local_ids:
        local_ids = _as_sequence(getattr(row, "factor_constraint_ids", None))
    constraint_id = _as_int(getattr(row, "constraint_id", None), default=-1)
    return str(constraint_type), _density_bucket(len(local_ids)), constraint_id


def _primary_factor_index(graph: Data, row: Any) -> int:
    graph_value = getattr(graph, "primary_factor_index", None)
    if graph_value is not None:
        return _as_int(graph_value, default=-1)
    return _as_int(getattr(row, "primary_factor_index", None), default=-1)


def _total_ops(metrics: dict[str, Any]) -> int:
    return int(metrics.get("add_count", 0)) + int(metrics.get("del_count", 0))


def _safe_flag(metrics: dict[str, Any], *, pre_gfr: float, max_disruption: int) -> bool:
    return (
        int(metrics.get("primary_satisfied", 0)) == 1
        and int(metrics.get("secondary_regressions", 0)) == 0
        and float(metrics.get("global_satisfied_fraction", 0.0)) >= pre_gfr
        and _total_ops(metrics) <= max_disruption
    )


def _pre_gfr(details: dict[str, Any]) -> float:
    pre_checkable = details.get("pre_checkable") or []
    pre_satisfied = details.get("pre_satisfied") or []
    denom = sum(1 for flag in pre_checkable if flag)
    if not denom:
        return 0.0
    return float(sum(int(v) for v, flag in zip(pre_satisfied, pre_checkable) if flag)) / denom


def _oracle_index(details: list[dict[str, Any]], scores: Sequence[float]) -> int:
    if not details:
        raise ValueError("Cannot choose an oracle candidate from an empty candidate set.")

    def key(idx: int) -> tuple[float, float, float, float, float, float]:
        item = details[idx]
        return (
            float(item.get("primary_satisfied", 0)),
            1.0 if int(item.get("secondary_regressions", 0)) == 0 else 0.0,
            float(item.get("global_satisfied_fraction", 0.0)),
            -float(_total_ops(item)),
            -float(item.get("srr", 0.0)),
            float(scores[idx]) if idx < len(scores) else 0.0,
        )

    return max(range(len(details)), key=key)


def _slot_list(slots: Sequence[int]) -> list[int]:
    return [int(v) for v in slots]


def _selected_from_argmax(logits: torch.Tensor) -> list[int]:
    return [int(v) for v in torch.argmax(logits, dim=-1).detach().cpu().tolist()]


def _selected_from_candidates(
    *,
    model: Any,
    graph_emb: torch.Tensor | None,
    logits: torch.Tensor,
    candidates: list[tuple[int, int, int, int, int, int]],
    candidate_scores: torch.Tensor,
    training_cfg: TrainingConfig,
) -> tuple[list[int], int | None]:
    if training_cfg.chooser.enabled:
        if graph_emb is None:
            raise RuntimeError("Chooser run requires graph_emb from model outputs.")
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
        scores = model.score_candidates(graph_emb, candidate_tensor)
        best_idx = int(torch.argmax(scores).item())
        return list(candidates[best_idx]), best_idx
    if training_cfg.direct_safety.enabled:
        best_idx = int(torch.argmax(candidate_scores).item())
        return list(candidates[best_idx]), best_idx
    return _selected_from_argmax(logits), None


class Aggregate:
    def __init__(self) -> None:
        self.support = 0
        self.candidate_nonempty = 0
        self.safe_available = 0
        self.selected_safe = 0
        self.oracle_primary = 0
        self.selected_primary = 0
        self.oracle_gfr_sum = 0.0
        self.selected_gfr_sum = 0.0
        self.oracle_regressions = 0
        self.oracle_regressions_denom = 0
        self.selected_regressions = 0
        self.selected_regressions_denom = 0
        self.oracle_improvements = 0
        self.oracle_improvements_denom = 0
        self.selected_improvements = 0
        self.selected_improvements_denom = 0
        self.oracle_disruption_sum = 0
        self.selected_disruption_sum = 0
        self.candidate_count_sum = 0

    def add(
        self,
        *,
        candidate_count: int,
        oracle: dict[str, Any],
        selected: dict[str, Any],
        oracle_safe: bool,
        selected_safe: bool,
    ) -> None:
        self.support += 1
        self.candidate_nonempty += int(candidate_count > 0)
        self.safe_available += int(oracle_safe)
        self.selected_safe += int(selected_safe)
        self.oracle_primary += int(oracle.get("primary_satisfied", 0))
        self.selected_primary += int(selected.get("primary_satisfied", 0))
        self.oracle_gfr_sum += float(oracle.get("global_satisfied_fraction", 0.0))
        self.selected_gfr_sum += float(selected.get("global_satisfied_fraction", 0.0))
        self.oracle_regressions += int(oracle.get("secondary_regressions", 0))
        self.oracle_regressions_denom += int(oracle.get("secondary_regressions_denom", 0))
        self.selected_regressions += int(selected.get("secondary_regressions", 0))
        self.selected_regressions_denom += int(selected.get("secondary_regressions_denom", 0))
        self.oracle_improvements += int(oracle.get("secondary_improvements", 0))
        self.oracle_improvements_denom += int(oracle.get("secondary_improvements_denom", 0))
        self.selected_improvements += int(selected.get("secondary_improvements", 0))
        self.selected_improvements_denom += int(selected.get("secondary_improvements_denom", 0))
        self.oracle_disruption_sum += _total_ops(oracle)
        self.selected_disruption_sum += _total_ops(selected)
        self.candidate_count_sum += int(candidate_count)

    def to_dict(self) -> dict[str, float | int]:
        support = self.support
        return {
            "support": support,
            "candidate_set_nonempty_rate": self.candidate_nonempty / support if support else 0.0,
            "candidate_count_mean": self.candidate_count_sum / support if support else 0.0,
            "oracle_safe_available_rate": self.safe_available / support if support else 0.0,
            "selected_safe_rate": self.selected_safe / support if support else 0.0,
            "oracle_primary_fix_rate": self.oracle_primary / support if support else 0.0,
            "selected_primary_fix_rate": self.selected_primary / support if support else 0.0,
            "oracle_gfr": self.oracle_gfr_sum / support if support else 0.0,
            "selected_gfr": self.selected_gfr_sum / support if support else 0.0,
            "oracle_srr": self.oracle_regressions / self.oracle_regressions_denom
            if self.oracle_regressions_denom
            else 0.0,
            "selected_srr": self.selected_regressions / self.selected_regressions_denom
            if self.selected_regressions_denom
            else 0.0,
            "oracle_sir": self.oracle_improvements / self.oracle_improvements_denom
            if self.oracle_improvements_denom
            else 0.0,
            "selected_sir": self.selected_improvements / self.selected_improvements_denom
            if self.selected_improvements_denom
            else 0.0,
            "oracle_mean_disruption": self.oracle_disruption_sum / support if support else 0.0,
            "selected_mean_disruption": self.selected_disruption_sum / support if support else 0.0,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_value(metrics: dict[str, Any], key: str) -> float | int:
    if key == "disruption":
        return _total_ops(metrics)
    value = metrics.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return 0


@torch.no_grad()
def run_analysis(args: argparse.Namespace) -> None:
    run_directory = Path(args.run_directory).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_directory / "evaluations" / "oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg, training_cfg = _load_config(run_directory)
    test_data = _load_test_data(model_cfg)
    _set_context_indices(test_data)

    repair_support = EVAL._maybe_prepare_repair_support(
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        split="test",
        none_class=NONE_CLASS_INDEX,
    )
    global_support = EVAL._maybe_prepare_global_support(
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        split="test",
        none_class=NONE_CLASS_INDEX,
        strict=bool(args.strict_global_metrics),
        registry_dataset=args.registry_dataset,
    )
    if global_support is None:
        raise RuntimeError("Candidate-oracle analysis requires symbolic global support.")

    device = EVAL.get_device()
    chooser_support = object() if training_cfg.chooser.enabled else None
    model = EVAL.load_trained_model_for_eval(
        run_directory=run_directory,
        model_cfg=model_cfg,
        device=device,
        chooser_support=chooser_support,
    )
    model.eval()

    candidate_cfg = _candidate_config(training_cfg)
    placeholder_ids = set(repair_support.heuristics.placeholder_ids.values())
    rows = global_support.rows
    contexts = repair_support.contexts

    overall = Aggregate()
    by_density: dict[str, Aggregate] = defaultdict(Aggregate)
    by_type: dict[str, Aggregate] = defaultdict(Aggregate)
    examples: list[dict[str, Any]] = []
    constraint_counter: Counter[str] = Counter()

    loader = DataLoader(test_data, batch_size=args.batch_size)
    processed = 0
    for batch in tqdm(loader, desc="Candidate oracle"):
        graphs = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
        batch = batch.to(device)
        out = model(batch)
        if isinstance(out, dict):
            logits_batch = out.get("edit_logits")
            graph_emb_batch = out.get("graph_emb")
            if logits_batch is None:
                raise KeyError("Model output dict missing edit_logits.")
        else:
            logits_batch = out
            graph_emb_batch = None

        for local_idx, graph in enumerate(graphs):
            if args.limit is not None and processed >= args.limit:
                break
            context_index = int(getattr(graph, "context_index", processed))
            if context_index >= len(contexts) or context_index >= len(rows):
                raise RuntimeError(f"Context index {context_index} out of bounds.")
            context = contexts[context_index]
            row = rows[context_index]
            logits = logits_batch[local_idx].detach()
            candidates, _gold_index = build_candidates(
                graph=graph,
                context=context,
                heuristics=repair_support.heuristics,
                proposal_logits=logits,
                cfg=candidate_cfg,
                placeholder_ids=placeholder_ids,
                num_target_ids=model.num_target_ids,
            )
            if not candidates:
                processed += 1
                continue

            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
            candidate_scores_tensor = score_candidates_from_logits(logits, candidate_tensor)
            candidate_scores = [float(v) for v in candidate_scores_tensor.detach().cpu().tolist()]
            selected_slots, selected_candidate_index = _selected_from_candidates(
                model=model,
                graph_emb=graph_emb_batch[local_idx] if graph_emb_batch is not None else None,
                logits=logits,
                candidates=candidates,
                candidate_scores=candidate_scores_tensor,
                training_cfg=training_cfg,
            )

            primary_index = _primary_factor_index(graph, row)
            candidate_details = global_support.evaluator.evaluate_candidates(
                row,
                candidates=candidates,
                primary_factor_index=primary_index,
            )
            selected_detail = global_support.evaluator.evaluate_candidates(
                row,
                candidates=[selected_slots],
                primary_factor_index=primary_index,
            )[0]
            oracle_candidate_index = _oracle_index(candidate_details, candidate_scores)
            oracle_detail = candidate_details[oracle_candidate_index]
            pre_gfr = _pre_gfr(oracle_detail)
            oracle_safe = _safe_flag(
                oracle_detail,
                pre_gfr=pre_gfr,
                max_disruption=args.max_safe_disruption,
            )
            selected_safe = _safe_flag(
                selected_detail,
                pre_gfr=pre_gfr,
                max_disruption=args.max_safe_disruption,
            )

            constraint_type, density_bucket, constraint_id = _row_context(row)
            constraint_counter[constraint_type] += 1
            overall.add(
                candidate_count=len(candidates),
                oracle=oracle_detail,
                selected=selected_detail,
                oracle_safe=oracle_safe,
                selected_safe=selected_safe,
            )
            by_density[density_bucket].add(
                candidate_count=len(candidates),
                oracle=oracle_detail,
                selected=selected_detail,
                oracle_safe=oracle_safe,
                selected_safe=selected_safe,
            )
            by_type[constraint_type].add(
                candidate_count=len(candidates),
                oracle=oracle_detail,
                selected=selected_detail,
                oracle_safe=oracle_safe,
                selected_safe=selected_safe,
            )

            selected_worse = (
                oracle_safe
                and not selected_safe
                or float(oracle_detail.get("global_satisfied_fraction", 0.0))
                > float(selected_detail.get("global_satisfied_fraction", 0.0))
                or int(oracle_detail.get("primary_satisfied", 0)) > int(selected_detail.get("primary_satisfied", 0))
            )
            if selected_worse and len(examples) < args.examples_limit:
                examples.append(
                    {
                        "index": context_index,
                        "constraint_id": constraint_id,
                        "constraint_type": constraint_type,
                        "density_bucket": density_bucket,
                        "candidate_count": len(candidates),
                        "selected_candidate_index": selected_candidate_index
                        if selected_candidate_index is not None
                        else "",
                        "oracle_candidate_index": oracle_candidate_index,
                        "selected_slots": json.dumps(_slot_list(selected_slots)),
                        "oracle_slots": json.dumps(_slot_list(candidates[oracle_candidate_index])),
                        "selected_primary_satisfied": selected_detail.get("primary_satisfied", 0),
                        "oracle_primary_satisfied": oracle_detail.get("primary_satisfied", 0),
                        "selected_gfr": selected_detail.get("global_satisfied_fraction", 0.0),
                        "oracle_gfr": oracle_detail.get("global_satisfied_fraction", 0.0),
                        "selected_srr": selected_detail.get("srr", 0.0),
                        "oracle_srr": oracle_detail.get("srr", 0.0),
                        "selected_regressions": selected_detail.get("secondary_regressions", 0),
                        "oracle_regressions": oracle_detail.get("secondary_regressions", 0),
                        "selected_disruption": _total_ops(selected_detail),
                        "oracle_disruption": _total_ops(oracle_detail),
                        "selected_safe": int(selected_safe),
                        "oracle_safe": int(oracle_safe),
                    }
                )

            processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    summary = {
        "run_directory": str(run_directory),
        "dataset_variant": model_cfg.dataset_variant,
        "min_occurrence": model_cfg.min_occurrence,
        "encoding": model_cfg.encoding,
        "constraint_representation": model_cfg.constraint_representation,
        "selection_mode": "chooser"
        if training_cfg.chooser.enabled
        else ("direct_safety" if training_cfg.direct_safety.enabled else "slot_argmax"),
        "candidate_config": {
            "topk_candidates": candidate_cfg.topk_candidates,
            "topk_per_slot": candidate_cfg.topk_per_slot,
            "heuristic_max_candidates": candidate_cfg.heuristic_max_candidates,
            "heuristic_max_values": candidate_cfg.heuristic_max_values,
            "include_gold": candidate_cfg.include_gold,
            "max_candidates_total": candidate_cfg.max_candidates_total,
        },
        "max_safe_disruption": args.max_safe_disruption,
        "constraint_type_support": dict(sorted(constraint_counter.items())),
        "overall": overall.to_dict(),
    }
    with (output_dir / "oracle_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    slice_fields = ["slice", *overall.to_dict().keys()]
    density_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_density.items())]
    type_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_type.items())]
    _write_csv(output_dir / "oracle_by_density.csv", density_rows, slice_fields)
    _write_csv(output_dir / "oracle_by_constraint_type.csv", type_rows, slice_fields)
    example_fields = [
        "index",
        "constraint_id",
        "constraint_type",
        "density_bucket",
        "candidate_count",
        "selected_candidate_index",
        "oracle_candidate_index",
        "selected_slots",
        "oracle_slots",
        "selected_primary_satisfied",
        "oracle_primary_satisfied",
        "selected_gfr",
        "oracle_gfr",
        "selected_srr",
        "oracle_srr",
        "selected_regressions",
        "oracle_regressions",
        "selected_disruption",
        "oracle_disruption",
        "selected_safe",
        "oracle_safe",
    ]
    _write_csv(output_dir / "oracle_examples.csv", examples, example_fields)
    logging.info("Wrote candidate-oracle outputs to %s", output_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(message)s")
    run_analysis(args)


if __name__ == "__main__":
    main()
