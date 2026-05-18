#!/usr/bin/env python3
"""Audit whether G0 behaves like the delete-focus H1 baseline.

The script is read-only with respect to model artifacts. It compares reranker
predictions against the constructed H1 delete-focus policy and evaluates both
through the shared symbolic candidate evaluator.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig
from modules.model_store import config_copy_path

NONE_CLASS_INDEX = 0
EQUIVALENCE_TOLERANCE = 1e-12


def _load_eval_module():
    module_path = ROOT / "src" / "09_eval.py"
    spec = importlib.util.spec_from_file_location("eval_09_for_deletion_degeneracy", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load eval helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit G0 delete-focus degeneracy against H1.")
    parser.add_argument("--g0-run-directory", required=True, help="G0 run directory under models/.")
    parser.add_argument(
        "--reranker-predictions",
        default=None,
        help="Optional predictions path. Defaults to <g0-run-directory>/reranker_predictions.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <g0-run-directory>/evaluations/deletion_degeneracy.",
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
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix length for smoke tests.")
    parser.add_argument(
        "--examples-limit",
        type=int,
        default=200,
        help="Maximum number of mismatch examples to write.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _load_model_config(run_directory: Path) -> ModelConfig:
    config_path = config_copy_path(run_directory)
    if not config_path.exists():
        raise FileNotFoundError(f"Stored configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return ModelConfig.from_mapping(payload["model_config"])


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split("|") if part]
    if isinstance(value, Sequence):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    return [value]


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


def _row_context(row: Any) -> tuple[str, str, int]:
    constraint_type = getattr(row, "constraint_type", None) or "UNKNOWN"
    local_ids = _as_sequence(getattr(row, "local_constraint_ids", None))
    if not local_ids:
        local_ids = _as_sequence(getattr(row, "factor_constraint_ids", None))
    constraint_id = _as_int(getattr(row, "constraint_id", None), default=-1)
    return str(constraint_type), _density_bucket(len(local_ids)), constraint_id


def _primary_factor_index(row: Any) -> int | None:
    value = getattr(row, "primary_factor_index", None)
    if value is None:
        return None
    return _as_int(value, default=-1)


def _slots_from_prediction(item: Any) -> list[int]:
    if isinstance(item, dict):
        add = item.get("add") or [NONE_CLASS_INDEX, NONE_CLASS_INDEX, NONE_CLASS_INDEX]
        delete = item.get("del") or item.get("delete") or [NONE_CLASS_INDEX, NONE_CLASS_INDEX, NONE_CLASS_INDEX]
        return [int(v) for v in [*add, *delete]]
    values = list(item)
    if len(values) != 6:
        raise ValueError(f"Expected 6-slot prediction, got {len(values)} values.")
    return [int(v) for v in values]


def _h1_slots(row: Any) -> list[int]:
    return [
        NONE_CLASS_INDEX,
        NONE_CLASS_INDEX,
        NONE_CLASS_INDEX,
        _as_int(getattr(row, "subject", None)),
        _as_int(getattr(row, "predicate", None)),
        _as_int(getattr(row, "object", None)),
    ]


def _total_ops(metrics: dict[str, Any]) -> int:
    return int(metrics.get("add_count", 0)) + int(metrics.get("del_count", 0))


def _metric_equivalent(g0: dict[str, Any], h1: dict[str, Any]) -> bool:
    numeric_keys = ("primary_satisfied", "global_satisfied_fraction", "sir", "srr")
    for key in numeric_keys:
        if abs(float(g0.get(key, 0.0)) - float(h1.get(key, 0.0))) > EQUIVALENCE_TOLERANCE:
            return False
    if _total_ops(g0) != _total_ops(h1):
        return False
    for key in ("focus_preserved", "focus_deleted", "candidate_deletes_focus"):
        if int(g0.get(key, 0)) != int(h1.get(key, 0)):
            return False
    return True


class Aggregate:
    def __init__(self) -> None:
        self.support = 0
        self.prediction_exact = 0
        self.metric_equivalent = 0
        self.g0_focus_deleted = 0
        self.h1_focus_deleted = 0
        self.g0_candidate_deletes_focus = 0
        self.h1_candidate_deletes_focus = 0
        self.g0_vacuous = 0
        self.h1_vacuous = 0
        self.g0_non_vacuous_primary_fix = 0
        self.h1_non_vacuous_primary_fix = 0

    def add(self, *, g0_slots: list[int], h1_slots: list[int], g0: dict[str, Any], h1: dict[str, Any]) -> None:
        self.support += 1
        self.prediction_exact += int(g0_slots == h1_slots)
        self.metric_equivalent += int(_metric_equivalent(g0, h1))
        self.g0_focus_deleted += int(g0.get("focus_deleted", 0))
        self.h1_focus_deleted += int(h1.get("focus_deleted", 0))
        self.g0_candidate_deletes_focus += int(g0.get("candidate_deletes_focus", 0))
        self.h1_candidate_deletes_focus += int(h1.get("candidate_deletes_focus", 0))
        self.g0_vacuous += int(g0.get("vacuous_satisfaction_improvement", 0))
        self.h1_vacuous += int(h1.get("vacuous_satisfaction_improvement", 0))
        self.g0_non_vacuous_primary_fix += int(g0.get("non_vacuous_primary_fix", 0))
        self.h1_non_vacuous_primary_fix += int(h1.get("non_vacuous_primary_fix", 0))

    def to_dict(self) -> dict[str, float | int]:
        support = self.support
        return {
            "support": support,
            "prediction_exact_match_rate": self.prediction_exact / support if support else 0.0,
            "metric_equivalent_rate": self.metric_equivalent / support if support else 0.0,
            "g0_focus_deleted_rate": self.g0_focus_deleted / support if support else 0.0,
            "h1_focus_deleted_rate": self.h1_focus_deleted / support if support else 0.0,
            "g0_candidate_deletes_focus_rate": self.g0_candidate_deletes_focus / support if support else 0.0,
            "h1_candidate_deletes_focus_rate": self.h1_candidate_deletes_focus / support if support else 0.0,
            "g0_vacuous_satisfaction_improvement_rate": self.g0_vacuous / support if support else 0.0,
            "h1_vacuous_satisfaction_improvement_rate": self.h1_vacuous / support if support else 0.0,
            "g0_non_vacuous_primary_fix_rate": self.g0_non_vacuous_primary_fix / support if support else 0.0,
            "h1_non_vacuous_primary_fix_rate": self.h1_non_vacuous_primary_fix / support if support else 0.0,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_analysis(args: argparse.Namespace) -> None:
    run_directory = Path(args.g0_run_directory).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_directory / "evaluations" / "deletion_degeneracy"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        Path(args.reranker_predictions).resolve()
        if args.reranker_predictions
        else run_directory / "reranker_predictions.json"
    )
    if not predictions_path.exists():
        raise FileNotFoundError(f"G0 reranker predictions not found at {predictions_path}")
    with predictions_path.open("r", encoding="utf-8") as fh:
        g0_predictions = json.load(fh)

    model_cfg = _load_model_config(run_directory)
    global_support = EVAL._maybe_prepare_global_support(
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        split="test",
        none_class=NONE_CLASS_INDEX,
        strict=bool(args.strict_global_metrics),
        registry_dataset=args.registry_dataset,
    )
    if global_support is None:
        raise RuntimeError("Deletion-degeneracy analysis requires symbolic global support.")

    rows = global_support.rows
    limit = min(len(rows), len(g0_predictions))
    if args.limit is not None:
        limit = min(limit, args.limit)

    overall = Aggregate()
    by_density: dict[str, Aggregate] = defaultdict(Aggregate)
    by_type: dict[str, Aggregate] = defaultdict(Aggregate)
    examples: list[dict[str, Any]] = []
    constraint_counter: Counter[str] = Counter()

    for idx in range(limit):
        row = rows[idx]
        g0_slots = _slots_from_prediction(g0_predictions[idx])
        h1_slots = _h1_slots(row)
        primary_idx = _primary_factor_index(row)
        g0_detail = global_support.evaluator.evaluate_full(
            row,
            candidate_slots=g0_slots,
            primary_factor_index=primary_idx,
        )
        h1_detail = global_support.evaluator.evaluate_full(
            row,
            candidate_slots=h1_slots,
            primary_factor_index=primary_idx,
        )
        constraint_type, density_bucket, constraint_id = _row_context(row)
        constraint_counter[constraint_type] += 1
        overall.add(g0_slots=g0_slots, h1_slots=h1_slots, g0=g0_detail, h1=h1_detail)
        by_density[density_bucket].add(g0_slots=g0_slots, h1_slots=h1_slots, g0=g0_detail, h1=h1_detail)
        by_type[constraint_type].add(g0_slots=g0_slots, h1_slots=h1_slots, g0=g0_detail, h1=h1_detail)

        equivalent = _metric_equivalent(g0_detail, h1_detail)
        if (g0_slots != h1_slots or not equivalent) and len(examples) < args.examples_limit:
            examples.append(
                {
                    "index": idx,
                    "constraint_id": constraint_id,
                    "constraint_type": constraint_type,
                    "density_bucket": density_bucket,
                    "prediction_exact_match": int(g0_slots == h1_slots),
                    "metric_equivalent": int(equivalent),
                    "g0_slots": json.dumps(g0_slots),
                    "h1_slots": json.dumps(h1_slots),
                    "g0_primary_satisfied": g0_detail.get("primary_satisfied", 0),
                    "h1_primary_satisfied": h1_detail.get("primary_satisfied", 0),
                    "g0_gfr": g0_detail.get("global_satisfied_fraction", 0.0),
                    "h1_gfr": h1_detail.get("global_satisfied_fraction", 0.0),
                    "g0_srr": g0_detail.get("srr", 0.0),
                    "h1_srr": h1_detail.get("srr", 0.0),
                    "g0_sir": g0_detail.get("sir", 0.0),
                    "h1_sir": h1_detail.get("sir", 0.0),
                    "g0_disruption": _total_ops(g0_detail),
                    "h1_disruption": _total_ops(h1_detail),
                    "g0_focus_deleted": g0_detail.get("focus_deleted", 0),
                    "h1_focus_deleted": h1_detail.get("focus_deleted", 0),
                    "g0_candidate_deletes_focus": g0_detail.get("candidate_deletes_focus", 0),
                    "h1_candidate_deletes_focus": h1_detail.get("candidate_deletes_focus", 0),
                    "g0_vacuous_satisfaction_improvement": g0_detail.get(
                        "vacuous_satisfaction_improvement", 0
                    ),
                    "h1_vacuous_satisfaction_improvement": h1_detail.get(
                        "vacuous_satisfaction_improvement", 0
                    ),
                    "g0_non_vacuous_primary_fix": g0_detail.get("non_vacuous_primary_fix", 0),
                    "h1_non_vacuous_primary_fix": h1_detail.get("non_vacuous_primary_fix", 0),
                }
            )

    summary = {
        "g0_run_directory": str(run_directory),
        "reranker_predictions": str(predictions_path),
        "dataset_variant": model_cfg.dataset_variant,
        "min_occurrence": model_cfg.min_occurrence,
        "encoding": model_cfg.encoding,
        "constraint_type_support": dict(sorted(constraint_counter.items())),
        "overall": overall.to_dict(),
    }
    with (output_dir / "deletion_degeneracy_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    slice_fields = ["slice", *overall.to_dict().keys()]
    density_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_density.items())]
    type_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_type.items())]
    _write_csv(output_dir / "deletion_degeneracy_by_density.csv", density_rows, slice_fields)
    _write_csv(output_dir / "deletion_degeneracy_by_constraint_type.csv", type_rows, slice_fields)
    _write_csv(
        output_dir / "deletion_degeneracy_examples.csv",
        examples,
        [
            "index",
            "constraint_id",
            "constraint_type",
            "density_bucket",
            "prediction_exact_match",
            "metric_equivalent",
            "g0_slots",
            "h1_slots",
            "g0_primary_satisfied",
            "h1_primary_satisfied",
            "g0_gfr",
            "h1_gfr",
            "g0_srr",
            "h1_srr",
            "g0_sir",
            "h1_sir",
            "g0_disruption",
            "h1_disruption",
            "g0_focus_deleted",
            "h1_focus_deleted",
            "g0_candidate_deletes_focus",
            "h1_candidate_deletes_focus",
            "g0_vacuous_satisfaction_improvement",
            "h1_vacuous_satisfaction_improvement",
            "g0_non_vacuous_primary_fix",
            "h1_non_vacuous_primary_fix",
        ],
    )
    logging.info("Wrote deletion-degeneracy outputs to %s", output_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(message)s")
    run_analysis(args)


if __name__ == "__main__":
    main()
