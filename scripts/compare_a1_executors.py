#!/usr/bin/env python3
"""Validate and compare the two neutral A1 executor runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_IGNORED_PATHS = {
    ("model_config", "factor_executor_impl"),
    ("model_config", "factor_adapter_rank"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _without_paths(
    payload: Mapping[str, Any],
    ignored: set[tuple[str, ...]],
    prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        path = (*prefix, str(key))
        if path in ignored:
            continue
        normalized[str(key)] = (
            _without_paths(value, ignored, path)
            if isinstance(value, Mapping)
            else value
        )
    return normalized


def _manifest_graph_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    records = manifest.get("graph_manifests", [])
    if not isinstance(records, list):
        return {}
    return {
        str(record.get("path")): str(record.get("sha256"))
        for record in records
        if isinstance(record, Mapping)
    }


def validate_comparison_contract(
    reference_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    reference_impl = reference_config.get("model_config", {}).get("factor_executor_impl")
    candidate_model = candidate_config.get("model_config", {})
    candidate_impl = candidate_model.get("factor_executor_impl")
    if reference_impl != "per_type_grouped_v2":
        raise ValueError("Reference run must use factor_executor_impl=per_type_grouped_v2")
    if candidate_impl != "shared_adapter_v1":
        raise ValueError("Candidate run must use factor_executor_impl=shared_adapter_v1")
    if int(candidate_model.get("factor_adapter_rank", -1)) != 16:
        raise ValueError("Candidate run must use factor_adapter_rank=16")

    normalized_reference = _without_paths(reference_config, CONFIG_IGNORED_PATHS)
    normalized_candidate = _without_paths(candidate_config, CONFIG_IGNORED_PATHS)
    if normalized_reference != normalized_candidate:
        raise ValueError(
            "Executor comparison configs differ outside factor_executor_impl/factor_adapter_rank"
        )

    checks = {
        "seed": reference_manifest.get("seed") == candidate_manifest.get("seed"),
        "dataset_manifest": reference_manifest.get("dataset_manifest")
        == candidate_manifest.get("dataset_manifest"),
        "graph_manifests": _manifest_graph_hashes(reference_manifest)
        == _manifest_graph_hashes(candidate_manifest),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Run provenance mismatch: " + ", ".join(failed))
    return checks


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return value


def _flatten_numbers(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        numeric = _finite_number(value)
        if numeric is not None:
            flattened[name] = numeric
        elif isinstance(value, Mapping):
            flattened.update(_flatten_numbers(value, prefix=name))
    return flattened


def _metric_delta(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    reference_flat = _flatten_numbers(reference)
    candidate_flat = _flatten_numbers(candidate)
    output: dict[str, dict[str, float | int]] = {}
    for metric in sorted(reference_flat.keys() & candidate_flat.keys()):
        reference_value = reference_flat[metric]
        candidate_value = candidate_flat[metric]
        output[metric] = {
            "reference": reference_value,
            "candidate": candidate_value,
            "delta": float(candidate_value) - float(reference_value),
        }
    return output


def _evaluation_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    selected = {
        key: metrics[key]
        for key in (
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "fallback_noop_count",
            "fallback_noop_rate",
        )
        if key in metrics
    }
    selected["model_selection"] = metrics.get("model_selection", {})
    selected["global_metrics"] = {
        "overall": (
            metrics.get("global_metrics", {}).get("overall", {})
            if isinstance(metrics.get("global_metrics"), Mapping)
            else {}
        )
    }
    return selected


def _history_summary(history: Mapping[str, Any]) -> dict[str, Any]:
    val_loss = history.get("val_loss", [])
    epoch_seconds = history.get("epoch_seconds", [])
    return {
        "epochs_completed": len(val_loss) if isinstance(val_loss, list) else 0,
        "best_val_loss": min(val_loss) if isinstance(val_loss, list) and val_loss else None,
        "total_epoch_seconds": (
            sum(float(value) for value in epoch_seconds)
            if isinstance(epoch_seconds, list)
            else None
        ),
        "mean_epoch_seconds": (
            sum(float(value) for value in epoch_seconds) / len(epoch_seconds)
            if isinstance(epoch_seconds, list) and epoch_seconds
            else None
        ),
        "peak_cuda_allocated_bytes": history.get("peak_cuda_allocated_bytes"),
        "peak_cuda_reserved_bytes": history.get("peak_cuda_reserved_bytes"),
        "factor_dispatch_backend": history.get("factor_dispatch_backend"),
    }


def _per_constraint_rows(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reference_per_type = reference.get("per_constraint_type", {})
    candidate_per_type = candidate.get("per_constraint_type", {})
    reference_global = reference.get("global_metrics_per_constraint_type", {})
    candidate_global = candidate.get("global_metrics_per_constraint_type", {})
    if not isinstance(reference_per_type, Mapping) or not isinstance(candidate_per_type, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for constraint_type in sorted(reference_per_type.keys() & candidate_per_type.keys()):
        reference_payload = {
            "fidelity": reference_per_type[constraint_type],
            "global": (
                reference_global.get(constraint_type, {})
                if isinstance(reference_global, Mapping)
                else {}
            ),
        }
        candidate_payload = {
            "fidelity": candidate_per_type[constraint_type],
            "global": (
                candidate_global.get(constraint_type, {})
                if isinstance(candidate_global, Mapping)
                else {}
            ),
        }
        for metric, values in _metric_delta(reference_payload, candidate_payload).items():
            rows.append(
                {
                    "constraint_type": constraint_type,
                    "metric": metric,
                    **values,
                }
            )
    return rows


def _keyed_h2_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    return {
        tuple(str(row.get(key, "")) for key in keys): row
        for row in rows
        if isinstance(row, Mapping)
    }


def _h2_delta_rows(
    reference_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    section: str,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    reference_rows = reference_report.get(section, [])
    candidate_rows = candidate_report.get(section, [])
    if not isinstance(reference_rows, list) or not isinstance(candidate_rows, list):
        return []
    reference_by_key = _keyed_h2_rows(reference_rows, keys=keys)
    candidate_by_key = _keyed_h2_rows(candidate_rows, keys=keys)
    output: list[dict[str, Any]] = []
    for row_key in sorted(reference_by_key.keys() & candidate_by_key.keys()):
        deltas = _metric_delta(reference_by_key[row_key], candidate_by_key[row_key])
        for metric, values in deltas.items():
            output.append(
                {
                    **dict(zip(keys, row_key, strict=True)),
                    "metric": metric,
                    **values,
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# A1 Executor Comparison",
        "",
        f"- Reference: `{report['reference_run']}`",
        f"- Candidate: `{report['candidate_run']}`",
        f"- Seed: `{report['seed']}`",
        "",
        "## Parameters and training",
        "",
        "| Measure | Per-type compact | Shared adapter | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in report["resource_deltas"].items():
        lines.append(
            f"| {metric} | {values['reference']} | {values['candidate']} | "
            f"{values['delta']:+.6g} |"
        )
    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            "| Metric | Per-type compact | Shared adapter | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, values in report["evaluation_deltas"].items():
        lines.append(
            f"| {metric} | {values['reference']:.6g} | "
            f"{values['candidate']:.6g} | {values['delta']:+.6g} |"
        )
    lines.extend(
        [
            "",
            "Positive deltas always mean candidate minus reference; interpret loss, "
            "latency, memory, and regression-rate deltas in the lower-is-better direction.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_runs(
    reference_run: Path,
    candidate_run: Path,
    output_directory: Path,
) -> dict[str, Any]:
    reference_config = _read_json(reference_run / "config.json")
    candidate_config = _read_json(candidate_run / "config.json")
    reference_manifest = _read_json(reference_run / "run_manifest.json")
    candidate_manifest = _read_json(candidate_run / "run_manifest.json")
    contract = validate_comparison_contract(
        reference_config,
        candidate_config,
        reference_manifest,
        candidate_manifest,
    )
    reference_history = _read_json(reference_run / "training_history.json")
    candidate_history = _read_json(candidate_run / "training_history.json")
    reference_evaluation = _read_json(reference_run / "evaluations" / "model.json")
    candidate_evaluation = _read_json(candidate_run / "evaluations" / "model.json")
    reference_h2 = _read_json(reference_run / "evaluations" / "h2" / "h2_report.json")
    candidate_h2 = _read_json(candidate_run / "evaluations" / "h2" / "h2_report.json")
    if reference_h2.get("status") != "ok" or candidate_h2.get("status") != "ok":
        raise ValueError("Both H2 reports must have status=ok")

    reference_training = _history_summary(reference_history)
    candidate_training = _history_summary(candidate_history)
    reference_resources = {
        "parameters_total": reference_manifest.get("parameters", {}).get("total"),
        "parameters_trainable": reference_manifest.get("parameters", {}).get("trainable"),
        **reference_training,
    }
    candidate_resources = {
        "parameters_total": candidate_manifest.get("parameters", {}).get("total"),
        "parameters_trainable": candidate_manifest.get("parameters", {}).get("trainable"),
        **candidate_training,
    }
    resource_deltas = _metric_delta(reference_resources, candidate_resources)
    evaluation_deltas = _metric_delta(
        _evaluation_summary(reference_evaluation),
        _evaluation_summary(candidate_evaluation),
    )
    per_constraint_rows = _per_constraint_rows(
        reference_evaluation,
        candidate_evaluation,
    )
    h2_overall_rows = _h2_delta_rows(
        reference_h2,
        candidate_h2,
        section="overall",
        keys=("variant",),
    )
    h2_semantic_rows = _h2_delta_rows(
        reference_h2,
        candidate_h2,
        section="factor_semantics",
        keys=("state", "factor_family", "factor_type"),
    )

    report = {
        "schema_version": 1,
        "reference_run": str(reference_run),
        "candidate_run": str(candidate_run),
        "seed": reference_manifest.get("seed"),
        "contract_checks": contract,
        "reference_executor": "per_type_grouped_v2",
        "candidate_executor": "shared_adapter_v1",
        "candidate_adapter_rank": 16,
        "reference_resources": reference_resources,
        "candidate_resources": candidate_resources,
        "resource_deltas": resource_deltas,
        "evaluation_deltas": evaluation_deltas,
        "h2_status": {
            "reference": reference_h2.get("status"),
            "candidate": candidate_h2.get("status"),
        },
        "artifacts": {
            "per_constraint_csv": "per_constraint_deltas.csv",
            "h2_overall_csv": "h2_overall_deltas.csv",
            "h2_factor_semantics_csv": "h2_factor_semantics_deltas.csv",
            "markdown": "comparison.md",
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "per_constraint_deltas.csv", per_constraint_rows)
    _write_csv(output_directory / "h2_overall_deltas.csv", h2_overall_rows)
    _write_csv(
        output_directory / "h2_factor_semantics_deltas.csv",
        h2_semantic_rows,
    )
    with (output_directory / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with (output_directory / "comparison.md").open("w", encoding="utf-8") as handle:
        handle.write(_markdown(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("models/paper_diagnostics/a1_executor_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_runs(
        args.reference_run,
        args.candidate_run,
        args.output_directory,
    )
    print(
        "[ok] compared "
        f"{report['reference_executor']} with {report['candidate_executor']} "
        f"at seed {report['seed']}"
    )


if __name__ == "__main__":
    main()
