#!/usr/bin/env python3
"""Validate paper-facing data, graph, run, and evaluation integrity contracts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig
from modules.class_hierarchy import (
    CLASS_HIERARCHY_FILENAME,
    CLASS_HIERARCHY_MANIFEST_FILENAME,
    CLASS_HIERARCHY_SCHEMA_VERSION,
    CLASS_HIERARCHY_SEMANTICS,
)
from modules.constraint_semantics import CONSTRAINT_SEMANTICS_VERSION
from modules.data_encoders import (
    DATASET_SCHEMA_VERSION,
    FEATURE_ENCODER_FILENAME,
    GRAPH_SCHEMA_VERSION,
    IDENTITY_ENCODER_FILENAME,
    IDENTITY_TO_FEATURE_FILENAME,
    ConstraintGraphData,
    discover_graph_artifacts,
    graph_dataset_filename,
)
from modules.provenance import (
    RUN_MANIFEST_FILENAME,
    canonical_json_hash,
    config_differences,
    file_sha256,
    json_file_hash,
)
from modules.training_utils import load_graph_dataset


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.checks.append(message)
        else:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": not self.errors,
            "checks_passed": len(self.checks),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _manifest_path(graph_path: Path) -> Path:
    return graph_path.with_suffix(graph_path.suffix + ".manifest.json")


def _iter_prefix(dataset: Iterable[Any], limit: int) -> list[Any]:
    rows: list[Any] = []
    for value in dataset:
        rows.append(value)
        if len(rows) >= limit:
            break
    return rows


def _validate_parquet_contract(interim: Path, report: ValidationReport) -> None:
    required_columns = {
        "constraint_id",
        "constraint_id_feature",
        "subject",
        "subject_feature",
        "predicate",
        "predicate_feature",
        "object",
        "object_feature",
        "add_subject",
        "add_subject_feature",
        "del_subject",
        "del_subject_feature",
    }
    for split in ("train", "val", "test"):
        path = interim / f"df_{split}.parquet"
        report.require(path.exists(), f"{split} parquet exists")
        if not path.exists():
            continue
        columns = set(pd.read_parquet(path, columns=[]).columns)
        if not columns:
            import pyarrow.parquet as pq

            columns = set(pq.ParquetFile(path).schema.names)
        missing = sorted(required_columns - columns)
        report.require(not missing, f"{split} carries separate identity and feature columns")

        primary_columns = {
            "primary_factor_index",
            "primary_checkable_pre",
            "primary_satisfied_pre",
            "primary_validation_reason",
            "primary_checkable_post_gold",
            "primary_satisfied_post_gold",
            "primary_gold_repair_status",
            "primary_gold_repair_verified",
        }
        if primary_columns <= columns:
            primary = pd.read_parquet(path, columns=sorted(primary_columns))
            valid = (
                primary["primary_checkable_pre"].astype(bool)
                & (primary["primary_satisfied_pre"].astype(int) == 0)
                & (primary["primary_validation_reason"].astype(str) == "valid")
                & (primary["primary_factor_index"].astype(int) >= 0)
            )
            report.require(bool(valid.all()), f"{split} primary rows are checkable pre-repair violations")
            expected_verified = (
                primary["primary_checkable_post_gold"].astype(bool)
                & (primary["primary_satisfied_post_gold"].astype(int) == 1)
            )
            report.require(
                bool(
                    (
                        primary["primary_gold_repair_verified"].astype(bool)
                        == expected_verified
                    ).all()
                ),
                f"{split} gold-repair verification matches executable POST_GOLD labels",
            )
            report.require(
                set(primary["primary_gold_repair_status"].astype(str).unique())
                <= {"verified", "post_uncheckable", "post_unsatisfied"},
                f"{split} retained rows carry an eligible gold-repair status",
            )
            expected_status = np.where(
                ~primary["primary_checkable_post_gold"].astype(bool),
                "post_uncheckable",
                np.where(
                    primary["primary_satisfied_post_gold"].astype(int) == 1,
                    "verified",
                    "post_unsatisfied",
                ),
            )
            report.require(
                bool(
                    (
                        primary["primary_gold_repair_status"].astype(str).to_numpy()
                        == expected_status
                    ).all()
                ),
                f"{split} gold-repair status matches POST_GOLD checkability and satisfaction",
            )
        else:
            report.errors.append(f"{split} is missing primary-validation columns: {sorted(primary_columns - columns)}")


def validate_dataset(
    *,
    dataset_variant: str,
    encoding: str,
    constraint_representation: str,
    primary_constraint_mode: str,
    verify_hashes: bool,
    validate_graphs: bool,
    report: ValidationReport,
) -> None:
    interim = ROOT / "data" / "interim" / dataset_variant
    processed = ROOT / "data" / "processed" / dataset_variant
    manifest_path = interim / "dataset_manifest.json"
    report.require(manifest_path.exists(), "dataset manifest exists")
    if not manifest_path.exists():
        return
    manifest = _read_json(manifest_path)
    report.require(
        int(manifest.get("schema_version", 0)) >= DATASET_SCHEMA_VERSION,
        f"dataset schema is at least v{DATASET_SCHEMA_VERSION}",
    )
    report.require(manifest.get("split_policy") == "preserve", "upstream train/dev/test splits are preserved")
    report.require(
        manifest.get("raw_split_mapping") == {"train": "train", "dev": "val", "test": "test"},
        "raw-to-local split mapping is explicit",
    )
    semantic_labeling = manifest.get("semantic_labeling", {})
    report.require(
        semantic_labeling.get("version") == CONSTRAINT_SEMANTICS_VERSION,
        f"constraint semantics are {CONSTRAINT_SEMANTICS_VERSION}",
    )
    validation_by_constraint = semantic_labeling.get(
        "primary_validation_by_constraint",
        {},
    )
    report.require(
        bool(validation_by_constraint),
        "per-family primary-validation audit is embedded in the manifest",
    )
    if validation_by_constraint:
        source_families: set[str] = set()
        valid_families: set[str] = set()
        for key, count in validation_by_constraint.items():
            try:
                _, family, reason = str(key).split("::", 2)
            except ValueError:
                continue
            if int(count) <= 0:
                continue
            source_families.add(family)
            if reason == "valid":
                valid_families.add(family)
        report.require(
            source_families == valid_families,
            "every source primary family has retained checkable violations",
        )
    gold_repair_by_constraint = semantic_labeling.get(
        "gold_repair_by_constraint",
        {},
    )
    report.require(
        bool(gold_repair_by_constraint),
        "per-family observed-edit repair audit is embedded in the manifest",
    )

    sampling = manifest.get("sampling", {})
    target_rows = sampling.get("target_rows") if isinstance(sampling, dict) else None
    if target_rows is not None:
        report.require(
            sum(int(value) for value in manifest.get("rows", {}).values())
            == int(target_rows),
            f"sample contains exactly {int(target_rows):,} rows",
        )
        sampled_primary = sampling.get("primary_validation_by_constraint", {})
        sampled_gold = sampling.get("gold_repair_by_constraint", {})
        report.require(
            isinstance(sampled_primary, dict)
            and sum(int(value) for value in sampled_primary.values()) == int(target_rows),
            "sampled primary-family audit covers every row",
        )
        report.require(
            isinstance(sampled_gold, dict)
            and sum(int(value) for value in sampled_gold.values()) == int(target_rows),
            "sampled observed-edit audit covers every row",
        )
    for filename in (
        IDENTITY_ENCODER_FILENAME,
        FEATURE_ENCODER_FILENAME,
        IDENTITY_TO_FEATURE_FILENAME,
        CLASS_HIERARCHY_FILENAME,
        CLASS_HIERARCHY_MANIFEST_FILENAME,
    ):
        report.require((interim / filename).exists(), f"{filename} exists")

    hierarchy_manifest_path = interim / CLASS_HIERARCHY_MANIFEST_FILENAME
    if hierarchy_manifest_path.exists():
        hierarchy_manifest = _read_json(hierarchy_manifest_path)
        report.require(
            int(hierarchy_manifest.get("schema_version", 0))
            >= CLASS_HIERARCHY_SCHEMA_VERSION,
            "class hierarchy schema is current",
        )
        report.require(
            hierarchy_manifest.get("semantics") == CLASS_HIERARCHY_SEMANTICS,
            "type constraints use the frozen training P279 closure",
        )
        report.require(
            hierarchy_manifest.get("source_split") == "train",
            "class hierarchy is derived from training context only",
        )
        report.require(
            int(hierarchy_manifest.get("direct_edge_count", 0)) > 0
            and int(hierarchy_manifest.get("child_count", 0)) > 0,
            "class hierarchy contains direct training evidence",
        )
        report.require(
            int(hierarchy_manifest.get("p279_predicate_id", 0)) > 0,
            "class hierarchy records the P279 identity",
        )
        report.require(
            len(str(hierarchy_manifest.get("source_manifest_sha256", ""))) == 64,
            "class hierarchy records its source dataset manifest hash",
        )
        hierarchy_outputs = hierarchy_manifest.get("outputs", {})
        hierarchy_path = interim / CLASS_HIERARCHY_FILENAME
        expected_hierarchy_hash = (
            hierarchy_outputs.get(CLASS_HIERARCHY_FILENAME)
            if isinstance(hierarchy_outputs, dict)
            else None
        )
        report.require(
            hierarchy_path.exists()
            and bool(expected_hierarchy_hash)
            and file_sha256(hierarchy_path) == expected_hierarchy_hash,
            "class hierarchy hash matches its manifest",
        )

    mapping_path = interim / IDENTITY_TO_FEATURE_FILENAME
    if mapping_path.exists():
        mapping = np.load(mapping_path)
        report.require(mapping.ndim == 1 and len(mapping) > 1, "identity-to-feature map is a nontrivial vector")
        report.require(int(mapping[0]) == 0, "NONE identity maps to NONE feature")

    if verify_hashes:
        for filename, expected in manifest.get("outputs", {}).items():
            path = interim / filename
            report.require(path.exists(), f"manifest output exists: {filename}")
            if path.exists():
                report.require(file_sha256(path) == expected, f"manifest hash matches: {filename}")

    _validate_parquet_contract(interim, report)

    if not validate_graphs:
        return

    for split in ("train", "val", "test"):
        graph_path = processed / graph_dataset_filename(
            split,
            encoding,
            constraint_representation=constraint_representation,
            primary_constraint_mode=primary_constraint_mode,
        )
        graph_manifest_path = _manifest_path(graph_path)
        report.require(graph_manifest_path.exists(), f"{split} graph manifest exists")
        if graph_manifest_path.exists():
            graph_manifest = _read_json(graph_manifest_path)
            report.require(
                int(graph_manifest.get("graph_schema_version", 0)) >= GRAPH_SCHEMA_VERSION,
                f"{split} graph schema is at least v{GRAPH_SCHEMA_VERSION}",
            )
            kept = set(graph_manifest.get("kept_fields", []))
            report.require(
                {"y_identity", "target_representable_mask", "node_identity_id"} <= kept,
                f"{split} graph manifest includes identity audit fields",
            )
        if split == "train" and discover_graph_artifacts(graph_path):
            sample = _iter_prefix(load_graph_dataset(graph_path), 2)
            report.require(bool(sample), "training graph sample is readable")
            if sample:
                report.require(
                    all(isinstance(graph, ConstraintGraphData) for graph in sample),
                    "graphs use explicit PyG increment semantics",
                )
                for graph in sample:
                    report.require(hasattr(graph, "y_identity"), "graph carries identity target")
                    report.require(
                        hasattr(graph, "target_representable_mask"),
                        "graph carries target representability mask",
                    )
            if len(sample) == 2:
                batch = next(iter(DataLoader(sample, batch_size=2)))
                predicates = batch.edge_attr_non_flattened.view(-1)
                edge_index = batch.edge_index_non_flattened
                if predicates.numel():
                    source_graphs = batch.batch[edge_index[0]]
                    target_graphs = batch.batch[edge_index[1]]
                    predicate_graphs = batch.batch[predicates]
                    report.require(
                        bool(
                            (source_graphs == target_graphs).all()
                            and (source_graphs == predicate_graphs).all()
                        ),
                        "batched non-flattened edge endpoints stay within their source graph",
                    )


def validate_run(run_directory: Path, report: ValidationReport) -> None:
    config_path = run_directory / "config.json"
    checkpoint_path = run_directory / "checkpoint.pth"
    manifest_path = run_directory / RUN_MANIFEST_FILENAME
    report.require(config_path.exists(), f"{run_directory.name}: config exists")
    report.require(checkpoint_path.exists(), f"{run_directory.name}: checkpoint exists")
    report.require(manifest_path.exists(), f"{run_directory.name}: run provenance exists")
    if not config_path.exists():
        return
    config = _read_json(config_path)
    model_config = ModelConfig.from_mapping(config.get("model_config", {}))
    if model_config.factor_executor_impl in {
        "per_type_grouped_v2",
        "shared_adapter_v1",
    }:
        active_ids = tuple(model_config.active_factor_type_ids or ())
        report.require(
            bool(active_ids)
            and active_ids == tuple(sorted(set(active_ids)))
            and active_ids[-1] < model_config.num_factor_types,
            f"{run_directory.name}: compact factor mapping is explicit and valid",
        )
        report.require(
            model_config.gold_edit_embedding_mode == "compact",
            f"{run_directory.name}: compact executor uses compact gold-edit embeddings",
        )
    if model_config.factor_executor_impl == "shared_adapter_v1":
        report.require(
            model_config.factor_adapter_rank == 16,
            f"{run_directory.name}: shared-adapter comparison uses locked rank 16",
        )
    if run_directory.name.startswith("a1_factorized_imitation"):
        report.require(
            model_config.pressure_module_sharing == "shared",
            "canonical A1 uses shared role-pressure blocks",
        )
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        recorded = checkpoint.get("model_cfg", {}) if isinstance(checkpoint, dict) else {}
        if recorded:
            recorded_config = ModelConfig.from_mapping(recorded)
            differences = config_differences(model_config.to_dict(), recorded_config.to_dict())
            report.require(not differences, f"{run_directory.name}: checkpoint model config matches config.json")
        else:
            report.errors.append(f"{run_directory.name}: checkpoint lacks embedded model config")
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        config_record = manifest.get("config", {})
        report.require(
            config_record.get("sha256") == canonical_json_hash(config),
            f"{run_directory.name}: run manifest matches effective config",
        )
        checkpoint_record = manifest.get("checkpoint", {})
        if checkpoint_path.exists() and checkpoint_record.get("sha256"):
            report.require(
                checkpoint_record["sha256"] == file_sha256(checkpoint_path),
                f"{run_directory.name}: checkpoint hash matches run manifest",
            )
        parameters = manifest.get("parameters", {})
        by_component = parameters.get("by_component", {}) if isinstance(parameters, dict) else {}
        if run_directory.name.startswith("a1_factorized_imitation"):
            role_parameters = int(by_component.get("_pressure_role_modules", 0))
            report.require(
                0 < role_parameters <= 1_500_000,
                "canonical A1 shared role-pressure component is at most 1.5M parameters",
            )
        if model_config.factor_executor_impl == "per_type_grouped_v2":
            report.require(
                0 < int(by_component.get("_factor_executors", 0)) <= 8_100_000,
                f"{run_directory.name}: compact per-type executors allocate only active types",
            )
            report.require(
                0 < int(by_component.get("_factor_post_heads", 0)) <= 3_300_000,
                f"{run_directory.name}: compact post heads allocate only active types",
            )
            report.require(
                0 < int(by_component.get("_gold_edit_embeddings", 0)) <= 1_700_000,
                f"{run_directory.name}: compact gold-edit table uses reachable targets",
            )
        if model_config.factor_executor_impl == "shared_adapter_v1":
            report.require(
                0 < int(by_component.get("_factor_executors", 0)) <= 1_100_000,
                f"{run_directory.name}: shared executor uses compact rank-16 adapters",
            )
            report.require(
                0 < int(by_component.get("_factor_post_heads", 0)) <= 500_000,
                f"{run_directory.name}: shared post head uses compact rank-16 adapters",
            )
            report.require(
                0 < int(by_component.get("_gold_edit_embeddings", 0)) <= 1_700_000,
                f"{run_directory.name}: compact gold-edit table uses reachable targets",
            )
        if str(model_config.model).upper() == "RERANKER" or "proposal_config" in config:
            extra = manifest.get("extra", {})
            proposal = extra.get("proposal_checkpoint", {}) if isinstance(extra, dict) else {}
            proposal_path = Path(proposal.get("path", "")) if isinstance(proposal, dict) else Path()
            if not proposal_path.is_absolute():
                proposal_path = ROOT / proposal_path
            report.require(
                isinstance(proposal, dict)
                and proposal_path.is_file()
                and proposal.get("sha256") == file_sha256(proposal_path),
                f"{run_directory.name}: proposal checkpoint hash matches run manifest",
            )


def validate_results(run_directory: Path, report: ValidationReport) -> None:
    result_path = run_directory / "evaluations" / "model.json"
    report.require(result_path.exists(), f"{run_directory.name}: evaluation result exists")
    if not result_path.exists():
        return
    metrics = _read_json(result_path)
    report.require(
        int(metrics.get("evaluation_schema_version", 0)) >= 2,
        f"{run_directory.name}: evaluation schema is v2",
    )
    report.require(
        metrics.get("fidelity_space") == "strict_identity",
        f"{run_directory.name}: headline fidelity is strict identity fidelity",
    )
    report.require(
        metrics.get("candidate_inference_gold_access") is False,
        f"{run_directory.name}: inference declares no gold candidate access",
    )
    evaluation_provenance = metrics.get("evaluation_provenance", {})
    run_manifest_path = run_directory / RUN_MANIFEST_FILENAME
    report.require(
        isinstance(evaluation_provenance, dict)
        and run_manifest_path.exists()
        and evaluation_provenance.get("run_manifest_sha256")
        == json_file_hash(run_manifest_path),
        f"{run_directory.name}: evaluation is tied to the current run manifest",
    )
    if run_manifest_path.exists() and isinstance(evaluation_provenance, dict):
        run_manifest = _read_json(run_manifest_path)
        checkpoint_record = run_manifest.get("checkpoint", {})
        expected_checkpoint = (
            checkpoint_record.get("sha256") if isinstance(checkpoint_record, dict) else None
        )
        report.require(
            bool(expected_checkpoint)
            and evaluation_provenance.get("checkpoint_sha256") == expected_checkpoint,
            f"{run_directory.name}: evaluation identifies the current checkpoint",
        )
        for key, label in (
            ("dataset_manifest", "dataset manifest"),
            ("test_graph_manifest", "test graph manifest"),
        ):
            record = evaluation_provenance.get(key)
            path = Path(record.get("path", "")) if isinstance(record, dict) else Path()
            if not path.is_absolute():
                path = ROOT / path
            report.require(
                isinstance(record, dict)
                and path.is_file()
                and record.get("sha256") == json_file_hash(path),
                f"{run_directory.name}: evaluation identifies the current {label}",
            )
    global_metrics = metrics.get("global_metrics", {})
    overall = global_metrics.get("overall", {}) if isinstance(global_metrics, dict) else {}
    report.require("primary_fix_rate" in overall, f"{run_directory.name}: transition-based primary fix is present")
    for metric in ("srr", "sir"):
        numerator = int(overall.get(f"{metric}_total", 0))
        denominator = int(overall.get(f"{metric}_denom_total", 0))
        expected = numerator / denominator if denominator else 0.0
        observed = float(overall.get(metric, float("nan")))
        report.require(
            abs(observed - expected) < 1e-12,
            f"{run_directory.name}: {metric.upper()} is the pooled ratio",
        )


def validate_baseline_directory(directory: Path, report: ValidationReport) -> None:
    paths = sorted(directory.glob("baseline-*.json"))
    report.require(bool(paths), f"{directory}: baseline result files exist")
    for path in paths:
        metrics = _read_json(path)
        label = path.stem
        report.require(
            int(metrics.get("evaluation_schema_version", 0)) >= 2,
            f"{label}: evaluation schema is v2",
        )
        report.require(
            metrics.get("fidelity_space") == "strict_identity",
            f"{label}: headline fidelity is strict identity fidelity",
        )
        report.require(
            metrics.get("candidate_inference_gold_access") is False,
            f"{label}: inference declares no gold candidate access",
        )
        provenance = metrics.get("evaluation_provenance", {})
        report.require(
            isinstance(provenance, dict) and provenance.get("deterministic_baseline") is True,
            f"{label}: deterministic baseline provenance is present",
        )
        if isinstance(provenance, dict):
            for key, hash_fn in (
                ("dataset_manifest", json_file_hash),
                ("constraint_registry", file_sha256),
            ):
                record = provenance.get(key)
                record_path = Path(record.get("path", "")) if isinstance(record, dict) else Path()
                if not record_path.is_absolute():
                    record_path = ROOT / record_path
                report.require(
                    isinstance(record, dict)
                    and record_path.is_file()
                    and record.get("sha256") == hash_fn(record_path),
                    f"{label}: {key} hash matches",
                )
        global_metrics = metrics.get("global_metrics", {})
        overall = global_metrics.get("overall", {}) if isinstance(global_metrics, dict) else {}
        report.require("primary_fix_rate" in overall, f"{label}: transition-based primary fix is present")
        for metric in ("srr", "sir"):
            numerator = int(overall.get(f"{metric}_total", 0))
            denominator = int(overall.get(f"{metric}_denom_total", 0))
            expected = numerator / denominator if denominator else 0.0
            report.require(
                abs(float(overall.get(metric, float("nan"))) - expected) < 1e-12,
                f"{label}: {metric.upper()} is the pooled ratio",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-variant", required=True)
    parser.add_argument("--encoding", default="node_id")
    parser.add_argument("--constraint-representation", default="factorized")
    parser.add_argument("--primary-constraint-mode", default="executable_factor")
    parser.add_argument("--run-directory", action="append", type=Path, default=[])
    parser.add_argument("--baseline-directory", action="append", type=Path, default=[])
    parser.add_argument(
        "--stage",
        choices=("interim", "data", "run", "results", "all"),
        default="all",
        help="interim validates Section 1 only; data also requires graph artifacts.",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = ValidationReport()
    if args.stage in {"interim", "data", "all"}:
        validate_dataset(
            dataset_variant=args.dataset_variant,
            encoding=args.encoding,
            constraint_representation=args.constraint_representation,
            primary_constraint_mode=args.primary_constraint_mode,
            verify_hashes=args.verify_hashes,
            validate_graphs=args.stage != "interim",
            report=report,
        )
    if args.stage in {"run", "all"}:
        for run_directory in args.run_directory:
            validate_run(run_directory.resolve(), report)
    if args.stage in {"results", "all"}:
        for run_directory in args.run_directory:
            validate_results(run_directory.resolve(), report)
        for baseline_directory in args.baseline_directory:
            validate_baseline_directory(baseline_directory.resolve(), report)

    payload = report.as_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
