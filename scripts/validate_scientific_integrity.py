#!/usr/bin/env python3
"""Validate paper-facing data, graph, run, and evaluation integrity contracts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
from modules.graph_manifest import (
    file_sha256 as graph_file_sha256,
    graph_incomplete_marker_path,
    graph_manifest_path,
    resolve_recorded_path,
    shard_index,
)
from modules.provenance import (
    RUN_MANIFEST_FILENAME,
    canonical_json_hash,
    config_differences,
    file_sha256,
    json_file_hash,
)
from modules.training_utils import load_graph_dataset
from modules.sampling_contract import ROW_ORDER_BUCKET_COUNT, ROW_ORDER_METHOD


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    validated_graph_manifests: list[dict[str, object]] = field(default_factory=list)

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
            "validated_graph_manifests": self.validated_graph_manifests,
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _manifest_path(graph_path: Path) -> Path:
    return graph_manifest_path(graph_path)


def _validate_sampled_row_order(
    *,
    interim: Path,
    manifest: dict[str, Any],
    report: ValidationReport,
) -> None:
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        return
    is_sampled = sampling.get("target_rows") is not None or sampling.get("sample_fraction") is not None
    if not is_sampled:
        return

    row_order = sampling.get("row_order")
    report.require(
        isinstance(row_order, dict),
        "sampled dataset records sampling.row_order provenance",
    )
    metadata_path = interim / "sampling_metadata.json"
    report.require(metadata_path.exists(), "sampled dataset has sampling_metadata.json")
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    metadata_row_order = metadata.get("row_order")
    report.require(
        isinstance(metadata_row_order, dict),
        "sampling metadata records row_order provenance",
    )

    manifest_method = row_order.get("method") if isinstance(row_order, dict) else None
    metadata_method = (
        metadata_row_order.get("method") if isinstance(metadata_row_order, dict) else None
    )
    report.require(
        manifest_method == ROW_ORDER_METHOD and metadata_method == ROW_ORDER_METHOD,
        f"sample row order method is exactly {ROW_ORDER_METHOD}",
    )
    sampling_seed = sampling.get("seed")
    manifest_seed = row_order.get("seed") if isinstance(row_order, dict) else None
    metadata_seed = metadata_row_order.get("seed") if isinstance(metadata_row_order, dict) else None
    report.require(
        sampling_seed is not None
        and manifest_seed == sampling_seed
        and metadata_seed == sampling_seed
        and metadata.get("seed") == sampling_seed,
        "sample row order seeds match sampling provenance",
    )
    manifest_buckets = (
        row_order.get("external_sort_buckets") if isinstance(row_order, dict) else None
    )
    metadata_buckets = (
        metadata_row_order.get("external_sort_buckets")
        if isinstance(metadata_row_order, dict)
        else None
    )
    report.require(
        isinstance(manifest_buckets, int)
        and manifest_buckets > 0
        and manifest_buckets == ROW_ORDER_BUCKET_COUNT
        and metadata_buckets == manifest_buckets,
        "sample row order bucket count is positive and matches metadata",
    )
    metadata_hash = manifest.get("outputs", {}).get("sampling_metadata.json")
    report.require(
        metadata_path.exists()
        and isinstance(metadata_hash, str)
        and len(metadata_hash) == 64
        and file_sha256(metadata_path) == metadata_hash,
        "sampling metadata is a hashed dataset-manifest output",
    )


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


def _graph_artifact_candidates(graph_path: Path) -> set[Path]:
    candidates: set[Path] = set()
    if graph_path.exists() or graph_path.is_symlink():
        candidates.add(graph_path.resolve(strict=False))
    for suffix in ("pkl", "pt"):
        for path in graph_path.parent.glob(f"{graph_path.stem}-shard*.{suffix}"):
            candidates.add(path.resolve(strict=False))
    return candidates


def _count_graph_objects(path: Path, *, sharded: bool) -> int:
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a list payload in graph shard {path}")
        return len(payload)

    with path.open("rb") as handle:
        try:
            first = pickle.load(handle)
        except EOFError:
            return 0
        if isinstance(first, list):
            return len(first)
        if sharded:
            raise TypeError(f"Expected a list payload in graph shard {path}")
        count = 1
        while True:
            try:
                pickle.load(handle)
            except EOFError:
                return count
            count += 1


def _validate_graph_derivation(
    *,
    graph_manifest_path: Path,
    graph_manifest: dict[str, Any],
    artifact_paths: list[Path],
    verify_hashes: bool,
    report: ValidationReport,
    label: str,
) -> None:
    derivation = graph_manifest.get("derivation")
    if derivation is None:
        report.checks.append(f"{label} graph is a canonical parquet-derived build")
        return
    if not isinstance(derivation, dict):
        report.errors.append(f"{label} graph derivation is malformed")
        return

    method = derivation.get("method")
    report.require(method in {"hard_link", "rewrite"}, f"{label} graph derivation method is supported")
    source_record = derivation.get("source_manifest")
    source_path: Path | None = None
    source_payload: dict[str, Any] = {}
    if isinstance(source_record, dict) and source_record.get("path"):
        source_path = resolve_recorded_path(
            source_record["path"],
            manifest_path=graph_manifest_path,
            repository_root=ROOT,
        )
    report.require(
        source_path is not None and source_path.exists(),
        f"{label} derivation source graph manifest exists",
    )
    if source_path is not None and source_path.exists():
        report.require(
            source_record.get("sha256") == graph_file_sha256(source_path),
            f"{label} derivation source graph manifest hash matches",
        )
        source_payload = _read_json(source_path)
        report.require(
            int(source_payload.get("graph_schema_version", 0)) == GRAPH_SCHEMA_VERSION,
            f"{label} derivation source graph manifest is schema v{GRAPH_SCHEMA_VERSION}",
        )
        report.require(
            source_payload.get("dataset_manifest") == graph_manifest.get("dataset_manifest")
            and source_payload.get("source_parquet") == graph_manifest.get("source_parquet")
            and source_payload.get("build_limit") == graph_manifest.get("build_limit")
            and source_payload.get("graph_count") == graph_manifest.get("graph_count"),
            f"{label} derivation preserves dataset and parquet lineage",
        )
        report.require(
            derivation.get("source_primary_constraint_mode")
            == source_payload.get("primary_constraint_mode")
            and derivation.get("target_primary_constraint_mode")
            == graph_manifest.get("primary_constraint_mode"),
            f"{label} derivation records source and target primary modes",
        )

    lineage = derivation.get("artifacts")
    report.require(
        isinstance(lineage, list) and len(lineage) == len(artifact_paths),
        f"{label} derivation records every artifact",
    )
    if not verify_hashes or not isinstance(lineage, list):
        return
    for index, record in enumerate(lineage):
        if not isinstance(record, dict):
            report.errors.append(f"{label} derivation artifact {index} is malformed")
            continue
        source_raw = record.get("source_path")
        target_raw = record.get("target_path")
        source_artifact = (
            resolve_recorded_path(
                source_raw,
                manifest_path=graph_manifest_path,
                repository_root=ROOT,
            )
            if source_raw
            else None
        )
        target_artifact = (
            resolve_recorded_path(
                target_raw,
                manifest_path=graph_manifest_path,
                repository_root=ROOT,
            )
            if target_raw
            else None
        )
        report.require(
            source_artifact is not None and source_artifact.exists(),
            f"{label} derivation source artifact {index:03d} exists",
        )
        report.require(
            target_artifact is not None
            and target_artifact in artifact_paths
            and target_artifact.exists(),
            f"{label} derivation target artifact {index:03d} matches the manifest",
        )
        if (
            method == "hard_link"
            and source_artifact is not None
            and target_artifact is not None
            and source_artifact.exists()
            and target_artifact.exists()
        ):
            report.require(
                os.path.samefile(source_artifact, target_artifact),
                f"{label} derivation artifact {index:03d} remains hard-linked",
            )


def _validate_graph_manifest(
    *,
    graph_path: Path,
    graph_manifest_path: Path,
    dataset_manifest_path: Path,
    parquet_path: Path,
    processed: Path,
    dataset_variant: str,
    split: str,
    encoding: str,
    constraint_representation: str,
    primary_constraint_mode: str,
    verify_hashes: bool,
    report: ValidationReport,
) -> None:
    label = split
    starting_error_count = len(report.errors)
    marker_path = graph_incomplete_marker_path(graph_path)
    report.require(not marker_path.exists(), f"{label} has no incomplete graph-build marker")
    report.require(graph_manifest_path.exists(), f"{label} graph manifest exists")
    if not graph_manifest_path.exists():
        return
    graph_manifest = _read_json(graph_manifest_path)
    report.require(
        int(graph_manifest.get("graph_schema_version", 0)) == GRAPH_SCHEMA_VERSION,
        f"{label} graph schema is exactly v{GRAPH_SCHEMA_VERSION}",
    )
    expected_fields = {
        "dataset_variant": dataset_variant,
        "split": split,
        "encoding": encoding,
        "constraint_representation": constraint_representation,
        "primary_constraint_mode": primary_constraint_mode,
    }
    for key, expected in expected_fields.items():
        report.require(
            graph_manifest.get(key) == expected,
            f"{label} graph manifest {key} matches the requested graph",
        )

    dataset_record = graph_manifest.get("dataset_manifest")
    recorded_dataset_path = None
    if isinstance(dataset_record, dict) and dataset_record.get("path"):
        recorded_dataset_path = resolve_recorded_path(
            dataset_record["path"],
            manifest_path=graph_manifest_path,
            repository_root=ROOT,
        )
    report.require(
        recorded_dataset_path == dataset_manifest_path.resolve()
        and isinstance(dataset_record, dict)
        and dataset_record.get("sha256") == graph_file_sha256(dataset_manifest_path),
        f"{label} graph commits to the current dataset manifest",
    )

    source_record = graph_manifest.get("source_parquet")
    recorded_parquet_path = None
    if isinstance(source_record, dict) and source_record.get("path"):
        recorded_parquet_path = resolve_recorded_path(
            source_record["path"],
            manifest_path=graph_manifest_path,
            repository_root=ROOT,
        )
    source_row_count = int(pq.ParquetFile(parquet_path).metadata.num_rows)
    report.require(
        recorded_parquet_path == parquet_path.resolve()
        and isinstance(source_record, dict)
        and source_record.get("sha256") == graph_file_sha256(parquet_path)
        and source_record.get("row_count") == source_row_count,
        f"{label} graph commits to the current split parquet and row count",
    )

    build_limit = graph_manifest.get("build_limit")
    valid_build_limit = build_limit is None or (
        isinstance(build_limit, int) and not isinstance(build_limit, bool) and build_limit > 0
    )
    report.require(valid_build_limit, f"{label} graph records a valid explicit build limit")
    expected_graph_count = (
        source_row_count if build_limit is None else min(source_row_count, int(build_limit))
    ) if valid_build_limit else source_row_count
    report.require(
        graph_manifest.get("graph_count") == expected_graph_count,
        f"{label} graph count is consistent with source rows and build limit",
    )

    kept = set(graph_manifest.get("kept_fields", []))
    report.require(
        {"y_identity", "target_representable_mask", "node_identity_id"} <= kept,
        f"{label} graph manifest includes identity audit fields",
    )

    artifact_records = graph_manifest.get("artifacts")
    report.require(isinstance(artifact_records, list), f"{label} graph manifest has an artifact list")
    if not isinstance(artifact_records, list):
        return
    report.require(
        graph_manifest.get("shard_count") == len(artifact_records),
        f"{label} graph shard count matches its artifact records",
    )
    recorded_paths: list[Path] = []
    resolved_artifacts: list[tuple[int, dict[str, Any], Path]] = []
    object_count_sum = 0
    processed_root = processed.resolve()
    for index, record in enumerate(artifact_records):
        if not isinstance(record, dict) or not record.get("path"):
            report.errors.append(f"{label} graph artifact {index:03d} is malformed")
            continue
        path = resolve_recorded_path(
            record["path"],
            manifest_path=graph_manifest_path,
            repository_root=ROOT,
        )
        recorded_paths.append(path)
        resolved_artifacts.append((index, record, path))
        try:
            confined = path.is_relative_to(processed_root)
        except ValueError:
            confined = False
        report.require(confined, f"{label} graph artifact {index:03d} is confined to its variant")
        object_count = record.get("object_count")
        report.require(
            isinstance(object_count, int) and not isinstance(object_count, bool) and object_count >= 0,
            f"{label} graph artifact {index:03d} records an object count",
        )
        if isinstance(object_count, int) and not isinstance(object_count, bool):
            object_count_sum += object_count
        report.require(
            isinstance(record.get("bytes"), int)
            and record.get("bytes", -1) >= 0
            and isinstance(record.get("sha256"), str)
            and len(record.get("sha256", "")) == 64,
            f"{label} graph artifact {index:03d} records size and full SHA-256",
        )

    report.require(
        len(recorded_paths) == len(set(recorded_paths)),
        f"{label} graph artifact paths are unique",
    )
    report.require(
        object_count_sum == graph_manifest.get("graph_count"),
        f"{label} graph artifact object counts sum to graph_count",
    )

    is_sharded = bool(graph_manifest.get("sharded"))
    indices = [shard_index(path) for path in recorded_paths]
    if is_sharded:
        report.require(
            indices == list(range(len(recorded_paths))),
            f"{label} graph shard numbering is contiguous from 000",
        )
    else:
        report.require(
            len(recorded_paths) == 1
            and recorded_paths[0] == graph_path.resolve(strict=False)
            and indices == [None],
            f"{label} monolithic graph artifact matches its base path",
        )

    if verify_hashes:
        actual_paths = _graph_artifact_candidates(graph_path)
        report.require(
            actual_paths == set(recorded_paths),
            f"{label} graph artifact set has no missing or extra payloads",
        )
        for index, record, path in resolved_artifacts:
            exists = path.exists()
            report.require(exists, f"{label} graph artifact {index:03d} exists")
            if not exists:
                continue
            report.require(
                path.stat().st_size == record.get("bytes"),
                f"{label} graph artifact {index:03d} size matches",
            )
            report.require(
                graph_file_sha256(path) == record.get("sha256"),
                f"{label} graph artifact {index:03d} full hash matches",
            )
            try:
                actual_count = _count_graph_objects(path, sharded=is_sharded)
            except Exception as exc:
                report.errors.append(
                    f"{label} graph artifact {index:03d} cannot be counted: {exc}"
                )
            else:
                report.require(
                    actual_count == record.get("object_count"),
                    f"{label} graph artifact {index:03d} object count matches",
                )

    _validate_graph_derivation(
        graph_manifest_path=graph_manifest_path,
        graph_manifest=graph_manifest,
        artifact_paths=recorded_paths,
        verify_hashes=verify_hashes,
        report=report,
        label=label,
    )
    if len(report.errors) == starting_error_count:
        report.validated_graph_manifests.append(
            {
                "path": str(graph_manifest_path.resolve()),
                "sha256": graph_file_sha256(graph_manifest_path),
                "hashes_verified": bool(verify_hashes),
            }
        )


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
    _validate_sampled_row_order(interim=interim, manifest=manifest, report=report)
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
        parquet_path = interim / f"df_{split}.parquet"
        graph_path = processed / graph_dataset_filename(
            split,
            encoding,
            constraint_representation=constraint_representation,
            primary_constraint_mode=primary_constraint_mode,
        )
        graph_manifest_path = _manifest_path(graph_path)
        graph_errors_before = len(report.errors)
        if parquet_path.exists():
            _validate_graph_manifest(
                graph_path=graph_path,
                graph_manifest_path=graph_manifest_path,
                dataset_manifest_path=manifest_path,
                parquet_path=parquet_path,
                processed=processed,
                dataset_variant=dataset_variant,
                split=split,
                encoding=encoding,
                constraint_representation=constraint_representation,
                primary_constraint_mode=primary_constraint_mode,
                verify_hashes=verify_hashes,
                report=report,
            )
        else:
            report.errors.append(f"{split} parquet is unavailable for graph lineage validation")
        if (
            split == "train"
            and len(report.errors) == graph_errors_before
            and discover_graph_artifacts(graph_path)
        ):
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


def _validate_global_results_contract(
    metrics: dict[str, Any],
    *,
    label: str,
    report: ValidationReport,
) -> None:
    global_metrics = metrics.get("global_metrics", {})
    overall = global_metrics.get("overall", {}) if isinstance(global_metrics, dict) else {}
    per_sample = (
        global_metrics.get("per_sample", {})
        if isinstance(global_metrics, dict)
        else {}
    )
    per_sample_gfr = per_sample.get("gfr") if isinstance(per_sample, dict) else None
    evaluated_rows = len(per_sample_gfr) if isinstance(per_sample_gfr, list) else 0
    report.require(
        evaluated_rows > 0,
        f"{label}: global metrics cover evaluation rows",
    )
    report.require(
        "primary_fix_rate" in overall,
        f"{label}: transition-based primary fix is present",
    )
    report.require(
        int(overall.get("primary_fix_denom_total", -1)) == evaluated_rows,
        f"{label}: primary-fix denominator covers every retained evaluation row",
    )

    pre_state_validation = (
        global_metrics.get("pre_state_validation", {})
        if isinstance(global_metrics, dict)
        else {}
    )
    checked_rows = (
        int(pre_state_validation.get("checked_rows", -1))
        if isinstance(pre_state_validation, dict)
        else -1
    )
    mismatch_count = (
        int(pre_state_validation.get("mismatch_count", -1))
        if isinstance(pre_state_validation, dict)
        else -1
    )
    source_counts = (
        pre_state_validation.get("source_counts", {})
        if isinstance(pre_state_validation, dict)
        else {}
    )
    report.require(
        checked_rows == evaluated_rows and mismatch_count == 0,
        f"{label}: every global PRE state matches authoritative labels",
    )
    valid_source_counts = isinstance(source_counts, dict) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in source_counts.values()
    )
    report.require(
        valid_source_counts
        and sum(source_counts.values()) == checked_rows,
        f"{label}: PRE-state validation source counts cover checked rows",
    )

    per_constraint = (
        global_metrics.get("per_constraint_type", {})
        if isinstance(global_metrics, dict)
        else {}
    )
    per_constraint_denom = (
        sum(
            int(values.get("primary_fix_denom_total", 0))
            for values in per_constraint.values()
            if isinstance(values, dict)
        )
        if isinstance(per_constraint, dict)
        else -1
    )
    report.require(
        per_constraint_denom == evaluated_rows,
        f"{label}: per-family primary-fix support sums to all evaluation rows",
    )

    for metric in ("srr", "sir"):
        numerator = int(overall.get(f"{metric}_total", 0))
        denominator = int(overall.get(f"{metric}_denom_total", 0))
        expected = numerator / denominator if denominator else 0.0
        observed = float(overall.get(metric, float("nan")))
        report.require(
            abs(observed - expected) < 1e-12,
            f"{label}: {metric.upper()} is the pooled ratio",
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
    report.require(
        isinstance(metrics.get("fallback_noop_count"), int)
        and int(metrics.get("fallback_noop_count", -1)) >= 0,
        f"{run_directory.name}: fallback no-op use is reported",
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
    _validate_global_results_contract(
        metrics,
        label=run_directory.name,
        report=report,
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
        report.require(
            isinstance(metrics.get("fallback_noop_count"), int)
            and int(metrics.get("fallback_noop_count", -1)) >= 0,
            f"{label}: fallback no-op use is reported",
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
        _validate_global_results_contract(metrics, label=label, report=report)


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
