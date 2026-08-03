from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.data_encoders import GRAPH_SCHEMA_VERSION, graph_dataset_filename
from modules.graph_manifest import (
    atomic_write_json,
    file_sha256,
    graph_incomplete_marker_path,
    graph_manifest_path,
)
from scripts.convert_primary_query_graph_mode import convert_split
from scripts.prune_graph_artifacts import prune_graph_artifacts
from scripts.validate_scientific_integrity import (
    ValidationReport,
    _validate_global_results_contract,
    _validate_graph_manifest,
    _validate_sampled_row_order,
)


def _graph(label: int) -> Data:
    return Data(
        x=torch.tensor([label], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([[label, 0, 0, 0, 0, 0]], dtype=torch.long),
    )


def test_results_contract_rejects_partial_primary_support_and_missing_pre_audit() -> None:
    metrics = {
        "global_metrics": {
            "overall": {
                "primary_fix_rate": 1.0,
                "primary_fix_denom_total": 1,
                "srr": 0.0,
                "srr_total": 0,
                "srr_denom_total": 0,
                "sir": 0.0,
                "sir_total": 0,
                "sir_denom_total": 0,
            },
            "per_constraint_type": {
                "single": {"primary_fix_denom_total": 1},
            },
            "per_sample": {"gfr": [1.0, 1.0]},
        }
    }
    report = ValidationReport()

    _validate_global_results_contract(metrics, label="toy", report=report)

    assert any("primary-fix denominator" in error for error in report.errors)
    assert any("PRE state" in error for error in report.errors)


def test_results_contract_accepts_full_support_and_pre_state_parity() -> None:
    metrics = {
        "global_metrics": {
            "overall": {
                "primary_fix_rate": 0.5,
                "primary_fix_denom_total": 2,
                "srr": 0.5,
                "srr_total": 1,
                "srr_denom_total": 2,
                "sir": 0.0,
                "sir_total": 0,
                "sir_denom_total": 0,
            },
            "per_constraint_type": {
                "single": {"primary_fix_denom_total": 2},
            },
            "pre_state_validation": {
                "checked_rows": 2,
                "mismatch_count": 0,
                "source_counts": {"parquet": 2},
            },
            "per_sample": {"gfr": [1.0, 0.0]},
        }
    }
    report = ValidationReport()

    _validate_global_results_contract(metrics, label="toy", report=report)

    assert report.errors == []


def _write_fixture(
    root: Path,
    *,
    primary_mode: str = "executable_factor",
) -> dict[str, Path]:
    variant = "toy"
    interim = root / "data" / "interim" / variant
    processed = root / "data" / "processed" / variant
    interim.mkdir(parents=True)
    processed.mkdir(parents=True)
    dataset_manifest = interim / "dataset_manifest.json"
    dataset_manifest.write_text('{"schema_version": 2}\n', encoding="utf-8")
    parquet = interim / "df_train.parquet"
    pd.DataFrame({"row": [0, 1, 2]}).to_parquet(parquet, index=False)

    graph_path = processed / graph_dataset_filename(
        "train",
        "node_id",
        primary_constraint_mode=primary_mode,
    )
    shard_paths = [
        graph_path.with_name(f"{graph_path.stem}-shard000.pt"),
        graph_path.with_name(f"{graph_path.stem}-shard001.pt"),
    ]
    torch.save([_graph(1), _graph(2)], shard_paths[0])
    torch.save([_graph(3)], shard_paths[1])
    manifest = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "split": "train",
        "dataset_variant": variant,
        "encoding": "node_id",
        "constraint_scope": "local",
        "constraint_representation": "factorized",
        "primary_constraint_mode": primary_mode,
        "build_limit": None,
        "dataset_manifest": {
            "path": str(dataset_manifest.resolve()),
            "sha256": file_sha256(dataset_manifest),
        },
        "source_parquet": {
            "path": str(parquet.resolve()),
            "sha256": file_sha256(parquet),
            "row_count": 3,
        },
        "derivation": None,
        "graph_count": 3,
        "shard_count": 2,
        "shard_size": 2,
        "sharded": True,
        "use_torch_save": True,
        "persistence_profile": "research_safe",
        "overwrite_mode": "atomic",
        "kept_fields": [
            "y_identity",
            "target_representable_mask",
            "node_identity_id",
        ],
        "dropped_fields": [],
        "artifacts": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "object_count": count,
                "sha256": file_sha256(path),
            }
            for path, count in zip(shard_paths, (2, 1))
        ],
    }
    manifest_path = graph_manifest_path(graph_path)
    atomic_write_json(manifest_path, manifest)
    return {
        "root": root,
        "interim": interim,
        "processed": processed,
        "dataset_manifest": dataset_manifest,
        "parquet": parquet,
        "graph_path": graph_path,
        "manifest": manifest_path,
        "shard0": shard_paths[0],
        "shard1": shard_paths[1],
    }


def _validate(paths: dict[str, Path], *, primary_mode: str = "executable_factor") -> ValidationReport:
    report = ValidationReport()
    _validate_graph_manifest(
        graph_path=paths["graph_path"],
        graph_manifest_path=paths["manifest"],
        dataset_manifest_path=paths["dataset_manifest"],
        parquet_path=paths["parquet"],
        processed=paths["processed"],
        dataset_variant="toy",
        split="train",
        encoding="node_id",
        constraint_representation="factorized",
        primary_constraint_mode=primary_mode,
        verify_hashes=True,
        report=report,
    )
    return report


def _mutate_manifest(paths: dict[str, Path], mutate) -> None:
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    mutate(payload)
    atomic_write_json(paths["manifest"], payload)


def test_schema_v3_graph_manifest_passes_full_integrity(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    report = _validate(paths)

    assert report.errors == []
    assert report.validated_graph_manifests == [
        {
            "path": str(paths["manifest"].resolve()),
            "sha256": file_sha256(paths["manifest"]),
            "hashes_verified": True,
        }
    ]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("missing", "no missing or extra payloads"),
        ("extra", "no missing or extra payloads"),
        ("size", "size matches"),
        ("hash", "full hash matches"),
        ("numbering", "numbering is contiguous"),
        ("count", "object counts sum to graph_count"),
        ("dataset", "current dataset manifest"),
        ("parquet", "current split parquet"),
        ("schema", "schema is exactly v3"),
        ("marker", "no incomplete graph-build marker"),
        ("graph_count", "consistent with source rows and build limit"),
    ],
)
def test_graph_integrity_rejects_corruption(
    tmp_path: Path,
    case: str,
    expected: str,
) -> None:
    paths = _write_fixture(tmp_path)
    if case == "missing":
        paths["shard1"].unlink()
    elif case == "extra":
        torch.save([_graph(4)], paths["graph_path"].with_name(f"{paths['graph_path'].stem}-shard002.pt"))
    elif case == "size":
        _mutate_manifest(paths, lambda payload: payload["artifacts"][0].__setitem__("bytes", 1))
    elif case == "hash":
        _mutate_manifest(paths, lambda payload: payload["artifacts"][0].__setitem__("sha256", "0" * 64))
    elif case == "numbering":
        new_path = paths["graph_path"].with_name(f"{paths['graph_path'].stem}-shard002.pt")
        paths["shard1"].rename(new_path)
        _mutate_manifest(
            paths,
            lambda payload: payload["artifacts"][1].__setitem__("path", str(new_path.resolve())),
        )
    elif case == "count":
        _mutate_manifest(
            paths,
            lambda payload: payload["artifacts"][0].__setitem__("object_count", 1),
        )
    elif case == "dataset":
        _mutate_manifest(
            paths,
            lambda payload: payload["dataset_manifest"].__setitem__("sha256", "0" * 64),
        )
    elif case == "parquet":
        _mutate_manifest(
            paths,
            lambda payload: payload["source_parquet"].__setitem__("sha256", "0" * 64),
        )
    elif case == "schema":
        _mutate_manifest(paths, lambda payload: payload.__setitem__("graph_schema_version", 2))
    elif case == "marker":
        graph_incomplete_marker_path(paths["graph_path"]).write_text("incomplete", encoding="utf-8")
    elif case == "graph_count":
        _mutate_manifest(paths, lambda payload: payload.__setitem__("graph_count", 2))

    report = _validate(paths)

    assert any(expected in error for error in report.errors), report.errors
    assert report.validated_graph_manifests == []


def test_converted_and_hard_linked_modes_preserve_lineage(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path, primary_mode="query_family")
    convert_split(
        split="train",
        processed_dir=paths["processed"],
        encoding="node_id",
        source_mode="query_family",
        target_mode="query_definition",
        encoder=None,
        overwrite=False,
        atomic=True,
        link_identical_structure=True,
    )
    linked = dict(paths)
    linked["graph_path"] = paths["processed"] / graph_dataset_filename(
        "train", "node_id", primary_constraint_mode="query_definition"
    )
    linked["manifest"] = graph_manifest_path(linked["graph_path"])
    linked_report = _validate(linked, primary_mode="query_definition")
    assert linked_report.errors == []
    linked_payload = json.loads(linked["manifest"].read_text(encoding="utf-8"))
    assert linked_payload["derivation"]["method"] == "hard_link"
    assert os.path.samefile(
        paths["shard0"],
        Path(linked_payload["artifacts"][0]["path"]),
    )

    convert_split(
        split="train",
        processed_dir=paths["processed"],
        encoding="node_id",
        source_mode="query_family",
        target_mode="none",
        encoder=None,
        overwrite=False,
        atomic=True,
        link_identical_structure=False,
    )
    rewritten = dict(paths)
    rewritten["graph_path"] = paths["processed"] / graph_dataset_filename(
        "train", "node_id", primary_constraint_mode="none"
    )
    rewritten["manifest"] = graph_manifest_path(rewritten["graph_path"])
    rewritten_report = _validate(rewritten, primary_mode="none")
    assert rewritten_report.errors == []
    rewritten_payload = json.loads(rewritten["manifest"].read_text(encoding="utf-8"))
    assert rewritten_payload["derivation"]["method"] == "rewrite"
    assert rewritten_payload["dataset_manifest"] == linked_payload["dataset_manifest"]
    assert rewritten_payload["source_parquet"] == linked_payload["source_parquet"]


def _write_integrity_report(path: Path, report: ValidationReport) -> None:
    atomic_write_json(path, report.as_dict())


def test_pruning_is_report_gated_and_hard_link_safe(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path, primary_mode="query_family")
    convert_split(
        split="train",
        processed_dir=paths["processed"],
        encoding="node_id",
        source_mode="query_family",
        target_mode="query_definition",
        encoder=None,
        overwrite=False,
        atomic=True,
        link_identical_structure=True,
    )
    linked = dict(paths)
    linked["graph_path"] = paths["processed"] / graph_dataset_filename(
        "train", "node_id", primary_constraint_mode="query_definition"
    )
    linked["manifest"] = graph_manifest_path(linked["graph_path"])
    report = _validate(linked, primary_mode="query_definition")
    integrity_path = tmp_path / "acceptance.json"
    _write_integrity_report(integrity_path, report)
    dry_receipt = tmp_path / "dry.json"

    dry = prune_graph_artifacts(
        graph_manifest_paths=[linked["manifest"]],
        integrity_report_path=integrity_path,
        receipt_path=dry_receipt,
        delete=False,
        repository_root=tmp_path,
    )
    target_shard = Path(dry["artifacts"][0]["path"])
    assert target_shard.exists()
    assert all(record["deleted"] is False for record in dry["artifacts"])
    assert dry_receipt.exists()

    delete_receipt = tmp_path / "delete.json"
    deleted = prune_graph_artifacts(
        graph_manifest_paths=[linked["manifest"]],
        integrity_report_path=integrity_path,
        receipt_path=delete_receipt,
        delete=True,
        repository_root=tmp_path,
    )
    assert all(record["deleted"] is True for record in deleted["artifacts"])
    assert not target_shard.exists()
    assert paths["shard0"].exists()
    assert paths["shard1"].exists()
    assert linked["manifest"].exists()
    receipt_payload = json.loads(delete_receipt.read_text(encoding="utf-8"))
    assert receipt_payload["integrity_report"]["sha256"] == file_sha256(integrity_path)
    assert receipt_payload["graph_manifests"][0]["retained"] is True


def test_pruning_rejects_report_mismatch_and_path_escape(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    report = _validate(paths)
    integrity_path = tmp_path / "acceptance.json"
    payload = report.as_dict()
    payload["validated_graph_manifests"][0]["sha256"] = "0" * 64
    atomic_write_json(integrity_path, payload)
    with pytest.raises(ValueError, match="hash does not match"):
        prune_graph_artifacts(
            graph_manifest_paths=[paths["manifest"]],
            integrity_report_path=integrity_path,
            receipt_path=tmp_path / "rejected.json",
            delete=False,
            repository_root=tmp_path,
        )

    outside = tmp_path / "outside.pt"
    torch.save([_graph(9)], outside)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["shard_count"] = 1
    manifest["graph_count"] = 1
    manifest["artifacts"] = [
        {
            "path": str(outside),
            "bytes": outside.stat().st_size,
            "object_count": 1,
            "sha256": file_sha256(outside),
        }
    ]
    atomic_write_json(paths["manifest"], manifest)
    atomic_write_json(
        integrity_path,
        {
            "ok": True,
            "validated_graph_manifests": [
                {
                    "path": str(paths["manifest"]),
                    "sha256": file_sha256(paths["manifest"]),
                    "hashes_verified": True,
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="outside data/processed"):
        prune_graph_artifacts(
            graph_manifest_paths=[paths["manifest"]],
            integrity_report_path=integrity_path,
            receipt_path=tmp_path / "escape.json",
            delete=False,
            repository_root=tmp_path,
        )
    assert outside.exists()


def test_sampled_row_order_contract_and_unsampled_compatibility(tmp_path: Path) -> None:
    metadata = {
        "seed": 42,
        "row_order": {
            "method": "splitmix64_source_index_v1",
            "seed": 42,
            "external_sort_buckets": 64,
        },
    }
    metadata_path = tmp_path / "sampling_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    sampled = {
        "sampling": {
            "seed": 42,
            "target_rows": 10,
            "sample_fraction": None,
            "row_order": metadata["row_order"],
        },
        "outputs": {"sampling_metadata.json": file_sha256(metadata_path)},
    }
    report = ValidationReport()
    _validate_sampled_row_order(interim=tmp_path, manifest=sampled, report=report)
    assert report.errors == []

    legacy = json.loads(json.dumps(sampled))
    legacy["sampling"].pop("row_order")
    metadata_path.write_text(json.dumps({"seed": 42}), encoding="utf-8")
    legacy["outputs"]["sampling_metadata.json"] = file_sha256(metadata_path)
    legacy_report = ValidationReport()
    _validate_sampled_row_order(interim=tmp_path, manifest=legacy, report=legacy_report)
    assert any("sampling.row_order provenance" in error for error in legacy_report.errors)

    unsampled_report = ValidationReport()
    _validate_sampled_row_order(
        interim=tmp_path / "missing",
        manifest={"sampling": None},
        report=unsampled_report,
    )
    assert unsampled_report.errors == []
