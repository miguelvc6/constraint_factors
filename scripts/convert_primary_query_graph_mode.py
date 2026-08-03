#!/usr/bin/env python3
"""Derive primary-query graph artifacts from an existing factorized graph mode.

This is intended for modes that are representational ablations over the same
row set. It avoids recomputing raw parquet rows when the source graph already
contains the primary-query metadata needed by the target mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from modules.data_encoders import (  # noqa: E402
    ArtifactWriteResult,
    GRAPH_SCHEMA_VERSION,
    GlobalIntEncoder,
    _torch_load_trusted,
    dataset_variant_name,
    graph_dataset_filename,
)
from modules.graph_manifest import (  # noqa: E402
    atomic_write_json,
    file_sha256,
    graph_incomplete_marker_path,
    graph_manifest_path,
    resolve_recorded_path,
    shard_index,
)

EDGE_FACTOR_TO_PARAM_PREDICATE = 2
EDGE_PARAM_PREDICATE_TO_OBJECT = 3


def _manifest_path(path: Path) -> Path:
    return graph_manifest_path(path)


def _load_objects(path: Path) -> list[Any]:
    if path.suffix == ".pt":
        payload = _torch_load_trusted(path)
    else:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}, got {type(payload)!r}")
    return payload


def _write_objects(
    objects: list[Any],
    target_base: Path,
    shard_index: int,
    use_torch_save: bool,
    atomic: bool,
) -> ArtifactWriteResult:
    suffix = ".pt" if use_torch_save else target_base.suffix
    target = target_base.with_name(f"{target_base.stem}-shard{shard_index:03d}{suffix}")
    destination = target.with_suffix(target.suffix + ".tmp") if atomic else target
    target.parent.mkdir(parents=True, exist_ok=True)
    if use_torch_save:
        torch.save(objects, destination)
    else:
        with destination.open("wb") as fh:
            pickle.dump(objects, fh, protocol=5)
    if atomic:
        destination.replace(target)
    return ArtifactWriteResult(
        path=target,
        bytes_written=target.stat().st_size,
        checksum=file_sha256(target),
        object_count=len(objects),
    )


def _clear_target_artifacts(target_base: Path) -> None:
    target_base.unlink(missing_ok=True)
    _manifest_path(target_base).unlink(missing_ok=True)
    graph_incomplete_marker_path(target_base).unlink(missing_ok=True)
    for suffix in ("pkl", "pt"):
        for path in target_base.parent.glob(f"{target_base.stem}-shard*.{suffix}"):
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def _node_attribute_for_global_id(encoder: GlobalIntEncoder, global_id: int) -> int:
    unknown_id = encoder.encode("unknown", add_new=False)
    if global_id in encoder._filtered_ids:
        global_id = unknown_id
    return int(encoder.get_unfiltered_global_id(global_id))


def _first_local_node_for_attribute(x: torch.Tensor, attr: int) -> int | None:
    matches = torch.nonzero(x.view(-1) == int(attr), as_tuple=False).view(-1)
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


def _primary_constraint_id(graph: Any) -> int:
    value = getattr(graph, "primary_constraint_id", None)
    if value is None:
        value = getattr(graph, "shape_id", None)
    if isinstance(value, torch.Tensor):
        return int(value.view(-1)[0].item())
    return int(value)


def _as_1d_long(value: Any) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.long)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=torch.long).view(-1)


def _append_node_id(graph: Any, attr: int) -> int:
    if not isinstance(graph.x, torch.Tensor) or graph.x.ndim != 1:
        raise ValueError("Passive conversion currently supports node_id graphs only.")
    new_index = int(graph.x.numel())
    graph.x = torch.cat([graph.x, torch.tensor([int(attr)], dtype=graph.x.dtype)])
    role_flags = getattr(graph, "role_flags", None)
    if isinstance(role_flags, torch.Tensor):
        graph.role_flags = torch.cat(
            [role_flags, torch.zeros(1, dtype=role_flags.dtype)]
        )
    is_factor_node = getattr(graph, "is_factor_node", None)
    if isinstance(is_factor_node, torch.Tensor):
        graph.is_factor_node = torch.cat(
            [is_factor_node, torch.zeros(1, dtype=is_factor_node.dtype)]
        )
    if hasattr(graph, "x_names"):
        graph.x_names = [*list(graph.x_names), ""]
    return new_index


def _append_edges(
    graph: Any,
    edge_pairs: list[tuple[int, int]],
    edge_types: list[int],
    non_flattened_pairs: list[tuple[int, int]],
    non_flattened_attrs: list[int],
) -> None:
    if edge_pairs:
        new_edges = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        graph.edge_type = torch.cat(
            [graph.edge_type, torch.tensor(edge_types, dtype=graph.edge_type.dtype)]
        )
    if non_flattened_pairs:
        nf_edges = torch.tensor(non_flattened_pairs, dtype=torch.long).t().contiguous()
        graph.edge_index_non_flattened = torch.cat(
            [graph.edge_index_non_flattened, nf_edges],
            dim=1,
        )
        graph.edge_attr_non_flattened = torch.cat(
            [
                graph.edge_attr_non_flattened,
                torch.tensor(
                    non_flattened_attrs,
                    dtype=graph.edge_attr_non_flattened.dtype,
                ),
            ]
        )


def _convert_query_metadata_only(graph: Any, target_mode: str) -> Any:
    graph.primary_constraint_mode = target_mode
    return graph


def _convert_passive_node(graph: Any, encoder: GlobalIntEncoder) -> Any:
    graph.primary_constraint_mode = "passive_node"
    primary_id = _primary_constraint_id(graph)
    constraint_token = encoder._decoding.get(primary_id)
    if constraint_token in (None, "", "unknown"):
        raise ValueError(f"Cannot resolve primary constraint token for id={primary_id}")
    factor_gid = encoder.encode(f"constraint_factor::{constraint_token}", add_new=False)
    if factor_gid == 0:
        raise ValueError(f"Missing factor token for primary constraint id={primary_id}")
    passive_index = _append_node_id(graph, _node_attribute_for_global_id(encoder, factor_gid))
    graph.passive_primary_node_index = int(passive_index)

    edge_pairs: list[tuple[int, int]] = []
    edge_types: list[int] = []
    non_flattened_pairs: list[tuple[int, int]] = []
    non_flattened_attrs: list[int] = []
    predicates = _as_1d_long(getattr(graph, "primary_param_predicate_ids", None))
    objects = _as_1d_long(getattr(graph, "primary_param_object_ids", None))
    if predicates.numel() != objects.numel():
        raise ValueError(
            "primary_param_predicate_ids and primary_param_object_ids length mismatch"
        )
    for predicate_gid_raw, object_gid_raw in zip(predicates.tolist(), objects.tolist()):
        predicate_attr = _node_attribute_for_global_id(encoder, int(predicate_gid_raw))
        object_attr = _node_attribute_for_global_id(encoder, int(object_gid_raw))
        predicate_index = _append_node_id(graph, predicate_attr)
        object_index = _first_local_node_for_attribute(graph.x, object_attr)
        if object_index is None:
            object_index = _append_node_id(graph, object_attr)
        edge_pairs.append((passive_index, predicate_index))
        edge_types.append(EDGE_FACTOR_TO_PARAM_PREDICATE)
        edge_pairs.append((predicate_index, object_index))
        edge_types.append(EDGE_PARAM_PREDICATE_TO_OBJECT)
        non_flattened_pairs.append((passive_index, object_index))
        non_flattened_attrs.append(predicate_index)
    _append_edges(graph, edge_pairs, edge_types, non_flattened_pairs, non_flattened_attrs)
    return graph


def _iter_source_shards(source_base: Path, use_torch_save: bool) -> list[Path]:
    suffix = ".pt" if use_torch_save else ".pkl"
    shards = sorted(source_base.parent.glob(f"{source_base.stem}-shard*{suffix}"))
    if not shards:
        raise FileNotFoundError(f"No source shards found for {source_base}")
    return shards


def _discover_source_artifacts(source_base: Path) -> set[Path]:
    artifacts: set[Path] = set()
    if source_base.exists() or source_base.is_symlink():
        artifacts.add(source_base.resolve(strict=False))
    for suffix in ("pkl", "pt"):
        artifacts.update(
            path.resolve(strict=False)
            for path in source_base.parent.glob(f"{source_base.stem}-shard*.{suffix}")
        )
    return artifacts


def _copy_manifest(
    source_manifest: Path,
    target_manifest: Path,
    *,
    artifact_writes: list[ArtifactWriteResult],
    graph_count: int,
    target_mode: str,
    derivation_method: str,
    artifact_lineage: list[dict[str, object]],
) -> None:
    with source_manifest.open("r", encoding="utf-8") as fh:
        source_payload = json.load(fh)
    if int(source_payload.get("graph_schema_version", 0)) != GRAPH_SCHEMA_VERSION:
        raise ValueError(
            f"Source graph manifest must use schema v{GRAPH_SCHEMA_VERSION}: {source_manifest}"
        )
    payload = dict(source_payload)
    payload.update(
        {
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "primary_constraint_mode": target_mode,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "graph_count": int(graph_count),
            "shard_count": len(artifact_writes),
            "derivation": {
                "method": derivation_method,
                "source_manifest": {
                    "path": str(source_manifest.resolve()),
                    "sha256": file_sha256(source_manifest),
                },
                "source_primary_constraint_mode": source_payload.get(
                    "primary_constraint_mode"
                ),
                "target_primary_constraint_mode": target_mode,
                "artifacts": artifact_lineage,
            },
            "artifacts": [
                {
                    "path": str(artifact.path.resolve()),
                    "bytes": int(artifact.bytes_written),
                    "object_count": int(artifact.object_count),
                    "sha256": artifact.checksum,
                }
                for artifact in artifact_writes
            ],
        }
    )
    atomic_write_json(target_manifest, payload)


def _load_source_manifest(source_base: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _manifest_path(source_base)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source graph manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in source graph manifest: {manifest_path}")
    if int(payload.get("graph_schema_version", 0)) != GRAPH_SCHEMA_VERSION:
        raise ValueError(
            f"Source graph manifest must use schema v{GRAPH_SCHEMA_VERSION}: {manifest_path}"
        )
    return manifest_path, payload


def _source_artifact_records(
    source_manifest: Path,
    payload: dict[str, Any],
) -> dict[Path, dict[str, Any]]:
    records: dict[Path, dict[str, Any]] = {}
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Source graph manifest has no artifact list: {source_manifest}")
    for record in artifacts:
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError(f"Malformed source artifact record in {source_manifest}")
        path = resolve_recorded_path(record["path"], manifest_path=source_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Source graph artifact not found: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"Source graph artifact size mismatch: {path}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"Source graph artifact hash mismatch: {path}")
        records[path] = record
    return records


def convert_split(
    split: str,
    processed_dir: Path,
    encoding: str,
    source_mode: str,
    target_mode: str,
    encoder: GlobalIntEncoder | None,
    overwrite: bool,
    atomic: bool,
    link_identical_structure: bool = False,
) -> None:
    source_base = processed_dir / graph_dataset_filename(
        split,
        encoding,
        primary_constraint_mode=source_mode,
    )
    target_base = processed_dir / graph_dataset_filename(
        split,
        encoding,
        primary_constraint_mode=target_mode,
    )
    use_torch_save = any(source_base.parent.glob(f"{source_base.stem}-shard*.pt"))
    source_shards = _iter_source_shards(source_base, use_torch_save=use_torch_save)
    indices = [shard_index(path) for path in source_shards]
    if indices != list(range(len(source_shards))):
        raise ValueError(f"Source shards must be contiguous from 000: {source_shards}")
    source_manifest, source_payload = _load_source_manifest(source_base)
    if source_payload.get("primary_constraint_mode") != source_mode:
        raise ValueError(
            "Source graph manifest primary mode does not match --source-mode: "
            f"{source_manifest}"
        )
    source_records = _source_artifact_records(source_manifest, source_payload)
    if _discover_source_artifacts(source_base) != set(source_records):
        raise ValueError("Source graph manifest artifact set does not match source payloads.")
    if set(path.resolve() for path in source_shards) != set(source_records):
        raise ValueError("Source graph manifest mixes unsupported payload formats.")

    target_exists = (
        target_base.exists()
        or _manifest_path(target_base).exists()
        or bool(list(target_base.parent.glob(f"{target_base.stem}-shard*.pt")))
        or bool(list(target_base.parent.glob(f"{target_base.stem}-shard*.pkl")))
    )
    if target_exists:
        if not overwrite:
            raise FileExistsError(f"Target artifacts already exist for {target_base}")
        _clear_target_artifacts(target_base)
    target_manifest = _manifest_path(target_base)
    target_manifest.unlink(missing_ok=True)
    incomplete_marker = graph_incomplete_marker_path(target_base)
    atomic_write_json(
        incomplete_marker,
        {
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "split": split,
            "encoding": encoding,
            "source_primary_constraint_mode": source_mode,
            "target_primary_constraint_mode": target_mode,
            "status": "incomplete",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    if link_identical_structure:
        if target_mode == "passive_node":
            raise ValueError("passive_node changes graph structure and cannot be hard-linked")
        artifact_writes = []
        artifact_lineage: list[dict[str, object]] = []
        graph_count = int(source_payload.get("graph_count") or 0)
        for shard_idx, source_shard in enumerate(source_shards):
            suffix = source_shard.suffix
            target = target_base.with_name(f"{target_base.stem}-shard{shard_idx:03d}{suffix}")
            logging.info("Linking %s -> %s", source_shard, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_shard, target)
            source_record = source_records[source_shard.resolve()]
            artifact_writes.append(
                ArtifactWriteResult(
                    path=target,
                    bytes_written=target.stat().st_size,
                    checksum=file_sha256(target),
                    object_count=int(source_record["object_count"]),
                )
            )
            artifact_lineage.append(
                {
                    "source_path": str(source_shard.resolve()),
                    "target_path": str(target.resolve()),
                    "link_type": "hard_link",
                    "sha256": source_record["sha256"],
                }
            )
        if graph_count <= 0:
            graph_count = sum(len(_load_objects(path)) for path in source_shards)
        _copy_manifest(
            source_manifest,
            target_manifest,
            artifact_writes=artifact_writes,
            graph_count=graph_count,
            target_mode=target_mode,
            derivation_method="hard_link",
            artifact_lineage=artifact_lineage,
        )
        incomplete_marker.unlink(missing_ok=True)
        return

    artifact_writes: list[ArtifactWriteResult] = []
    artifact_lineage = []
    graph_count = 0
    for shard_idx, source_shard in enumerate(source_shards):
        logging.info("Converting %s -> %s", source_shard, target_base.name)
        objects = _load_objects(source_shard)
        converted = []
        for graph in objects:
            if target_mode == "passive_node":
                if encoder is None:
                    raise ValueError("passive_node conversion requires a GlobalIntEncoder")
                converted.append(_convert_passive_node(graph, encoder))
            else:
                converted.append(_convert_query_metadata_only(graph, target_mode))
        artifact_writes.append(
            _write_objects(
                converted,
                target_base,
                shard_idx,
                use_torch_save=use_torch_save,
                atomic=atomic,
            )
        )
        graph_count += len(converted)
        artifact_lineage.append(
            {
                "source_path": str(source_shard.resolve()),
                "target_path": str(artifact_writes[-1].path.resolve()),
                "link_type": "rewritten",
                "source_sha256": source_records[source_shard.resolve()]["sha256"],
                "target_sha256": artifact_writes[-1].checksum,
            }
        )
        del objects
        del converted

    _copy_manifest(
        _manifest_path(source_base),
        target_manifest,
        artifact_writes=artifact_writes,
        graph_count=graph_count,
        target_mode=target_mode,
        derivation_method="rewrite",
        artifact_lineage=artifact_lineage,
    )
    incomplete_marker.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="full_strat1m")
    parser.add_argument("--min-occurrence", type=int, default=100)
    parser.add_argument("--encoding", default="node_id")
    parser.add_argument("--source-mode", default="query_family")
    parser.add_argument(
        "--target-mode",
        choices=["query_definition", "query_family", "passive_node", "none"],
        required=True,
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--unsafe-write", action="store_true")
    parser.add_argument(
        "--link-identical-structure",
        action="store_true",
        help=(
            "Create hard-linked target shards for modes with identical graph "
            "structure. The graph payload keeps the source mode label; the manifest and "
            "artifact names use the target mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    variant = dataset_variant_name(args.dataset, args.min_occurrence)
    processed_dir = Path("data/processed") / variant
    interim_dir = Path("data/interim") / variant
    encoder = None
    if args.target_mode == "passive_node":
        encoder = GlobalIntEncoder()
        encoder.load(interim_dir / "globalintencoder.txt")
    for split in args.splits:
        convert_split(
            split=split,
            processed_dir=processed_dir,
            encoding=args.encoding,
            source_mode=args.source_mode,
            target_mode=args.target_mode,
            encoder=encoder,
            overwrite=args.overwrite,
            atomic=not args.unsafe_write,
            link_identical_structure=args.link_identical_structure,
        )


if __name__ == "__main__":
    main()
