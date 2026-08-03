#!/usr/bin/env python3
"""Safely unlink validated graph payloads while retaining provenance manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.graph_manifest import (  # noqa: E402
    GRAPH_SCHEMA_VERSION,
    atomic_write_json,
    file_sha256,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _lexical_recorded_path(
    recorded: str | Path,
    *,
    manifest_path: Path,
    repository_root: Path,
) -> Path:
    raw = Path(recorded)
    if raw.is_absolute():
        return Path(os.path.abspath(raw))
    repository_candidate = repository_root / raw
    manifest_candidate = manifest_path.parent / raw
    if repository_candidate.exists() or repository_candidate.is_symlink():
        return Path(os.path.abspath(repository_candidate))
    return Path(os.path.abspath(manifest_candidate))


def _validated_manifest_records(
    integrity_report: dict[str, Any],
    *,
    integrity_report_path: Path,
    repository_root: Path,
) -> dict[Path, dict[str, Any]]:
    if integrity_report.get("ok") is not True:
        raise ValueError(f"Integrity report is not successful: {integrity_report_path}")
    raw_records = integrity_report.get("validated_graph_manifests")
    if not isinstance(raw_records, list):
        raise ValueError("Integrity report has no validated_graph_manifests list.")
    records: dict[Path, dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError("Integrity report contains a malformed graph-manifest record.")
        path = _lexical_recorded_path(
            record["path"],
            manifest_path=integrity_report_path,
            repository_root=repository_root,
        )
        if not record.get("hashes_verified"):
            continue
        records[path] = record
    return records


def prune_graph_artifacts(
    *,
    graph_manifest_paths: list[Path],
    integrity_report_path: Path,
    receipt_path: Path,
    delete: bool,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate the complete prune plan, optionally unlink it, and write a receipt."""

    repository_root = repository_root.resolve()
    processed_root = (repository_root / "data" / "processed").resolve()
    integrity_report_path = integrity_report_path.resolve()
    integrity_report = _read_json(integrity_report_path)
    validated_records = _validated_manifest_records(
        integrity_report,
        integrity_report_path=integrity_report_path,
        repository_root=repository_root,
    )
    if not graph_manifest_paths:
        raise ValueError("At least one graph manifest is required.")

    manifest_receipts: list[dict[str, Any]] = []
    artifact_receipts: list[dict[str, Any]] = []
    target_paths: set[Path] = set()
    for raw_manifest_path in graph_manifest_paths:
        manifest_path = raw_manifest_path.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Graph manifest not found: {manifest_path}")
        try:
            relative_manifest = manifest_path.relative_to(processed_root)
        except ValueError as exc:
            raise ValueError(
                f"Graph manifest is outside data/processed: {manifest_path}"
            ) from exc
        if len(relative_manifest.parts) != 2:
            raise ValueError(
                "Graph manifests must be direct children of data/processed/<variant>/: "
                f"{manifest_path}"
            )
        variant_root = processed_root / relative_manifest.parts[0]

        manifest_hash = file_sha256(manifest_path)
        validated = validated_records.get(manifest_path)
        if validated is None:
            raise ValueError(
                f"Integrity report did not hash-verify graph manifest: {manifest_path}"
            )
        if validated.get("sha256") != manifest_hash:
            raise ValueError(
                f"Graph manifest hash does not match integrity report: {manifest_path}"
            )

        manifest = _read_json(manifest_path)
        if int(manifest.get("graph_schema_version", 0)) != GRAPH_SCHEMA_VERSION:
            raise ValueError(
                f"Graph manifest must use schema v{GRAPH_SCHEMA_VERSION}: {manifest_path}"
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(f"Graph manifest has no artifact list: {manifest_path}")

        manifest_artifacts: list[str] = []
        for index, record in enumerate(artifacts):
            if not isinstance(record, dict) or not record.get("path"):
                raise ValueError(
                    f"Malformed artifact record {index} in graph manifest {manifest_path}"
                )
            target = _lexical_recorded_path(
                record["path"],
                manifest_path=manifest_path,
                repository_root=repository_root,
            )
            resolved_target = target.resolve(strict=False)
            try:
                resolved_target.relative_to(variant_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Refusing artifact path outside data/processed/{variant_root.name}/: {target}"
                ) from exc
            if target == manifest_path or target.name.endswith(".manifest.json"):
                raise ValueError(f"Refusing to prune a graph manifest: {target}")
            if target in target_paths:
                raise ValueError(f"Artifact path is listed more than once: {target}")
            if not target.exists() and not target.is_symlink():
                raise FileNotFoundError(f"Graph artifact not found: {target}")
            if target.is_dir():
                raise ValueError(f"Graph artifact path is a directory: {target}")
            size = target.stat().st_size
            digest = file_sha256(target)
            if size != record.get("bytes"):
                raise ValueError(f"Graph artifact size changed since validation: {target}")
            if digest != record.get("sha256"):
                raise ValueError(f"Graph artifact hash changed since validation: {target}")
            target_paths.add(target)
            manifest_artifacts.append(str(target))
            artifact_receipts.append(
                {
                    "path": str(target),
                    "bytes": int(size),
                    "sha256": digest,
                    "graph_manifest_path": str(manifest_path),
                    "graph_manifest_sha256": manifest_hash,
                    "deleted": False,
                }
            )

        manifest_receipts.append(
            {
                "path": str(manifest_path),
                "sha256": manifest_hash,
                "retained": True,
                "artifact_paths": manifest_artifacts,
            }
        )

    if delete:
        by_path = {Path(record["path"]): record for record in artifact_receipts}
        for target in sorted(target_paths):
            target.unlink()
            by_path[target]["deleted"] = True

    receipt = {
        "pruning_receipt_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "delete" if delete else "dry_run",
        "integrity_report": {
            "path": str(integrity_report_path),
            "sha256": file_sha256(integrity_report_path),
            "ok": True,
        },
        "graph_manifests": manifest_receipts,
        "artifacts": artifact_receipts,
        "artifact_count": len(artifact_receipts),
        "total_bytes": sum(int(record["bytes"]) for record in artifact_receipts),
    }
    atomic_write_json(receipt_path.resolve(), receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-manifest",
        type=Path,
        nargs="+",
        required=True,
        help="One or more schema-v3 graph manifests whose payload names may be unlinked.",
    )
    parser.add_argument("--integrity-report", type=Path, required=True)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Receipt path (defaults beside the integrity report).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--delete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt_path = args.receipt
    if receipt_path is None:
        receipt_path = args.integrity_report.with_name(
            f"{args.integrity_report.stem}.pruning_receipt.json"
        )
    receipt = prune_graph_artifacts(
        graph_manifest_paths=args.graph_manifest,
        integrity_report_path=args.integrity_report,
        receipt_path=receipt_path,
        delete=bool(args.delete),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
