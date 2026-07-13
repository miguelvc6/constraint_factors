from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 1


def canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_hash(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return canonical_json_hash(json.load(handle))


def parameter_counts(model: torch.nn.Module) -> dict[str, object]:
    by_component: dict[str, int] = {}
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
        component = name.split(".", 1)[0]
        by_component[component] = by_component.get(component, 0) + count
    return {
        "total": total,
        "trainable": trainable,
        "by_component": dict(sorted(by_component.items())),
    }


def _git_state(repository_root: Path) -> dict[str, object]:
    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain=v1", "--untracked-files=all")
    diff = _run("diff", "--binary", "HEAD", "--", "src", "scripts", "pyproject.toml", "uv.lock")
    source_digest = hashlib.sha256()
    source_files = [
        path
        for directory in (repository_root / "src", repository_root / "scripts")
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    ]
    for filename in ("pyproject.toml", "uv.lock"):
        path = repository_root / filename
        if path.exists():
            source_files.append(path)
    for path in sorted(set(source_files)):
        source_digest.update(str(path.relative_to(repository_root)).encode("utf-8"))
        source_digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                source_digest.update(chunk)

    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status_sha256": hashlib.sha256((status or "").encode("utf-8")).hexdigest(),
        "source_diff_sha256": hashlib.sha256((diff or "").encode("utf-8")).hexdigest(),
        "source_tree_sha256": source_digest.hexdigest(),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "pyarrow", "torch", "torch-geometric"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _optional_json_record(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return {
        "path": str(path),
        "sha256": json_file_hash(path),
    }


def build_run_provenance(
    *,
    repository_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    model: torch.nn.Module,
    seed: int,
    dataset_manifest_path: Path | None,
    graph_manifest_paths: Sequence[Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        effective_config = json.load(handle)
    dataset_manifest = (
        _optional_json_record(dataset_manifest_path)
        if dataset_manifest_path is not None
        else None
    )
    graph_manifests = [
        record
        for path in graph_manifest_paths
        if (record := _optional_json_record(path)) is not None
    ]
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": int(seed),
        "config": {
            "path": str(config_path),
            "sha256": canonical_json_hash(effective_config),
            "model_config_sha256": canonical_json_hash(effective_config.get("model_config", {})),
            "training_config_sha256": canonical_json_hash(effective_config.get("training_config", {})),
        },
        "checkpoint": _artifact_record(checkpoint_path),
        "parameters": parameter_counts(model),
        "dataset_manifest": dataset_manifest,
        "graph_manifests": graph_manifests,
        "source": _git_state(repository_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
            "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        },
        "extra": dict(extra or {}),
    }


def write_run_manifest(run_directory: Path, payload: Mapping[str, object]) -> Path:
    path = run_directory / RUN_MANIFEST_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def config_differences(
    requested: Mapping[str, Any],
    recorded: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(requested) | set(recorded)):
        requested_value = requested.get(key)
        recorded_value = recorded.get(key)
        if canonical_json_hash(requested_value) != canonical_json_hash(recorded_value):
            differences[key] = (requested_value, recorded_value)
    return differences
