from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


GRAPH_SCHEMA_VERSION = 3
_SHARD_INDEX_RE = re.compile(r"-shard(?P<index>\d+)\.(?:pkl|pt)$")


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the full SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def graph_manifest_path(graph_path: Path) -> Path:
    path = Path(graph_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def graph_incomplete_marker_path(graph_path: Path) -> Path:
    path = Path(graph_path)
    return path.with_suffix(path.suffix + ".incomplete")


def shard_index(path: Path) -> int | None:
    match = _SHARD_INDEX_RE.search(Path(path).name)
    return int(match.group("index")) if match is not None else None


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file and publish with rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def resolve_recorded_path(
    recorded_path: str | Path,
    *,
    manifest_path: Path,
    repository_root: Path | None = None,
) -> Path:
    """Resolve a manifest path recorded as absolute, repository-relative, or local."""

    raw = Path(recorded_path)
    if raw.is_absolute():
        return raw.resolve(strict=False)

    candidates: list[Path] = []
    if repository_root is not None:
        candidates.append(Path(repository_root) / raw)
    candidates.append(Path(manifest_path).parent / raw)
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            return candidate.resolve(strict=False)
    return candidates[0].resolve(strict=False)
