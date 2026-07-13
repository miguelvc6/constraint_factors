"""Frozen training-split class hierarchy for executable type constraints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set

import pyarrow as pa
import pyarrow.parquet as pq

from .evidence_edits import resolve_other_entity_id


CLASS_HIERARCHY_FILENAME = "class_hierarchy.parquet"
CLASS_HIERARCHY_MANIFEST_FILENAME = "class_hierarchy_manifest.json"
CLASS_HIERARCHY_SCHEMA_VERSION = 1
CLASS_HIERARCHY_SEMANTICS = "p279-reflexive-transitive-train-v1"

_ENTITY_FACT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("subject", "subject_predicates", "subject_objects"),
    ("object", "object_predicates", "object_objects"),
)
_OTHER_ENTITY_COLUMNS = (
    "other_subject",
    "other_object",
    "other_entity_predicates",
    "other_entity_objects",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ClassHierarchy:
    """Direct ``P279`` edges with cached reflexive-transitive reachability."""

    parents_by_child: Mapping[int, frozenset[int]]
    _ancestor_cache: Dict[int, frozenset[int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[int, int]]) -> "ClassHierarchy":
        parents: Dict[int, Set[int]] = {}
        for child, parent in edges:
            child_id = int(child)
            parent_id = int(parent)
            if child_id <= 0 or parent_id <= 0 or child_id == parent_id:
                continue
            parents.setdefault(child_id, set()).add(parent_id)
        return cls(
            parents_by_child={
                child: frozenset(values)
                for child, values in parents.items()
            }
        )

    @classmethod
    def load(cls, path: Path) -> "ClassHierarchy":
        table = pq.read_table(path, columns=["child_id", "parent_id"])
        return cls.from_edges(
            zip(table["child_id"].to_pylist(), table["parent_id"].to_pylist())
        )

    @property
    def direct_edge_count(self) -> int:
        return sum(len(parents) for parents in self.parents_by_child.values())

    @property
    def child_count(self) -> int:
        return len(self.parents_by_child)

    def direct_edges(self) -> Iterable[tuple[int, int]]:
        for child in sorted(self.parents_by_child):
            for parent in sorted(self.parents_by_child[child]):
                yield child, parent

    def ancestors_including_self(self, class_id: int) -> frozenset[int]:
        class_id = int(class_id)
        cached = self._ancestor_cache.get(class_id)
        if cached is not None:
            return cached

        seen = {class_id}
        pending = [class_id]
        while pending:
            child = pending.pop()
            for parent in self.parents_by_child.get(child, ()):
                if parent in seen:
                    continue
                seen.add(parent)
                pending.append(parent)
        result = frozenset(seen)
        self._ancestor_cache[class_id] = result
        return result

    def reaches_any(
        self,
        class_ids: Iterable[int],
        allowed_classes: Set[int] | frozenset[int],
    ) -> bool:
        allowed = set(allowed_classes)
        if not allowed:
            return False
        return any(
            bool(self.ancestors_including_self(int(class_id)) & allowed)
            for class_id in class_ids
            if int(class_id) > 0
        )


def build_training_class_hierarchy(
    train_parquet: Path,
    *,
    p279_predicate_id: int,
    batch_size: int = 100_000,
) -> ClassHierarchy:
    """Collect the union of direct ``P279`` facts in training entity context."""

    parquet_file = pq.ParquetFile(train_parquet)
    required = {
        column
        for columns in _ENTITY_FACT_COLUMNS
        for column in columns
    } | set(_OTHER_ENTITY_COLUMNS)
    missing = required - set(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(
            f"{train_parquet} is missing hierarchy source columns: {sorted(missing)}"
        )
    if int(p279_predicate_id) <= 0:
        raise ValueError("The identity encoder does not contain the P279 predicate.")

    parents: Dict[int, Set[int]] = {}
    columns = list(
        dict.fromkeys(
            [column for group in _ENTITY_FACT_COLUMNS for column in group]
            + list(_OTHER_ENTITY_COLUMNS)
        )
    )

    def _collect_entity_edges(
        entity_id: int | None,
        predicates: list[int] | None,
        objects: list[int] | None,
    ) -> None:
        if not entity_id:
            return
        for predicate_id, object_id in zip(predicates or (), objects or ()):
            if (
                int(predicate_id or 0) != int(p279_predicate_id)
                or not object_id
                or int(entity_id) == int(object_id)
            ):
                continue
            parents.setdefault(int(entity_id), set()).add(int(object_id))

    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        values = batch.to_pydict()
        for entity_column, predicates_column, objects_column in _ENTITY_FACT_COLUMNS:
            for entity_id, predicates, objects in zip(
                values[entity_column],
                values[predicates_column],
                values[objects_column],
            ):
                _collect_entity_edges(entity_id, predicates, objects)
        for subject, other_subject, other_object, predicates, objects in zip(
            values["subject"],
            values["other_subject"],
            values["other_object"],
            values["other_entity_predicates"],
            values["other_entity_objects"],
        ):
            entity_id = resolve_other_entity_id(
                subject=subject,
                other_subject=other_subject,
                other_object=other_object,
            )
            _collect_entity_edges(entity_id, predicates, objects)

    return ClassHierarchy(
        parents_by_child={
            child: frozenset(parent_ids)
            for child, parent_ids in parents.items()
        }
    )


def write_class_hierarchy(
    hierarchy: ClassHierarchy,
    output_root: Path,
    *,
    p279_predicate_id: int,
    source_dataset_variant: str,
    source_manifest_path: Path | None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    hierarchy_path = output_root / CLASS_HIERARCHY_FILENAME
    edges = list(hierarchy.direct_edges())
    table = pa.table(
        {
            "child_id": pa.array((child for child, _ in edges), type=pa.int64()),
            "parent_id": pa.array((parent for _, parent in edges), type=pa.int64()),
        }
    )
    pq.write_table(table, hierarchy_path, compression="zstd")

    payload: dict[str, object] = {
        "schema_version": CLASS_HIERARCHY_SCHEMA_VERSION,
        "semantics": CLASS_HIERARCHY_SEMANTICS,
        "source_dataset_variant": source_dataset_variant,
        "source_split": "train",
        "source_columns": [
            *[
                column
                for columns in _ENTITY_FACT_COLUMNS
                for column in columns
            ],
            *_OTHER_ENTITY_COLUMNS,
        ],
        "p279_predicate_id": int(p279_predicate_id),
        "child_count": hierarchy.child_count,
        "direct_edge_count": hierarchy.direct_edge_count,
        "source_manifest_sha256": (
            file_sha256(source_manifest_path)
            if source_manifest_path is not None and source_manifest_path.exists()
            else None
        ),
        "outputs": {
            CLASS_HIERARCHY_FILENAME: file_sha256(hierarchy_path),
        },
    }
    manifest_path = output_root / CLASS_HIERARCHY_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
