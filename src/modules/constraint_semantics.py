"""Shared registry parsing for executable Wikidata constraint semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd

from .constraint_checkers import ConstraintInstance, normalize_token
from .data_encoders import GlobalIntEncoder


PARAM_EXCEPTIONS = "P2303"
PARAM_ITEMS = "P2305"
PARAM_PROPERTY = "P2306"
PARAM_CLASS = "P2308"
PARAM_RELATION = "P2309"
PARAM_SCOPE = "P4680"

RELATION_MODE_INSTANCE_OF = "Q21503252"
RELATION_MODE_SUBCLASS_OF = "Q21514624"
RELATION_MODE_INSTANCE_OR_SUBCLASS_OF = "Q30208840"
CONSTRAINT_SEMANTICS_VERSION = "wikidata-main-v4"

RELATION_MODE_PREDICATES: dict[str, tuple[str, ...]] = {
    RELATION_MODE_INSTANCE_OF: ("P31",),
    RELATION_MODE_SUBCLASS_OF: ("P279",),
    RELATION_MODE_INSTANCE_OR_SUBCLASS_OF: ("P31", "P279"),
    # Accept direct property values for compatibility with noncanonical dumps.
    "P31": ("P31",),
    "P279": ("P279",),
}

# Both the historical dump values and current constraint-scope items are
# accepted. The correction corpus contains main-value statements only.
MAIN_VALUE_SCOPE_ITEMS = frozenset({"Q54828448", "Q46466787"})


@dataclass(frozen=True)
class RegistryEntry:
    constraint_type_raw: str
    constraint_type_item: str
    constraint_type_index: int
    constraint_family: str
    constraint_label: str
    constraint_family_supported: bool
    constrained_property_raw: str
    param_predicates_raw: Tuple[str, ...]
    param_objects_raw: Tuple[str, ...]


def load_registry(path: str | Path) -> Dict[str, RegistryEntry]:
    registry_df = pd.read_parquet(path)
    registry_json = registry_df["registry_json"].iloc[0]
    registry = json.loads(registry_json) if isinstance(registry_json, str) else registry_json
    type_items = sorted(
        {
            str(entry.get("constraint_type_item", "")).strip()
            for entry in registry.values()
            if str(entry.get("constraint_type_item", "")).strip()
        }
    )
    fallback_type_index = {type_item: idx for idx, type_item in enumerate(type_items)}
    parsed: Dict[str, RegistryEntry] = {}
    for constraint_id, entry in registry.items():
        family = entry.get("constraint_family") or entry.get("constraint_type_name", "")
        supported = entry.get("constraint_family_supported")
        if supported is None:
            supported = entry.get("constraint_type_supported", False)
        type_item = str(entry.get("constraint_type_item", ""))
        type_index = entry.get("constraint_type_index")
        if type_index is None:
            type_index = fallback_type_index.get(type_item.strip(), -1)
        parsed[str(constraint_id)] = RegistryEntry(
            constraint_type_raw=str(entry.get("constraint_type", "")),
            constraint_type_item=type_item,
            constraint_type_index=int(type_index),
            constraint_family=str(family or ""),
            constraint_label=str(entry.get("constraint_label", "")),
            constraint_family_supported=bool(supported),
            constrained_property_raw=str(entry.get("constrained_property", "")),
            param_predicates_raw=tuple(entry.get("param_predicates") or ()),
            param_objects_raw=tuple(entry.get("param_objects") or ()),
        )
    return parsed


def resolve_registry_id(raw_id: str | None, encoder: GlobalIntEncoder | None) -> int:
    if encoder is None or not raw_id:
        return 0
    raw = str(raw_id).strip().strip("<>").strip()
    if raw.startswith("http://www.wikidata.org/prop/direct/"):
        raw = raw.replace(
            "http://www.wikidata.org/prop/direct/",
            "http://www.wikidata.org/entity/",
        )
    candidates: List[str] = []
    if raw.startswith(("http://", "https://")):
        candidates.extend([raw, f"<{raw}>"])
        tail = raw.rsplit("/", 1)[-1]
        if tail and tail[0] in ("P", "Q") and tail[1:].isdigit():
            candidates.append(tail)
    else:
        if raw and raw[0] in ("P", "Q") and raw[1:].isdigit():
            uri = f"http://www.wikidata.org/entity/{raw}"
            candidates.extend([uri, f"<{uri}>"])
        candidates.append(raw)
    for candidate in dict.fromkeys(candidates):
        token_id = encoder.encode(candidate, add_new=False)
        if token_id:
            return token_id
    return 0


def resolve_registry_mapping(
    registry: Dict[str, RegistryEntry],
    *,
    encoder: GlobalIntEncoder | None,
    use_encoded_ids: bool,
) -> Dict[int, RegistryEntry] | Dict[str, RegistryEntry]:
    if use_encoded_ids:
        if encoder is None:
            raise ValueError("Identity encoder required for encoded constraint IDs.")
        mapped: Dict[int, RegistryEntry] = {}
        for constraint_id, entry in registry.items():
            encoded = resolve_registry_id(constraint_id, encoder)
            if encoded:
                mapped[encoded] = entry
        return mapped
    return {
        normalize_token(constraint_id) or constraint_id: entry
        for constraint_id, entry in registry.items()
    }


def lookup_registry_entry(
    constraint_id: Any,
    registry_by_id: Dict[int, RegistryEntry] | Dict[str, RegistryEntry],
    *,
    use_encoded_ids: bool,
) -> RegistryEntry | None:
    if use_encoded_ids:
        try:
            return registry_by_id.get(int(constraint_id))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    key = normalize_token(str(constraint_id)) or str(constraint_id)
    return registry_by_id.get(key)  # type: ignore[arg-type]


def _resolved_objects(
    pairs: Iterable[tuple[str, str]],
    predicate: str,
    encoder: GlobalIntEncoder | None,
) -> list[int]:
    values: list[int] = []
    for pred_raw, obj_raw in pairs:
        if (normalize_token(pred_raw) or pred_raw) != predicate:
            continue
        resolved = resolve_registry_id(obj_raw, encoder)
        if resolved:
            values.append(resolved)
    return values


def _normalized_objects(
    pairs: Iterable[tuple[str, str]],
    predicate: str,
) -> set[str]:
    return {
        normalize_token(obj_raw) or str(obj_raw)
        for pred_raw, obj_raw in pairs
        if (normalize_token(pred_raw) or pred_raw) == predicate
    }


def resolve_relation_predicate_ids(
    relation_values: Iterable[Any],
    *,
    encoder: GlobalIntEncoder | None,
) -> list[int]:
    """Translate P2309 relation-mode items into executable predicates.

    P2309 values are Q-items describing a relation mode, not predicates that
    can occur on entity statements. P2309 is mandatory for type constraints;
    an absent or explicitly unknown mode remains unresolved and uncheckable.
    """

    values = list(relation_values)
    if not values:
        return []

    predicate_ids: list[int] = []
    for value in values:
        raw_value: Any = value
        if isinstance(value, int) and encoder is not None:
            raw_value = encoder.decode(value)
        mode = normalize_token(str(raw_value)) if raw_value is not None else None
        for predicate in RELATION_MODE_PREDICATES.get(mode or "", ()):
            predicate_id = resolve_registry_id(predicate, encoder)
            if predicate_id and predicate_id not in predicate_ids:
                predicate_ids.append(predicate_id)
    return predicate_ids


def build_constraint_instance(
    constraint_id: int,
    registry_entry: RegistryEntry,
    *,
    encoder: GlobalIntEncoder | None,
    constraint_type_name: str,
    constraint_type_id: int,
) -> ConstraintInstance:
    constrained_property = resolve_registry_id(
        registry_entry.constrained_property_raw,
        encoder,
    )
    pairs = list(
        zip(registry_entry.param_predicates_raw, registry_entry.param_objects_raw)
    )
    property_ids = {
        value
        for value in _resolved_objects(pairs, PARAM_PROPERTY, encoder)
        if value
    }
    item_ids = set(_resolved_objects(pairs, PARAM_ITEMS, encoder))
    class_ids = set(_resolved_objects(pairs, PARAM_CLASS, encoder))
    relation_values = [
        obj_raw
        for pred_raw, obj_raw in pairs
        if (normalize_token(pred_raw) or pred_raw) == PARAM_RELATION
    ]
    relation_ids = resolve_relation_predicate_ids(
        relation_values,
        encoder=encoder,
    )
    exception_ids = set(_resolved_objects(pairs, PARAM_EXCEPTIONS, encoder))
    scope_items = _normalized_objects(pairs, PARAM_SCOPE)
    applies_to_main = not scope_items or bool(scope_items & MAIN_VALUE_SCOPE_ITEMS)

    required_properties: Set[int] = set()
    inverse_properties: List[int] = []
    conflict_properties: Set[int] = set()
    if constraint_type_name in {"inverse", "symmetric"}:
        inverse_properties = sorted(property_ids)
    elif constraint_type_name in {"itemRequiresStatement", "valueRequiresStatement"}:
        required_properties = set(property_ids)
    elif constraint_type_name == "conflictWith":
        conflict_properties = set(property_ids)

    return ConstraintInstance(
        constraint_id=int(constraint_id),
        constraint_type=constraint_type_name,
        constraint_type_id=int(constraint_type_id),
        constrained_property=constrained_property,
        required_properties=required_properties,
        allowed_items=item_ids,
        allowed_classes=class_ids,
        relation_predicates=relation_ids,
        inverse_properties=inverse_properties,
        conflict_properties=conflict_properties,
        exceptions=exception_ids,
        applies_to_main_value=applies_to_main,
    )
