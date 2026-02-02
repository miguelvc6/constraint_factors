from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

from modules.constraint_checkers import ConstraintInstance, EvidenceState, evaluate_constraint, normalize_property_id, normalize_token
from modules.data_encoders import GlobalIntEncoder

PARAM_P2306 = "P2306"
PARAM_P2309 = "P2309"
PARAM_P2308 = "P2308"
PARAM_P2305 = "P2305"
PARAM_P1696 = "P1696"


@dataclass(frozen=True)
class RegistryEntry:
    constraint_type_raw: str
    constraint_type_item: str
    constraint_family: str
    constraint_label: str
    constraint_family_supported: bool
    constrained_property_raw: str
    param_predicates_raw: Tuple[str, ...]
    param_objects_raw: Tuple[str, ...]


@dataclass
class CandidateMetrics:
    primary_satisfied: int
    global_satisfied_fraction: float
    secondary_regressions: int


def _load_registry(path: str | None) -> Dict[str, RegistryEntry]:
    if path is None:
        return {}
    registry_df = pd.read_parquet(path)
    registry_json = registry_df["registry_json"].iloc[0]
    if isinstance(registry_json, str):
        registry = json.loads(registry_json)
    else:
        registry = registry_json
    parsed: Dict[str, RegistryEntry] = {}
    for constraint_id, entry in registry.items():
        constraint_family = entry.get("constraint_family")
        if not constraint_family:
            constraint_family = entry.get("constraint_type_name", "")
        constraint_family_supported = entry.get("constraint_family_supported")
        if constraint_family_supported is None:
            constraint_family_supported = entry.get("constraint_type_supported", False)
        parsed[constraint_id] = RegistryEntry(
            constraint_type_raw=str(entry.get("constraint_type", "")),
            constraint_type_item=str(entry.get("constraint_type_item", "")),
            constraint_family=str(constraint_family or ""),
            constraint_label=str(entry.get("constraint_label", "")),
            constraint_family_supported=bool(constraint_family_supported),
            constrained_property_raw=str(entry.get("constrained_property", "")),
            param_predicates_raw=tuple(entry.get("param_predicates") or ()),
            param_objects_raw=tuple(entry.get("param_objects") or ()),
        )
    return parsed


def _resolve_registry_id(raw_id: str | None, encoder: GlobalIntEncoder | None) -> int:
    if encoder is None or not raw_id:
        return 0
    raw = raw_id.strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1].strip()
    if raw.startswith("http://www.wikidata.org/prop/direct/"):
        raw = raw.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")
    candidates: List[str] = []
    seen: Set[str] = set()
    if raw.startswith("http://") or raw.startswith("https://"):
        candidates.extend([raw, f"<{raw}>"])
        tail = raw.rsplit("/", 1)[-1]
        if tail and tail[0] in ("P", "Q") and tail[1:].isdigit():
            candidates.append(tail)
    else:
        if raw and raw[0] in ("P", "Q") and raw[1:].isdigit():
            entity_uri = f"http://www.wikidata.org/entity/{raw}"
            candidates.extend([entity_uri, f"<{entity_uri}>"])
        candidates.append(raw)
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        token_id = encoder.encode(candidate, add_new=False)
        if token_id:
            return token_id
    return 0


def _resolve_registry_mapping(
    registry: Dict[str, RegistryEntry],
    *,
    encoder: GlobalIntEncoder | None,
    use_encoded_ids: bool,
) -> Dict[int, RegistryEntry] | Dict[str, RegistryEntry]:
    if use_encoded_ids:
        if encoder is None:
            raise ValueError("Encoder required to map registry constraint ids to dataset ids.")
        mapped: Dict[int, RegistryEntry] = {}
        for constraint_id, entry in registry.items():
            cid = _resolve_registry_id(constraint_id, encoder)
            if cid == 0:
                continue
            mapped[cid] = entry
        return mapped
    mapped_str: Dict[str, RegistryEntry] = {}
    for constraint_id, entry in registry.items():
        key = normalize_token(constraint_id) or constraint_id
        mapped_str[key] = entry
    return mapped_str


def _lookup_registry_entry(
    constraint_id: Any,
    registry_by_id: Dict[int, RegistryEntry] | Dict[str, RegistryEntry],
    *,
    use_encoded_ids: bool,
) -> RegistryEntry | None:
    if use_encoded_ids:
        try:
            cid = int(constraint_id)
        except (TypeError, ValueError):
            return None
        return registry_by_id.get(cid)  # type: ignore[arg-type]
    key = normalize_token(str(constraint_id)) or str(constraint_id)
    return registry_by_id.get(key)  # type: ignore[arg-type]


def _coerce_sequence(value: Any, *, cast_int: bool = True) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        seq = list(value)
    else:
        seq = [value]
    if not cast_int:
        return seq
    return [int(v) for v in seq]


def _coerce_value(value: Any, *, cast_int: bool = True) -> Any:
    if not cast_int:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compute_p_local(row: Any, *, cast_int: bool = True) -> Set[Any]:
    p_local: Set[Any] = set()
    for name in ("predicate", "other_predicate"):
        value = _coerce_value(getattr(row, name, None), cast_int=cast_int)
        if value not in (None, "", 0):
            p_local.add(value)
    for name in ("subject_predicates", "object_predicates", "other_entity_predicates"):
        for pred in _coerce_sequence(getattr(row, name, None), cast_int=cast_int):
            if pred not in (None, "", 0):
                p_local.add(pred)
    return p_local


def _pick_other_entity_id(row: Any, *, cast_int: bool = True) -> Any:
    other_subject = _coerce_value(getattr(row, "other_subject", None), cast_int=cast_int)
    if other_subject not in (None, "", 0):
        return other_subject
    other_object = _coerce_value(getattr(row, "other_object", None), cast_int=cast_int)
    if other_object not in (None, "", 0):
        return other_object
    return 0


def _build_facts_for_entity(
    predicates: Sequence[Any],
    objects: Sequence[Any],
    *,
    p_local: Set[Any],
    cast_int: bool,
) -> Tuple[Dict[Any, Set[Any]], Set[Any]]:
    facts: Dict[Any, Set[Any]] = {}
    predicates_present: Set[Any] = set()
    for pred, obj in zip(predicates, objects):
        pred_id = _coerce_value(pred, cast_int=cast_int)
        obj_id = _coerce_value(obj, cast_int=cast_int)
        if pred_id in (None, "", 0) or obj_id in (None, "", 0):
            continue
        if pred_id not in p_local:
            continue
        facts.setdefault(pred_id, set()).add(obj_id)
        predicates_present.add(pred_id)
    return facts, predicates_present


def _build_facts_state(
    row: Any,
    *,
    p_local: Set[Any],
    assume_complete: bool,
    cast_int: bool,
) -> Tuple[Dict[int, Dict[int, Set[int]]], Dict[int, Set[int]]]:
    facts_by_entity: Dict[int, Dict[int, Set[int]]] = {}
    predicates_present: Dict[int, Set[int]] = {}

    subject_id = _coerce_value(getattr(row, "subject", None), cast_int=cast_int)
    object_id = _coerce_value(getattr(row, "object", None), cast_int=cast_int)
    other_entity_id = _pick_other_entity_id(row, cast_int=cast_int)

    subject_preds = _coerce_sequence(getattr(row, "subject_predicates", None), cast_int=cast_int)
    subject_objs = _coerce_sequence(getattr(row, "subject_objects", None), cast_int=cast_int)
    subject_facts, subject_present = _build_facts_for_entity(
        subject_preds, subject_objs, p_local=p_local, cast_int=cast_int
    )
    facts_by_entity[subject_id] = subject_facts
    predicates_present[subject_id] = subject_present

    if object_id not in (None, "", 0):
        object_preds = _coerce_sequence(getattr(row, "object_predicates", None), cast_int=cast_int)
        object_objs = _coerce_sequence(getattr(row, "object_objects", None), cast_int=cast_int)
        object_facts, object_present = _build_facts_for_entity(
            object_preds, object_objs, p_local=p_local, cast_int=cast_int
        )
        facts_by_entity[object_id] = object_facts
        predicates_present[object_id] = object_present

    if other_entity_id not in (None, "", 0):
        other_preds = _coerce_sequence(getattr(row, "other_entity_predicates", None), cast_int=cast_int)
        other_objs = _coerce_sequence(getattr(row, "other_entity_objects", None), cast_int=cast_int)
        other_facts, other_present = _build_facts_for_entity(
            other_preds, other_objs, p_local=p_local, cast_int=cast_int
        )
        facts_by_entity[other_entity_id] = other_facts
        predicates_present[other_entity_id] = other_present

    return facts_by_entity, predicates_present


def _build_placeholder_map(encoder: GlobalIntEncoder | None, row: Any) -> Dict[Any, Any]:
    mapping: Dict[Any, Any] = {}
    if encoder is None:
        return {
            "subject": getattr(row, "subject", 0),
            "predicate": getattr(row, "predicate", 0),
            "object": getattr(row, "object", 0),
            "other_subject": getattr(row, "other_subject", 0),
            "other_predicate": getattr(row, "other_predicate", 0),
            "other_object": getattr(row, "other_object", 0),
        }

    placeholders = [
        "subject",
        "predicate",
        "object",
        "other_subject",
        "other_predicate",
        "other_object",
    ]
    for label in placeholders:
        token_id = encoder.encode(label, add_new=False)
        if token_id:
            mapping[token_id] = int(getattr(row, label, 0) or 0)
    return mapping


def _resolve_placeholder(value: Any, placeholder_map: Dict[Any, Any]) -> Any:
    if value in placeholder_map:
        return placeholder_map[value]
    return value


def _apply_candidate_edit(
    facts_by_entity: Dict[int, Dict[int, Set[int]]],
    predicates_present: Dict[int, Set[int]],
    p_local: Set[int],
    *,
    candidate_slots: Sequence[int],
    placeholder_map: Dict[Any, Any],
    assume_complete: bool,
) -> Set[Tuple[int, int]]:
    missing_edits: Set[Tuple[int, int]] = set()

    def _apply(kind: str, subj: int, pred: int, obj: int) -> None:
        subj = _resolve_placeholder(subj, placeholder_map)
        pred = _resolve_placeholder(pred, placeholder_map)
        obj = _resolve_placeholder(obj, placeholder_map)
        if subj in (None, "", 0) or pred in (None, "", 0) or obj in (None, "", 0):
            return
        if not assume_complete and pred not in predicates_present.get(subj, set()):
            missing_edits.add((subj, pred))
            return
        if pred not in p_local:
            missing_edits.add((subj, pred))
            return
        if subj not in facts_by_entity:
            missing_edits.add((subj, pred))
            return
        entity_facts = facts_by_entity[subj]
        if kind == "del":
            if pred in entity_facts and obj in entity_facts[pred]:
                entity_facts[pred].discard(obj)
        else:
            entity_facts.setdefault(pred, set()).add(obj)
            predicates_present.setdefault(subj, set()).add(pred)

    add = candidate_slots[:3]
    delete = candidate_slots[3:]
    _apply("del", int(delete[0]), int(delete[1]), int(delete[2]))
    _apply("add", int(add[0]), int(add[1]), int(add[2]))

    return missing_edits


def _resolve_default_relations(encoder: GlobalIntEncoder | None) -> List[int]:
    if encoder is None:
        return []
    p31 = _resolve_registry_id("P31", encoder)
    p279 = _resolve_registry_id("P279", encoder)
    defaults = [pid for pid in (p31, p279) if pid]
    return defaults


def _constraint_type_id_from_registry(
    registry_entry: RegistryEntry,
    encoder: GlobalIntEncoder | None,
) -> int:
    if encoder is None:
        return 0
    return _resolve_registry_id(registry_entry.constraint_type_item, encoder)


def _build_constraint_instance(
    constraint_id: int,
    registry_entry: RegistryEntry,
    *,
    encoder: GlobalIntEncoder | None,
    constraint_type_name: str,
    constraint_type_id: int,
    default_relation_predicates: List[int],
) -> ConstraintInstance:
    constrained_property_id = _resolve_registry_id(registry_entry.constrained_property_raw, encoder)

    param_predicates = registry_entry.param_predicates_raw
    param_objects = registry_entry.param_objects_raw
    param_pairs = list(zip(param_predicates, param_objects))

    required_properties: Set[int] = set()
    allowed_items: Set[int] = set()
    allowed_classes: Set[int] = set()
    relation_predicates: List[int] = []
    inverse_properties: List[int] = []
    conflict_properties: Set[int] = set()

    for pred_raw, obj_raw in param_pairs:
        pred_norm = normalize_token(pred_raw)
        obj_norm = normalize_token(obj_raw)
        pred_key = pred_norm or pred_raw
        obj_key = obj_norm or obj_raw
        obj_id = _resolve_registry_id(obj_raw, encoder) if encoder else 0

        if pred_key == PARAM_P2306:
            if normalize_property_id(obj_key):
                if obj_id:
                    required_properties.add(obj_id)
        elif pred_key == PARAM_P2305:
            if obj_id:
                allowed_items.add(obj_id)
        elif pred_key == PARAM_P2308:
            if obj_id:
                allowed_classes.add(obj_id)
        elif pred_key == PARAM_P2309:
            if normalize_property_id(obj_key) and obj_id:
                relation_predicates.append(obj_id)
        elif pred_key == PARAM_P1696:
            if normalize_property_id(obj_key) and obj_id:
                inverse_properties.append(obj_id)

        if normalize_property_id(obj_key) and obj_id:
            conflict_properties.add(obj_id)

    if not relation_predicates:
        relation_predicates = list(default_relation_predicates)
    if not inverse_properties and constrained_property_id:
        inverse_properties = [constrained_property_id]

    return ConstraintInstance(
        constraint_id=constraint_id,
        constraint_type=constraint_type_name,
        constraint_type_id=constraint_type_id,
        constrained_property=constrained_property_id,
        required_properties=required_properties,
        allowed_items=allowed_items,
        allowed_classes=allowed_classes,
        relation_predicates=relation_predicates,
        inverse_properties=inverse_properties,
        conflict_properties=conflict_properties,
    )


class CandidateConstraintEvaluator:
    def __init__(
        self,
        registry_path: str,
        *,
        encoder: GlobalIntEncoder | None,
        assume_complete: bool,
        constraint_scope: str,
        use_encoded_ids: bool,
    ) -> None:
        registry_raw = _load_registry(registry_path)
        self._registry_by_id = _resolve_registry_mapping(
            registry_raw, encoder=encoder, use_encoded_ids=use_encoded_ids
        )
        self._encoder = encoder
        self._assume_complete = assume_complete
        self._constraint_scope = constraint_scope
        self._use_encoded_ids = use_encoded_ids
        self._default_relations = _resolve_default_relations(encoder)
        self._constraint_cache: Dict[str, ConstraintInstance] = {}

    def _get_constraint_instance(self, constraint_id: Any) -> ConstraintInstance | None:
        entry = _lookup_registry_entry(
            constraint_id, self._registry_by_id, use_encoded_ids=self._use_encoded_ids
        )
        if entry is None:
            return None
        cache_key = str(int(constraint_id)) if self._use_encoded_ids else str(constraint_id)
        cached = self._constraint_cache.get(cache_key)
        if cached is not None:
            return cached
        type_name = entry.constraint_family or ""
        constraint_type_id = _constraint_type_id_from_registry(entry, self._encoder)
        instance = _build_constraint_instance(
            int(constraint_id) if self._use_encoded_ids else 0,
            entry,
            encoder=self._encoder,
            constraint_type_name=type_name,
            constraint_type_id=constraint_type_id,
            default_relation_predicates=self._default_relations,
        )
        self._constraint_cache[cache_key] = instance
        return instance

    def evaluate(
        self,
        row: Any,
        *,
        candidate_slots: Sequence[int],
        primary_factor_index: int,
    ) -> CandidateMetrics:
        details = self.evaluate_full(
            row,
            candidate_slots=candidate_slots,
            primary_factor_index=primary_factor_index,
        )
        return CandidateMetrics(
            primary_satisfied=details["primary_satisfied"],
            global_satisfied_fraction=details["global_satisfied_fraction"],
            secondary_regressions=details["secondary_regressions"],
        )

    def evaluate_full(
        self,
        row: Any,
        *,
        candidate_slots: Sequence[int],
        primary_factor_index: int | None = None,
    ) -> Dict[str, Any]:
        p_local = _compute_p_local(row, cast_int=self._use_encoded_ids)
        facts_by_entity, predicates_present = _build_facts_state(
            row,
            p_local=p_local,
            assume_complete=self._assume_complete,
            cast_int=self._use_encoded_ids,
        )
        subject = _coerce_value(getattr(row, "subject", 0), cast_int=self._use_encoded_ids)
        predicate = _coerce_value(getattr(row, "predicate", 0), cast_int=self._use_encoded_ids)
        obj = _coerce_value(getattr(row, "object", 0), cast_int=self._use_encoded_ids)
        other_subject = _coerce_value(getattr(row, "other_subject", 0), cast_int=self._use_encoded_ids)
        other_predicate = _coerce_value(getattr(row, "other_predicate", 0), cast_int=self._use_encoded_ids)
        other_object = _coerce_value(getattr(row, "other_object", 0), cast_int=self._use_encoded_ids)

        pre_state = EvidenceState(
            facts_by_entity=facts_by_entity,
            predicates_present=predicates_present,
            assume_complete=self._assume_complete,
            missing_edits=set(),
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
        )

        post_facts = {ent: {pred: set(values) for pred, values in facts.items()} for ent, facts in facts_by_entity.items()}
        post_predicates = {ent: set(preds) for ent, preds in predicates_present.items()}
        placeholder_map = _build_placeholder_map(self._encoder, row)
        missing_edits = _apply_candidate_edit(
            post_facts,
            post_predicates,
            p_local,
            candidate_slots=candidate_slots,
            placeholder_map=placeholder_map,
            assume_complete=self._assume_complete,
        )
        post_state = EvidenceState(
            facts_by_entity=post_facts,
            predicates_present=post_predicates,
            assume_complete=self._assume_complete,
            missing_edits=missing_edits,
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
        )

        if self._constraint_scope == "focus":
            constraint_ids_raw = getattr(row, "local_constraint_ids_focus", None)
            if constraint_ids_raw is None:
                constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        else:
            constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        local_constraint_ids = _coerce_sequence(constraint_ids_raw, cast_int=self._use_encoded_ids)
        if not local_constraint_ids:
            return {
                "local_constraint_ids": [],
                "primary_factor_index": -1,
                "pre_checkable": [],
                "pre_satisfied": [],
                "post_checkable": [],
                "post_satisfied": [],
                "primary_satisfied": 0,
                "global_satisfied_fraction": 0.0,
                "secondary_regressions": 0,
                "secondary_improvements": 0,
            }

        pre_checkable: List[bool] = []
        pre_satisfied: List[int] = []
        post_checkable: List[bool] = []
        post_satisfied: List[int] = []

        for constraint_id in local_constraint_ids:
            instance = self._get_constraint_instance(constraint_id)
            if instance is None:
                pre_checkable.append(False)
                pre_satisfied.append(0)
                post_checkable.append(False)
                post_satisfied.append(0)
                continue
            checkable_pre, satisfied_pre = evaluate_constraint(pre_state, instance, set(p_local))
            checkable_post, satisfied_post = evaluate_constraint(post_state, instance, set(p_local))
            pre_checkable.append(bool(checkable_pre))
            pre_satisfied.append(int(satisfied_pre))
            post_checkable.append(bool(checkable_post))
            post_satisfied.append(int(satisfied_post))

        resolved_primary_index = -1
        if primary_factor_index is not None and 0 <= primary_factor_index < len(local_constraint_ids):
            resolved_primary_index = int(primary_factor_index)
        else:
            constraint_id = _coerce_value(getattr(row, "constraint_id", None), cast_int=self._use_encoded_ids)
            try:
                resolved_primary_index = local_constraint_ids.index(constraint_id)
            except ValueError:
                resolved_primary_index = -1

        primary_satisfied = 0
        if 0 <= resolved_primary_index < len(post_satisfied):
            primary_satisfied = post_satisfied[resolved_primary_index]

        checkable_total = sum(1 for flag in post_checkable if flag)
        if checkable_total:
            global_satisfied_fraction = sum(post_satisfied) / float(checkable_total)
        else:
            global_satisfied_fraction = 0.0

        secondary_regressions = 0
        secondary_improvements = 0
        for idx in range(len(local_constraint_ids)):
            if idx == resolved_primary_index:
                continue
            if pre_checkable[idx]:
                if pre_satisfied[idx] and post_checkable[idx] and not post_satisfied[idx]:
                    secondary_regressions += 1
                if (not pre_satisfied[idx]) and post_checkable[idx] and post_satisfied[idx]:
                    secondary_improvements += 1

        return {
            "local_constraint_ids": local_constraint_ids,
            "primary_factor_index": resolved_primary_index,
            "pre_checkable": pre_checkable,
            "pre_satisfied": pre_satisfied,
            "post_checkable": post_checkable,
            "post_satisfied": post_satisfied,
            "primary_satisfied": primary_satisfied,
            "global_satisfied_fraction": global_satisfied_fraction,
            "secondary_regressions": secondary_regressions,
            "secondary_improvements": secondary_improvements,
        }
