

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from modules.class_hierarchy import ClassHierarchy
from modules.constraint_checkers import (
    CHECKERS,
    ConstraintInstance,
    EvidenceState,
    evaluate_constraint,
)
from modules.constraint_semantics import (
    RegistryEntry,
    build_constraint_instance as build_registry_constraint_instance,
    load_registry,
    lookup_registry_entry,
    resolve_registry_id,
    resolve_registry_mapping,
)
from modules.data_encoders import GlobalIntEncoder
from modules.evidence_edits import (
    normalize_pre_edit_state,
    resolve_other_entity_id,
    resolve_row_edits,
)

PLACEHOLDER_LABELS: tuple[str, ...] = (
    "subject",
    "predicate",
    "object",
    "other_subject",
    "other_predicate",
    "other_object",
)

_EMPTY_MISSING_EDITS = frozenset()


@dataclass
class CandidateMetrics:
    primary_satisfied: int
    global_satisfied_fraction: float
    secondary_regressions: int
    secondary_regressions_denom: int
    secondary_improvements: int
    secondary_improvements_denom: int
    srr: float
    sir: float
    add_count: int
    del_count: int


class _ReusableEvidenceState:
    __slots__ = (
        "facts_by_entity",
        "predicates_present",
        "assume_complete",
        "missing_edits",
        "focus_subject",
        "focus_predicate",
        "focus_object",
        "other_subject",
        "other_predicate",
        "other_object",
        "class_hierarchy",
    )

    def __init__(
        self,
        *,
        facts_by_entity: Dict[int, Dict[int, Set[int]]],
        predicates_present: Dict[int, Set[int]],
        assume_complete: bool,
        missing_edits: Set[Tuple[int, int]],
        focus_subject: int,
        focus_predicate: int,
        focus_object: int,
        other_subject: int,
        other_predicate: int,
        other_object: int,
        class_hierarchy: ClassHierarchy | None,
    ) -> None:
        self.facts_by_entity = facts_by_entity
        self.predicates_present = predicates_present
        self.assume_complete = assume_complete
        self.missing_edits = missing_edits
        self.focus_subject = focus_subject
        self.focus_predicate = focus_predicate
        self.focus_object = focus_object
        self.other_subject = other_subject
        self.other_predicate = other_predicate
        self.other_object = other_object
        self.class_hierarchy = class_hierarchy

    def entity_in_scope(self, entity_id: int) -> bool:
        return entity_id in self.facts_by_entity

    def property_complete(self, entity_id: int, predicate_id: int) -> bool:
        if self.assume_complete:
            return self.entity_in_scope(entity_id)
        return predicate_id in self.predicates_present.get(entity_id, set())

    def has_property(self, entity_id: int, predicate_id: int) -> bool:
        return len(self.facts_by_entity.get(entity_id, {}).get(predicate_id, set())) > 0

    def values_for(self, entity_id: int, predicate_id: int) -> Set[int]:
        return self.facts_by_entity.get(entity_id, {}).get(predicate_id, set())

    def has_statement(self, entity_id: int, predicate_id: int, object_id: int) -> bool:
        return object_id in self.facts_by_entity.get(entity_id, {}).get(predicate_id, set())

    def edit_unknown(self, entity_id: int, predicate_id: int) -> bool:
        return (entity_id, predicate_id) in self.missing_edits

    def focus_statement_present(self) -> bool:
        return self.has_statement(self.focus_subject, self.focus_predicate, self.focus_object)


def _load_registry(path: str | None) -> Dict[str, RegistryEntry]:
    if path is None:
        return {}
    return load_registry(path)


def _resolve_registry_id(raw_id: str | None, encoder: GlobalIntEncoder | None) -> int:
    return resolve_registry_id(raw_id, encoder)


def _resolve_registry_mapping(
    registry: Dict[str, RegistryEntry],
    *,
    encoder: GlobalIntEncoder | None,
    use_encoded_ids: bool,
) -> Dict[int, RegistryEntry] | Dict[str, RegistryEntry]:
    return resolve_registry_mapping(
        registry,
        encoder=encoder,
        use_encoded_ids=use_encoded_ids,
    )


def _lookup_registry_entry(
    constraint_id: Any,
    registry_by_id: Dict[int, RegistryEntry] | Dict[str, RegistryEntry],
    *,
    use_encoded_ids: bool,
) -> RegistryEntry | None:
    return lookup_registry_entry(
        constraint_id,
        registry_by_id,
        use_encoded_ids=use_encoded_ids,
    )


def _coerce_sequence(value: Any, *, cast_int: bool = True) -> List[Any]:
    # Fast path for common parquet payloads: 1-D primitive ndarrays.
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.dtype != object:
            if not cast_int:
                return value.tolist()
            if np.issubdtype(value.dtype, np.integer):
                return value.astype(np.int64, copy=False).tolist()
            coerced = []
            for v in value.tolist():
                try:
                    coerced.append(int(v))
                except (TypeError, ValueError):
                    coerced.append(0)
            return coerced
        if value.ndim == 0:
            scalar = value.item()
            if not cast_int:
                return [scalar]
            try:
                return [int(scalar)]
            except (TypeError, ValueError):
                return [0]

    # Fast path for flat Python sequences.
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(not isinstance(v, (list, tuple, np.ndarray)) for v in value):
            if not cast_int:
                return list(value)
            coerced = []
            for v in value:
                try:
                    coerced.append(int(v))
                except (TypeError, ValueError):
                    coerced.append(0)
            return coerced

    def _flatten(item: Any) -> List[Any]:
        if item is None:
            return []
        if isinstance(item, np.ndarray):
            return _flatten(item.tolist())
        if isinstance(item, (list, tuple)):
            flattened: List[Any] = []
            for sub in item:
                flattened.extend(_flatten(sub))
            return flattened
        return [item]

    seq = _flatten(value)
    if not cast_int:
        return seq
    coerced: List[Any] = []
    for v in seq:
        try:
            coerced.append(int(v))
        except (TypeError, ValueError):
            coerced.append(0)
    return coerced


def _coerce_value(value: Any, *, cast_int: bool = True) -> Any:
    if not cast_int:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _resolve_placeholder_token_ids(encoder: GlobalIntEncoder | None) -> Dict[str, int]:
    if encoder is None:
        return {}
    token_ids: Dict[str, int] = {}
    for label in PLACEHOLDER_LABELS:
        token_id = encoder.encode(label, add_new=False)
        if token_id:
            token_ids[label] = int(token_id)
    return token_ids


def _compute_p_local(row: Any, *, cast_int: bool = True) -> Set[Any]:
    p_local: Set[Any] = set()
    for name in ("predicate", "other_predicate"):
        value = _coerce_value(getattr(row, name, None), cast_int=cast_int)
        if value not in (None, "", 0):
            p_local.add(value)
    for name in ("subject_predicates", "object_predicates", "other_entity_predicates"):
        raw = getattr(row, name, None)
        if isinstance(raw, np.ndarray) and raw.ndim == 1 and raw.dtype != object:
            if cast_int and np.issubdtype(raw.dtype, np.integer):
                for pred_id in raw:
                    value = int(pred_id)
                    if value != 0:
                        p_local.add(value)
            else:
                for pred in raw.tolist():
                    value = _coerce_value(pred, cast_int=cast_int)
                    if value not in (None, "", 0):
                        p_local.add(value)
            continue
        for pred in _coerce_sequence(raw, cast_int=cast_int):
            if pred not in (None, "", 0):
                p_local.add(pred)
    return p_local


def _pick_other_entity_id(row: Any, *, cast_int: bool = True) -> Any:
    subject = _coerce_value(getattr(row, "subject", None), cast_int=cast_int)
    other_subject = _coerce_value(getattr(row, "other_subject", None), cast_int=cast_int)
    other_object = _coerce_value(getattr(row, "other_object", None), cast_int=cast_int)
    return resolve_other_entity_id(
        subject=subject,
        other_subject=other_subject,
        other_object=other_object,
    )


def _build_facts_for_entity(
    predicates: Sequence[Any] | np.ndarray | None,
    objects: Sequence[Any] | np.ndarray | None,
    *,
    p_local: Set[Any],
    cast_int: bool,
) -> Tuple[Dict[Any, Set[Any]], Set[Any]]:
    facts: Dict[Any, Set[Any]] = {}
    predicates_present: Set[Any] = set()
    if predicates is None or objects is None:
        return facts, predicates_present

    if (
        isinstance(predicates, np.ndarray)
        and isinstance(objects, np.ndarray)
        and predicates.ndim == 1
        and objects.ndim == 1
    ):
        limit = min(int(predicates.shape[0]), int(objects.shape[0]))
        if cast_int and np.issubdtype(predicates.dtype, np.integer) and np.issubdtype(objects.dtype, np.integer):
            for idx in range(limit):
                pred_id = int(predicates[idx])
                obj_id = int(objects[idx])
                if pred_id == 0 or obj_id == 0:
                    continue
                if pred_id not in p_local:
                    continue
                facts.setdefault(pred_id, set()).add(obj_id)
                predicates_present.add(pred_id)
            return facts, predicates_present

        for idx in range(limit):
            pred_id = _coerce_value(predicates[idx], cast_int=cast_int)
            obj_id = _coerce_value(objects[idx], cast_int=cast_int)
            if pred_id in (None, "", 0) or obj_id in (None, "", 0):
                continue
            if pred_id not in p_local:
                continue
            facts.setdefault(pred_id, set()).add(obj_id)
            predicates_present.add(pred_id)
        return facts, predicates_present

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

    def _merge_entity(entity_id: Any, facts: Dict[Any, Set[Any]], present: Set[Any]) -> None:
        if entity_id in (None, "", 0):
            return
        target = facts_by_entity.setdefault(entity_id, {})
        for pred, values in facts.items():
            target.setdefault(pred, set()).update(values)
        predicates_present.setdefault(entity_id, set()).update(present)

    def _add_explicit_statement(entity_id: Any, predicate_id: Any, object_value: Any) -> None:
        if entity_id in (None, "", 0) or predicate_id in (None, "", 0) or object_value in (None, "", 0):
            return
        facts_by_entity.setdefault(entity_id, {}).setdefault(predicate_id, set()).add(object_value)
        predicates_present.setdefault(entity_id, set()).add(predicate_id)

    subject_id = _coerce_value(getattr(row, "subject", None), cast_int=cast_int)
    object_id = _coerce_value(getattr(row, "object", None), cast_int=cast_int)
    other_entity_id = _pick_other_entity_id(row, cast_int=cast_int)

    subject_preds = getattr(row, "subject_predicates", None)
    subject_objs = getattr(row, "subject_objects", None)
    subject_facts, subject_present = _build_facts_for_entity(
        subject_preds, subject_objs, p_local=p_local, cast_int=cast_int
    )
    _merge_entity(subject_id, subject_facts, subject_present)

    if object_id not in (None, "", 0):
        object_preds = getattr(row, "object_predicates", None)
        object_objs = getattr(row, "object_objects", None)
        object_facts, object_present = _build_facts_for_entity(
            object_preds, object_objs, p_local=p_local, cast_int=cast_int
        )
        _merge_entity(object_id, object_facts, object_present)

    if other_entity_id not in (None, "", 0):
        other_preds = getattr(row, "other_entity_predicates", None)
        other_objs = getattr(row, "other_entity_objects", None)
        other_facts, other_present = _build_facts_for_entity(
            other_preds, other_objs, p_local=p_local, cast_int=cast_int
        )
        _merge_entity(other_entity_id, other_facts, other_present)

    _add_explicit_statement(
        subject_id,
        _coerce_value(getattr(row, "predicate", None), cast_int=cast_int),
        object_id,
    )
    _add_explicit_statement(
        _coerce_value(getattr(row, "other_subject", None), cast_int=cast_int),
        _coerce_value(getattr(row, "other_predicate", None), cast_int=cast_int),
        _coerce_value(getattr(row, "other_object", None), cast_int=cast_int),
    )

    return facts_by_entity, predicates_present


def _build_placeholder_map(
    encoder: GlobalIntEncoder | None,
    row: Any,
    placeholder_token_ids: Dict[str, int] | None = None,
) -> Dict[Any, Any]:
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

    token_ids = placeholder_token_ids if placeholder_token_ids is not None else _resolve_placeholder_token_ids(encoder)
    for label, token_id in token_ids.items():
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


def _build_post_state_for_candidate(
    base_facts_by_entity: Dict[Any, Dict[Any, Set[Any]]],
    base_predicates_present: Dict[Any, Set[Any]],
    p_local: Set[Any],
    *,
    candidate_slots: Sequence[int],
    placeholder_map: Dict[Any, Any],
    assume_complete: bool,
) -> Tuple[Dict[Any, Dict[Any, Set[Any]]], Dict[Any, Set[Any]], Set[Tuple[Any, Any]]]:
    post_facts = base_facts_by_entity
    post_predicates = base_predicates_present
    missing_edits: Set[Tuple[Any, Any]] | None = None

    copied_entity_maps: Dict[Any, Dict[Any, Set[Any]]] = {}
    copied_value_sets: Set[Tuple[Any, Any]] = set()
    copied_predicate_sets: Dict[Any, Set[Any]] = {}

    for kind, base_idx in (("del", 3), ("add", 0)):
        subj = _resolve_placeholder(int(candidate_slots[base_idx]), placeholder_map)
        pred = _resolve_placeholder(int(candidate_slots[base_idx + 1]), placeholder_map)
        obj = _resolve_placeholder(int(candidate_slots[base_idx + 2]), placeholder_map)
        if subj in (None, "", 0) or pred in (None, "", 0) or obj in (None, "", 0):
            continue
        if not assume_complete and pred not in base_predicates_present.get(subj, set()):
            if missing_edits is None:
                missing_edits = set()
            missing_edits.add((subj, pred))
            continue
        if pred not in p_local:
            if missing_edits is None:
                missing_edits = set()
            missing_edits.add((subj, pred))
            continue

        entity_facts = copied_entity_maps.get(subj)
        if entity_facts is None:
            source_facts = base_facts_by_entity.get(subj)
            if source_facts is None:
                if missing_edits is None:
                    missing_edits = set()
                missing_edits.add((subj, pred))
                continue
            if post_facts is base_facts_by_entity:
                post_facts = dict(base_facts_by_entity)
            entity_facts = dict(source_facts)
            post_facts[subj] = entity_facts
            copied_entity_maps[subj] = entity_facts

        key = (subj, pred)
        values = entity_facts.get(pred)
        if kind == "del":
            if values is None or obj not in values:
                continue
            if key not in copied_value_sets:
                entity_facts[pred] = set(values)
                copied_value_sets.add(key)
                values = entity_facts[pred]
            values.discard(obj)
            continue

        if values is None:
            entity_facts[pred] = {obj}
            copied_value_sets.add(key)
        else:
            if obj in values:
                pass
            else:
                if key not in copied_value_sets:
                    entity_facts[pred] = set(values)
                    copied_value_sets.add(key)
                    values = entity_facts[pred]
                values.add(obj)

        base_subject_preds = base_predicates_present.get(subj, set())
        if pred in base_subject_preds:
            continue

        subject_preds = copied_predicate_sets.get(subj)
        if subject_preds is None:
            if post_predicates is base_predicates_present:
                post_predicates = dict(base_predicates_present)
            subject_preds = set(base_subject_preds)
            post_predicates[subj] = subject_preds
            copied_predicate_sets[subj] = subject_preds
        subject_preds.add(pred)

    return post_facts, post_predicates, missing_edits if missing_edits is not None else _EMPTY_MISSING_EDITS


def _local_satisfied_fraction(checkable: Sequence[bool], satisfied: Sequence[int]) -> float:
    denom = sum(1 for flag in checkable if flag)
    if not denom:
        return 0.0
    return float(sum(int(value) for value, flag in zip(satisfied, checkable) if flag)) / float(denom)


def _resolved_candidate_triples(
    candidate_slots: Sequence[int],
    placeholder_map: Dict[Any, Any],
) -> Dict[str, Tuple[Any, Any, Any] | None]:
    if len(candidate_slots) < 6:
        return {"add": None, "del": None}

    def _triple(base_idx: int) -> Tuple[Any, Any, Any] | None:
        subj = _resolve_placeholder(int(candidate_slots[base_idx]), placeholder_map)
        pred = _resolve_placeholder(int(candidate_slots[base_idx + 1]), placeholder_map)
        obj = _resolve_placeholder(int(candidate_slots[base_idx + 2]), placeholder_map)
        if subj in (None, "", 0) or pred in (None, "", 0) or obj in (None, "", 0):
            return None
        return subj, pred, obj

    return {"add": _triple(0), "del": _triple(3)}


def _evidence_preservation_details(
    *,
    pre_state: EvidenceState,
    post_state: EvidenceState,
    candidate_slots: Sequence[int],
    placeholder_map: Dict[Any, Any],
    primary_satisfied: int,
    pre_global_satisfied_fraction: float,
    post_global_satisfied_fraction: float,
) -> Dict[str, Any]:
    pre_focus_present = bool(pre_state.focus_statement_present())
    post_focus_present = bool(post_state.focus_statement_present())
    focus_preserved = pre_focus_present and post_focus_present
    focus_deleted = pre_focus_present and not post_focus_present
    resolved = _resolved_candidate_triples(candidate_slots, placeholder_map)
    candidate_deletes_focus = resolved.get("del") == (
        pre_state.focus_subject,
        pre_state.focus_predicate,
        pre_state.focus_object,
    )
    return {
        "pre_global_satisfied_fraction": pre_global_satisfied_fraction,
        "post_global_satisfied_fraction": post_global_satisfied_fraction,
        "pre_focus_present": int(pre_focus_present),
        "post_focus_present": int(post_focus_present),
        "focus_preserved": int(focus_preserved),
        "focus_deleted": int(focus_deleted),
        "candidate_deletes_focus": int(candidate_deletes_focus),
        "non_vacuous_primary_fix": int(int(primary_satisfied) == 1 and focus_preserved),
        "vacuous_satisfaction_improvement": int(
            focus_deleted and post_global_satisfied_fraction > pre_global_satisfied_fraction
        ),
    }


def _constraint_type_id_from_registry(
    registry_entry: RegistryEntry,
    encoder: GlobalIntEncoder | None,
) -> int:
    del encoder
    return int(registry_entry.constraint_type_index)


def _build_constraint_instance(
    constraint_id: int,
    registry_entry: RegistryEntry,
    *,
    encoder: GlobalIntEncoder | None,
    constraint_type_name: str,
    constraint_type_id: int,
) -> ConstraintInstance:
    return build_registry_constraint_instance(
        constraint_id,
        registry_entry,
        encoder=encoder,
        constraint_type_name=constraint_type_name,
        constraint_type_id=constraint_type_id,
    )


def _prebind_constraint_checker(
    instance: ConstraintInstance | None,
) -> tuple[Any, Any, ConstraintInstance] | None:
    if instance is None:
        return None
    checker = CHECKERS.get(instance.constraint_type)
    if checker is None:
        return None
    is_checkable, is_satisfied = checker
    return is_checkable, is_satisfied, instance


def _evaluate_constraint_prebound(
    state: EvidenceState,
    checker_tuple: tuple[Any, Any, ConstraintInstance] | None,
    p_local: Set[Any],
) -> tuple[bool, int]:
    if checker_tuple is None:
        return False, 0
    is_checkable, is_satisfied, instance = checker_tuple
    checkable = bool(is_checkable(state, instance, p_local))
    if not checkable:
        return False, 0
    return True, 1 if bool(is_satisfied(state, instance, p_local)) else 0


class CandidateConstraintEvaluator:
    def __init__(
        self,
        registry_path: str,
        *,
        encoder: GlobalIntEncoder | None,
        assume_complete: bool,
        constraint_scope: str,
        use_encoded_ids: bool,
        class_hierarchy_path: str | Path | None = None,
    ) -> None:
        registry_raw = _load_registry(registry_path)
        self._registry_by_id = _resolve_registry_mapping(
            registry_raw, encoder=encoder, use_encoded_ids=use_encoded_ids
        )
        self._encoder = encoder
        self._assume_complete = assume_complete
        self._constraint_scope = constraint_scope
        self._use_encoded_ids = use_encoded_ids
        self._placeholder_token_ids = _resolve_placeholder_token_ids(encoder)
        self._constraint_cache: Dict[str, ConstraintInstance] = {}
        self._class_hierarchy = (
            ClassHierarchy.load(Path(class_hierarchy_path))
            if class_hierarchy_path is not None
            else None
        )

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
        return _metrics_from_details(details)

    def evaluate_candidates_loss_terms(
        self,
        row: Any,
        *,
        candidates: Sequence[Sequence[int]],
        gold_index: int,
        primary_factor_index: int | None = None,
        need_regression: bool = True,
        need_primary: bool = False,
    ) -> Tuple[List[float], List[float] | None]:
        """
        Compute only the chooser loss terms required by training.

        Returns:
            regression_rates: per-candidate secondary regression rates (zeros when disabled).
            primary_flags: per-candidate primary satisfaction flags (or None when disabled).
        """
        candidate_count = len(candidates)
        if candidate_count == 0:
            return [], [] if need_primary else None
        if not need_regression and not need_primary:
            return [0.0] * candidate_count, None
        if gold_index < 0 or gold_index >= candidate_count:
            raise ValueError(
                f"gold_index {gold_index} out of range for {candidate_count} candidates."
            )

        placeholder_map = _build_placeholder_map(self._encoder, row, self._placeholder_token_ids)
        resolved_edits = resolve_row_edits(
            row,
            placeholder_map=placeholder_map,
            coerce_value=lambda value: _coerce_value(
                value,
                cast_int=self._use_encoded_ids,
            ),
        )
        p_local = _compute_p_local(row, cast_int=self._use_encoded_ids)
        p_local.update(resolved_edits.predicates)
        p_local_set = p_local
        facts_by_entity, predicates_present = _build_facts_state(
            row,
            p_local=p_local,
            assume_complete=self._assume_complete,
            cast_int=self._use_encoded_ids,
        )
        normalize_pre_edit_state(facts_by_entity, predicates_present, resolved_edits)
        subject = _coerce_value(getattr(row, "subject", 0), cast_int=self._use_encoded_ids)
        predicate = _coerce_value(getattr(row, "predicate", 0), cast_int=self._use_encoded_ids)
        obj = _coerce_value(getattr(row, "object", 0), cast_int=self._use_encoded_ids)
        other_subject = _coerce_value(getattr(row, "other_subject", 0), cast_int=self._use_encoded_ids)
        other_predicate = _coerce_value(getattr(row, "other_predicate", 0), cast_int=self._use_encoded_ids)
        other_object = _coerce_value(getattr(row, "other_object", 0), cast_int=self._use_encoded_ids)

        if getattr(row, "eval_factor_constraint_ids", None) is not None:
            constraint_ids_raw = getattr(row, "eval_factor_constraint_ids", None)
        elif self._constraint_scope == "focus":
            constraint_ids_raw = getattr(row, "local_constraint_ids_focus", None)
            if constraint_ids_raw is None:
                constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        else:
            constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        local_constraint_ids = _coerce_sequence(constraint_ids_raw, cast_int=self._use_encoded_ids)
        if not local_constraint_ids:
            zeros = [0.0] * candidate_count
            return zeros, (list(zeros) if need_primary else None)

        constraint_instances: List[ConstraintInstance | None] = [
            self._get_constraint_instance(cid) for cid in local_constraint_ids
        ]
        constraint_checkers: List[tuple[Any, Any, ConstraintInstance] | None] = [
            _prebind_constraint_checker(instance) for instance in constraint_instances
        ]
        resolved_primary_index = -1
        if primary_factor_index is not None and 0 <= primary_factor_index < len(local_constraint_ids):
            resolved_primary_index = int(primary_factor_index)
        elif getattr(row, "eval_primary_factor_index", None) is not None:
            value = getattr(row, "eval_primary_factor_index")
            try:
                if 0 <= int(value) < len(local_constraint_ids):
                    resolved_primary_index = int(value)
            except (TypeError, ValueError):
                resolved_primary_index = -1
        else:
            constraint_id = _coerce_value(getattr(row, "constraint_id", None), cast_int=self._use_encoded_ids)
            try:
                resolved_primary_index = local_constraint_ids.index(constraint_id)
            except ValueError:
                resolved_primary_index = -1

        primary_checker = (
            constraint_checkers[resolved_primary_index]
            if 0 <= resolved_primary_index < len(constraint_checkers)
            else None
        )

        state = _ReusableEvidenceState(
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
            class_hierarchy=self._class_hierarchy,
        )

        tracked_checkers: List[tuple[Any, Any, ConstraintInstance]] = []
        if need_regression:
            gold_facts, gold_predicates, gold_missing = _build_post_state_for_candidate(
                facts_by_entity,
                predicates_present,
                p_local,
                candidate_slots=candidates[gold_index],
                placeholder_map=placeholder_map,
                assume_complete=self._assume_complete,
            )
            state.facts_by_entity = gold_facts
            state.predicates_present = gold_predicates
            state.missing_edits = gold_missing
            for idx, checker_tuple in enumerate(constraint_checkers):
                if idx == resolved_primary_index or checker_tuple is None:
                    continue
                is_checkable, is_satisfied, instance = checker_tuple
                checkable_post = bool(is_checkable(state, instance, p_local_set))
                if not checkable_post:
                    continue
                if bool(is_satisfied(state, instance, p_local_set)):
                    tracked_checkers.append(checker_tuple)
            if not tracked_checkers and not need_primary:
                return [0.0] * candidate_count, None

        if need_primary and primary_checker is None and not need_regression:
            zeros = [0.0] * candidate_count
            return zeros, list(zeros)
        primary_is_checkable = None
        primary_is_satisfied = None
        primary_instance: ConstraintInstance | None = None
        if primary_checker is not None:
            primary_is_checkable, primary_is_satisfied, primary_instance = primary_checker

        regression_rates: List[float] = []
        primary_flags: List[float] | None = [] if need_primary else None

        for candidate_slots in candidates:
            post_facts, post_predicates, missing_edits = _build_post_state_for_candidate(
                facts_by_entity,
                predicates_present,
                p_local,
                candidate_slots=candidate_slots,
                placeholder_map=placeholder_map,
                assume_complete=self._assume_complete,
            )
            state.facts_by_entity = post_facts
            state.predicates_present = post_predicates
            state.missing_edits = missing_edits

            if need_regression:
                regress = 0
                denom = 0
                for is_checkable, is_satisfied, instance in tracked_checkers:
                    checkable_post = bool(is_checkable(state, instance, p_local_set))
                    if not checkable_post:
                        continue
                    denom += 1
                    if not bool(is_satisfied(state, instance, p_local_set)):
                        regress += 1
                regression_rates.append(float(regress) / float(denom) if denom else 0.0)
            else:
                regression_rates.append(0.0)

            if need_primary and primary_flags is not None:
                primary_value = 0.0
                if (
                    primary_is_checkable is not None
                    and primary_is_satisfied is not None
                    and primary_instance is not None
                ):
                    checkable_post = bool(primary_is_checkable(state, primary_instance, p_local_set))
                    if checkable_post:
                        primary_value = float(bool(primary_is_satisfied(state, primary_instance, p_local_set)))
                primary_flags.append(primary_value)

        return regression_rates, primary_flags

    def evaluate_full(
        self,
        row: Any,
        *,
        candidate_slots: Sequence[int],
        primary_factor_index: int | None = None,
        factor_constraint_ids: Sequence[int] | None = None,
    ) -> Dict[str, Any]:
        placeholder_map = _build_placeholder_map(self._encoder, row, self._placeholder_token_ids)
        resolved_edits = resolve_row_edits(
            row,
            placeholder_map=placeholder_map,
            coerce_value=lambda value: _coerce_value(
                value,
                cast_int=self._use_encoded_ids,
            ),
        )
        p_local = _compute_p_local(row, cast_int=self._use_encoded_ids)
        p_local.update(resolved_edits.predicates)
        p_local_set = p_local
        facts_by_entity, predicates_present = _build_facts_state(
            row,
            p_local=p_local,
            assume_complete=self._assume_complete,
            cast_int=self._use_encoded_ids,
        )
        normalize_pre_edit_state(facts_by_entity, predicates_present, resolved_edits)
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
            class_hierarchy=self._class_hierarchy,
        )

        post_facts, post_predicates, missing_edits = _build_post_state_for_candidate(
            facts_by_entity,
            predicates_present,
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
            class_hierarchy=self._class_hierarchy,
        )

        if factor_constraint_ids is not None:
            constraint_ids_raw = factor_constraint_ids
        elif getattr(row, "eval_factor_constraint_ids", None) is not None:
            constraint_ids_raw = getattr(row, "eval_factor_constraint_ids", None)
        elif getattr(row, "factor_constraint_ids", None) is not None:
            constraint_ids_raw = getattr(row, "factor_constraint_ids", None)
        elif self._constraint_scope == "focus":
            constraint_ids_raw = getattr(row, "local_constraint_ids_focus", None)
            if constraint_ids_raw is None:
                constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        else:
            constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        local_constraint_ids = _coerce_sequence(constraint_ids_raw, cast_int=self._use_encoded_ids)
        if not local_constraint_ids:
            pre_global_satisfied_fraction = 0.0
            post_global_satisfied_fraction = 0.0
            evidence_details = _evidence_preservation_details(
                pre_state=pre_state,
                post_state=post_state,
                candidate_slots=candidate_slots,
                placeholder_map=placeholder_map,
                primary_satisfied=0,
                pre_global_satisfied_fraction=pre_global_satisfied_fraction,
                post_global_satisfied_fraction=post_global_satisfied_fraction,
            )
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
                "secondary_regressions_denom": 0,
                "secondary_improvements_denom": 0,
                "srr": 0.0,
                "sir": 0.0,
                "add_count": 0,
                "del_count": 0,
                **evidence_details,
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
            checkable_pre, satisfied_pre = evaluate_constraint(pre_state, instance, p_local_set)
            checkable_post, satisfied_post = evaluate_constraint(post_state, instance, p_local_set)
            pre_checkable.append(bool(checkable_pre))
            pre_satisfied.append(int(satisfied_pre))
            post_checkable.append(bool(checkable_post))
            post_satisfied.append(int(satisfied_post))

        resolved_primary_index = -1
        if primary_factor_index is not None and 0 <= primary_factor_index < len(local_constraint_ids):
            resolved_primary_index = int(primary_factor_index)
        elif getattr(row, "eval_primary_factor_index", None) is not None:
            value = getattr(row, "eval_primary_factor_index")
            try:
                if 0 <= int(value) < len(local_constraint_ids):
                    resolved_primary_index = int(value)
            except (TypeError, ValueError):
                resolved_primary_index = -1
        else:
            constraint_id = _coerce_value(getattr(row, "constraint_id", None), cast_int=self._use_encoded_ids)
            try:
                resolved_primary_index = local_constraint_ids.index(constraint_id)
            except ValueError:
                resolved_primary_index = -1

        primary_satisfied = 0
        if 0 <= resolved_primary_index < len(post_satisfied):
            primary_satisfied = post_satisfied[resolved_primary_index]

        pre_global_satisfied_fraction = _local_satisfied_fraction(pre_checkable, pre_satisfied)
        global_satisfied_fraction = _local_satisfied_fraction(post_checkable, post_satisfied)

        secondary_regressions = 0
        secondary_improvements = 0
        secondary_regressions_denom = 0
        secondary_improvements_denom = 0
        for idx in range(len(local_constraint_ids)):
            if idx == resolved_primary_index:
                continue
            if not pre_checkable[idx]:
                continue
            if not post_checkable[idx]:
                continue
            if pre_satisfied[idx]:
                secondary_regressions_denom += 1
                if not post_satisfied[idx]:
                    secondary_regressions += 1
            else:
                secondary_improvements_denom += 1
                if post_satisfied[idx]:
                    secondary_improvements += 1

        srr = float(secondary_regressions) / secondary_regressions_denom if secondary_regressions_denom else 0.0
        sir = float(secondary_improvements) / secondary_improvements_denom if secondary_improvements_denom else 0.0

        add_count = 0
        del_count = 0
        if len(candidate_slots) >= 6:
            if all(int(v) != 0 for v in candidate_slots[:3]):
                add_count = 1
            if all(int(v) != 0 for v in candidate_slots[3:6]):
                del_count = 1

        evidence_details = _evidence_preservation_details(
            pre_state=pre_state,
            post_state=post_state,
            candidate_slots=candidate_slots,
            placeholder_map=placeholder_map,
            primary_satisfied=primary_satisfied,
            pre_global_satisfied_fraction=pre_global_satisfied_fraction,
            post_global_satisfied_fraction=global_satisfied_fraction,
        )

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
            "secondary_regressions_denom": secondary_regressions_denom,
            "secondary_improvements_denom": secondary_improvements_denom,
            "srr": srr,
            "sir": sir,
            "add_count": add_count,
            "del_count": del_count,
            **evidence_details,
        }

    def evaluate_candidates(
        self,
        row: Any,
        *,
        candidates: Sequence[Sequence[int]],
        primary_factor_index: int | None = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        placeholder_map = _build_placeholder_map(self._encoder, row, self._placeholder_token_ids)
        resolved_edits = resolve_row_edits(
            row,
            placeholder_map=placeholder_map,
            coerce_value=lambda value: _coerce_value(
                value,
                cast_int=self._use_encoded_ids,
            ),
        )
        p_local = _compute_p_local(row, cast_int=self._use_encoded_ids)
        p_local.update(resolved_edits.predicates)
        p_local_set = p_local
        facts_by_entity, predicates_present = _build_facts_state(
            row,
            p_local=p_local,
            assume_complete=self._assume_complete,
            cast_int=self._use_encoded_ids,
        )
        normalize_pre_edit_state(facts_by_entity, predicates_present, resolved_edits)
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
            class_hierarchy=self._class_hierarchy,
        )

        if self._constraint_scope == "focus":
            constraint_ids_raw = getattr(row, "local_constraint_ids_focus", None)
            if constraint_ids_raw is None:
                constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        else:
            constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        local_constraint_ids = _coerce_sequence(constraint_ids_raw, cast_int=self._use_encoded_ids)
        if not local_constraint_ids:
            results: List[Dict[str, Any]] = []
            for candidate_slots in candidates:
                post_facts, post_predicates, missing_edits = _build_post_state_for_candidate(
                    facts_by_entity,
                    predicates_present,
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
                    class_hierarchy=self._class_hierarchy,
                )
                evidence_details = _evidence_preservation_details(
                    pre_state=pre_state,
                    post_state=post_state,
                    candidate_slots=candidate_slots,
                    placeholder_map=placeholder_map,
                    primary_satisfied=0,
                    pre_global_satisfied_fraction=0.0,
                    post_global_satisfied_fraction=0.0,
                )
                results.append(
                    {
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
                        "secondary_regressions_denom": 0,
                        "secondary_improvements_denom": 0,
                        "srr": 0.0,
                        "sir": 0.0,
                        "add_count": 0,
                        "del_count": 0,
                        **evidence_details,
                    }
                )
            return results

        constraint_instances: List[ConstraintInstance | None] = [
            self._get_constraint_instance(cid) for cid in local_constraint_ids
        ]
        pre_checkable: List[bool] = []
        pre_satisfied: List[int] = []
        for instance in constraint_instances:
            if instance is None:
                pre_checkable.append(False)
                pre_satisfied.append(0)
            else:
                checkable_pre, satisfied_pre = evaluate_constraint(pre_state, instance, p_local_set)
                pre_checkable.append(bool(checkable_pre))
                pre_satisfied.append(int(satisfied_pre))

        resolved_primary_index = -1
        if primary_factor_index is not None and 0 <= primary_factor_index < len(local_constraint_ids):
            resolved_primary_index = int(primary_factor_index)
        elif getattr(row, "eval_primary_factor_index", None) is not None:
            value = getattr(row, "eval_primary_factor_index")
            try:
                if 0 <= int(value) < len(local_constraint_ids):
                    resolved_primary_index = int(value)
            except (TypeError, ValueError):
                resolved_primary_index = -1
        else:
            constraint_id = _coerce_value(getattr(row, "constraint_id", None), cast_int=self._use_encoded_ids)
            try:
                resolved_primary_index = local_constraint_ids.index(constraint_id)
            except ValueError:
                resolved_primary_index = -1

        results: List[Dict[str, Any]] = []

        for candidate_slots in candidates:
            post_facts, post_predicates, missing_edits = _build_post_state_for_candidate(
                facts_by_entity,
                predicates_present,
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
                class_hierarchy=self._class_hierarchy,
            )

            post_checkable: List[bool] = []
            post_satisfied: List[int] = []
            for instance in constraint_instances:
                if instance is None:
                    post_checkable.append(False)
                    post_satisfied.append(0)
                else:
                    checkable_post, satisfied_post = evaluate_constraint(post_state, instance, p_local_set)
                    post_checkable.append(bool(checkable_post))
                    post_satisfied.append(int(satisfied_post))

            primary_satisfied = 0
            if 0 <= resolved_primary_index < len(post_satisfied):
                primary_satisfied = post_satisfied[resolved_primary_index]

            pre_global_satisfied_fraction = _local_satisfied_fraction(pre_checkable, pre_satisfied)
            global_satisfied_fraction = _local_satisfied_fraction(post_checkable, post_satisfied)

            secondary_regressions = 0
            secondary_improvements = 0
            secondary_regressions_denom = 0
            secondary_improvements_denom = 0
            for idx in range(len(local_constraint_ids)):
                if idx == resolved_primary_index:
                    continue
                if not pre_checkable[idx]:
                    continue
                if not post_checkable[idx]:
                    continue
                if pre_satisfied[idx]:
                    secondary_regressions_denom += 1
                    if not post_satisfied[idx]:
                        secondary_regressions += 1
                else:
                    secondary_improvements_denom += 1
                    if post_satisfied[idx]:
                        secondary_improvements += 1

            srr = float(secondary_regressions) / secondary_regressions_denom if secondary_regressions_denom else 0.0
            sir = float(secondary_improvements) / secondary_improvements_denom if secondary_improvements_denom else 0.0

            add_count = 0
            del_count = 0
            if len(candidate_slots) >= 6:
                if all(int(v) != 0 for v in candidate_slots[:3]):
                    add_count = 1
                if all(int(v) != 0 for v in candidate_slots[3:6]):
                    del_count = 1

            evidence_details = _evidence_preservation_details(
                pre_state=pre_state,
                post_state=post_state,
                candidate_slots=candidate_slots,
                placeholder_map=placeholder_map,
                primary_satisfied=primary_satisfied,
                pre_global_satisfied_fraction=pre_global_satisfied_fraction,
                post_global_satisfied_fraction=global_satisfied_fraction,
            )

            results.append(
                {
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
                    "secondary_regressions_denom": secondary_regressions_denom,
                    "secondary_improvements_denom": secondary_improvements_denom,
                    "srr": srr,
                    "sir": sir,
                    "add_count": add_count,
                    "del_count": del_count,
                    **evidence_details,
                }
            )

        return results

    def evaluate_candidate_metrics(
        self,
        row: Any,
        *,
        candidates: Sequence[Sequence[int]],
        primary_factor_index: int | None = None,
    ) -> List[CandidateMetrics]:
        details = self.evaluate_candidates(
            row,
            candidates=candidates,
            primary_factor_index=primary_factor_index,
        )
        return [_metrics_from_details(detail) for detail in details]


def _metrics_from_details(details: Dict[str, Any]) -> CandidateMetrics:
    return CandidateMetrics(
        primary_satisfied=int(details.get("primary_satisfied", 0)),
        global_satisfied_fraction=float(details.get("global_satisfied_fraction", 0.0)),
        secondary_regressions=int(details.get("secondary_regressions", 0)),
        secondary_regressions_denom=int(details.get("secondary_regressions_denom", 0)),
        secondary_improvements=int(details.get("secondary_improvements", 0)),
        secondary_improvements_denom=int(details.get("secondary_improvements_denom", 0)),
        srr=float(details.get("srr", 0.0)),
        sir=float(details.get("sir", 0.0)),
        add_count=int(details.get("add_count", 0)),
        del_count=int(details.get("del_count", 0)),
    )
