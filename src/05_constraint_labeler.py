#!/usr/bin/env python3
"""
05_constraint_labeler.py
========================
Generate per-factor constraint satisfaction labels (pre + post gold edit)
without rebuilding graphs.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from modules.class_hierarchy import (
    CLASS_HIERARCHY_FILENAME,
    CLASS_HIERARCHY_MANIFEST_FILENAME,
    ClassHierarchy,
    build_training_class_hierarchy,
    write_class_hierarchy,
)
from modules.constraint_checkers import (
    ConstraintInstance,
    EvidenceState,
    evaluate_constraint,
)
from modules.constraint_semantics import (
    CONSTRAINT_SEMANTICS_VERSION,
    RegistryEntry,
    build_constraint_instance as build_registry_constraint_instance,
    load_registry,
    lookup_registry_entry,
    resolve_registry_id,
    resolve_registry_mapping,
)
from modules.evidence_edits import (
    normalize_pre_edit_state,
    resolve_other_entity_id,
    resolve_row_edits,
)
from modules.data_encoders import GlobalIntEncoder, encoder_path
from modules.data_encoders import (
    DATASET_SCHEMA_VERSION,
    FEATURE_ENCODER_FILENAME,
    IDENTITY_ENCODER_FILENAME,
    IDENTITY_TO_FEATURE_FILENAME,
    LEGACY_ENCODER_FILENAME,
)

def _load_registry(path: Path) -> Dict[str, RegistryEntry]:
    return load_registry(path)


def _resolve_registry_id(raw_id: str | None, encoder: GlobalIntEncoder | None) -> int:
    return resolve_registry_id(raw_id, encoder)


def _coerce_sequence(value: Any, *, cast_int: bool = True) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        seq = value.tolist()
    elif isinstance(value, (list, tuple)):
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


def _build_facts_for_entity(
    predicates: Sequence[Any],
    objects: Sequence[Any],
    *,
    p_local: Set[Any],
    cast_int: bool,
) -> Tuple[Dict[Any, Set[Any]], Set[Any]]:
    facts: Dict[Any, Set[Any]] = defaultdict(set)
    predicates_present: Set[Any] = set()
    for pred, obj in zip(predicates, objects):
        pred_id = _coerce_value(pred, cast_int=cast_int)
        obj_id = _coerce_value(obj, cast_int=cast_int)
        if pred_id in (None, "", 0) or obj_id in (None, "", 0):
            continue
        if pred_id not in p_local:
            continue
        facts[pred_id].add(obj_id)
        predicates_present.add(pred_id)
    return facts, predicates_present


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
    subject = _coerce_value(getattr(row, "subject", None), cast_int=cast_int)
    other_subject = _coerce_value(getattr(row, "other_subject", None), cast_int=cast_int)
    other_object = _coerce_value(getattr(row, "other_object", None), cast_int=cast_int)
    return resolve_other_entity_id(
        subject=subject,
        other_subject=other_subject,
        other_object=other_object,
    )


def _build_facts_state(
    row: Any,
    *,
    p_local: Set[Any],
    assume_complete: bool,
    cast_int: bool,
) -> Tuple[Dict[int, Dict[int, Set[int]]], Dict[int, Set[int]]]:
    facts_by_entity: Dict[int, Dict[int, Set[int]]] = {}
    predicates_present: Dict[int, Set[int]] = {}

    def _merge_entity(
        entity_id: Any,
        facts: Dict[Any, Set[Any]],
        present: Set[Any],
    ) -> None:
        if entity_id in (None, "", 0):
            return
        target = facts_by_entity.setdefault(entity_id, {})
        for pred, values in facts.items():
            target.setdefault(pred, set()).update(values)
        predicates_present.setdefault(entity_id, set()).update(present)

    def _add_explicit_statement(entity_id: Any, predicate_id: Any, object_id: Any) -> None:
        if entity_id in (None, "", 0) or predicate_id in (None, "", 0) or object_id in (None, "", 0):
            return
        facts_by_entity.setdefault(entity_id, {}).setdefault(predicate_id, set()).add(object_id)
        predicates_present.setdefault(entity_id, set()).add(predicate_id)

    subject_id = _coerce_value(getattr(row, "subject", None), cast_int=cast_int)
    object_id = _coerce_value(getattr(row, "object", None), cast_int=cast_int)
    other_entity_id = _pick_other_entity_id(row, cast_int=cast_int)

    subject_preds = _coerce_sequence(getattr(row, "subject_predicates", None), cast_int=cast_int)
    subject_objs = _coerce_sequence(getattr(row, "subject_objects", None), cast_int=cast_int)
    subject_facts, subject_present = _build_facts_for_entity(
        subject_preds, subject_objs, p_local=p_local, cast_int=cast_int
    )
    _merge_entity(subject_id, subject_facts, subject_present)

    if object_id not in (None, "", 0):
        object_preds = _coerce_sequence(getattr(row, "object_predicates", None), cast_int=cast_int)
        object_objs = _coerce_sequence(getattr(row, "object_objects", None), cast_int=cast_int)
        object_facts, object_present = _build_facts_for_entity(
            object_preds, object_objs, p_local=p_local, cast_int=cast_int
        )
        _merge_entity(object_id, object_facts, object_present)

    if other_entity_id not in (None, "", 0):
        other_preds = _coerce_sequence(getattr(row, "other_entity_predicates", None), cast_int=cast_int)
        other_objs = _coerce_sequence(getattr(row, "other_entity_objects", None), cast_int=cast_int)
        other_facts, other_present = _build_facts_for_entity(
            other_preds, other_objs, p_local=p_local, cast_int=cast_int
        )
        _merge_entity(other_entity_id, other_facts, other_present)

    # The correction row's focus and comparison triples are authoritative
    # pre-edit statements even when the serialized neighborhood omits them.
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


def _resolve_placeholder(
    value: Any,
    row: Any,
    placeholder_map: Dict[Any, Any],
    *,
    cast_int: bool,
) -> Any:
    if value is None:
        return 0
    if value in placeholder_map:
        return placeholder_map[value]
    if not cast_int:
        return value
    return _coerce_value(value, cast_int=True)


def _apply_edit(
    facts_by_entity: Dict[int, Dict[int, Set[int]]],
    predicates_present: Dict[int, Set[int]],
    p_local: Set[Any],
    row: Any,
    *,
    placeholder_map: Dict[Any, Any],
    assume_complete: bool,
    cast_int: bool,
) -> Set[Tuple[int, int]]:
    missing_edits: Set[Tuple[int, int]] = set()

    def _apply(kind: str) -> None:
        subj = _resolve_placeholder(getattr(row, f"{kind}_subject", 0), row, placeholder_map, cast_int=cast_int)
        pred = _resolve_placeholder(getattr(row, f"{kind}_predicate", 0), row, placeholder_map, cast_int=cast_int)
        obj = _resolve_placeholder(getattr(row, f"{kind}_object", 0), row, placeholder_map, cast_int=cast_int)
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

    _apply("del")
    _apply("add")
    return missing_edits


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


def _load_encoder(path: Path | None) -> GlobalIntEncoder | None:
    if path is None:
        return None
    encoder = GlobalIntEncoder()
    encoder.load(path)
    return encoder


def _build_placeholder_map(encoder: GlobalIntEncoder | None, row: Any) -> Dict[Any, Any]:
    mapping: Dict[Any, Any] = {}
    if encoder is None:
        mapping = {
            "subject": getattr(row, "subject", 0),
            "predicate": getattr(row, "predicate", 0),
            "object": getattr(row, "object", 0),
            "other_subject": getattr(row, "other_subject", 0),
            "other_predicate": getattr(row, "other_predicate", 0),
            "other_object": getattr(row, "other_object", 0),
        }
        return mapping

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


def _constraint_type_id_from_registry(
    registry_entry: RegistryEntry,
    encoder: GlobalIntEncoder | None,
) -> int:
    del encoder
    return int(registry_entry.constraint_type_index)


def _process_dataframe(
    df: pd.DataFrame,
    registry_by_id: Dict[int, RegistryEntry] | Dict[str, RegistryEntry],
    *,
    encoder: GlobalIntEncoder | None,
    assume_complete: bool,
    use_encoded_ids: bool,
    constraint_scope: str,
    factor_family_policy: str,
    class_hierarchy: ClassHierarchy | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Counter[str]], Counter[str]]:
    constraint_cache: Dict[str, ConstraintInstance] = {}
    coverage: Dict[str, Counter[str]] = defaultdict(Counter)
    filter_stats: Counter[str] = Counter()

    factor_checkable_pre: List[List[bool]] = []
    factor_satisfied_pre: List[List[int]] = []
    factor_checkable_post: List[List[bool]] = []
    factor_satisfied_post: List[List[int]] = []
    factor_types: List[List[int]] = []
    factor_constraint_ids: List[List[int]] = []
    num_checkable_pre: List[int] = []
    num_checkable_post: List[int] = []
    coverage_pre: List[float] = []
    coverage_post: List[float] = []
    primary_factor_indices: List[int] = []
    primary_checkable_pre: List[bool] = []
    primary_satisfied_pre: List[int] = []
    primary_checkable_post: List[bool] = []
    primary_satisfied_post: List[int] = []
    primary_validation_reasons: List[str] = []
    primary_gold_repair_statuses: List[str] = []

    for row in df.itertuples(index=False):
        placeholder_map = _build_placeholder_map(encoder, row)
        resolved_edits = resolve_row_edits(
            row,
            placeholder_map=placeholder_map,
            coerce_value=lambda value: _coerce_value(value, cast_int=use_encoded_ids),
        )
        p_local = _compute_p_local(row, cast_int=use_encoded_ids)
        p_local.update(resolved_edits.predicates)
        facts_by_entity, predicates_present = _build_facts_state(
            row, p_local=p_local, assume_complete=assume_complete, cast_int=use_encoded_ids
        )
        normalize_pre_edit_state(facts_by_entity, predicates_present, resolved_edits)

        subject = _coerce_value(getattr(row, "subject", 0), cast_int=use_encoded_ids)
        predicate = _coerce_value(getattr(row, "predicate", 0), cast_int=use_encoded_ids)
        obj = _coerce_value(getattr(row, "object", 0), cast_int=use_encoded_ids)
        other_subject = _coerce_value(getattr(row, "other_subject", 0), cast_int=use_encoded_ids)
        other_predicate = _coerce_value(getattr(row, "other_predicate", 0), cast_int=use_encoded_ids)
        other_object = _coerce_value(getattr(row, "other_object", 0), cast_int=use_encoded_ids)

        pre_state = EvidenceState(
            facts_by_entity=facts_by_entity,
            predicates_present=predicates_present,
            assume_complete=assume_complete,
            missing_edits=set(),
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
            class_hierarchy=class_hierarchy,
        )

        post_facts = {
            ent: {pred: set(values) for pred, values in facts.items()} for ent, facts in facts_by_entity.items()
        }
        post_predicates = {ent: set(preds) for ent, preds in predicates_present.items()}
        missing_edits = _apply_edit(
            post_facts,
            post_predicates,
            p_local,
            row,
            placeholder_map=placeholder_map,
            assume_complete=assume_complete,
            cast_int=use_encoded_ids,
        )
        post_state = EvidenceState(
            facts_by_entity=post_facts,
            predicates_present=post_predicates,
            assume_complete=assume_complete,
            missing_edits=missing_edits,
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
            class_hierarchy=class_hierarchy,
        )

        if constraint_scope == "focus":
            constraint_ids_raw = getattr(row, "local_constraint_ids_focus", None)
            if constraint_ids_raw is None:
                constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        else:
            constraint_ids_raw = getattr(row, "local_constraint_ids", None)
        local_constraint_ids = _coerce_sequence(constraint_ids_raw, cast_int=use_encoded_ids)
        primary_constraint_id = _coerce_value(getattr(row, "constraint_id", 0), cast_int=use_encoded_ids)
        retained_constraint_ids: List[int] = []
        checkable_pre_row: List[bool] = []
        satisfied_pre_row: List[int] = []
        checkable_post_row: List[bool] = []
        satisfied_post_row: List[int] = []
        types_row: List[int] = []
        primary_reason = "missing_primary_factor"

        for constraint_id in local_constraint_ids:
            entry = _lookup_registry_entry(constraint_id, registry_by_id, use_encoded_ids=use_encoded_ids)
            is_primary = constraint_id == primary_constraint_id
            if entry is None:
                filter_stats["missing_registry_total"] += 1
                if factor_family_policy == "supported_only" and not is_primary:
                    filter_stats["missing_registry_filtered"] += 1
                    continue
                if is_primary:
                    filter_stats["missing_registry_primary_retained"] += 1
                retained_constraint_ids.append(int(constraint_id))
                checkable_pre_row.append(False)
                satisfied_pre_row.append(0)
                checkable_post_row.append(False)
                satisfied_post_row.append(0)
                types_row.append(-1)
                coverage["missing_registry"]["total"] += 1
                if is_primary:
                    primary_reason = "missing_registry"
                continue

            cache_key = str(int(constraint_id)) if use_encoded_ids else str(constraint_id)
            if cache_key not in constraint_cache:
                type_name = entry.constraint_family or ""
                constraint_type_id = _constraint_type_id_from_registry(entry, encoder)
                constraint_cache[cache_key] = _build_constraint_instance(
                    int(constraint_id) if use_encoded_ids else 0,
                    entry,
                    encoder=encoder,
                    constraint_type_name=type_name,
                    constraint_type_id=constraint_type_id,
                )

            constraint_instance = constraint_cache[cache_key]
            if not entry.constraint_family_supported:
                filter_stats["unsupported_total"] += 1
                filter_stats[f"unsupported_family::{constraint_instance.constraint_type}"] += 1
                if factor_family_policy == "supported_only" and not is_primary:
                    filter_stats["unsupported_filtered"] += 1
                    filter_stats[f"unsupported_family_filtered::{constraint_instance.constraint_type}"] += 1
                    continue
                if is_primary:
                    filter_stats["unsupported_primary_retained"] += 1
                    primary_reason = "unsupported_family"
                checkable_pre = False
                satisfied_pre = 0
                checkable_post = False
                satisfied_post = 0
            else:
                filter_stats["supported_retained"] += 1
                checkable_pre, satisfied_pre = evaluate_constraint(pre_state, constraint_instance, p_local)
                checkable_post, satisfied_post = evaluate_constraint(post_state, constraint_instance, p_local)

            if is_primary and entry.constraint_family_supported:
                if subject in constraint_instance.exceptions:
                    primary_reason = "exempt"
                elif not constraint_instance.applies_to_main_value:
                    primary_reason = "unsupported_scope"
                elif not checkable_pre:
                    primary_reason = "uncheckable_pre"
                elif satisfied_pre:
                    primary_reason = "already_satisfied_pre"
                else:
                    primary_reason = "valid"

            retained_constraint_ids.append(int(constraint_id))
            checkable_pre_row.append(bool(checkable_pre))
            satisfied_pre_row.append(int(satisfied_pre))
            checkable_post_row.append(bool(checkable_post))
            satisfied_post_row.append(int(satisfied_post))
            types_row.append(int(constraint_instance.constraint_type_id))

            ctype = constraint_instance.constraint_type or "unknown"
            coverage[ctype]["total"] += 1
            coverage[ctype]["checkable_pre"] += int(checkable_pre)
            coverage[ctype]["checkable_post"] += int(checkable_post)
            coverage[ctype]["satisfied_pre"] += int(satisfied_pre) if checkable_pre else 0
            coverage[ctype]["satisfied_post"] += int(satisfied_post) if checkable_post else 0

        filter_stats["raw_factor_total"] += len(local_constraint_ids)
        filter_stats["retained_factor_total"] += len(retained_constraint_ids)
        factor_checkable_pre.append(checkable_pre_row)
        factor_satisfied_pre.append(satisfied_pre_row)
        factor_checkable_post.append(checkable_post_row)
        factor_satisfied_post.append(satisfied_post_row)
        factor_types.append(types_row)
        factor_constraint_ids.append(retained_constraint_ids)

        total = len(retained_constraint_ids)
        num_checkable = sum(1 for flag in checkable_pre_row if flag)
        num_checkable_post_row = sum(1 for flag in checkable_post_row if flag)
        num_checkable_pre.append(num_checkable)
        num_checkable_post.append(num_checkable_post_row)
        coverage_pre.append(num_checkable / total if total else 0.0)
        coverage_post.append(num_checkable_post_row / total if total else 0.0)

        try:
            primary_index = retained_constraint_ids.index(int(primary_constraint_id))
        except (ValueError, TypeError):
            primary_index = -1
        primary_factor_indices.append(primary_index)
        if primary_index >= 0:
            primary_checkable_pre.append(bool(checkable_pre_row[primary_index]))
            primary_satisfied_pre.append(int(satisfied_pre_row[primary_index]))
            primary_checkable_post.append(bool(checkable_post_row[primary_index]))
            primary_satisfied_post.append(int(satisfied_post_row[primary_index]))
        else:
            primary_checkable_pre.append(False)
            primary_satisfied_pre.append(0)
            primary_checkable_post.append(False)
            primary_satisfied_post.append(0)
        primary_validation_reasons.append(primary_reason)
        if primary_reason != "valid":
            gold_repair_status = "ineligible_pre"
        elif primary_index < 0 or not checkable_post_row[primary_index]:
            gold_repair_status = "post_uncheckable"
        elif satisfied_post_row[primary_index]:
            gold_repair_status = "verified"
        else:
            gold_repair_status = "post_unsatisfied"
        primary_gold_repair_statuses.append(gold_repair_status)
        filter_stats[f"primary_validation::{primary_reason}"] += 1
        filter_stats[f"primary_gold_repair::{gold_repair_status}"] += 1

    df = df.copy()
    df["factor_checkable_pre"] = factor_checkable_pre
    df["factor_satisfied_pre"] = factor_satisfied_pre
    df["factor_checkable_post_gold"] = factor_checkable_post
    df["factor_satisfied_post_gold"] = factor_satisfied_post
    df["factor_types"] = factor_types
    df["factor_constraint_ids"] = factor_constraint_ids
    df["num_checkable_factors_pre"] = num_checkable_pre
    df["coverage_pre"] = coverage_pre
    df["num_checkable_factors_post_gold"] = num_checkable_post
    df["coverage_post_gold"] = coverage_post
    df["primary_factor_index"] = primary_factor_indices
    df["primary_checkable_pre"] = primary_checkable_pre
    df["primary_satisfied_pre"] = primary_satisfied_pre
    df["primary_checkable_post_gold"] = primary_checkable_post
    df["primary_satisfied_post_gold"] = primary_satisfied_post
    df["primary_validation_reason"] = primary_validation_reasons
    df["primary_gold_repair_status"] = primary_gold_repair_statuses
    df["primary_gold_repair_verified"] = [
        status == "verified" for status in primary_gold_repair_statuses
    ]

    return df, coverage, filter_stats


def _print_coverage(coverage: Dict[str, Counter[str]]) -> None:
    if not coverage:
        print("No coverage statistics collected.")
        return
    print("\nConstraint coverage summary:")
    for ctype in sorted(coverage.keys()):
        stats = coverage[ctype]
        total = stats.get("total", 0)
        if total == 0:
            continue
        checkable_pre = stats.get("checkable_pre", 0)
        checkable_post = stats.get("checkable_post", 0)
        satisfied_pre = stats.get("satisfied_pre", 0)
        satisfied_post = stats.get("satisfied_post", 0)
        pre_rate = checkable_pre / total if total else 0.0
        post_rate = checkable_post / total if total else 0.0
        pre_sat = satisfied_pre / checkable_pre if checkable_pre else 0.0
        post_sat = satisfied_post / checkable_post if checkable_post else 0.0
        print(
            f"- {ctype:20s} total={total:<6d} "
            f"checkable_pre={pre_rate:.2%} satisfied_pre={pre_sat:.2%} "
            f"checkable_post={post_rate:.2%} satisfied_post={post_sat:.2%}"
        )


def _coverage_rows(coverage: Dict[str, Counter[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ctype in sorted(coverage.keys()):
        stats = coverage[ctype]
        total = int(stats.get("total", 0))
        if total == 0:
            continue
        checkable_pre = int(stats.get("checkable_pre", 0))
        checkable_post = int(stats.get("checkable_post", 0))
        satisfied_pre = int(stats.get("satisfied_pre", 0))
        satisfied_post = int(stats.get("satisfied_post", 0))
        rows.append(
            {
                "constraint_family": ctype,
                "total": total,
                "checkable_pre": checkable_pre,
                "checkable_post": checkable_post,
                "satisfied_pre": satisfied_pre,
                "satisfied_post": satisfied_post,
                "checkable_pre_rate": checkable_pre / total if total else 0.0,
                "checkable_post_rate": checkable_post / total if total else 0.0,
            }
        )
    return rows


def _print_coverage_table(coverage: Dict[str, Counter[str]]) -> None:
    if not coverage:
        return
    rows = _coverage_rows(coverage)
    if not rows:
        return

    print("\nConstraint coverage table:")
    header = (
        "constraint_family",
        "total",
        "checkable_pre",
        "checkable_post",
        "satisfied_pre",
        "satisfied_post",
        "checkable_pre_rate",
        "checkable_post_rate",
    )
    print("  ".join(f"{col:>18s}" for col in header))
    for row in sorted(rows, key=lambda r: (r["checkable_pre_rate"], r["constraint_family"])):
        ctype = row["constraint_family"]
        total = row["total"]
        checkable_pre = row["checkable_pre"]
        checkable_post = row["checkable_post"]
        satisfied_pre = row["satisfied_pre"]
        satisfied_post = row["satisfied_post"]
        checkable_pre_rate = row["checkable_pre_rate"]
        checkable_post_rate = row["checkable_post_rate"]
        print(
            f"{ctype:>18s}  {total:18d}  {checkable_pre:18d}  {checkable_post:18d}  "
            f"{satisfied_pre:18d}  {satisfied_post:18d}  "
            f"{checkable_pre_rate:18.2%}  {checkable_post_rate:18.2%}"
        )


def _write_coverage_report(
    coverage: Dict[str, Counter[str]],
    output_root: Path,
    constraint_scope: str,
) -> None:
    rows = _coverage_rows(coverage)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["checkable_pre_rate", "constraint_family"])
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"coverage_{constraint_scope}.csv"
    md_path = output_root / f"coverage_{constraint_scope}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")


def _filtered_factor_rows(filter_stats: Counter[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_total = int(filter_stats.get("raw_factor_total", 0))
    retained_total = int(filter_stats.get("retained_factor_total", 0))
    filtered_total = raw_total - retained_total
    rows.append(
        {
            "metric": "raw_factor_total",
            "count": raw_total,
            "rate": 1.0 if raw_total else 0.0,
        }
    )
    rows.append(
        {
            "metric": "retained_factor_total",
            "count": retained_total,
            "rate": retained_total / raw_total if raw_total else 0.0,
        }
    )
    rows.append(
        {
            "metric": "filtered_factor_total",
            "count": filtered_total,
            "rate": filtered_total / raw_total if raw_total else 0.0,
        }
    )
    for key in sorted(filter_stats):
        if key.startswith("unsupported_family::") or key.startswith("unsupported_family_filtered::"):
            continue
        if key in {"raw_factor_total", "retained_factor_total"}:
            continue
        rows.append(
            {
                "metric": key,
                "count": int(filter_stats[key]),
                "rate": int(filter_stats[key]) / raw_total if raw_total else 0.0,
            }
        )
    return rows


def _filtered_family_rows(filter_stats: Counter[str]) -> List[Dict[str, Any]]:
    families = sorted(
        {
            key.split("::", 1)[1]
            for key in filter_stats
            if key.startswith("unsupported_family::")
        }
    )
    rows: List[Dict[str, Any]] = []
    raw_total = int(filter_stats.get("raw_factor_total", 0))
    for family in families:
        total = int(filter_stats.get(f"unsupported_family::{family}", 0))
        filtered = int(filter_stats.get(f"unsupported_family_filtered::{family}", 0))
        rows.append(
            {
                "constraint_family": family,
                "unsupported_occurrences": total,
                "filtered_occurrences": filtered,
                "retained_occurrences": total - filtered,
                "occurrence_rate": total / raw_total if raw_total else 0.0,
            }
        )
    return rows


def _write_filtered_factor_report(
    filter_stats: Counter[str],
    output_root: Path,
    constraint_scope: str,
    factor_family_policy: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = _filtered_factor_rows(filter_stats)
    family_rows = _filtered_family_rows(filter_stats)
    summary_df = pd.DataFrame(summary_rows)
    family_df = pd.DataFrame(family_rows)
    if not family_df.empty:
        family_df = family_df.sort_values(["filtered_occurrences", "constraint_family"], ascending=[False, True])

    summary_csv = output_root / f"filtered_factors_{constraint_scope}.csv"
    family_csv = output_root / f"filtered_factor_families_{constraint_scope}.csv"
    md_path = output_root / f"filtered_factors_{constraint_scope}.md"
    summary_df.to_csv(summary_csv, index=False)
    family_df.to_csv(family_csv, index=False)

    raw_total = int(filter_stats.get("raw_factor_total", 0))
    retained_total = int(filter_stats.get("retained_factor_total", 0))
    filtered_total = raw_total - retained_total
    lines = [
        "# Filtered Factor Report",
        "",
        f"- factor_family_policy: `{factor_family_policy}`",
        f"- raw_factor_total: {raw_total:,}",
        f"- retained_factor_total: {retained_total:,}",
        f"- filtered_factor_total: {filtered_total:,}",
        f"- filtered_factor_rate: {(filtered_total / raw_total if raw_total else 0.0):.2%}",
        f"- unsupported_primary_retained: {int(filter_stats.get('unsupported_primary_retained', 0)):,}",
        f"- missing_registry_primary_retained: {int(filter_stats.get('missing_registry_primary_retained', 0)):,}",
        "",
        "## Unsupported Families",
        "",
    ]
    if family_df.empty:
        lines.append("No unsupported families were encountered.")
    else:
        lines.append(family_df.to_markdown(index=False))
    md_path.write_text("\n".join(lines), encoding="utf-8")


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


def _iter_parquet_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    candidates = sorted(input_path.glob("df_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {input_path}")
    return candidates


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_dataset_contract(input_dir: Path, output_root: Path) -> None:
    for filename in (
        IDENTITY_ENCODER_FILENAME,
        FEATURE_ENCODER_FILENAME,
        LEGACY_ENCODER_FILENAME,
        IDENTITY_TO_FEATURE_FILENAME,
    ):
        source = input_dir / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)


def _write_labeled_manifest(
    *,
    input_dir: Path,
    output_root: Path,
    rows_by_split: dict[str, int],
    exclusions: Counter[str],
    exclusions_by_constraint: Counter[str],
    gold_repairs_by_constraint: Counter[str],
    filter_invalid_primary: bool,
) -> None:
    source_manifest_path = input_dir / "dataset_manifest.json"
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    payload = {
        **source_manifest,
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_variant": output_root.name,
        "parent_dataset_variant": input_dir.name,
        "parent_manifest_sha256": _sha256_file(source_manifest_path)
        if source_manifest_path.exists()
        else None,
        "semantic_labeling": {
            "version": CONSTRAINT_SEMANTICS_VERSION,
            "filter_invalid_primary": bool(filter_invalid_primary),
            "exclusions": dict(sorted(exclusions.items())),
            "primary_validation_by_constraint": dict(
                sorted(exclusions_by_constraint.items())
            ),
            "gold_repair_by_constraint": dict(
                sorted(gold_repairs_by_constraint.items())
            ),
        },
        "rows": dict(sorted(rows_by_split.items())),
        "outputs": {
            path.name: _sha256_file(path)
            for path in sorted(
                [*output_root.glob("df_*.parquet")]
                + [
                    output_root / filename
                    for filename in (
                        IDENTITY_ENCODER_FILENAME,
                        FEATURE_ENCODER_FILENAME,
                        LEGACY_ENCODER_FILENAME,
                        IDENTITY_TO_FEATURE_FILENAME,
                        CLASS_HIERARCHY_FILENAME,
                        CLASS_HIERARCHY_MANIFEST_FILENAME,
                        "primary_validation_audit.csv",
                        "primary_validation_audit_by_constraint.csv",
                        "primary_gold_repair_audit_by_constraint.csv",
                    )
                    if (output_root / filename).exists()
                ]
            )
        },
    }
    with (output_root / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label constraint satisfaction for local factors.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset variant to label, e.g. full or full_strat1m.",
    )
    parser.add_argument(
        "--registry-dataset",
        default=None,
        help="Raw dataset name for constraint_registry_<dataset>.parquet. Defaults to --dataset.",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Write a standalone dataset variant instead of the legacy <variant>_labeled directory.",
    )
    parser.add_argument(
        "--filter-invalid-primary",
        action="store_true",
        help="Exclude rows whose primary constraint is exempt, uncheckable, or already satisfied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset directory.",
    )
    parser.add_argument(
        "--min-occurrence",
        "--min-occurence",
        type=int,
        default=100,
        help="Minimum occurrence threshold used to build the parquet dataset.",
    )
    parser.add_argument(
        "--assume-complete-entity-facts",
        dest="assume_complete_entity_facts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assume entity facts are complete for all properties in scope (default).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows per parquet for debugging.",
    )
    parser.add_argument(
        "--constraint-scope",
        choices=["local", "focus"],
        default="local",
        help="Which constraint neighborhood to label: local (default) or focus predicate scope.",
    )
    parser.add_argument(
        "--factor-family-policy",
        choices=["supported_only", "all"],
        default="supported_only",
        help=(
            "Which attached constraints to write as supervised factors. "
            "supported_only drops unsupported secondary factors but retains unsupported primary factors; "
            "all preserves every local/focus constraint id."
        ),
    )
    args = parser.parse_args()

    from modules.data_encoders import base_dataset_name, dataset_variant_name

    dataset_variant = dataset_variant_name(args.dataset, args.min_occurrence)
    input_dir = Path("data") / "interim" / dataset_variant
    if args.output_dataset:
        output_variant = dataset_variant_name(args.output_dataset, args.min_occurrence)
    else:
        output_variant = f"{dataset_variant}_labeled"
    output_root = Path("data") / "interim" / output_variant
    registry_candidates = []
    if args.registry_dataset:
        registry_candidates.append(args.registry_dataset)
    registry_candidates.extend([args.dataset, base_dataset_name(args.dataset)])
    if "_strat" in base_dataset_name(args.dataset):
        registry_candidates.append(base_dataset_name(args.dataset).split("_strat", 1)[0])
    registry_path = None
    for candidate in dict.fromkeys(registry_candidates):
        candidate_path = Path("data") / "interim" / f"constraint_registry_{candidate}.parquet"
        if candidate_path.exists():
            registry_path = candidate_path
            break
    if registry_path is None:
        raise FileNotFoundError(f"No constraint registry found for candidates: {', '.join(dict.fromkeys(registry_candidates))}")
    resolved_encoder_path = encoder_path(input_dir, identity=True)

    registry_raw = _load_registry(registry_path)
    encoder = _load_encoder(resolved_encoder_path if resolved_encoder_path.exists() else None)

    parquet_paths = _iter_parquet_paths(input_dir)
    first_df = pd.read_parquet(parquet_paths[0], columns=["constraint_id"])
    use_encoded_ids = pd.api.types.is_integer_dtype(first_df["constraint_id"])
    if use_encoded_ids and encoder is None:
        raise SystemExit("Encoder is required to resolve registry ids for encoded parquet data.")
    registry_by_id = _resolve_registry_mapping(registry_raw, encoder=encoder, use_encoded_ids=use_encoded_ids)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_dataset_contract(input_dir, output_root)

    train_parquet = input_dir / "df_train.parquet"
    if not use_encoded_ids or encoder is None:
        raise SystemExit(
            f"{CONSTRAINT_SEMANTICS_VERSION} requires encoded identities and a frozen training hierarchy."
        )
    p279_predicate_id = _resolve_registry_id("P279", encoder)
    class_hierarchy = build_training_class_hierarchy(
        train_parquet,
        p279_predicate_id=p279_predicate_id,
    )
    write_class_hierarchy(
        class_hierarchy,
        output_root,
        p279_predicate_id=p279_predicate_id,
        source_dataset_variant=input_dir.name,
        source_manifest_path=input_dir / "dataset_manifest.json",
    )
    print(
        "Frozen training class hierarchy: "
        f"{class_hierarchy.direct_edge_count:,} direct P279 edges across "
        f"{class_hierarchy.child_count:,} child classes."
    )

    combined_coverage: Dict[str, Counter[str]] = defaultdict(Counter)
    combined_filter_stats: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    exclusion_counts_by_constraint: Counter[str] = Counter()
    gold_repair_counts_by_constraint: Counter[str] = Counter()
    rows_by_split: dict[str, int] = {}

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        if args.constraint_scope == "focus" and "local_constraint_ids_focus" not in df.columns:
            print("Warning: local_constraint_ids_focus missing; falling back to local_constraint_ids.")
        if args.max_rows is not None and args.max_rows > 0:
            df = df.iloc[: args.max_rows].copy()

        labeled_df, coverage, filter_stats = _process_dataframe(
            df,
            registry_by_id,
            encoder=encoder,
            class_hierarchy=class_hierarchy,
            assume_complete=args.assume_complete_entity_facts,
            use_encoded_ids=use_encoded_ids,
            constraint_scope=args.constraint_scope,
            factor_family_policy=args.factor_family_policy,
        )
        for ctype, stats in coverage.items():
            combined_coverage[ctype].update(stats)
        combined_filter_stats.update(filter_stats)

        split = parquet_path.stem.removeprefix("df_")
        reason_counts = labeled_df["primary_validation_reason"].value_counts()
        for reason, count in reason_counts.items():
            exclusion_counts[f"{split}::{reason}"] += int(count)
        grouped_reasons = labeled_df.groupby(
            ["constraint_type", "primary_validation_reason"],
            dropna=False,
        ).size()
        for (constraint_type, reason), count in grouped_reasons.items():
            exclusion_counts_by_constraint[
                f"{split}::{constraint_type}::{reason}"
            ] += int(count)
        grouped_gold_repairs = labeled_df.groupby(
            ["constraint_type", "primary_gold_repair_status"],
            dropna=False,
        ).size()
        for (constraint_type, status), count in grouped_gold_repairs.items():
            gold_repair_counts_by_constraint[
                f"{split}::{constraint_type}::{status}"
            ] += int(count)
        if args.filter_invalid_primary:
            labeled_df = labeled_df.loc[
                labeled_df["primary_validation_reason"] == "valid"
            ].reset_index(drop=True)
        rows_by_split[split] = int(len(labeled_df))

        output_path = output_root / parquet_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_parquet(output_path, index=False)
        print(f"Wrote labeled parquet to {output_path}")

    _print_coverage(combined_coverage)
    _print_coverage_table(combined_coverage)
    _write_coverage_report(combined_coverage, output_root, args.constraint_scope)
    _write_filtered_factor_report(
        combined_filter_stats,
        output_root,
        args.constraint_scope,
        args.factor_family_policy,
    )
    exclusion_rows = []
    for key, count in sorted(exclusion_counts.items()):
        split, reason = key.split("::", 1)
        exclusion_rows.append({"split": split, "reason": reason, "count": int(count)})
    pd.DataFrame(exclusion_rows).to_csv(
        output_root / "primary_validation_audit.csv",
        index=False,
    )
    exclusion_rows_by_constraint = []
    for key, count in sorted(exclusion_counts_by_constraint.items()):
        split, constraint_type, reason = key.split("::", 2)
        exclusion_rows_by_constraint.append(
            {
                "split": split,
                "constraint_type": constraint_type,
                "reason": reason,
                "count": int(count),
            }
        )
    pd.DataFrame(exclusion_rows_by_constraint).to_csv(
        output_root / "primary_validation_audit_by_constraint.csv",
        index=False,
    )
    gold_repair_rows_by_constraint = []
    for key, count in sorted(gold_repair_counts_by_constraint.items()):
        split, constraint_type, status = key.split("::", 2)
        gold_repair_rows_by_constraint.append(
            {
                "split": split,
                "constraint_type": constraint_type,
                "status": status,
                "count": int(count),
            }
        )
    pd.DataFrame(gold_repair_rows_by_constraint).to_csv(
        output_root / "primary_gold_repair_audit_by_constraint.csv",
        index=False,
    )
    _write_labeled_manifest(
        input_dir=input_dir,
        output_root=output_root,
        rows_by_split=rows_by_split,
        exclusions=exclusion_counts,
        exclusions_by_constraint=exclusion_counts_by_constraint,
        gold_repairs_by_constraint=gold_repair_counts_by_constraint,
        filter_invalid_primary=args.filter_invalid_primary,
    )


if __name__ == "__main__":
    main()
