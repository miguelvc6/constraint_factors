#!/usr/bin/env python3
"""
04_constraint_labeler.py
========================
Generate per-factor constraint satisfaction labels (pre + post gold edit)
without rebuilding graphs.
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from modules.constraint_checkers import (
    ConstraintInstance,
    EvidenceState,
    evaluate_constraint,
    normalize_property_id,
    normalize_token,
)
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


def _load_registry(path: Path) -> Dict[str, RegistryEntry]:
    registry_df = pd.read_parquet(path)
    registry_json = registry_df["registry_json"].iloc[0]
    registry = json.loads(registry_json) if isinstance(registry_json, str) else registry_json
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
    other_subject = _coerce_value(getattr(row, "other_subject", None), cast_int=cast_int)
    if other_subject not in (None, "", 0):
        return other_subject
    other_object = _coerce_value(getattr(row, "other_object", None), cast_int=cast_int)
    if other_object not in (None, "", 0):
        return other_object
    return 0


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
    if encoder is None:
        return 0
    return _resolve_registry_id(registry_entry.constraint_type_item, encoder)


def _resolve_default_relations(encoder: GlobalIntEncoder | None) -> List[int]:
    if encoder is None:
        return []
    p31 = _resolve_registry_id("P31", encoder)
    p279 = _resolve_registry_id("P279", encoder)
    defaults = [pid for pid in (p31, p279) if pid]
    return defaults


def _process_dataframe(
    df: pd.DataFrame,
    registry_by_id: Dict[int, RegistryEntry] | Dict[str, RegistryEntry],
    *,
    encoder: GlobalIntEncoder | None,
    assume_complete: bool,
    use_encoded_ids: bool,
) -> Tuple[pd.DataFrame, Dict[str, Counter[str]]]:
    default_relation_predicates = _resolve_default_relations(encoder)
    constraint_cache: Dict[str, ConstraintInstance] = {}
    coverage: Dict[str, Counter[str]] = defaultdict(Counter)

    factor_checkable_pre: List[List[bool]] = []
    factor_satisfied_pre: List[List[int]] = []
    factor_checkable_post: List[List[bool]] = []
    factor_satisfied_post: List[List[int]] = []
    factor_types: List[List[int]] = []
    num_checkable_pre: List[int] = []
    num_checkable_post: List[int] = []
    coverage_pre: List[float] = []
    coverage_post: List[float] = []

    for row in df.itertuples(index=False):
        p_local = _compute_p_local(row, cast_int=use_encoded_ids)
        facts_by_entity, predicates_present = _build_facts_state(
            row, p_local=p_local, assume_complete=assume_complete, cast_int=use_encoded_ids
        )

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
        )

        post_facts = {
            ent: {pred: set(values) for pred, values in facts.items()} for ent, facts in facts_by_entity.items()
        }
        post_predicates = {ent: set(preds) for ent, preds in predicates_present.items()}
        placeholder_map = _build_placeholder_map(encoder, row)
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
        )

        local_constraint_ids = _coerce_sequence(getattr(row, "local_constraint_ids", None), cast_int=use_encoded_ids)
        checkable_pre_row: List[bool] = []
        satisfied_pre_row: List[int] = []
        checkable_post_row: List[bool] = []
        satisfied_post_row: List[int] = []
        types_row: List[int] = []

        for constraint_id in local_constraint_ids:
            entry = _lookup_registry_entry(constraint_id, registry_by_id, use_encoded_ids=use_encoded_ids)
            if entry is None:
                checkable_pre_row.append(False)
                satisfied_pre_row.append(0)
                checkable_post_row.append(False)
                satisfied_post_row.append(0)
                types_row.append(0)
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
                    default_relation_predicates=default_relation_predicates,
                )

            constraint_instance = constraint_cache[cache_key]
            if not entry.constraint_family_supported:
                checkable_pre = False
                satisfied_pre = 0
                checkable_post = False
                satisfied_post = 0
            else:
                checkable_pre, satisfied_pre = evaluate_constraint(pre_state, constraint_instance, p_local)
                checkable_post, satisfied_post = evaluate_constraint(post_state, constraint_instance, p_local)

            checkable_pre_row.append(bool(checkable_pre))
            satisfied_pre_row.append(int(satisfied_pre))
            checkable_post_row.append(bool(checkable_post))
            satisfied_post_row.append(int(satisfied_post))
            types_row.append(int(constraint_instance.constraint_type_id or 0))

            ctype = constraint_instance.constraint_type or "unknown"
            coverage[ctype]["total"] += 1
            coverage[ctype]["checkable_pre"] += int(checkable_pre)
            coverage[ctype]["checkable_post"] += int(checkable_post)
            coverage[ctype]["satisfied_pre"] += int(satisfied_pre) if checkable_pre else 0
            coverage[ctype]["satisfied_post"] += int(satisfied_post) if checkable_post else 0

        factor_checkable_pre.append(checkable_pre_row)
        factor_satisfied_pre.append(satisfied_pre_row)
        factor_checkable_post.append(checkable_post_row)
        factor_satisfied_post.append(satisfied_post_row)
        factor_types.append(types_row)

        total = len(local_constraint_ids)
        num_checkable = sum(1 for flag in checkable_pre_row if flag)
        num_checkable_post_row = sum(1 for flag in checkable_post_row if flag)
        num_checkable_pre.append(num_checkable)
        num_checkable_post.append(num_checkable_post_row)
        coverage_pre.append(num_checkable / total if total else 0.0)
        coverage_post.append(num_checkable_post_row / total if total else 0.0)

    df = df.copy()
    df["factor_checkable_pre"] = factor_checkable_pre
    df["factor_satisfied_pre"] = factor_satisfied_pre
    df["factor_checkable_post_gold"] = factor_checkable_post
    df["factor_satisfied_post_gold"] = factor_satisfied_post
    df["factor_types"] = factor_types
    df["num_checkable_factors_pre"] = num_checkable_pre
    df["coverage_pre"] = coverage_pre
    df["num_checkable_factors_post_gold"] = num_checkable_post
    df["coverage_post_gold"] = coverage_post

    return df, coverage


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
        missing: List[str] = []
        for constraint_id, entry in registry.items():
            cid = _resolve_registry_id(constraint_id, encoder)
            if cid == 0:
                missing.append(constraint_id)
                continue
            mapped[cid] = entry
        if missing:
            print(f"Warning: {len(missing)} registry ids could not be resolved via encoder.")
        return mapped
    mapped_str: Dict[str, RegistryEntry] = {}
    for constraint_id, entry in registry.items():
        key = normalize_token(constraint_id) or constraint_id
        mapped_str[key] = entry
    return mapped_str


def _iter_parquet_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    candidates = sorted(input_path.glob("df_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {input_path}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Label constraint satisfaction for local factors.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Which dataset to label.",
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
    args = parser.parse_args()

    from modules.data_encoders import dataset_variant_name

    dataset_variant = dataset_variant_name(args.dataset, args.min_occurrence)
    input_dir = Path("data") / "interim" / dataset_variant
    output_root = Path("data") / "interim" / f"{dataset_variant}_labeled"
    registry_path = Path("data") / "interim" / f"constraint_registry_{args.dataset}.parquet"
    encoder_path = input_dir / "globalintencoder.txt"

    registry_raw = _load_registry(registry_path)
    encoder = _load_encoder(encoder_path if encoder_path.exists() else None)

    parquet_paths = _iter_parquet_paths(input_dir)
    first_df = pd.read_parquet(parquet_paths[0], columns=["constraint_id"])
    use_encoded_ids = pd.api.types.is_integer_dtype(first_df["constraint_id"])
    if use_encoded_ids and encoder is None:
        raise SystemExit("Encoder is required to resolve registry ids for encoded parquet data.")
    registry_by_id = _resolve_registry_mapping(registry_raw, encoder=encoder, use_encoded_ids=use_encoded_ids)
    output_root.mkdir(parents=True, exist_ok=True)

    combined_coverage: Dict[str, Counter[str]] = defaultdict(Counter)

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        if args.max_rows is not None and args.max_rows > 0:
            df = df.iloc[: args.max_rows].copy()

        labeled_df, coverage = _process_dataframe(
            df,
            registry_by_id,
            encoder=encoder,
            assume_complete=args.assume_complete_entity_facts,
            use_encoded_ids=use_encoded_ids,
        )
        for ctype, stats in coverage.items():
            combined_coverage[ctype].update(stats)

        output_path = output_root / parquet_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_parquet(output_path)
        print(f"Wrote labeled parquet to {output_path}")

    _print_coverage(combined_coverage)


if __name__ == "__main__":
    main()
