#!/usr/bin/env python3
"""
02a_constraint_registry.py
==========================
Build a standalone constraint registry from constraints.tsv.

Usage
-----
python src/02a_constraint_registry.py --dataset {sample,full}
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

CONSTRAINT_TYPE_PREDICATE = "<http://www.wikidata.org/entity/P2302>"


def _load_dataframe_builder() -> Any:
    module_path = Path(__file__).with_name("02_dataframe_builder.py")
    spec = importlib.util.spec_from_file_location("dataframe_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_entity_id(value: str) -> str:
    raw = value.strip().strip("<>").strip()
    if raw.startswith("http://www.wikidata.org/entity/"):
        raw = raw.rsplit("/", 1)[-1]
    return raw


def _normalize_token(value: str) -> str:
    """Normalize tokens to match the encoder's string form."""
    raw = value.strip()
    if raw.startswith("http://www.wikidata.org/prop/direct/"):
        raw = raw.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")
    return raw


def build_registry(
    constraints_def: dict[str, dict[str, list[str]]],
    constraints_by_property: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    constraint_to_property: dict[str, str] = {}
    for prop, constraint_ids in constraints_by_property.items():
        for constraint_id in constraint_ids:
            if constraint_id in constraint_to_property:
                raise ValueError(f"Constraint id {constraint_id} appears in multiple properties.")
            constraint_to_property[constraint_id] = prop

    registry: dict[str, dict[str, Any]] = {}
    for constraint_id in sorted(constraints_def):
        definition = constraints_def[constraint_id]
        predicates = definition["predicates"]
        objects = definition["objects"]
        if len(predicates) != len(objects):
            raise ValueError(f"Predicate/object mismatch for {constraint_id}.")

        type_objects = [obj for pred, obj in zip(predicates, objects) if pred == CONSTRAINT_TYPE_PREDICATE]
        if len(type_objects) != 1:
            raise ValueError(f"Constraint {constraint_id} has {len(type_objects)} type objects; expected 1.")
        constraint_type = _normalize_token(type_objects[0])

        constrained_property = constraint_to_property.get(constraint_id)
        if constrained_property is None:
            raise ValueError(f"Constraint {constraint_id} missing constrained property.")

        param_predicates: list[str] = []
        param_objects: list[str] = []
        for pred, obj in zip(predicates, objects):
            if pred == CONSTRAINT_TYPE_PREDICATE:
                continue
            if pred.startswith("^") and _normalize_entity_id(obj) == constrained_property:
                continue
            param_predicates.append(_normalize_token(pred))
            param_objects.append(_normalize_token(obj))

        registry[constraint_id] = {
            "constraint_type": constraint_type,
            "constrained_property": constrained_property,
            "param_predicates": param_predicates,
            "param_objects": param_objects,
        }

    return registry


def validate_registry(
    registry: dict[str, dict[str, Any]],
    constraints_def: dict[str, dict[str, list[str]]],
    constraints_by_property: dict[str, list[str]],
) -> None:
    if len(registry) != len(constraints_def):
        raise ValueError("Registry entry count does not match parsed constraint instances.")

    if len(registry) != len(set(registry.keys())):
        raise ValueError("Constraint ids appear more than once in the registry.")

    constraint_to_property = {}
    for prop, constraint_ids in constraints_by_property.items():
        for constraint_id in constraint_ids:
            if constraint_id in constraint_to_property:
                raise ValueError(f"Constraint {constraint_id} has multiple constrained properties.")
            constraint_to_property[constraint_id] = prop
    if set(constraint_to_property.keys()) != set(registry.keys()):
        raise ValueError("Registry constraint ids differ from constrained properties index.")

    for constraint_id, entry in registry.items():
        if entry.get("constrained_property") != constraint_to_property.get(constraint_id):
            raise ValueError(f"Constraint {constraint_id} has inconsistent constrained property.")
        for field in ("param_predicates", "param_objects"):
            values = entry.get(field)
            if not isinstance(values, list):
                raise TypeError(f"{constraint_id} field {field} must be a list.")
            if any(value in (None, "") for value in values):
                raise ValueError(f"{constraint_id} field {field} contains null/empty values.")
            if not all(isinstance(value, str) for value in values):
                raise TypeError(f"{constraint_id} field {field} must contain strings.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a constraint registry artifact.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Which dataset to read constraints.tsv from.",
    )
    args = parser.parse_args()

    raw_data_path = Path("data/raw") / args.dataset
    interim_root = Path("data/interim")
    interim_root.mkdir(parents=True, exist_ok=True)

    builder = _load_dataframe_builder()
    builder.RAW_DATA_PATH = raw_data_path
    constraints_def, constraints_by_property = builder.load_constraint_data()

    registry = build_registry(constraints_def, constraints_by_property)
    validate_registry(registry, constraints_def, constraints_by_property)

    registry_json = json.dumps(registry, sort_keys=True)
    output_path = interim_root / f"constraint_registry_{args.dataset}.parquet"
    pd.DataFrame({"registry_json": [registry_json]}).to_parquet(output_path)

    print(f"Wrote constraint registry with {len(registry)} entries to {output_path}")


if __name__ == "__main__":
    main()
