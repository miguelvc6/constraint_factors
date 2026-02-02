#!/usr/bin/env python3
"""List constraint type items and their canonical families from a registry parquet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="List constraint type items from registry.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Which dataset registry to inspect.",
    )
    args = parser.parse_args()

    registry_path = Path("data") / "interim" / f"constraint_registry_{args.dataset}.parquet"
    df = pd.read_parquet(registry_path)
    registry_json = df["registry_json"].iloc[0]
    registry = json.loads(registry_json) if isinstance(registry_json, str) else registry_json
    if not registry:
        print("Registry is empty. Rebuild it with 03_constraint_registry.py.")
        return

    type_counts = Counter()
    name_counts = Counter()
    for entry in registry.values():
        type_item = entry.get("constraint_type_item")
        type_name = entry.get("constraint_family")
        if type_item is None or type_name is None:
            print(
                "Registry entries are missing constraint_type_item/constraint_family. "
                "Rebuild the registry with the updated 03_constraint_registry.py."
            )
            return
        type_item = type_item or ""
        type_name = type_name or ""
        type_counts[type_item] += 1
        name_counts[(type_item, type_name)] += 1

    print("Constraint type items (count):")
    for qid, count in type_counts.most_common():
        print(f"- {qid}: {count}")

    print("\nType item → name assignments:")
    for (qid, name), count in name_counts.most_common():
        print(f"- {qid} -> {name}: {count}")


if __name__ == "__main__":
    main()
