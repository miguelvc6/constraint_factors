#!/usr/bin/env python3
"""Build a constraint type catalog with labels from Wikidata (one-time script)."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import requests

from modules.wikidata_utils import query_wikidata_sparql


def _normalize_qid(raw: str) -> str | None:
    value = raw.strip()
    if not value:
        return None
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    if value.startswith("http://www.wikidata.org/entity/"):
        value = value.rsplit("/", 1)[-1]
    if value.startswith("Q") and value[1:].isdigit():
        return value
    return None


def _chunk(items: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _to_family(label: str) -> str:
    cleaned = "".join(ch for ch in label if ch.isalnum() or ch.isspace()).strip().lower()
    if not cleaned:
        return "unsupported"
    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0]
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build constraint type catalog from Wikidata.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        default="full",
        help="Dataset to scan for constraints.tsv (default: full).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/static/constraint_type_catalog.json"),
        help="Output path for the catalog JSON.",
    )
    args = parser.parse_args()

    constraints_path = Path("data/raw") / args.dataset / "constraints.tsv"
    if not constraints_path.exists():
        raise SystemExit(f"constraints.tsv not found at {constraints_path}")

    qids: set[str] = set()
    with constraints_path.open(newline="") as fh:
        reader = csv.DictReader(fh, dialect="excel-tab")
        for row in reader:
            raw = row.get(" constraint type id", "")
            qid = _normalize_qid(raw) if raw else None
            if qid:
                qids.add(qid)

    if not qids:
        raise SystemExit("No constraint_type_item values found.")

    session = requests.Session()
    headers = {
        "User-Agent": "ConstraintFactorBot/1.0 (https://github.com/your-repo; miguel.vazquez@wu.ac.at) Python-Requests/2.31",
        "Accept": "application/sparql-results+json",
    }

    labels: dict[str, str] = {}
    counts = Counter()
    qids_sorted = sorted(qids)

    for chunk in _chunk(qids_sorted, 200):
        values_clause = " ".join(f"wd:{qid}" for qid in chunk)
        query = f"""
            SELECT ?entity ?entityLabel WHERE {{
                VALUES ?entity {{ {values_clause} }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\" . }}
            }}
        """
        bindings = query_wikidata_sparql(session, query, headers=headers, base_delay=1.0, min_delay=1.0)
        for binding in bindings:
            entity_uri = binding["entity"]["value"]
            qid = entity_uri.rsplit("/", 1)[-1]
            label = binding.get("entityLabel", {}).get("value", qid)
            labels[qid] = label
        counts["chunks"] += 1

    catalog: dict[str, dict[str, str]] = {}
    for qid in qids_sorted:
        label = labels.get(qid, qid)
        family = _to_family(label)
        catalog[qid] = {
            "family": family,
            "label": label,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote catalog with {len(catalog)} entries to {args.output}")


if __name__ == "__main__":
    main()
