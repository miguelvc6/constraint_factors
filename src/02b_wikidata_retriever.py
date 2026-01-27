#!/usr/bin/env python3
"""
02b_wikidata_retriever.py
=========================
Pre-compute Wikidata labels and embeddings used by 03_graph.py.

The script scans the interim parquet files, collects every referenced
identifier and literal text, resolves their human-readable labels via
Wikidata, embeds them with the SentenceTransformer model, and persists the
result as a parquet cache under data/raw/wikidata_text.parquet.


Usage
-----
python src/02b_wikidata_retriever.py --dataset sample
"""

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.data_encoders import (
    SCALAR_FEATURES,
    SEQUENCE_FEATURES,
    GlobalIntEncoder,
    dataset_variant_name,
    discover_min_occurrence,
)
from modules.wikidata_utils import WikidataUriEmbedder

TEXT_COLUMN_SUFFIX = "_text"
DEFAULT_OUTPUT = Path("data/interim/wikidata_text.parquet")


class CacheEntry(NamedTuple):
    key: str
    kind: str
    global_id: int | None
    text: str
    embedding: np.ndarray


CacheKey = tuple[str, str, int | None]


def _ensure_iterable(value: Any) -> Iterable[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return value
    if value is None or value == "":
        return []
    return [value]


def _gather_from_dataframe(
    dataframe: pd.DataFrame,
    unique_ids: set[int],
    literal_texts: set[str],
) -> None:
    """Scan the dataframe and collect all unique integer IDs and literal texts in-place."""
    # Gather all unique non-zero integer IDs from the dataframe.
    for column in SCALAR_FEATURES:
        if column not in dataframe:
            continue
        for value in dataframe[column]:
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            ivalue = int(value)
            if ivalue != 0:
                unique_ids.add(ivalue)

    # Also gather from sequence features.
    for column in SEQUENCE_FEATURES:
        if column not in dataframe:
            continue
        for value in dataframe[column]:
            for element in _ensure_iterable(value):
                if element is None:
                    continue
                if isinstance(element, float) and np.isnan(element):
                    continue
                ivalue = int(element)
                if ivalue != 0:
                    unique_ids.add(ivalue)

    # Gather all literal texts from the dataframe.
    for column in dataframe.columns:
        if not column.endswith(TEXT_COLUMN_SUFFIX):
            continue
        for text in dataframe[column]:
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            if stripped and stripped.lower() != "nan":
                literal_texts.add(stripped)


def _prepare_identifier_sets(
    encoder: GlobalIntEncoder,
    dataframe_root: Path,
) -> tuple[set[int], set[str]]:
    """Scan the parquet files and collect all unique integer IDs and literal texts."""
    unique_ids: set[int] = set()
    literal_texts: set[str] = set()

    for split in ("train", "val", "test"):
        parquet_path = dataframe_root / f"df_{split}.parquet"
        if not parquet_path.exists():
            logging.warning("Skipping missing parquet file: %s", parquet_path)
            continue
        logging.info("Scanning %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        _gather_from_dataframe(df, unique_ids, literal_texts)
        del df

    # Remove filtered IDs from the set, because we will never need their wikidata information.
    unique_ids = unique_ids - encoder._filtered_ids

    placeholders = (
        "unknown",
        "subject",
        "predicate",
        "object",
        "other_subject",
        "other_predicate",
        "other_object",
        "LITERAL_OBJECT",
    )
    for placeholder in placeholders:
        placeholder_id = encoder.encode(placeholder, add_new=False)
        if placeholder_id:
            unique_ids.add(placeholder_id)

    unique_ids.discard(0)
    literal_texts.discard("")
    return unique_ids, literal_texts


def _resolve_registry_id(encoder: GlobalIntEncoder, raw_id: str) -> int:
    """Resolve a registry identifier into an existing global id."""
    candidates: list[str] = []
    if raw_id.startswith("http://") or raw_id.startswith("https://"):
        candidates.append(raw_id)
        candidates.append(f"<{raw_id}>")
    else:
        candidates.append(raw_id)
        if raw_id and raw_id[0] in ("P", "Q") and raw_id[1:].isdigit():
            candidates.append(f"http://www.wikidata.org/entity/{raw_id}")
            candidates.append(f"<http://www.wikidata.org/entity/{raw_id}>")

    for candidate in candidates:
        gid = encoder.encode(candidate, add_new=False)
        if gid:
            return gid
    raise ValueError(f"Registry id '{raw_id}' not found in the global encoder.")


def _load_constraint_registry(interim_root: Path) -> dict[str, dict[str, Any]]:
    registry_path = interim_root / "constraint_registry.parquet"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Constraint registry not found at {registry_path}. "
            "Run src/02a_constraint_registry.py first."
        )
    registry_df = pd.read_parquet(registry_path)
    if "registry_json" not in registry_df.columns or registry_df.empty:
        raise ValueError(f"Constraint registry is missing 'registry_json' in {registry_path}")
    return json.loads(str(registry_df.loc[0, "registry_json"]))


def _collect_registry_ids(
    encoder: GlobalIntEncoder,
    registry: dict[str, dict[str, Any]],
) -> set[int]:
    registry_ids: set[int] = set()
    for entry in registry.values():
        constrained_property = entry.get("constrained_property")
        if constrained_property:
            registry_ids.add(_resolve_registry_id(encoder, constrained_property))

        for predicate in entry.get("param_predicates", []):
            registry_ids.add(_resolve_registry_id(encoder, predicate))
        for obj in entry.get("param_objects", []):
            registry_ids.add(_resolve_registry_id(encoder, obj))

    registry_ids.discard(0)
    return registry_ids


def _load_existing_cache(path: Path) -> tuple[dict[CacheKey, CacheEntry], dict[tuple[str, str], CacheEntry]]:
    """Load an existing parquet cache if available, returning two lookup dictionaries."""
    if not path.exists():
        return {}, {}

    logging.info("Loading existing cache from %s", path)
    dataframe = pd.read_parquet(path)

    cache: dict[CacheKey, CacheEntry] = {}
    by_kind_key: dict[tuple[str, str], CacheEntry] = {}

    for row in dataframe.itertuples(index=False):
        raw_gid = getattr(row, "global_id", None)
        gid = None if pd.isna(raw_gid) else int(raw_gid)

        embedding_array = np.asarray(row.embedding, dtype=np.float16)
        entry = CacheEntry(row.key, row.kind, gid, row.text, embedding_array)
        key = (entry.kind, entry.key, entry.global_id)
        cache[key] = entry
        by_kind_key.setdefault((entry.kind, entry.key), entry)

    logging.info("Loaded %s cached entries", len(cache))
    return cache, by_kind_key


def _materialise_entries(
    encoder: GlobalIntEncoder,
    embedder: WikidataUriEmbedder,
    unique_ids: set[int],
    literal_texts: set[str],
    existing_cache: dict[CacheKey, CacheEntry],
    existing_by_kind_key: dict[tuple[str, str], CacheEntry],
) -> list[CacheEntry]:
    """Materialise cache entries for the given identifiers and literal texts."""
    records: list[CacheEntry] = []

    uri_to_ids: dict[str, list[int]] = {}
    placeholder_to_ids: dict[str, list[int]] = {}

    def register(record: CacheEntry) -> None:
        """Register a new cache entry if not already present."""
        cache_key = (record.kind, record.key, record.global_id)
        if cache_key in existing_cache:
            return
        existing_cache[cache_key] = record
        existing_by_kind_key.setdefault((record.kind, record.key), record)
        records.append(record)

    # First, try to reuse existing entries or prepare for new lookups.
    for gid in sorted(unique_ids):
        decoded = encoder.decode(gid)
        if not decoded:
            continue
        if "wikidata.org" in decoded:
            cache_key = ("uri", decoded, gid)
            if cache_key in existing_cache:
                continue
            existing_entry = existing_by_kind_key.get(("uri", decoded))
            if existing_entry:
                register(
                    CacheEntry(
                        decoded,
                        "uri",
                        gid,
                        existing_entry.text,
                        existing_entry.embedding.copy(),
                    )
                )
            else:
                uri_to_ids.setdefault(decoded, []).append(gid)
        else:
            cache_key = ("placeholder", decoded, gid)
            if cache_key in existing_cache:
                continue
            existing_entry = existing_by_kind_key.get(("placeholder", decoded))
            if existing_entry:
                register(
                    CacheEntry(
                        decoded,
                        "placeholder",
                        gid,
                        decoded,
                        existing_entry.embedding.copy(),
                    )
                )
            else:
                placeholder_to_ids.setdefault(decoded, []).append(gid)

    # Now, resolve and embed new entries as needed.
    if uri_to_ids:
        logging.info("Resolving %s new Wikidata URIs", len(uri_to_ids))
        all_uris = list(uri_to_ids.keys())
        uri_embeddings = embedder.embed_uris(all_uris)
        for uri, ids in uri_to_ids.items():
            text = embedder.uri2text.get(uri, uri)
            embedding = np.asarray(uri_embeddings[uri], dtype=np.float16)
            for gid in ids:
                register(CacheEntry(uri, "uri", gid, text, embedding.copy()))

    # Embed placeholder tokens
    if placeholder_to_ids:
        tokens = list(placeholder_to_ids.keys())
        logging.info("Embedding %s placeholder identifiers", len(tokens))
        for start in tqdm(range(0, len(tokens), embedder.batch_size)):
            batch_tokens = tokens[start : start + embedder.batch_size]
            batch_embeddings = embedder.get_embeddings_from_texts(batch_tokens).astype(np.float16, copy=False)
            for token, embedding in zip(batch_tokens, batch_embeddings):
                for gid in placeholder_to_ids[token]:
                    register(CacheEntry(token, "placeholder", gid, token, embedding.copy()))

    # Finally, embed literal texts
    pending_literals = []
    for text in sorted(literal_texts):
        cache_key = ("literal", text, None)
        if cache_key in existing_cache:
            continue
        pending_literals.append(text)

    if pending_literals:
        logging.info("Embedding %s literal texts", len(pending_literals))
        for start in tqdm(range(0, len(pending_literals), embedder.batch_size)):
            batch_texts = pending_literals[start : start + embedder.batch_size]
            batch_embeddings = embedder.get_embeddings_from_texts(batch_texts).astype(np.float16, copy=False)
            for text, embedding in zip(batch_texts, batch_embeddings):
                register(CacheEntry(text, "literal", None, text, embedding.copy()))

    return records


def _persist_records(records: list[CacheEntry], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda e: (e.kind, e.key, e.global_id or -1))

    # Build columns
    keys = [r.key for r in sorted_records]
    kinds = [r.kind for r in sorted_records]
    gids_raw = [r.global_id for r in sorted_records]
    texts = [r.text for r in sorted_records]
    embeds = [r.embedding.astype(np.float16).tolist() for r in sorted_records]
    gid_series = pd.Series([pd.NA if g is None else g for g in gids_raw], dtype="Int64")

    # Persist as parquet
    dataframe = pd.DataFrame(
        {
            "key": keys,
            "kind": kinds,
            "global_id": gid_series,
            "text": texts,
            "embedding": embeds,
        }
    )
    dataframe.to_parquet(output, index=False)
    logging.info("Persisted %s cached entries to %s", len(sorted_records), output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialise Wikidata lookups for 03_graph.py")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Dataset split to scan (same as other pipeline stages).",
    )
    parser.add_argument(
        "--min-occurrence",
        type=int,
        default=None,
        help=(
            "Min occurrence threshold used for the dataframe builder stage. "
            "Defaults to the smallest available parquet dataset."
        ),
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimensionality for the SentenceTransformer model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for Wikidata label resolution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Location of the parquet cache (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()

    # If min_occurrence is not provided, try to discover it automatically.
    if args.min_occurrence is None:
        args.min_occurrence = discover_min_occurrence(args.dataset)
        logging.info(
            "Discovered min occurrence %s for dataset '%s'",
            args.min_occurrence,
            args.dataset,
        )

    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Input paths
    dataset_variant = dataset_variant_name(args.dataset, max(1, args.min_occurrence))
    interim_root = Path("data/interim") / dataset_variant
    if not interim_root.exists():
        raise FileNotFoundError(f"Interim data directory not found: {interim_root}")

    # Load the global integer encoder
    encoder = GlobalIntEncoder()
    encoder.load(interim_root / "globalintencoder.txt")
    encoder.freeze()

    # Load existing cache if available
    existing_cache, existing_by_kind_key = _load_existing_cache(args.output)

    # Gather all unique IDs and literal texts from the dataframes
    unique_ids, literal_texts = _prepare_identifier_sets(encoder, interim_root)

    # Merge constraint-registry identifiers needed for factor nodes.
    registry = _load_constraint_registry(interim_root)
    registry_ids = _collect_registry_ids(encoder, registry)
    pre_merge_count = len(unique_ids)
    unique_ids.update(registry_ids)

    if not registry_ids.issubset(unique_ids):
        raise RuntimeError("Constraint registry identifiers missing from the retrieval set.")
    logging.info(
        "Merged %s registry identifiers into the retrieval set",
        len(unique_ids) - pre_merge_count,
    )

    logging.info(
        "Collected %s unique identifiers and %s literal texts",
        len(unique_ids),
        len(literal_texts),
    )

    # Materialise cache entries
    embedder = WikidataUriEmbedder(batch_size=args.batch_size, embed_dim=args.embed_dim)
    new_records = _materialise_entries(
        encoder,
        embedder,
        unique_ids,
        literal_texts,
        existing_cache,
        existing_by_kind_key,
    )

    total_records = list(existing_cache.values())
    if not total_records:
        raise RuntimeError("No Wikidata entries were collected; aborting cache creation.")

    new_count = len(new_records)
    logging.info(
        "Prepared %s total cached entries (%s new, %s reused)",
        len(total_records),
        new_count,
        len(total_records) - new_count,
    )

    # Persist the cache
    _persist_records(total_records, args.output)


if __name__ == "__main__":
    main()
