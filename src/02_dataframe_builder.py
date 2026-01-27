#!/usr/bin/env python3
"""
02_dataframe_builder.py
====================
Transform the raw datasets into parquet dataframes

Datasets
--------
* **sample** - the sample corpus shipped in the GitHub repository
  `Tpt/bass-materials` -> 1'80GB
* **full**   - the full dump hosted on Figshare
  (article ID 13338743) -> 10'50GB

Usage
-----
python src/02_dataframe_builder.py --dataset {sample,full} --min-occurrence N

"""

import argparse
import csv
import gc
import gzip
import json
import re
from collections import Counter, defaultdict
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.data_encoders import SCALAR_FEATURES, SEQUENCE_FEATURES, GlobalIntEncoder, dataset_variant_name

UNKNOWN_TOKEN_STRING = "unknown"

RESERVED_TOKEN_STRINGS: tuple[str, ...] = (
    "subject",
    "predicate",
    "object",
    "other_subject",
    "other_predicate",
    "other_object",
    "LITERAL_OBJECT",
    UNKNOWN_TOKEN_STRING,
)
FACTOR_TOKEN_PREFIX = "constraint_factor::"
FACTOR_TOKEN_STRINGS: tuple[str, ...] = ()
CONSTRAINT_ID_TOKENS: tuple[str, ...] = ()

ALLOW_NEW_TOKENS = True
RECORD_FREQUENCIES = False
TOKEN_FREQUENCY: Counter[int] = Counter()
MIN_OCCURRENCE = 1
UNKNOWN_TOKEN_ID = 0
VALIDATE_CONSTRAINTS = True
PROP_RE = re.compile(r"^P[1-9]\d*$")

# Includes the new per-row constraint neighborhood IDs.
SEQUENCE_FEATURE_KEYS: tuple[str, ...] = SEQUENCE_FEATURES + ("local_constraint_ids",)
REGISTRY_RESERVED_IDS: set[int] = set()


def _normalize_property_id(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip().strip("<>").strip()
    if raw.startswith("http://www.wikidata.org/prop/direct/"):
        raw = raw.rsplit("/", 1)[-1]
    elif raw.startswith("http://www.wikidata.org/entity/"):
        raw = raw.rsplit("/", 1)[-1]
    if PROP_RE.match(raw):
        return raw
    return None


def load_constraint_data() -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
    # load the constraints data
    constraints_def: dict[str, dict[str, list[str]]] = {}
    constraints_by_property: dict[str, list[str]] = defaultdict(list)  # property -> constraint ids

    mapping_to_wikidata = {
        "property id": "^<http://www.wikidata.org/entity/P2302>",
        " constraint type id": "<http://www.wikidata.org/entity/P2302>",
        "regex": "<http://www.wikidata.org/entity/P1793>",
        "exceptions": "<http://www.wikidata.org/entity/P2303>",
        "group by": "<http://www.wikidata.org/entity/P2304>",
        "items": "<http://www.wikidata.org/entity/P2305>",
        "property": "<http://www.wikidata.org/entity/P2306>",
        "namespace": "<http://www.wikidata.org/entity/P2307>",
        "class": "<http://www.wikidata.org/entity/P2308>",
        "relation": "<http://www.wikidata.org/entity/P2309>",
        "minimal date": "<http://www.wikidata.org/entity/P2310>",
        "maximum date": "<http://www.wikidata.org/entity/P2311>",
        "maximum value": "<http://www.wikidata.org/entity/P2312>",
        "minimal value": "<http://www.wikidata.org/entity/P2313>",
        "status": "<http://www.wikidata.org/entity/P2316>",
        "separator": "<http://www.wikidata.org/entity/P4155>",
        "scope": "<http://www.wikidata.org/entity/P4680>",
    }
    with open(RAW_DATA_PATH / "constraints.tsv", newline="") as fp:
        for row in csv.DictReader(fp, dialect="excel-tab"):
            predicates: list[str] = []
            objects: list[str] = []
            for k, vs in row.items():
                if k != "constraint id":
                    for v in vs.split(" "):
                        v = v.strip()
                        if v:
                            predicates.append(mapping_to_wikidata[k])
                            objects.append(v)
            constraint_id = row["constraint id"]
            constrained_property = _normalize_property_id(row["property id"])
            constraints_def[constraint_id] = {
                "predicates": predicates,
                "objects": objects,
            }
            if constrained_property is not None:
                constraints_by_property[constrained_property].append(constraint_id)

    constraints_by_property = {
        prop: sorted(constraint_ids) for prop, constraint_ids in constraints_by_property.items()
    }

    if VALIDATE_CONSTRAINTS:
        constraint_ids = set(constraints_def.keys())
        indexed_ids = [cid for ids in constraints_by_property.values() for cid in ids]
        indexed_set = set(indexed_ids)
        assert len(indexed_ids) == len(constraints_def), "Constraint count mismatch in constraints_by_property"
        assert indexed_set == constraint_ids, "constraints_by_property must index every constraint id exactly once"
        assert all(PROP_RE.match(prop) for prop in constraints_by_property), (
            "Invalid property id in constraints_by_property"
        )

    return constraints_def, constraints_by_property


def _encode_constraints_by_property(constraints_by_property: dict[str, list[str]]) -> dict[str, list[int]]:
    encoded: dict[str, list[int]] = {}
    for prop, constraint_ids in constraints_by_property.items():
        encoded_ids = {_convert_value(cid) for cid in constraint_ids}
        encoded[prop] = sorted(encoded_ids)
    return encoded


def _collect_registry_reserved_ids(
    constraints_def: dict[str, dict[str, list[str]]],
    constraints_by_property_raw: dict[str, list[str]],
) -> set[int]:
    reserved: set[int] = set()
    for prop_qid in constraints_by_property_raw.keys():
        reserved.add(_convert_value(prop_qid))
    for constraint in constraints_def.values():
        for predicate in constraint["predicates"]:
            reserved.add(_convert_value(predicate))
        for obj in constraint["objects"]:
            reserved.add(_convert_value(obj))
    return reserved


def _register_token(token_id: int) -> int:
    if token_id == 0:
        return 0
    if RECORD_FREQUENCIES:
        TOKEN_FREQUENCY[token_id] += 1
    return token_id


def _convert_value(
    value: str | None,
    subject: Optional[int] = None,
    predicate: Optional[int] = None,
    obj: Optional[int] = None,
    other_subject: Optional[int] = None,
    other_predicate: Optional[int] = None,
    other_object: Optional[int] = None,
) -> int:
    """Convert *value* to its global integer id, handling placeholder tokens.

    The function first maps *value* through the GlobalIntEncoder. If the raw
    string equals any of the *subject / predicate / object* parameters (or
    their *other_* counterparts) it is instead replaced by a synthetic
    placeholder token such as 'subject' or 'other_object' before
    encoding so that identical placeholders share a single id.

    Returns
    -------
    int
        Encoded identifier suitable for numerical ML pipelines.
    """
    global UNKNOWN_TOKEN_ID
    if value is None or value == "":
        return 0

    normalized = value.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")
    encoded_value = ENCODER.encode(
        normalized,
        add_new=ALLOW_NEW_TOKENS,
    )

    if encoded_value == 0 and not ALLOW_NEW_TOKENS:
        encoded_value = UNKNOWN_TOKEN_ID

    if encoded_value == subject:
        return _register_token(ENCODER.encode("subject", add_new=ALLOW_NEW_TOKENS))
    if encoded_value == predicate:
        return _register_token(ENCODER.encode("predicate", add_new=ALLOW_NEW_TOKENS))
    if encoded_value == obj:
        return _register_token(ENCODER.encode("object", add_new=ALLOW_NEW_TOKENS))
    if encoded_value == other_subject:
        return _register_token(ENCODER.encode("other_subject", add_new=ALLOW_NEW_TOKENS))
    if encoded_value == other_predicate:
        return _register_token(ENCODER.encode("other_predicate", add_new=ALLOW_NEW_TOKENS))
    if encoded_value == other_object:
        return _register_token(ENCODER.encode("other_object", add_new=ALLOW_NEW_TOKENS))

    return _register_token(encoded_value)


def _seed_constraint_factor_tokens(constraints_def: dict[str, dict[str, list[str]]]) -> None:
    """Seed constraint factor tokens into the encoder and reserved list."""
    global FACTOR_TOKEN_STRINGS
    global CONSTRAINT_ID_TOKENS
    constraint_ids = sorted(constraints_def.keys(), key=str)
    CONSTRAINT_ID_TOKENS = tuple(constraint_ids)
    FACTOR_TOKEN_STRINGS = tuple(f"{FACTOR_TOKEN_PREFIX}{cid}" for cid in constraint_ids)
    assert len(FACTOR_TOKEN_STRINGS) == len(constraints_def), "Constraint factor token count mismatch."
    for token in FACTOR_TOKEN_STRINGS:
        ENCODER.encode(token, add_new=True)


def _read_entity_desc(line: list[str], desc_position: int) -> dict[str, Any]:
    """Parse the JSON entity/page description found at *desc_position* in *line*.

    Depending on the "type" field the description can either refer to a
    Wikidata **entity** (containing labels and facts) or to a **page**
    (containing HTTP status code and raw HTML content).

    The function translates every URI it touches via :pyfunc:`_convert_value`
    and returns four lists which are later merged into the main dataset.

    Parameters
    ----------
    line
        Entire TSV row already split on \\t.
    desc_position
        Index of the field holding the JSON description.

    Returns
    -------
    dict
        Keys:
            - entity_predicates : list[int]
            - entity_objects   : list[int]
            - entity_labels    : list[str]
            - http_content     : str
    """
    desc = line[desc_position].strip()
    result: dict[str, Any] = {
        "entity_predicates": [],
        "entity_objects": [],
        "entity_labels": [],
        "entity_property_qids": [],
        "http_content": "",
    }
    if not desc:
        return result
    try:
        desc = json.loads(desc)
    except ValueError:
        print("Invalid description: {}".format(desc))
        return result
    if desc.get("type") == "page":
        # robust status handling (covers non-standard codes like 520, 599, 999)
        code = desc.get("statusCode")
        try:
            phrase = HTTPStatus(code).phrase  # standard codes
        except Exception:
            # fallback: still create a name; you can also map specific vendor codes here
            phrase = f"Status{code}"
        status_name = phrase.title().replace(" ", "").replace("-", "")
        status_iri = f"<http://www.w3.org/2011/http-statusCodes#{status_name}>"

        result["entity_predicates"].append(_convert_value("<http://wikiba.se/history/ontology#pageStatusCode>"))
        result["entity_objects"].append(_convert_value(status_iri))
        result["http_content"] = desc.get("content", "")

    elif desc.get("type") == "entity":
        result["entity_labels"].extend(desc.get("labels", {}).values())
        for predicate, objects in desc.get("facts", {}).items():
            prop_qid = _normalize_property_id(predicate)
            if prop_qid is not None:
                result["entity_property_qids"].append(prop_qid)
            for obj in objects:
                result["entity_predicates"].append(_convert_value(predicate))
                result["entity_objects"].append(_convert_value(obj))
    else:
        # unknown description type; ignore silently or log once upstream
        pass

    return result


def load_dataset(file_path: Path | str, max_size: int = -1) -> dict[str, list[Any]]:
    """Read a gzipped TSV file and build an in-memory dataset dict.

    Every line describes a constraint-violation *correction* plus rich
    context.  All string values are mapped to integer ids through the global
    encoder.  Processing stops after *max_size* lines (handy for debugging).

    Parameters
    ----------
    file_path
        Path to a .tsv.gz file produced by the Wikidata constraint corpus.
    max_size
        Maximum number of lines to process.

    Returns
    -------
    dict[str, list[Any]]
        Mapping from feature name to a list of encoded / raw values.
        lists are kept as Python lists to allow concatenation; conversion to
        numpy happens later in :pyfunc:`load`.
    """

    constraint_type = str(file_path).split("-")[-1].split(".")[0]
    dataset: dict[str, list[Any]] = {
        "constraint_type": [],
        "constraint_id": [],
        "constraint_predicates": [],
        "constraint_objects": [],
        "subject": [],
        "predicate": [],
        "object": [],
        "object_text": [],
        "other_subject": [],
        "other_predicate": [],
        "other_object": [],
        "other_object_text": [],
        "subject_predicates": [],
        "subject_objects": [],
        "object_predicates": [],
        "object_objects": [],
        "other_entity_predicates": [],
        "other_entity_objects": [],
        "add_subject": [],
        "add_predicate": [],
        "add_object": [],
        "del_subject": [],
        "del_predicate": [],
        "del_object": [],
        "local_constraint_ids": [],
    }
    with gzip.open(file_path, "rt", encoding="utf-8") as fp:
        for line_i, line in enumerate(fp):
            if line_i == max_size:
                break

            elements = line.split("\t")
            if elements[0] not in constraints_def:
                continue

            # Encode main triple & auxiliary data
            constraint = constraints_def[elements[0]]
            subject = _convert_value(elements[2])
            predicate = _convert_value(elements[3])
            obj = _convert_value(elements[4])
            other_subject = _convert_value(elements[5])
            other_predicate = _convert_value(elements[6])
            other_object = _convert_value(elements[7])
            constraint_id = _convert_value(elements[0])
            add_subject = None
            add_predicate = None
            add_object = None
            del_subject = None
            del_predicate = None
            del_object = None
            i = 12
            while i < len(elements):
                if elements[i] == "<http://wikiba.se/history/ontology#addition>":
                    add_subject = elements[i - 3]
                    add_predicate = elements[i - 2]
                    add_object = elements[i - 1]
                elif elements[i] == "<http://wikiba.se/history/ontology#deletion>":
                    del_subject = elements[i - 3]
                    del_predicate = elements[i - 2]
                    del_object = elements[i - 1]
                else:
                    print("Unexpected entity: {}".format(elements[i - 3 : i + 1]))
                    i += 4
                    continue
                i += 4

            # Parse JSON descriptions (page/entity)
            subject_desc = _read_entity_desc(elements, -3)
            object_desc = _read_entity_desc(elements, -2)
            other_entity_desc = _read_entity_desc(elements, -1)
            if any(label in object_desc["http_content"] for label in subject_desc["entity_labels"]):
                object_desc["entity_predicates"].append(
                    _convert_value("<http://wikiba.se/history/ontology#pageContainsLabel>")
                )
                object_desc["entity_objects"].append(subject)
            if any(label in object_desc["http_content"] for label in other_entity_desc["entity_labels"]):
                object_desc["entity_predicates"].append(
                    _convert_value("<http://wikiba.se/history/ontology#pageContainsLabel>")
                )
                object_desc["entity_objects"].append(other_subject)
            if any(label in other_entity_desc["http_content"] for label in subject_desc["entity_labels"]):
                other_entity_desc["entity_predicates"].append(
                    _convert_value("<http://wikiba.se/history/ontology#pageContainsLabel>")
                )
                other_entity_desc["entity_objects"].append(subject)
            if any(label in other_entity_desc["http_content"] for label in object_desc["entity_labels"]):
                other_entity_desc["entity_predicates"].append(
                    _convert_value("<http://wikiba.se/history/ontology#pageContainsLabel>")
                )
                other_entity_desc["entity_objects"].append(obj)

            # Final append
            dataset["constraint_type"].append(constraint_type)
            dataset["constraint_id"].append(constraint_id)
            dataset["constraint_predicates"].append([_convert_value(v) for v in constraint["predicates"]])
            dataset["constraint_objects"].append([_convert_value(v) for v in constraint["objects"]])
            dataset["subject"].append(subject)
            dataset["predicate"].append(predicate)
            if elements[4].startswith("<http://www.wikidata.org/entity/"):
                dataset["object"].append(obj)
                dataset["object_text"].append("")
            else:
                dataset["object"].append(0)
                dataset["object_text"].append(elements[4].split("^^")[0])
            dataset["other_subject"].append(other_subject)
            dataset["other_predicate"].append(other_predicate)
            if elements[7].startswith("<http://www.wikidata.org/entity/"):
                dataset["other_object"].append(other_object)
                dataset["other_object_text"].append("")
            else:
                dataset["other_object"].append(0)
                dataset["other_object_text"].append(elements[7].split("^^")[0])

            operations = {
                "add_subject": add_subject,
                "add_predicate": add_predicate,
                "add_object": add_object,
                "del_subject": del_subject,
                "del_predicate": del_predicate,
                "del_object": del_object,
            }

            for key, value in operations.items():
                dataset[key].append(
                    _convert_value(
                        value,
                        subject,
                        predicate,
                        obj,
                        other_subject,
                        other_predicate,
                        other_object,
                    )
                )

            dataset["subject_predicates"].append(subject_desc["entity_predicates"])
            dataset["subject_objects"].append(subject_desc["entity_objects"])
            dataset["object_predicates"].append(object_desc["entity_predicates"])
            dataset["object_objects"].append(object_desc["entity_objects"])
            dataset["other_entity_predicates"].append(other_entity_desc["entity_predicates"])
            dataset["other_entity_objects"].append(other_entity_desc["entity_objects"])

            # P_local = predicates in the local neighborhood (main/other + entity predicate lists).
            # C_local = union of constraints_by_property for P_local plus the central constraint_id.
            p_local: set[str] = set()
            main_prop = _normalize_property_id(elements[3])
            if main_prop:
                p_local.add(main_prop)
            other_prop = _normalize_property_id(elements[6])
            if other_prop:
                p_local.add(other_prop)
            p_local.update(subject_desc["entity_property_qids"])
            p_local.update(object_desc["entity_property_qids"])
            p_local.update(other_entity_desc["entity_property_qids"])

            local_constraint_ids: set[int] = {constraint_id}
            for prop in p_local:
                for cid in constraints_by_property.get(prop, []):
                    local_constraint_ids.add(cid)
            dataset["local_constraint_ids"].append(sorted(local_constraint_ids))
    return dataset


def _ensure_reserved_tokens() -> set[int]:
    global UNKNOWN_TOKEN_ID
    reserved_ids: set[int] = set()
    for token in RESERVED_TOKEN_STRINGS + FACTOR_TOKEN_STRINGS + CONSTRAINT_ID_TOKENS:
        reserved_ids.add(ENCODER.encode(token, add_new=True))
    UNKNOWN_TOKEN_ID = ENCODER.encode(UNKNOWN_TOKEN_STRING, add_new=True)
    return reserved_ids


def _reindex_encoder(allowed_ids: set[int]) -> dict[int, int]:
    """Rebuild encoder mappings so surviving token ids are contiguous."""
    old_to_new = {0: 0}
    new_encoding: dict[str, int] = {"": 0}
    new_decoding: dict[int, str] = {0: ""}

    next_id = 1
    for token, old_id in sorted(ENCODER._encoding.items(), key=lambda item: item[1]):
        if old_id == 0 or old_id not in allowed_ids:
            continue
        new_encoding[token] = next_id
        new_decoding[next_id] = token
        old_to_new[old_id] = next_id
        next_id += 1

    ENCODER._encoding = new_encoding
    ENCODER._decoding = new_decoding
    ENCODER._frozen = False

    global UNKNOWN_TOKEN_ID
    UNKNOWN_TOKEN_ID = ENCODER._encoding.get(UNKNOWN_TOKEN_STRING, 0)
    return old_to_new


def _remap_scalar_array(values: Any, id_map: dict[int, int]) -> np.ndarray:
    """Apply *id_map* to a scalar feature array."""
    if isinstance(values, np.ndarray) and values.dtype != object:
        vectorised = np.vectorize(lambda x: id_map.get(int(x), UNKNOWN_TOKEN_ID), otypes=[np.int64])
        return vectorised(values)
    return np.asarray([id_map.get(int(v), UNKNOWN_TOKEN_ID) for v in values], dtype=np.int64)


def _remap_sequence_array(values: Any, id_map: dict[int, int]) -> np.ndarray:
    """Apply *id_map* to a sequence feature array (list of lists)."""
    remapped: list[list[int]] = []
    for seq in values:
        if isinstance(seq, np.ndarray):
            seq_iter = seq.tolist()
        else:
            seq_iter = list(seq)
        remapped.append([id_map.get(int(v), UNKNOWN_TOKEN_ID) for v in seq_iter])
    return np.array(remapped, dtype=object)


def _remap_dataset_inplace(dataset: dict[str, Any], id_map: dict[int, int]) -> None:
    for key in SCALAR_FEATURES:
        if key in dataset:
            dataset[key] = _remap_scalar_array(dataset[key], id_map)

    for key in SEQUENCE_FEATURE_KEYS:
        if key in dataset:
            dataset[key] = _remap_sequence_array(dataset[key], id_map)


def _apply_frequency_filter_inplace(dataset: dict[str, Any], allowed_ids: set[int]) -> None:
    allowed_array = np.fromiter(allowed_ids, dtype=np.int64)
    unknown_id = UNKNOWN_TOKEN_ID

    for key in SCALAR_FEATURES:
        if key not in dataset:
            continue
        values = dataset[key]
        if isinstance(values, np.ndarray):
            dataset[key] = np.where(np.isin(values, allowed_array), values, unknown_id)
        else:
            dataset[key] = np.asarray([v if v in allowed_ids else unknown_id for v in values])

    for key in SEQUENCE_FEATURE_KEYS:
        if key not in dataset:
            continue
        sequences = dataset[key]
        filtered = []
        for seq in sequences:
            if isinstance(seq, np.ndarray):
                seq_list = seq.tolist()
            else:
                seq_list = list(seq)
            filtered.append([v if v == 0 or v in allowed_ids else unknown_id for v in seq_list])
        dataset[key] = np.array(filtered, dtype=object)


def _prune_encoder(allowed_ids: set[int]) -> None:
    removable: list[str] = []
    for token_str, token_id in ENCODER._encoding.items():
        if token_id == 0:
            continue
        if token_id not in allowed_ids:
            removable.append(token_str)

    for token_str in removable:
        token_id = ENCODER._encoding.pop(token_str)
        ENCODER._decoding.pop(token_id, None)


def _concatenate_datasets(
    datasets: Sequence[dict[str, Any]],
    *,
    reuse_first: bool = True,
) -> dict[str, Any]:
    """
    Concatenate multiple dataset dictionaries row-wise with a single
    allocation per key. Replaces arrays in the first dict (if reuse_first=True)
    to minimize extra memory.
    """
    if not datasets:
        return {}

    # Keys from the first dataset define the schema
    first = dict(datasets[0])  # shallow copy of mapping -> dict
    keys = list(first.keys())

    # Compute total lengths and validate shapes/dtypes
    total_len: dict[str, int] = {k: 0 for k in keys}
    dtypes: dict[str, np.dtype] = {}
    tails: dict[str, tuple[int, ...]] = {}

    for k in keys:
        for ds in datasets:
            arr = ds[k]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Value for key '{k}' must be np.ndarray, got {type(arr)}")
            if k not in dtypes:
                dtypes[k] = arr.dtype
                tails[k] = arr.shape[1:]
            else:
                if arr.dtype != dtypes[k]:
                    raise TypeError(f"Key '{k}' has mismatched dtypes: {dtypes[k]} vs {arr.dtype}")
                if arr.shape[1:] != tails[k]:
                    raise ValueError(f"Key '{k}' has incompatible shapes after axis 0: {tails[k]} vs {arr.shape[1:]}")
            total_len[k] += arr.shape[0]

    # Prepare output dict (optionally reusing the first dict object)
    out = first if reuse_first else {}

    # Allocate once per key, then copy slices
    for k in keys:
        dtype = dtypes[k]
        tail = tails[k]
        n_total = total_len[k]

        # Allocate destination; empty() is slightly faster than zeros()
        dst = np.empty((n_total, *tail), dtype=dtype)

        # Fill by slices
        off = 0
        for ds in datasets:
            src = ds[k]
            n = src.shape[0]
            dst[off : off + n] = src
            off += n

        # Replace in the (first) dict to keep the interface identical
        out[k] = dst

    gc.collect()
    return out


def _slice_dataset(dataset: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    """Return a dataset subset indexed by *indices* preserving array types."""
    subset: dict[str, Any] = {}
    index_array = np.asarray(indices, dtype=np.int64)
    for key, values in dataset.items():
        subset[key] = values[index_array]
    return subset


def _compute_token_frequency(dataset: dict[str, Any]) -> Counter[int]:
    """Count token appearances across scalar and sequence features."""
    frequency: Counter[int] = Counter()

    for key in SCALAR_FEATURES:
        if key not in dataset:
            continue
        values = dataset[key]
        if isinstance(values, np.ndarray) and values.dtype != object:
            flat = values.ravel()
        else:
            flat = np.asarray(values, dtype=object).ravel()
        for token_id in flat:
            int_id = int(token_id)
            if int_id != 0:
                frequency[int_id] += 1

    for key in SEQUENCE_FEATURE_KEYS:
        if key not in dataset:
            continue
        sequences = dataset[key]
        for seq in sequences:
            seq_iter = seq.tolist() if isinstance(seq, np.ndarray) else list(seq)
            for token_id in seq_iter:
                int_id = int(token_id)
                if int_id != 0:
                    frequency[int_id] += 1

    return frequency


def stratified_train_val_test_split(df_full, stratify_feature, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split DataFrame into train/validation/test sets with stratification

    Parameters:
    df: Dictionary representing the full dataset
    stratify_feature: column name to stratify on
    train_size: proportion for training set (default 0.7)
    val_size: proportion for validation set (default 0.15)
    test_size: proportion for test set (default 0.15)
    random_state: random seed for reproducibility

    Returns:
    df_train, df_val, df_test: three DataFrames
    """
    df = pd.DataFrame(df_full)

    # First split: separate train from (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=(val_size + test_size),  # 0.3 total for val+test
        stratify=df[stratify_feature],
        random_state=random_state,
        shuffle=True,
    )

    # Second split: separate val from test
    val_ratio = val_size / (val_size + test_size)

    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_ratio),
        stratify=df_temp[stratify_feature],
        random_state=random_state,
        shuffle=True,
    )

    # Slice the original dataset to get the same structure
    train_indices = df_train.index.to_numpy()
    val_indices = df_val.index.to_numpy()
    test_indices = df_test.index.to_numpy()

    train_dataset = _slice_dataset(df_full, train_indices)
    val_dataset = _slice_dataset(df_full, val_indices)
    test_dataset = _slice_dataset(df_full, test_indices)

    return train_dataset, val_dataset, test_dataset


def load(kind: str, targets: Union[list[Any], np.ndarray]) -> dict[str, Any]:
    """Load *kind* split (train / dev / test) for multiple constraint *targets*.

    The function concatenates the raw dictionaries produced by
    :pyfunc:`load_dataset`, converts them to numpy arrays and returns the
    combined mapping.

    Parameters
    ----------
    kind
        Dataset split to load - one of "train", "dev", or "test".
    targets
        list of constraint type names, e.g. ["single", "conflictWith"].

    Returns
    -------
    dict[str, Any]
        dictionary whose values are numpy arrays; nested-list features are
        stored with dtype=object to preserve list structure.
    """
    result: dict[str, Union[list[Any], np.ndarray]] = defaultdict(list)
    for target in targets:
        print("Loading {} {}".format(target, kind))
        for k, v in load_dataset(
            RAW_DATA_PATH / ("constraint-corrections-" + target + ".tsv.gz.full." + kind + ".tsv.gz")
        ).items():
            result[k] += v
    print()
    gc.collect()

    for k, v in result.items():
        if k in SEQUENCE_FEATURE_KEYS:
            result[k] = np.array(v, dtype=object)
        elif k.endswith("_text") or k == "constraint_type":
            result[k] = np.array(v, dtype=object)
        else:
            result[k] = np.asarray(v)
    gc.collect()
    return result


def main():
    global ALLOW_NEW_TOKENS, RECORD_FREQUENCIES, UNKNOWN_TOKEN_ID, ENCODER, REGISTRY_RESERVED_IDS

    # Types of contraints
    targets = [
        "conflictWith",
        "distinct",
        "inverse",
        "itemRequiresStatement",
        "oneOf",
        "single",
        "type",
        "valueRequiresStatement",
        "valueType",
    ]

    # Build dictionaries
    TOKEN_FREQUENCY.clear()
    ALLOW_NEW_TOKENS = True
    RECORD_FREQUENCIES = False

    # Load raw datasets
    raw_train_dataset = load("train", targets)
    raw_dev_dataset = load("dev", targets)
    raw_test_dataset = load("test", targets)
    print("\nRaw dataset loading complete.\n")
    gc.collect()

    # Concatenate and stratified split
    full_dataset = _concatenate_datasets([raw_train_dataset, raw_dev_dataset, raw_test_dataset], reuse_first=True)
    print("Full dataset size: {}\n".format(len(full_dataset["constraint_type"])))
    del raw_train_dataset, raw_dev_dataset, raw_test_dataset
    gc.collect()

    train_dataset, val_dataset, test_dataset = stratified_train_val_test_split(
        full_dataset, "constraint_type", random_state=42
    )
    print(
        "Train size: {}, Val size: {}, Test size: {}\n".format(
            len(train_dataset["constraint_type"]),
            len(val_dataset["constraint_type"]),
            len(test_dataset["constraint_type"]),
        )
    )
    del full_dataset
    gc.collect()

    # Compute token frequencies on training set
    print("Computing token frequencies on training set...")
    train_frequency = _compute_token_frequency(train_dataset)
    TOKEN_FREQUENCY.clear()
    TOKEN_FREQUENCY.update(train_frequency)

    allowed_ids: set[int] = {0}
    allowed_ids.update(token for token, count in train_frequency.items() if count >= MIN_OCCURRENCE)
    base_reserved_ids = _ensure_reserved_tokens()
    allowed_ids.update(base_reserved_ids)
    allowed_ids.update(REGISTRY_RESERVED_IDS)

    print("Applying frequency filter to all datasets...")
    for dataset in (train_dataset, val_dataset, test_dataset):
        _apply_frequency_filter_inplace(dataset, allowed_ids)

    _prune_encoder(allowed_ids)
    id_map = _reindex_encoder(allowed_ids)

    for dataset in (train_dataset, val_dataset, test_dataset):
        _remap_dataset_inplace(dataset, id_map)

    allowed_ids = set(id_map.values())
    print("Frequency filter retained {} token IDs (including unknown)".format(len(allowed_ids)))
    print("Encoder size after pruning: {}".format(len(ENCODER._encoding)))
    print(
        "Reserved base tokens: {}, reserved registry tokens: {}, final vocab size: {}".format(
            len(base_reserved_ids),
            len(REGISTRY_RESERVED_IDS),
            len(ENCODER._encoding),
        )
    )

    # Finalize settings
    ENCODER.freeze()
    ALLOW_NEW_TOKENS = False
    RECORD_FREQUENCIES = False

    df_train = pd.DataFrame(train_dataset)
    df_val = pd.DataFrame(val_dataset)
    df_test = pd.DataFrame(test_dataset)
    del train_dataset, val_dataset, test_dataset
    gc.collect()

    print(f"Number of rows in df_train: {df_train.shape}")
    print(f"Number of rows in df_val: {df_val.shape}")
    print(f"Number of rows in df_test: {df_test.shape}")

    # Save to disk
    INTERIM_DATA_PATH.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(INTERIM_DATA_PATH / "df_train.parquet")
    df_val.to_parquet(INTERIM_DATA_PATH / "df_val.parquet")
    df_test.to_parquet(INTERIM_DATA_PATH / "df_test.parquet")

    ENCODER.save(INTERIM_DATA_PATH / "globalintencoder.txt")
    print("Number of constraints: {}".format(len(constraints_def)))
    print("Number of seeded factor tokens: {}".format(len(FACTOR_TOKEN_STRINGS)))
    print("Number of tokens in final encoder: {}".format(len(ENCODER._encoding)))

    if not FACTOR_TOKEN_STRINGS:
        raise AssertionError("No constraint factor tokens were seeded.")
    sample_constraint_ids = list(constraints_def.keys())
    if len(sample_constraint_ids) > 50:
        rng = np.random.default_rng(42)
        sample_constraint_ids = rng.choice(sample_constraint_ids, size=50, replace=False).tolist()
    for constraint_id in sample_constraint_ids:
        token = f"{FACTOR_TOKEN_PREFIX}{constraint_id}"
        token_id = ENCODER.encode(token, add_new=False)
        assert token_id != 0, f"Missing constraint factor token in encoder: {token}"
    print("Validated {} constraint factor tokens in encoder.".format(len(sample_constraint_ids)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforms raw data into pandas dataframes")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Which dataset to fetch.",
    )
    parser.add_argument(
        "--min-occurrence",
        type=int,
        default=100,
        help="Minimum number of occurrences in the training split required for a token to keep its ID.",
    )
    args = parser.parse_args()

    DATASET = args.dataset
    MIN_OCCURRENCE = max(1, args.min_occurrence)
    dataset_variant = dataset_variant_name(DATASET, MIN_OCCURRENCE)
    print(f"Processing {DATASET} data (min_occurrence={MIN_OCCURRENCE})...\n")

    # Data paths
    RAW_DATA_PATH = Path("data/raw/") / DATASET
    INTERIM_DATA_PATH = Path("data/interim/") / dataset_variant
    INTERIM_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Build DataFrames
    ENCODER = GlobalIntEncoder()
    constraints_def, constraints_by_property_raw = load_constraint_data()
    _seed_constraint_factor_tokens(constraints_def)
    REGISTRY_RESERVED_IDS = _collect_registry_reserved_ids(constraints_def, constraints_by_property_raw)
    constraints_by_property = _encode_constraints_by_property(constraints_by_property_raw)
    print("Number of constrained properties:", len(constraints_by_property))
    print("Total constraint instances:",
          sum(len(v) for v in constraints_by_property.values()))

    main()
