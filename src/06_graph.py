#!/usr/bin/env python3
"""
06_graph.py
====================
Transforms the interim dataframes into a NetworkX-backed, PyTorch Geometric
`Data` object and stores the resulting list of graphs as pickle files.

Datasets
--------
* **sample** - the sample corpus shipped in the GitHub repository
  `Tpt/bass-materials` ≈ 1 .80 GB
* **full**   - the full dump hosted on Figshare
  (article ID 13338743) ≈ 10 .50 GB

Encoding Options
----------------
* **node_id**     - represent each URI by its global integer node_id
* **text_embedding** - represent each URI by a sentence-transformer text_embedding

Usage
-----
python src/06_graph.py --dataset {sample,full} --encoding {node_id,text_embedding} \
    [--min-occurrence N] [--shard-size N] [--use-torch-save] [--embedding-dtype {float32,float16}]
"""

import os

# Tell HF Transformers to skip TensorFlow entirely
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import gc
import itertools
import json
import logging
import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import datasets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from modules.data_encoders import (
    ROLE_NONE,
    ROLE_OBJECT,
    ROLE_PREDICATE,
    ROLE_SUBJECT,
    GlobalIntEncoder,
    GlobalToLocalNodeMap,
    PrecomputedWikidataCache,
    dataset_variant_name,
    dump_in_shards,
    dump_stream,
    iter_stream,
)


LITERAL_ID = 0


def _normalize_target_id(value: Any, encoder: GlobalIntEncoder, unknown_id: int) -> int:
    """Return a sanitized class id compatible with the filtered encoder."""
    if value is None:
        return 0
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return 0
    if idx <= 0:
        return 0
    if idx in getattr(encoder, "_filtered_ids", set()):
        return unknown_id
    return idx


def _is_id_iterable(value: Any) -> bool:
    """True when value should be treated as a list of ids."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _normalize_id_sequence(value: Any) -> list[Any]:
    """Normalize singletons and iterables into a concrete list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _is_id_iterable(value):
        return list(value)
    return [value]


def _is_literal_node(graph: dict[str, Any], key: str) -> bool:
    text_value = graph.get(f"{key}_text")
    if isinstance(text_value, str) and text_value != "":
        return True
    if LITERAL_ID and graph.get(key) == LITERAL_ID:
        return True
    return False


def create_graph(
    graph: dict[str, Any],
    wikidata_cache: PrecomputedWikidataCache | None,
    global_int_encoder: GlobalIntEncoder,
    constraint_registry: Dict[str, Dict[str, Any]],
    encoding: Literal["node_id", "text_embedding"] = "text_embedding",
    constraint_scope: Literal["local", "focus"] = "local",
    store_node_names: bool = False,
    embedding_dtype: np.dtype | None = None,
    debug_factor_wiring: bool = False,
) -> Data:
    """
    Convert a single Wikidata conflict record into a homogeneous multigraph
    represented as a `torch_geometric.data.Data` object.

    Each triple in the record is rewritten as a length-2 path
    *(subject → predicate → object)*.  All triples that appear in the local
    neighbourhood of the conflict (main statement, neighbour statements,
    constraint statements, …) are merged into one multirelational graph.
    Optionally, the graph can be visualised with NetworkX/Matplotlib.

    Parameters
    ----------
    graph:
        A dictionary obtained from a row in the intermediate parquet file
        whose keys correspond to the columns described in the BASS paper
        (e.g. "subject", "predicate", "object", …).
    wikidata_cache:
        Cache that provides pre-computed labels and embeddings for all
        identifiers encountered in the dataset.
    global_int_encoder:
        An instance of `GlobalIntEncoder` used to encode URIs to global integer IDs.
    encode_predicates_as_nodes:
        If *True* (recommended) the predicate URI is represented as a proper
        node, leading to a uniform first-order graph. TODO: Setting this to
        *False* is currently not supported and will raise an AssertionError.
    used_attribute:
        Controls the node feature:
        - "node_id"    → one-hot integer label
        - "text_embedding" → 768-d sentence-transformer embedding
    embedding_dtype:
        Optional numpy dtype used when `encoding` is ``"text_embedding"`` to
        control the precision (e.g., ``np.float16`` or ``np.float32``).

    Returns
    -------
    torch_geometric.data.Data
        A directed graph with
        1. x -- holding node features,
        2. edge_index -- holding source/target indices,
        3. y -- holding the 6-way multi-label classification target.
    """

    global_to_local_id_encoder = GlobalToLocalNodeMap()
    unknown_global_id = global_int_encoder.encode("unknown", add_new=False)
    if unknown_global_id == 0:
        unknown_global_id = global_int_encoder.encode("unknown", add_new=True)
    EDGE_SUBJECT_TO_PREDICATE = 0
    EDGE_PREDICATE_TO_OBJECT = 1
    EDGE_FACTOR_TO_PARAM_PREDICATE = 2
    EDGE_PARAM_PREDICATE_TO_OBJECT = 3
    EDGE_FACTOR_TO_LOCAL_PREDICATE = 4
    EDGE_FACTOR_TO_LOCAL_SUBJECT = 5
    EDGE_FACTOR_TO_LOCAL_OBJECT = 6
    EDGE_LOCAL_PREDICATE_TO_FACTOR = 7
    EDGE_LOCAL_SUBJECT_TO_FACTOR = 8
    EDGE_LOCAL_OBJECT_TO_FACTOR = 9

    edges: List[tuple[int, int]] = []
    edge_types: List[int] = []
    edges_non_flattened: List[tuple[int, int]] = []
    non_flattened_edge_attributes: List[Any] = []
    resolved_embedding_dtype: np.dtype | None = None
    if encoding == "text_embedding":
        resolved_embedding_dtype = np.dtype(embedding_dtype or np.float16)
    focus_local_nodes: dict[str, int] = {}
    factor_constraint_ids: list[int] = []
    factor_constraint_types: list[str] = []
    factor_local_ids: list[int] = []
    primary_factor_index: int = -1
    pred_global_to_local_triples: dict[int, list[tuple[int, int, int]]] = {}
    pred_global_to_pred_local_ids: dict[int, list[int]] = {}
    subj_local_to_pred_global_to_pred_local_ids: dict[int, dict[int, list[int]]] = {}
    debug_entries: list[dict[str, Any]] = []
    primary_factor_focus_scope_ok: bool | None = None
    factor_checkable_pre_tensor: torch.Tensor | None = None
    factor_satisfied_pre_tensor: torch.Tensor | None = None
    factor_checkable_post_gold_tensor: torch.Tensor | None = None
    factor_satisfied_post_gold_tensor: torch.Tensor | None = None
    factor_types_tensor: torch.Tensor | None = None

    def _resolve_registry_id(raw_id: str) -> int | None:
        raw = raw_id.strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1].strip()
        if raw.startswith("http://www.wikidata.org/prop/direct/"):
            raw = raw.replace(
                "http://www.wikidata.org/prop/direct/",
                "http://www.wikidata.org/entity/",
            )
        candidates: list[str] = []
        seen: set[str] = set()
        if raw.startswith("http://") or raw.startswith("https://"):
            candidates.extend([raw, f"<{raw}>"])
            last = raw.rsplit("/", 1)[-1]
            if last and last[0] in ("P", "Q") and last[1:].isdigit():
                candidates.append(last)
        else:
            if raw and raw[0] in ("P", "Q") and raw[1:].isdigit():
                entity_uri = f"http://www.wikidata.org/entity/{raw}"
                candidates.extend([entity_uri, f"<{entity_uri}>"])
            candidates.append(raw)
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            gid = global_int_encoder.encode(candidate, add_new=False)
            if gid:
                return gid
        return None

    def _maybe_encode_registry_token(raw_token: str | None) -> int:
        if not raw_token:
            return 0
        try:
            return global_int_encoder.encode(raw_token, add_new=False)
        except Exception:
            return 0

    def _is_property_token(raw_value: str | None) -> bool:
        if not raw_value:
            return False
        raw = raw_value.strip().strip("<>").strip()
        if raw.startswith("http://www.wikidata.org/entity/"):
            raw = raw.rsplit("/", 1)[-1]
        if raw.startswith("http://www.wikidata.org/prop/direct/"):
            raw = raw.rsplit("/", 1)[-1]
        return raw.startswith("P") and raw[1:].isdigit()

    if encoding == "text_embedding" and wikidata_cache is None:
        raise ValueError("wikidata_cache is required when encoding='text_embedding'")
    if store_node_names and wikidata_cache is None:
        raise ValueError("wikidata_cache is required to store node names")

    def get_node_attribute(global_node_id: int, node_text: str | None) -> Any:
        """Get the node attribute for a global node ID."""
        if encoding == "text_embedding":
            assert wikidata_cache is not None
            if node_text is not None and node_text != "":
                return wikidata_cache.get_embedding_for_literal(
                    node_text,
                    dtype=resolved_embedding_dtype,
                    fallback_id=unknown_global_id,
                )
            return wikidata_cache.get_embedding_for_id(
                global_node_id,
                dtype=resolved_embedding_dtype,
                fallback_id=unknown_global_id,
            )
        else:
            if global_node_id in global_int_encoder._filtered_ids:
                global_node_id = unknown_global_id
            return global_int_encoder.get_unfiltered_global_id(global_node_id)

    def add_edge_from_ids(
        subject_global_id: int,
        predicate_global_id: int,
        object_global_id: int,
        subject_text: str | None = None,
        predicate_text: str | None = None,
        object_text: str | None = None,
        assign_focus: bool = False,
    ):
        if subject_global_id == 0:
            if not subject_text:
                subject_global_id = unknown_global_id
            else:
                # Fall back to the literal placeholder when a text literal lacks a global id.
                subject_global_id = LITERAL_ID
        if predicate_global_id == 0:
            if not predicate_text:
                predicate_global_id = unknown_global_id
            else:
                predicate_global_id = LITERAL_ID
        if object_global_id == 0:
            if not object_text:
                object_global_id = unknown_global_id
            else:
                object_global_id = LITERAL_ID

        subject_name = None
        predicate_name = None
        object_name = None
        if store_node_names:
            assert wikidata_cache is not None
            subject_name = subject_text or wikidata_cache.get_text_for_id(
                subject_global_id,
                fallback_id=unknown_global_id,
            )
            predicate_name = predicate_text or wikidata_cache.get_text_for_id(
                predicate_global_id,
                fallback_id=unknown_global_id,
            )
            if object_text:
                object_name = object_text
            else:
                object_name = wikidata_cache.get_text_for_id(
                    object_global_id,
                    fallback_id=unknown_global_id,
                )

        subject_id = global_to_local_id_encoder.store(
            subject_global_id,
            get_node_attribute(subject_global_id, subject_text),
            name=subject_name,
        )
        predicate_id = global_to_local_id_encoder.store(
            predicate_global_id,
            get_node_attribute(predicate_global_id, predicate_text),
            # because we encode predicates as nodes, we need to duplicate them to not change the graph structure
            force_create=True,
            name=predicate_name,
        )

        object_id = global_to_local_id_encoder.store(
            object_global_id,
            get_node_attribute(object_global_id, object_text),
            name=object_name,
        )

        if assign_focus:
            focus_local_nodes["subject"] = subject_id
            focus_local_nodes["predicate"] = predicate_id
            focus_local_nodes["object"] = object_id

        edges.append((subject_id, predicate_id))
        edge_types.append(EDGE_SUBJECT_TO_PREDICATE)
        edges.append((predicate_id, object_id))
        edge_types.append(EDGE_PREDICATE_TO_OBJECT)

        pred_global_to_pred_local_ids.setdefault(predicate_global_id, []).append(predicate_id)
        pred_global_to_local_triples.setdefault(predicate_global_id, []).append((subject_id, predicate_id, object_id))
        subj_local_to_pred_global_to_pred_local_ids.setdefault(subject_id, {}).setdefault(
            predicate_global_id, []
        ).append(predicate_id)

        edges_non_flattened.append((subject_id, object_id))
        non_flattened_edge_attributes.append(predicate_id)

    def add_edge(
        graph: dict[str, Any],
        subject_key: str,
        predicate_key: str,
        object_key: str,
        assign_focus: bool = False,
    ) -> None:
        subject_global_id = graph[subject_key]
        predicate_global_id = graph[predicate_key]
        object_global_id = graph[object_key]

        # handle iterable ids
        if (
            _is_id_iterable(subject_global_id)
            or _is_id_iterable(object_global_id)
            or _is_id_iterable(predicate_global_id)
        ):
            subject_ids = _normalize_id_sequence(subject_global_id)
            object_ids = _normalize_id_sequence(object_global_id)
            predicate_ids = _normalize_id_sequence(predicate_global_id)
            if len(subject_ids) == 0 or len(object_ids) == 0 or len(predicate_ids) == 0:
                # skip creation, triple doesn't exist
                return
            length = max(len(subject_ids), len(object_ids), len(predicate_ids))
            if length > 1:
                # broadcast singletons
                if len(subject_ids) == 1:
                    subject_ids = subject_ids * length
                if len(object_ids) == 1:
                    object_ids = object_ids * length
                if len(predicate_ids) == 1:
                    predicate_ids = predicate_ids * length
            assert len(subject_ids) == len(object_ids) == len(predicate_ids), (
                f"Length mismatch when adding edge: {len(subject_ids)} subjects, {len(predicate_ids)} predicates, {len(object_ids)} objects."
            )

            for subject_global, predicate_global, object_global in zip(subject_ids, predicate_ids, object_ids):
                add_edge_from_ids(
                    subject_global,
                    predicate_global,
                    object_global,
                    subject_text=graph.get(f"{subject_key}_text"),
                    predicate_text=graph.get(f"{predicate_key}_text"),
                    object_text=graph.get(f"{object_key}_text"),
                    assign_focus=assign_focus,
                )
        else:
            if object_global_id == 0 and graph.get(f"{object_key}_text") in ["", None]:
                assert predicate_global_id == 0, f"Predicate {predicate_global_id} without object {object_global_id}"
                # skip creation, triple doesn't exist
                return
            add_edge_from_ids(
                subject_global_id,
                predicate_global_id,
                object_global_id,
                subject_text=graph.get(f"{subject_key}_text"),
                predicate_text=graph.get(f"{predicate_key}_text"),
                object_text=graph.get(f"{object_key}_text"),
                assign_focus=assign_focus,
            )

    def add_factor_definition_edge(
        factor_local_id: int,
        predicate_global_id: int,
        object_global_id: int,
    ) -> None:
        if predicate_global_id == 0:
            predicate_global_id = unknown_global_id
        if object_global_id == 0:
            object_global_id = unknown_global_id

        predicate_name = None
        object_name = None
        if store_node_names:
            assert wikidata_cache is not None
            cache = wikidata_cache
            predicate_name = cache.get_text_for_id(
                predicate_global_id,
                fallback_id=unknown_global_id,
            )
            object_name = cache.get_text_for_id(
                object_global_id,
                fallback_id=unknown_global_id,
            )

        predicate_id = global_to_local_id_encoder.store(
            predicate_global_id,
            get_node_attribute(predicate_global_id, None),
            force_create=True,
            name=predicate_name,
        )
        object_id = global_to_local_id_encoder.store(
            object_global_id,
            get_node_attribute(object_global_id, None),
            name=object_name,
        )

        edges.append((factor_local_id, predicate_id))
        edge_types.append(EDGE_FACTOR_TO_PARAM_PREDICATE)
        edges.append((predicate_id, object_id))
        edge_types.append(EDGE_PARAM_PREDICATE_TO_OBJECT)

        edges_non_flattened.append((factor_local_id, object_id))
        non_flattened_edge_attributes.append(predicate_id)

    # --- Node construction -------------------------------------------------
    add_edge(graph, "subject", "predicate", "object", assign_focus=True)
    add_edge(graph, "other_subject", "other_predicate", "other_object")

    # add subject neighbours
    add_edge(graph, "subject", "subject_predicates", "subject_objects")
    add_edge(graph, "object", "object_predicates", "object_objects")
    # FIXME: not sure which direction the edges should go
    # add_edge(graph, "subject_objects", "subject_predicates", "subject")
    # add_edge(graph, "object_objects", "object_predicates", "object")

    # pick the "other_entity"
    if graph["other_subject"] == graph["subject"]:
        other_entity_name = "other_object"
    elif graph["other_object"] == graph["object"]:
        other_entity_name = "other_subject"
    else:
        assert graph["other_subject"] == graph["other_predicate"] == graph["other_object"] == 0
        other_entity_name = None

    if other_entity_name is not None:
        add_edge(graph, other_entity_name, "other_entity_predicates", "other_entity_objects")

    param_property_gid = _maybe_encode_registry_token("<http://www.wikidata.org/entity/P2306>")
    param_relation_gid = _maybe_encode_registry_token("<http://www.wikidata.org/entity/P2309>")
    param_inverse_gid = _maybe_encode_registry_token("<http://www.wikidata.org/entity/P1696>")
    default_relation_gids = [
        gid
        for gid in (
            _resolve_registry_id("P31"),
            _resolve_registry_id("P279"),
        )
        if gid is not None
    ]

    # add constraint factor branches
    if constraint_scope == "focus":
        constraint_ids_raw = graph.get("local_constraint_ids_focus")
        if constraint_ids_raw is None:
            constraint_ids_raw = graph.get("local_constraint_ids")
    else:
        constraint_ids_raw = graph.get("local_constraint_ids")
    if isinstance(constraint_ids_raw, Iterable) and not isinstance(constraint_ids_raw, (str, bytes)):
        factor_ids = [int(cid) for cid in constraint_ids_raw if cid is not None]
    else:
        factor_ids = []
    if "factor_constraint_ids" in graph:
        expected_ids_raw = graph.get("factor_constraint_ids") or []
        if isinstance(expected_ids_raw, Iterable) and not isinstance(expected_ids_raw, (str, bytes)):
            expected_ids = [int(cid) for cid in expected_ids_raw if cid is not None]
        else:
            expected_ids = []
        if expected_ids and expected_ids != factor_ids:
            raise AssertionError(
                "Factor constraint id order mismatch between labeled data and graph builder."
            )
    if not factor_ids:
        factor_ids = [int(graph["constraint_id"])]

    required_factor_fields = (
        "factor_checkable_pre",
        "factor_satisfied_pre",
        "factor_checkable_post_gold",
        "factor_satisfied_post_gold",
        "factor_types",
        "factor_constraint_ids",
    )
    if all(graph.get(field) is not None for field in required_factor_fields):
        expected_len = len(factor_ids)

        def _normalize_factor_list(value: Any, name: str) -> list[Any]:
            if isinstance(value, list):
                items = value
            elif isinstance(value, tuple):
                items = list(value)
            elif _is_id_iterable(value):
                items = list(value)
            else:
                items = [value]
            if len(items) != expected_len:
                raise AssertionError(
                    f"Factor label length mismatch for {name}: expected {expected_len}, got {len(items)}."
                )
            return items

        factor_ids_from_row = _normalize_factor_list(graph.get("factor_constraint_ids"), "factor_constraint_ids")
        if factor_ids_from_row and [int(cid) for cid in factor_ids_from_row] != factor_ids:
            raise AssertionError(
                "Factor constraint id order mismatch between labeled data and graph builder."
            )

        factor_checkable_pre_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_checkable_pre"), "factor_checkable_pre"),
            dtype=torch.bool,
        )
        factor_satisfied_pre_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_satisfied_pre"), "factor_satisfied_pre"),
            dtype=torch.long,
        )
        factor_checkable_post_gold_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_checkable_post_gold"), "factor_checkable_post_gold"),
            dtype=torch.bool,
        )
        factor_satisfied_post_gold_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_satisfied_post_gold"), "factor_satisfied_post_gold"),
            dtype=torch.long,
        )
        factor_types_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_types"), "factor_types"),
            dtype=torch.long,
        )

    for idx, constraint_id in enumerate(factor_ids):
        constraint_token = global_int_encoder._decoding.get(constraint_id)
        if constraint_token in (None, "", "unknown") or constraint_id == unknown_global_id:
            raise AssertionError(
                "Constraint id token is unknown; rebuild interim data with constraint ids preserved."
            )
        registry_entry = constraint_registry.get(constraint_token)
        if registry_entry is None:
            registry_entry = constraint_registry.get(constraint_token.strip("<>"))
        assert registry_entry is not None, f"Missing constraint_id={constraint_token} in constraint registry."

        try:
            factor_gid = global_int_encoder.encode(f"constraint_factor::{constraint_token}", add_new=False)
        except Exception as exc:
            raise AssertionError(f"Missing factor token for constraint_id={constraint_token} in encoder.") from exc
        assert factor_gid != 0, f"Invalid factor global id for constraint_id={constraint_id}."

        factor_local_id = global_to_local_id_encoder.store(
            factor_gid,
            get_node_attribute(factor_gid, None),
            name=f"constraint_factor::{constraint_token}" if store_node_names else None,
            force_create=True,
        )

        factor_constraint_ids.append(constraint_id)
        factor_local_ids.append(factor_local_id)
        constraint_type = str(registry_entry.get("constraint_type", ""))
        factor_constraint_types.append(constraint_type)
        if constraint_id == int(graph["constraint_id"]):
            primary_factor_index = idx

        constrained_property = registry_entry.get("constrained_property")

        param_predicates = registry_entry.get("param_predicates") or []
        param_objects = registry_entry.get("param_objects") or []
        assert len(param_predicates) == len(param_objects), (
            f"Constraint registry param list length mismatch for constraint_id={constraint_id}."
        )

        observed_predicates: list[int] = []
        observed_predicates_set: set[int] = set()

        def _add_observed(predicate_gid: int | None) -> None:
            if predicate_gid is None:
                return
            if predicate_gid in observed_predicates_set:
                return
            observed_predicates_set.add(predicate_gid)
            observed_predicates.append(predicate_gid)

        def _collect_param_object_gids(param_predicate_gid: int) -> list[int]:
            if not param_predicate_gid:
                return []
            matches: list[int] = []
            for pred_raw, obj_raw in zip(param_predicates, param_objects):
                pred_gid = _maybe_encode_registry_token(pred_raw)
                if pred_gid != param_predicate_gid:
                    continue
                obj_gid = _resolve_registry_id(obj_raw)
                if obj_gid is not None:
                    matches.append(obj_gid)
            return matches

        constrained_gid = None
        if constrained_property:
            constrained_gid = _resolve_registry_id(constrained_property)
            if constrained_gid is not None:
                _add_observed(constrained_gid)

        if constraint_type == "conflictWith":
            for obj_raw in param_objects:
                if _is_property_token(obj_raw):
                    obj_gid = _resolve_registry_id(obj_raw)
                    if obj_gid is not None:
                        _add_observed(obj_gid)
            other_predicate_gid = int(graph.get("other_predicate") or 0)
            if other_predicate_gid:
                _add_observed(other_predicate_gid)
        elif constraint_type == "inverse":
            inverse_gids = _collect_param_object_gids(param_inverse_gid)
            if not inverse_gids and constrained_gid is not None:
                inverse_gids = [constrained_gid]
            for obj_gid in inverse_gids:
                _add_observed(obj_gid)
        elif constraint_type == "itemRequiresStatement":
            for obj_gid in _collect_param_object_gids(param_property_gid):
                _add_observed(obj_gid)
        elif constraint_type == "valueRequiresStatement":
            for obj_gid in _collect_param_object_gids(param_property_gid):
                _add_observed(obj_gid)
        elif constraint_type == "type":
            relation_gids = _collect_param_object_gids(param_relation_gid)
            if not relation_gids:
                relation_gids = list(default_relation_gids)
            for obj_gid in relation_gids:
                _add_observed(obj_gid)
        elif constraint_type == "valueType":
            relation_gids = _collect_param_object_gids(param_relation_gid)
            if not relation_gids:
                relation_gids = list(default_relation_gids)
            for obj_gid in relation_gids:
                _add_observed(obj_gid)

        subject_scope_ids: set[int] | None
        if constraint_type in {"single", "conflictWith", "itemRequiresStatement"}:
            subject_id = focus_local_nodes.get("subject")
            subject_scope_ids = {subject_id} if subject_id is not None else set()
        elif constraint_type in {"valueRequiresStatement", "valueType"}:
            object_id = focus_local_nodes.get("object")
            if object_id is not None and not _is_literal_node(graph, "object"):
                subject_scope_ids = {object_id}
            else:
                subject_scope_ids = set()
        elif constraint_type == "distinct":
            subject_scope_ids = set()
            focus_subject_id = focus_local_nodes.get("subject")
            if focus_subject_id is not None:
                subject_scope_ids.add(focus_subject_id)
            other_subject_gid = int(graph.get("other_subject") or 0)
            if other_subject_gid:
                other_subject_id = global_to_local_id_encoder.global_to_local.get(other_subject_gid)
                if other_subject_id is not None:
                    subject_scope_ids.add(other_subject_id)
        else:
            subject_scope_ids = None

        matched_predicate_local_ids = 0
        wiring_edges_created = 0
        matched_focus_predicate = False
        scope_predicate_counts: dict[int, int] = {}

        def _collect_pred_local_ids(predicate_gid: int, subject_ids: set[int] | None) -> list[int]:
            if subject_ids is None:
                return list(pred_global_to_pred_local_ids.get(predicate_gid, []))
            pred_ids: list[int] = []
            for subj_id in sorted(subject_ids):
                pred_ids.extend(subj_local_to_pred_global_to_pred_local_ids.get(subj_id, {}).get(predicate_gid, []))
            return pred_ids

        scope_pred_local_ids: list[int] = []
        scope_pred_local_ids_seen: set[int] = set()
        for predicate_gid in observed_predicates:
            local_pred_ids = _collect_pred_local_ids(predicate_gid, subject_scope_ids)
            if local_pred_ids:
                scope_predicate_counts[predicate_gid] = len(local_pred_ids)
            for pred_local_id in local_pred_ids:
                if pred_local_id in scope_pred_local_ids_seen:
                    continue
                scope_pred_local_ids_seen.add(pred_local_id)
                scope_pred_local_ids.append(pred_local_id)

        matched_predicate_local_ids = len(scope_pred_local_ids)
        for pred_local_id in scope_pred_local_ids:
            edges.append((factor_local_id, pred_local_id))
            edge_types.append(EDGE_FACTOR_TO_LOCAL_PREDICATE)
            edges.append((pred_local_id, factor_local_id))
            edge_types.append(EDGE_LOCAL_PREDICATE_TO_FACTOR)
            wiring_edges_created += 1
            if focus_local_nodes.get("predicate") == pred_local_id:
                matched_focus_predicate = True

        if constrained_gid is not None:
            local_triples = pred_global_to_local_triples.get(constrained_gid, [])
            for subject_id, predicate_id, object_id in local_triples:
                edges.append((factor_local_id, subject_id))
                edge_types.append(EDGE_FACTOR_TO_LOCAL_SUBJECT)
                edges.append((subject_id, factor_local_id))
                edge_types.append(EDGE_LOCAL_SUBJECT_TO_FACTOR)
                edges.append((factor_local_id, object_id))
                edge_types.append(EDGE_FACTOR_TO_LOCAL_OBJECT)
                edges.append((object_id, factor_local_id))
                edge_types.append(EDGE_LOCAL_OBJECT_TO_FACTOR)
                wiring_edges_created += 2
                if focus_local_nodes.get("predicate") == predicate_id:
                    matched_focus_predicate = True

        for predicate_id_raw, object_id_raw in zip(param_predicates, param_objects):
            try:
                predicate_gid = global_int_encoder.encode(predicate_id_raw, add_new=False)
            except Exception as exc:
                raise AssertionError(f"Missing predicate token '{predicate_id_raw}' in encoder.") from exc
            try:
                object_gid = global_int_encoder.encode(object_id_raw, add_new=False)
            except Exception as exc:
                raise AssertionError(f"Missing object token '{object_id_raw}' in encoder.") from exc

            add_factor_definition_edge(
                factor_local_id=factor_local_id,
                predicate_global_id=predicate_gid,
                object_global_id=object_gid,
            )
        if debug_factor_wiring:
            debug_entries.append(
                {
                    "constraint_id": int(constraint_id),
                    "constraint_token": str(constraint_token),
                    "constraint_type": constraint_type,
                    "constrained_property": constrained_property,
                    "observed_predicates": [int(gid) for gid in observed_predicates],
                    "matched_pred_local_ids": int(matched_predicate_local_ids),
                    "wiring_edges_created": int(wiring_edges_created),
                    "matched_focus_predicate": bool(matched_focus_predicate),
                    "scope_predicate_counts": {
                        str(int(gid)): int(count) for gid, count in scope_predicate_counts.items()
                    },
                }
            )
            if constraint_id == int(graph["constraint_id"]):
                primary_factor_focus_scope_ok = matched_focus_predicate

    assert primary_factor_index >= 0, "Primary constraint_id missing from factor list."
    if debug_factor_wiring and primary_factor_focus_scope_ok is not None:
        assert primary_factor_focus_scope_ok, "Primary factor missing scope edge to focus predicate."
        if factor_local_ids:
            factor_set = set(factor_local_ids)
            reverse_edge_types = {
                EDGE_LOCAL_PREDICATE_TO_FACTOR,
                EDGE_LOCAL_SUBJECT_TO_FACTOR,
                EDGE_LOCAL_OBJECT_TO_FACTOR,
            }
            has_incoming = False
            for (src, dst), etype in zip(edges, edge_types):
                if dst in factor_set and etype in reverse_edge_types:
                    has_incoming = True
                    break
            assert has_incoming, "Factor nodes missing incoming variable edges."

    # --- Finalise -----------------------------------------------------------
    num_nodes = len(global_to_local_id_encoder.local_attributes)
    role_flags = torch.full((num_nodes,), ROLE_NONE, dtype=torch.long)
    for key, role_value in (
        ("object", ROLE_OBJECT),
        ("predicate", ROLE_PREDICATE),
        ("subject", ROLE_SUBJECT),
    ):
        local_id = focus_local_nodes.get(key)
        if local_id is not None:
            role_flags[local_id] |= role_value

    if encoding == "text_embedding":
        x_np = np.asarray(global_to_local_id_encoder.local_attributes, dtype=resolved_embedding_dtype)
        if not x_np.flags["C_CONTIGUOUS"]:
            x_np = np.ascontiguousarray(x_np)
        x_tensor = torch.from_numpy(x_np)
    else:
        # integer node ids case
        x_tensor = torch.tensor(global_to_local_id_encoder.local_attributes, dtype=torch.long)

    data_graph = Data(
        x=x_tensor,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        edge_index_non_flattened=torch.tensor(edges_non_flattened, dtype=torch.long).t().contiguous(),
        edge_attr_non_flattened=torch.tensor(non_flattened_edge_attributes, dtype=torch.long),
        y=torch.tensor(
            [
                [
                    _normalize_target_id(graph["add_subject"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["add_predicate"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["add_object"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["del_subject"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["del_predicate"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["del_object"], global_int_encoder, unknown_global_id),
                ]
            ],
            dtype=torch.long,
        ),
        x_names=global_to_local_id_encoder.local_names,
        role_flags=role_flags,
    )

    # Baseline metadata
    # Violating triple in global ID space
    data_graph.focus_triple = torch.tensor([graph["subject"], graph["predicate"], graph["object"]], dtype=torch.long)
    data_graph.shape_id = int(graph["constraint_id"])
    # Standardize on `constraint_type` across the pipeline
    data_graph.constraint_type = str(graph["constraint_type"])
    data_graph.factor_constraint_ids = torch.tensor(factor_constraint_ids, dtype=torch.long)
    data_graph.factor_node_index = torch.tensor(factor_local_ids, dtype=torch.long)
    data_graph.primary_factor_index = int(primary_factor_index)
    data_graph.factor_constraint_types = factor_constraint_types
    if factor_checkable_pre_tensor is not None:
        data_graph.factor_checkable_pre = factor_checkable_pre_tensor
        data_graph.factor_satisfied_pre = factor_satisfied_pre_tensor
        data_graph.factor_checkable_post_gold = factor_checkable_post_gold_tensor
        data_graph.factor_satisfied_post_gold = factor_satisfied_post_gold_tensor
        data_graph.factor_types = factor_types_tensor
    if debug_factor_wiring:
        data_graph.factor_wiring_debug = {
            "constraint_id": int(graph["constraint_id"]),
            "primary_constraint_id": int(graph["constraint_id"]),
            "primary_factor_index": int(primary_factor_index),
            "local_constraint_count": int(len(factor_ids)),
            "focus_predicate_local_id": int(focus_local_nodes.get("predicate", -1)),
            "focus_predicate_global_id": int(graph.get("predicate") or 0),
            "other_predicate_global_id": int(graph.get("other_predicate") or 0),
            "factors": debug_entries,
        }

    if factor_local_ids:
        is_factor_node = torch.zeros(num_nodes, dtype=torch.bool)
        is_factor_node[data_graph.factor_node_index] = True
        data_graph.is_factor_node = is_factor_node

    return data_graph


def pandas_to_dataset(dataframe: pd.DataFrame) -> datasets.Dataset:
    """
    Convert a Pandas DataFrame with BASS columns into a *datasets* Dataset.
    """
    dataset = datasets.Dataset.from_pandas(dataframe)
    return dataset


def _maybe_add_target(value: Any, store: set[int]) -> None:
    """Add a scalar integer target to ``store`` if possible."""
    if value is None:
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        # Target slots should be scalar but guard against accidental iterables.
        for item in value:
            _maybe_add_target(item, store)
        return

    try:
        idx = int(value)
    except (TypeError, ValueError):
        return
    if idx >= 0:
        store.add(idx)


def _update_target_sets(graph: dict[str, Any], entity_store: set[int], predicate_store: set[int]) -> None:
    """Collect target class ids from a raw graph row."""
    for key in ("add_subject", "add_object", "del_subject", "del_object"):
        _maybe_add_target(graph.get(key), entity_store)
    for key in ("add_predicate", "del_predicate"):
        _maybe_add_target(graph.get(key), predicate_store)


def compute_torch_geometric_objects(
    data: datasets.Dataset,
    wikidata_cache: PrecomputedWikidataCache | None,
    global_int_encoder: GlobalIntEncoder,
    constraint_registry: Dict[str, Dict[str, Any]],
    encoding: Literal["node_id", "text_embedding"],
    constraint_scope: Literal["local", "focus"] = "local",
    store_node_names: bool = False,
    embedding_dtype: np.dtype | None = None,
    on_sample: Optional[Callable[[Data], None]] = None,
    debug_state: dict[str, Any] | None = None,
) -> Iterator[Data]:
    """
    Build a stream of torch_geometric.data.Data graphs from a *datasets* object.

    Yields
    -----
    torch_geometric.data.Data
        Individual graphs emitted one-by-one to avoid materialising the full
        dataset in memory.
    """
    for row_idx in tqdm(range(len(data)), desc="Processing rows"):
        row = dict(data[row_idx])
        graph = create_graph(
            row,
            wikidata_cache=wikidata_cache,
            global_int_encoder=global_int_encoder,
            constraint_registry=constraint_registry,
            encoding=encoding,
            constraint_scope=constraint_scope,
            store_node_names=store_node_names,
            embedding_dtype=embedding_dtype,
            debug_factor_wiring=bool(debug_state and debug_state.get("enabled")),
        )
        if on_sample is not None:
            on_sample(graph)
        if debug_state and debug_state.get("enabled") and not debug_state.get("written"):
            debug_payload = getattr(graph, "factor_wiring_debug", None)
            if debug_payload and debug_payload.get("local_constraint_count", 0) > 1:
                debug_payload = dict(debug_payload)
                debug_payload["row_index"] = int(row_idx)
                for entry in debug_payload.get("factors", []):
                    logging.info(
                        "Factor wiring debug: constraint_id=%s constraint_type=%s constrained_property=%s observed_predicates=%s matched_pred_local_ids=%s wiring_edges_created=%s",
                        entry.get("constraint_id"),
                        entry.get("constraint_type"),
                        entry.get("constrained_property"),
                        entry.get("observed_predicates"),
                        entry.get("matched_pred_local_ids"),
                        entry.get("wiring_edges_created"),
                    )
                debug_path = Path(debug_state["path"])
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with debug_path.open("w", encoding="utf-8") as fh:
                    json.dump(debug_payload, fh)
                debug_state["written"] = True
        yield graph


def check_data_graph(geometric_objects: Iterable[Data], batch_size: int = 32) -> None:
    """
    Sanity-check that the graphs can be batched without shape mismatches.
    """
    objects = list(geometric_objects)
    if not objects:
        logging.warning("No graph objects available for sanity check.")
        return

    loader = DataLoader(objects, batch_size=batch_size)

    for batch in loader:
        logging.info("Sample batch size: %s", batch.num_graphs)
        logging.info("Sample edge index shape: %s", batch.edge_index.shape)
        logging.info("Sample node features shape: %s", batch.x.shape)
        logging.info("Sample label shape: %s", batch.y.shape)
        break  # Just to check the first batch


def collect_sample_for_check(
    output_path: Path,
    shard_size: int,
    use_torch_save: bool,
    sample_size: int,
) -> list[Data]:
    """Collect up to ``sample_size`` graphs from the written artifacts."""
    if sample_size <= 0:
        return []

    output_path = Path(output_path)

    if shard_size <= 0:
        return list(itertools.islice(iter_stream(output_path), sample_size))

    suffix = ".pt" if use_torch_save else output_path.suffix
    sample: list[Data] = []
    for shard_path in sorted(output_path.parent.glob(f"{output_path.stem}-shard*{suffix}")):
        if use_torch_save:
            shard_objects = torch.load(shard_path, map_location="cpu")
        else:
            with open(shard_path, "rb") as f:
                shard_objects = pickle.load(f)
        for obj in shard_objects:
            sample.append(obj)
            if len(sample) >= sample_size:
                del shard_objects
                return sample

        del shard_objects

    return sample


def main(
    wikidata_cache: PrecomputedWikidataCache | None,
    global_int_encoder: GlobalIntEncoder,
    constraint_registry: Dict[str, Dict[str, Any]],
    encoding: Literal["node_id", "text_embedding"],
    constraint_scope: Literal["local", "focus"] = "local",
    store_node_names: bool = False,
    shard_size: int = 0,
    use_torch_save: bool = False,
    embedding_dtype: np.dtype | None = None,
    check_sample_size: int = 32,
    max_instances: int = 0,
    debug_factor_wiring: bool = False,
) -> dict[str, Path]:
    """Sequentially build graphs per split and persist them to disk."""

    split_to_parquet = {
        "train": "df_train.parquet",
        "val": "df_val.parquet",
        "test": "df_test.parquet",
    }

    outputs: dict[str, Path] = {}

    # Record target vocabularies seen in the labels (`add_*` / `del_*`)
    # so training can precompute class vocabularies.
    total_entity_targets: set[int] = {0}
    total_predicate_targets: set[int] = {0}
    split_targets: dict[str, dict[str, list[int]]] = {}

    debug_state = {
        "enabled": debug_factor_wiring,
        "written": False,
        "path": PROCESSED_DATA_PATH / "factor_wiring_debug.json",
    }

    for split, parquet_name in split_to_parquet.items():
        logging.info("Processing %s split", split)

        # Load dataframe
        dataframe = pd.read_parquet(INTERIM_DATA_PATH / parquet_name)
        dataset = pandas_to_dataset(dataframe)
        del dataframe
        if max_instances and max_instances > 0:
            limit = min(max_instances, len(dataset))
            dataset = dataset.select(range(limit))
            logging.info("Limiting %s split to first %s instances", split, limit)

        split_entity_targets: set[int] = {0}
        split_predicate_targets: set[int] = {0}

        def collect_targets(graph: Data):
            y = graph.y
            if y is None:
                raise ValueError("Missing graph.y targets for sanity check.")
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            if y.ndim == 1:
                y = y.unsqueeze(0)
            if y.shape != (1, 6):
                raise ValueError(f"Expected y to be of shape (1, 6), got {tuple(y.shape)}")
            add_subject, add_predicate, add_object, del_subject, del_predicate, del_object = y[0].tolist()
            for idx in (add_subject, add_object, del_subject, del_object):
                split_entity_targets.add(idx)
                total_entity_targets.add(idx)
            for idx in (add_predicate, del_predicate):
                split_predicate_targets.add(idx)
                total_predicate_targets.add(idx)

        # Build graphs (the heavy lifting happens here)
        generator = compute_torch_geometric_objects(
            dataset,
            wikidata_cache,
            global_int_encoder,
            constraint_registry,
            encoding=encoding,
            constraint_scope=constraint_scope,
            store_node_names=store_node_names,
            embedding_dtype=embedding_dtype,
            on_sample=collect_targets,
            debug_state=debug_state,
        )
        del dataset
        output_path = PROCESSED_DATA_PATH / f"{split}_graph-{encoding}.pkl"

        # Persist graphs
        total_objects = 0
        shard_count = 0
        if shard_size > 0:
            total_objects, shard_count = dump_in_shards(
                generator,
                output_path,
                shard_size=shard_size,
                use_torch_save=use_torch_save,
            )
        else:
            total_objects = dump_stream(generator, output_path)
            shard_count = 1 if total_objects else 0

        logging.info(
            "Finished writing %s split (%s graphs across %s shard%s)",
            split,
            total_objects,
            shard_count,
            "s" if shard_count != 1 else "",
        )

        del generator

        # Sanity check
        sample_size = min(check_sample_size, shard_size) if shard_size > 0 else check_sample_size
        sample_graphs = collect_sample_for_check(
            output_path,
            shard_size=shard_size,
            use_torch_save=use_torch_save,
            sample_size=sample_size,
        )
        if sample_graphs:
            check_data_graph(sample_graphs)
        else:
            logging.warning("No data available for sanity checking %s split", split)
        del sample_graphs

        outputs[split] = output_path
        gc.collect()

        split_targets[split] = {
            "entity_class_ids": sorted(split_entity_targets),
            "predicate_class_ids": sorted(split_predicate_targets),
        }

    vocab_path = PROCESSED_DATA_PATH / "target_vocabs.json"
    vocab_payload = {
        "entity_class_ids": sorted(total_entity_targets),
        "predicate_class_ids": sorted(total_predicate_targets),
        "per_split": split_targets,
    }
    with vocab_path.open("w", encoding="utf-8") as fh:
        json.dump(vocab_payload, fh)
    logging.info(
        "Stored target vocabularies in %s | entities=%s predicates=%s",
        vocab_path,
        len(vocab_payload["entity_class_ids"]),
        len(vocab_payload["predicate_class_ids"]),
    )

    return outputs


def display_graph(encoder: GlobalIntEncoder) -> None:
    train_output = outputs.get("train")
    graph_to_show: Optional[Data] = None

    if train_output is None:
        logging.warning("Train split output not found; skipping graph visualisation.")
    elif args.shard_size > 0:
        suffix = ".pt" if args.use_torch_save else train_output.suffix
        for shard_path in sorted(train_output.parent.glob(f"{train_output.stem}-shard*{suffix}")):
            if args.use_torch_save:
                shard_objects = torch.load(shard_path, map_location="cpu")
            else:
                with open(shard_path, "rb") as f:
                    shard_objects = pickle.load(f)
            if shard_objects:
                graph_to_show = shard_objects[0]
                del shard_objects
                break
            del shard_objects
    else:
        graph_to_show = next(iter_stream(train_output), None)

    if graph_to_show is None:
        logging.warning("No graph available to display.")
    else:
        graph_to_show.validate()

        print("Graph to show:")
        print(graph_to_show)

        g = utils.to_networkx(
            graph_to_show,
            to_undirected=False,
            # node_attrs=graph_to_show.x_names)
            node_attrs=["x_names"],
        )
        x_names = getattr(graph_to_show, "x_names", None)
        labels = dict(enumerate(x_names)) if x_names is not None else None
        nx.draw(
            g,
            pos=nx.spring_layout(g, k=7 / g.order() ** (1 / 2)),
            with_labels=True,
            node_size=500,
            font_size=8,
            labels=labels,
        )
        plt.savefig(PROCESSED_DATA_PATH / "graph_visualization.png")
        plt.show()

        # show non-flattened graph
        if graph_to_show.edge_index_non_flattened is None:
            raise ValueError("Missing edge_index_non_flattened for non-flattened display.")
        graph_to_show.edge_index = graph_to_show.edge_index_non_flattened
        graph_to_show.validate()
        g = utils.to_networkx(
            graph_to_show,
            to_undirected=False,
            node_attrs=["x_names"],
            edge_attrs=["edge_attr_non_flattened"],
        )
        pos = nx.spring_layout(g, k=7 / g.order() ** (1 / 2))
        x = graph_to_show.x
        if x is None:
            raise ValueError("Missing node features for graph visualization.")
        edge_attr = graph_to_show.edge_attr_non_flattened
        if edge_attr is None:
            raise ValueError("Missing edge_attr_non_flattened for edge label display.")
        nx.draw(
            g,
            pos=pos,
            with_labels=True,
            node_size=500,
            font_size=8,
            labels={
                i: encoder.decode(int(global_id.item()), use_filtered_id_mapping=True)
                for i, global_id in enumerate(x)
            },
        )
        nx.draw_networkx_edge_labels(
            g,
            pos=pos,
            edge_labels={
                (u, v): encoder.decode(
                    int(x[int(edge_attr[i].item())].item()),
                    use_filtered_id_mapping=True,
                )
                for i, (u, v) in enumerate(g.edges())
            },
        )
        plt.savefig(PROCESSED_DATA_PATH / "graph_visualization-non_flattened.png")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Build PyG graphs from interim dataframes")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Which dataset to fetch.",
    )
    parser.add_argument(
        "--encoding",
        choices=["node_id", "text_embedding"],
        required=True,
        help="Encoding options for the Wikidata URIs",
    )
    parser.add_argument(
        "--min-occurrence",
        type=int,
        default=100,
        help="Dataset variant produced with the given training frequency threshold.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Number of graphs per shard when streaming to disk (0 disables sharding).",
    )
    parser.add_argument(
        "--max_instances",
        "--max-instances",
        type=int,
        default=0,
        help="Limit the number of rows converted per split (0 uses the full split).",
    )
    parser.add_argument(
        "--use-torch-save",
        action="store_true",
        help="Persist shards with torch.save instead of pickle for reduced overhead.",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Visualise the first graph using NetworkX and Matplotlib.",
    )
    parser.add_argument(
        "--wikidata-cache-path",
        type=str,
        default="data/interim/wikidata_text.parquet",
        help="Path to the precomputed Wikidata cache parquet file.",
    )
    parser.add_argument(
        "--debug_factor_wiring",
        "--debug-factor-wiring",
        action="store_true",
        help="Write factor wiring diagnostics for the first multi-constraint instance.",
    )
    parser.add_argument(
        "--constraint-scope",
        choices=["local", "focus"],
        default="local",
        help="Which constraint neighborhood to use for factor nodes (default: local).",
    )

    args = parser.parse_args()

    if args.shard_size < 0:
        parser.error("--shard-size must be non-negative")
    if args.max_instances < 0:
        parser.error("--max-instances must be non-negative")

    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    text_embedding_dtype: np.dtype = np.dtype("float16")

    # Parse command-line arguments
    args = parse_args()
    DATASET: str = args.dataset
    MIN_OCCURRENCE = max(1, args.min_occurrence)
    dataset_variant = dataset_variant_name(DATASET, MIN_OCCURRENCE)

    # .gitignore the data files
    gitignore = Path("data/.gitignore")
    gitignore.write_text("*\n!.gitignore\n")

    # Define data paths
    INTERIM_DATA_PATH = Path("data/interim/") / dataset_variant  # Load data
    PROCESSED_DATA_PATH = Path("data/processed/") / dataset_variant  # Save data
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Building graphs for dataset=%s (variant=%s, min_occurrence=%s)",
        DATASET,
        dataset_variant,
        MIN_OCCURRENCE,
    )

    # Load and freeze int encoder
    encoder = GlobalIntEncoder()
    encoder.load(INTERIM_DATA_PATH / "globalintencoder.txt")
    encoder.freeze()

    registry_path = Path("data/interim") / f"constraint_registry_{DATASET}.parquet"
    registry_df = pd.read_parquet(registry_path)
    if "registry_json" not in registry_df.columns:
        raise KeyError(f"Missing registry_json column in {registry_path}")
    constraint_registry: Dict[str, Dict[str, Any]] = {}
    for registry_payload in registry_df["registry_json"]:
        if isinstance(registry_payload, str):
            registry_payload = json.loads(registry_payload)
        if not isinstance(registry_payload, dict):
            raise TypeError(f"Unexpected registry_json payload type: {type(registry_payload)}")
        for key, value in registry_payload.items():
            constraint_registry[str(key)] = value

    # Graph
    LITERAL_ID = encoder.encode("LITERAL_OBJECT", add_new=False)
    if args.encoding == "node_id":
        logging.info("Using node ID encoding for graph nodes.")
        wikidata_cache = None
    else:
        logging.info("Using text embedding encoding for graph nodes.")
        wikidata_cache = PrecomputedWikidataCache(Path(args.wikidata_cache_path))

    outputs = main(
        wikidata_cache=wikidata_cache,
        global_int_encoder=encoder,
        constraint_registry=constraint_registry,
        encoding=args.encoding,
        constraint_scope=args.constraint_scope,
        shard_size=args.shard_size,
        use_torch_save=args.use_torch_save,
        embedding_dtype=text_embedding_dtype,
        max_instances=args.max_instances,
        debug_factor_wiring=args.debug_factor_wiring,
    )

    if args.show_graph:
        display_graph(encoder)
