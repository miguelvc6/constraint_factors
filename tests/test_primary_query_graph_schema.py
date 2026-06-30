from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.data_encoders import GlobalIntEncoder
from scripts.convert_primary_query_graph_mode import _convert_passive_node, _convert_query_metadata_only


def _load_graph_module():
    spec = importlib.util.spec_from_file_location("graph_06_for_primary_query_test", ROOT / "src" / "06_graph.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load src/06_graph.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoder_and_registry():
    encoder = GlobalIntEncoder()

    def enc(token: str) -> int:
        return encoder.encode(token, add_new=True)

    subject = enc("http://www.wikidata.org/entity/Q1")
    predicate = enc("http://www.wikidata.org/entity/P10")
    obj = enc("http://www.wikidata.org/entity/Q2")
    add_subject = enc("http://www.wikidata.org/entity/Q3")
    add_predicate = enc("http://www.wikidata.org/entity/P11")
    add_object = enc("http://www.wikidata.org/entity/Q4")
    c_primary = enc("C_PRIMARY")
    c_secondary = enc("C_SECONDARY")
    enc("constraint_factor::C_PRIMARY")
    enc("constraint_factor::C_SECONDARY")
    for token in (
        "P2306",
        "P2309",
        "P1696",
        "Q10",
        "Q20",
        "http://www.wikidata.org/entity/P10",
        "http://www.wikidata.org/entity/P20",
        "http://www.wikidata.org/entity/Q10",
        "http://www.wikidata.org/entity/Q20",
    ):
        enc(token)
        if token.startswith("http://"):
            enc(f"<{token}>")

    encoder.freeze()
    registry = {
        "C_PRIMARY": {
            "constraint_family": "single",
            "constraint_type_index": 0,
            "constrained_property": "P10",
            "param_predicates": ["P2306", "P2309"],
            "param_objects": ["Q10", "Q20"],
        },
        "C_SECONDARY": {
            "constraint_family": "valueType",
            "constraint_type_index": 1,
            "constrained_property": "P20",
            "param_predicates": ["P2306"],
            "param_objects": ["Q20"],
        },
    }
    row = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "other_subject": 0,
        "other_predicate": 0,
        "other_object": 0,
        "subject_predicates": [],
        "subject_objects": [],
        "object_predicates": [],
        "object_objects": [],
        "other_entity_predicates": [],
        "other_entity_objects": [],
        "add_subject": add_subject,
        "add_predicate": add_predicate,
        "add_object": add_object,
        "del_subject": subject,
        "del_predicate": predicate,
        "del_object": obj,
        "constraint_id": c_primary,
        "constraint_type": "single",
        "factor_constraint_ids": [c_primary, c_secondary],
        "factor_types": [0, 1],
        "factor_checkable_pre": [True, True],
        "factor_satisfied_pre": [0, 1],
        "factor_checkable_post_gold": [True, True],
        "factor_satisfied_post_gold": [1, 1],
    }
    return encoder, registry, row, c_primary, c_secondary


def _make_graph(mode: str):
    graph_module = _load_graph_module()
    encoder, registry, row, primary_id, secondary_id = _encoder_and_registry()
    data = graph_module.create_graph(
        row,
        wikidata_cache=None,
        global_int_encoder=encoder,
        constraint_registry=registry,
        encoding="node_id",
        constraint_scope="local",
        constraint_representation="factorized",
        primary_constraint_mode=mode,
    )
    return data, primary_id, secondary_id


def _make_primary_only_graph(mode: str):
    graph_module = _load_graph_module()
    encoder, registry, row, primary_id, _ = _encoder_and_registry()
    row = dict(row)
    row["factor_constraint_ids"] = [primary_id]
    row["factor_types"] = [0]
    row["factor_checkable_pre"] = [True]
    row["factor_satisfied_pre"] = [0]
    row["factor_checkable_post_gold"] = [True]
    row["factor_satisfied_post_gold"] = [1]
    data = graph_module.create_graph(
        row,
        wikidata_cache=None,
        global_int_encoder=encoder,
        constraint_registry=registry,
        encoding="node_id",
        constraint_scope="local",
        constraint_representation="factorized",
        primary_constraint_mode=mode,
    )
    return data, primary_id


def test_primary_modes_split_executable_and_eval_factors() -> None:
    for mode in ("query_definition", "query_family", "passive_node"):
        data, primary_id, secondary_id = _make_graph(mode)
        assert primary_id not in data.factor_constraint_ids.tolist()
        assert secondary_id in data.factor_constraint_ids.tolist()
        assert primary_id in data.eval_factor_constraint_ids.tolist()
        assert int(data.primary_factor_index) == -1
        assert int(data.eval_primary_factor_index) >= 0
        assert data.factor_constraint_ids.numel() == data.factor_types.numel()
        assert data.factor_constraint_ids.numel() == data.factor_satisfied_pre.numel()
        assert data.eval_factor_constraint_ids.numel() == data.eval_factor_satisfied_pre.numel()
        assert data.primary_constraint_id.tolist() == [primary_id]
        assert data.primary_constraint_type_id.tolist() == [0]
        assert data.primary_param_count.tolist() == [2]


def test_executable_factor_mode_is_backward_compatible() -> None:
    data, primary_id, _ = _make_graph("executable_factor")
    assert primary_id in data.factor_constraint_ids.tolist()
    assert int(data.primary_factor_index) >= 0
    assert data.eval_factor_constraint_ids.tolist() == data.factor_constraint_ids.tolist()
    assert int(data.eval_primary_factor_index) == int(data.primary_factor_index)


def test_passive_primary_node_is_not_executable_and_has_no_pressure_edges() -> None:
    data, _, _ = _make_graph("passive_node")
    assert hasattr(data, "passive_primary_node_index")
    passive_idx = int(data.passive_primary_node_index)
    assert passive_idx not in data.factor_node_index.tolist()
    pressure_mask = torch.isin(data.edge_type, torch.tensor([4, 5, 6], dtype=torch.long))
    if pressure_mask.any():
        assert not torch.any(data.edge_index[0, pressure_mask] == passive_idx)


def test_query_mode_allows_primary_only_local_set_with_no_executable_factors() -> None:
    data, primary_id = _make_primary_only_graph("query_definition")
    assert data.factor_constraint_ids.numel() == 0
    assert data.factor_node_index.numel() == 0
    assert data.factor_types.numel() == 0
    assert int(data.primary_factor_index) == -1
    assert data.eval_factor_constraint_ids.tolist() == [primary_id]
    assert int(data.eval_primary_factor_index) == 0


def test_primary_query_converter_derives_query_definition_and_passive_node() -> None:
    graph_module = _load_graph_module()
    encoder, registry, row, _, _ = _encoder_and_registry()
    query_family = graph_module.create_graph(
        row,
        wikidata_cache=None,
        global_int_encoder=encoder,
        constraint_registry=registry,
        encoding="node_id",
        constraint_scope="local",
        constraint_representation="factorized",
        primary_constraint_mode="query_family",
    )
    query_definition = _convert_query_metadata_only(query_family.clone(), "query_definition")
    assert query_definition.primary_constraint_mode == "query_definition"
    assert not hasattr(query_definition, "passive_primary_node_index")

    before_nodes = int(query_family.x.numel())
    passive = _convert_passive_node(query_family, encoder)
    passive_idx = int(passive.passive_primary_node_index)
    assert passive.primary_constraint_mode == "passive_node"
    assert passive_idx == before_nodes
    assert passive_idx not in passive.factor_node_index.tolist()
    assert not bool(passive.is_factor_node[passive_idx].item())
    pressure_mask = torch.isin(passive.edge_type, torch.tensor([4, 5, 6], dtype=torch.long))
    if pressure_mask.any():
        assert not torch.any(passive.edge_index[0, pressure_mask] == passive_idx)
    definition_mask = passive.edge_index[0] == passive_idx
    assert int(definition_mask.sum().item()) == int(passive.primary_param_count.item())


if __name__ == "__main__":
    test_primary_modes_split_executable_and_eval_factors()
    test_executable_factor_mode_is_backward_compatible()
    test_passive_primary_node_is_not_executable_and_has_no_pressure_edges()
    test_query_mode_allows_primary_only_local_set_with_no_executable_factors()
    test_primary_query_converter_derives_query_definition_and_passive_node()
    print("primary query graph schema tests passed")
