from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for path in (SRC, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.config import ModelConfig
from modules.models import build_model
from test_primary_query_graph_schema import _make_graph, _make_primary_only_graph


def _batched_pair(mode: str) -> Batch:
    g1, _, _ = _make_graph(mode)
    g2, _, _ = _make_graph(mode)
    g2.primary_param_predicate_ids = g2.primary_param_predicate_ids[:1]
    g2.primary_param_object_ids = g2.primary_param_object_ids[:1]
    g2.primary_param_count = torch.tensor([1], dtype=torch.long)
    return Batch.from_data_list([g1, g2])


def test_query_metadata_batches_with_variable_parameter_counts() -> None:
    batch = _batched_pair("query_definition")
    assert batch.primary_constraint_type_id.shape[0] == 2
    assert batch.primary_constrained_property_id.shape[0] == 2
    assert batch.primary_param_count.shape[0] == 2
    assert int(batch.primary_param_count.sum().item()) == len(batch.primary_param_predicate_ids)
    assert len(batch.primary_param_predicate_ids) == 3
    assert len(batch.primary_param_object_ids) == 3


def test_primary_modes_forward_emit_edit_logits() -> None:
    for mode in ("query_definition", "query_family", "passive_node"):
        batch = _batched_pair(mode)
        num_nodes = int(batch.x.max().item()) + 1
        cfg = ModelConfig(
            model="GIN_PRESSURE",
            num_embedding_size=8,
            num_layers=2,
            hidden_channels=8,
            head_hidden=8,
            dropout=0.0,
            use_node_embeddings=True,
            use_edge_attributes=False,
            use_role_embeddings=True,
            num_role_types=8,
            role_embedding_dim=4,
            entity_class_ids=tuple(range(num_nodes)),
            predicate_class_ids=tuple(range(num_nodes)),
            num_factor_types=2,
            factor_type_embedding_dim=4,
            factor_executor_impl="per_type_v1",
            pressure_enabled=True,
            primary_constraint_mode=mode,
        )
        model = build_model("GIN_PRESSURE", num_nodes, cfg)
        outputs = model(batch)
        assert outputs["edit_logits"].shape == (2, 6, model.num_target_ids)


def test_query_forward_handles_batch_with_no_executable_factors() -> None:
    g1, _ = _make_primary_only_graph("query_definition")
    g2, _ = _make_primary_only_graph("query_definition")
    batch = Batch.from_data_list([g1, g2])
    num_nodes = int(batch.x.max().item()) + 1
    cfg = ModelConfig(
        model="GIN_PRESSURE",
        num_embedding_size=8,
        num_layers=2,
        hidden_channels=8,
        head_hidden=8,
        dropout=0.0,
        use_node_embeddings=True,
        use_edge_attributes=False,
        entity_class_ids=tuple(range(num_nodes)),
        predicate_class_ids=tuple(range(num_nodes)),
        num_factor_types=2,
        factor_executor_impl="per_type_v1",
        pressure_enabled=True,
        primary_constraint_mode="query_definition",
    )
    model = build_model("GIN_PRESSURE", num_nodes, cfg)
    outputs = model(batch)
    assert outputs["edit_logits"].shape == (2, 6, model.num_target_ids)
    assert outputs["factor_logits_pre"] is not None
    assert outputs["factor_logits_pre"].numel() == 0


if __name__ == "__main__":
    test_query_metadata_batches_with_variable_parameter_counts()
    test_primary_modes_forward_emit_edit_logits()
    test_query_forward_handles_batch_with_no_executable_factors()
    print("primary query model forward tests passed")
