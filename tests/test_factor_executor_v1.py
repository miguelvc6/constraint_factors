import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig
from modules.models import RepairGINFactorPressure, _build_factor_scope_runtime


def _make_factor_graph(*, factor_type: int, subject_id: int, predicate_id: int, object_id: int) -> Data:
    x = torch.tensor([subject_id, predicate_id, object_id, subject_id + predicate_id + object_id], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 3, 3, 3, 1, 0, 2],
            [1, 2, 1, 0, 2, 3, 3, 3],
        ],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([[subject_id, predicate_id, object_id, subject_id, predicate_id, object_id]], dtype=torch.long),
    )
    data.factor_node_index = torch.tensor([3], dtype=torch.long)
    data.is_factor_node = torch.tensor([False, False, False, True], dtype=torch.bool)
    data.factor_constraint_ids = torch.tensor([100 + factor_type], dtype=torch.long)
    data.primary_factor_index = 0
    data.factor_types = torch.tensor([factor_type], dtype=torch.long)
    data.factor_checkable_pre = torch.tensor([True], dtype=torch.bool)
    data.factor_satisfied_pre = torch.tensor([1], dtype=torch.long)
    data.factor_checkable_post_gold = torch.tensor([True], dtype=torch.bool)
    data.factor_satisfied_post_gold = torch.tensor([1], dtype=torch.long)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data


def test_scope_runtime_roles_and_batch_order() -> None:
    g1 = _make_factor_graph(factor_type=0, subject_id=1, predicate_id=2, object_id=3)
    g2 = _make_factor_graph(factor_type=1, subject_id=4, predicate_id=5, object_id=6)
    batch = Batch.from_data_list([g1, g2])
    runtime = _build_factor_scope_runtime(
        batch.x.float().view(-1, 1),
        batch,
        factor_edge_types=(4, 5, 6),
        num_factor_types=2,
    )

    assert runtime is not None
    assert runtime.factor_graph_index.tolist() == [0, 1]
    assert runtime.predicate_counts.view(-1).tolist() == [1.0, 1.0]
    assert runtime.subject_counts.view(-1).tolist() == [1.0, 1.0]
    assert runtime.object_counts.view(-1).tolist() == [1.0, 1.0]
    assert runtime.predicate_factor_pos.tolist() == [0, 1]
    assert runtime.subject_factor_pos.tolist() == [0, 1]
    assert runtime.object_factor_pos.tolist() == [0, 1]


def test_unknown_factor_type_maps_to_fallback() -> None:
    graph = _make_factor_graph(factor_type=9, subject_id=1, predicate_id=2, object_id=3)
    runtime = _build_factor_scope_runtime(
        graph.x.float().view(-1, 1),
        graph,
        factor_edge_types=(4, 5, 6),
        num_factor_types=2,
    )

    assert runtime is not None
    assert runtime.factor_type_ids.tolist() == [2]


def test_per_type_executor_forward_emits_post_gold() -> None:
    cfg = ModelConfig(
        num_embedding_size=8,
        num_layers=2,
        hidden_channels=8,
        head_hidden=8,
        dropout=0.1,
        use_node_embeddings=True,
        use_edge_attributes=False,
        entity_class_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        predicate_class_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        num_factor_types=2,
        factor_executor_impl="per_type_v1",
        pressure_enabled=True,
    )
    model = RepairGINFactorPressure(
        num_input_graph_nodes=16,
        num_embedding_size=cfg.num_embedding_size,
        num_layers=cfg.num_layers,
        hidden=cfg.hidden_channels,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
        use_node_embeddings=cfg.use_node_embeddings,
        use_role_embeddings=cfg.use_role_embeddings,
        num_role_types=cfg.num_role_types,
        role_embedding_dim=cfg.role_embedding_dim,
        use_edge_attributes=cfg.use_edge_attributes,
        use_edge_subtraction=cfg.use_edge_subtraction,
        entity_class_ids=cfg.entity_class_ids,
        predicate_class_ids=cfg.predicate_class_ids,
        num_factor_types=cfg.num_factor_types,
        factor_type_embedding_dim=cfg.factor_type_embedding_dim,
        factor_executor_impl=cfg.factor_executor_impl,
        pressure_enabled=cfg.pressure_enabled,
    )

    data = _make_factor_graph(factor_type=1, subject_id=1, predicate_id=2, object_id=3)
    outputs = model(data)

    assert "factor_logits_pre" in outputs
    assert "factor_logits_post_gold" in outputs
    assert outputs["factor_logits_pre"] is not None
    assert outputs["factor_logits_post_gold"] is not None
    assert tuple(outputs["factor_logits_pre"].shape) == (1,)
    assert tuple(outputs["factor_logits_post_gold"].shape) == (1,)

    loss = outputs["edit_logits"].sum() + outputs["factor_logits_pre"].sum() + outputs["factor_logits_post_gold"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


if __name__ == "__main__":
    test_scope_runtime_roles_and_batch_order()
    test_unknown_factor_type_maps_to_fallback()
    test_per_type_executor_forward_emits_post_gold()
    print("factor executor v1 tests passed")
