import torch
from torch_geometric.data import Data

from modules.config import ModelConfig
from modules.models import RepairGINFactorPressure


def _make_sample_graph() -> Data:
    # 4 nodes: subject(0), predicate(1), object(2), factor(3)
    x = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 3, 3], [1, 2, 1, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 4, 5], dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([[1, 2, 3, 1, 2, 3]], dtype=torch.long),
    )
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    data.factor_node_index = torch.tensor([3], dtype=torch.long)
    data.factor_constraint_ids = torch.tensor([42], dtype=torch.long)
    data.primary_factor_index = 0
    data.factor_types = torch.tensor([1], dtype=torch.long)
    data.factor_checkable_pre = torch.tensor([True], dtype=torch.bool)
    data.factor_satisfied_pre = torch.tensor([1], dtype=torch.long)
    return data


def test_pressure_forward_and_backward():
    cfg = ModelConfig(
        num_embedding_size=8,
        num_layers=2,
        hidden_channels=8,
        head_hidden=8,
        dropout=0.1,
        use_node_embeddings=True,
        use_edge_attributes=False,
        entity_class_ids=(0, 1, 2, 3, 4),
        predicate_class_ids=(0, 1, 2, 3),
        num_factor_types=3,
        factor_type_embedding_dim=4,
        factor_executor_impl="per_type_v1",
        pressure_enabled=True,
    )
    model = RepairGINFactorPressure(
        num_input_graph_nodes=10,
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

    data = _make_sample_graph()
    outputs = model(data)
    assert "edit_logits" in outputs
    assert "factor_logits_pre" in outputs
    assert "factor_logits_post_gold" in outputs
    edit_logits = outputs["edit_logits"]
    factor_logits = outputs["factor_logits_pre"]
    factor_logits_post_gold = outputs["factor_logits_post_gold"]
    assert edit_logits.shape == (1, 6, model.num_target_ids)
    assert factor_logits.shape == (1,)
    assert factor_logits_post_gold.shape == (1,)

    loss = edit_logits.sum() + factor_logits.sum() + factor_logits_post_gold.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    print("test_pressure_forward_and_backward passed")
