from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig
from modules.factor_dispatch import build_grouped_dispatch
from modules.factor_types import scan_factor_type_ids
from modules.models import (
    FactorPostEditHead,
    FactorTypeExecutor,
    GroupedFactorPostEditHead,
    GroupedFactorTypeExecutor,
    RepairGINFactorPressure,
    SharedAdapterFactorPostEditHead,
    SharedAdapterFactorTypeExecutor,
)


ACTIVE_IDS = (0, 2)


def _compact_model(*, gold_mode: str = "compact") -> RepairGINFactorPressure:
    return RepairGINFactorPressure(
        num_input_graph_nodes=16,
        num_embedding_size=8,
        num_layers=2,
        hidden=8,
        head_hidden=8,
        dropout=0.0,
        use_node_embeddings=True,
        use_edge_attributes=False,
        entity_class_ids=(0, 2, 7),
        predicate_class_ids=(0, 5),
        num_factor_types=3,
        active_factor_type_ids=ACTIVE_IDS,
        factor_executor_impl="per_type_grouped_v2",
        gold_edit_embedding_mode=gold_mode,
        pressure_enabled=True,
        pressure_module_sharing="shared",
    )


def _factor_graph(factor_type: int) -> Data:
    graph = Data(
        x=torch.tensor([2, 5, 7, 9], dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 3, 3, 3], [1, 2, 1, 0, 2]],
            dtype=torch.long,
        ),
        edge_type=torch.tensor([0, 1, 4, 5, 6], dtype=torch.long),
        y=torch.tensor([[2, 5, 7, 0, 0, 0]], dtype=torch.long),
    )
    graph.batch = torch.zeros(4, dtype=torch.long)
    graph.factor_node_index = torch.tensor([3], dtype=torch.long)
    graph.is_factor_node = torch.tensor([False, False, False, True])
    graph.factor_constraint_ids = torch.tensor([100])
    graph.factor_types = torch.tensor([factor_type])
    graph.factor_checkable_pre = torch.tensor([True])
    graph.factor_satisfied_pre = torch.tensor([0])
    graph.factor_checkable_post_gold = torch.tensor([True])
    graph.factor_satisfied_post_gold = torch.tensor([1])
    graph.primary_factor_index = 0
    return graph


def _copy_executor(
    legacy: nn.ModuleList,
    grouped: GroupedFactorTypeExecutor,
) -> None:
    for type_index, source in enumerate(legacy):
        grouped.input_layer.weight.data[type_index].copy_(source.state_mlp[0].weight)
        grouped.input_layer.bias.data[type_index].copy_(source.state_mlp[0].bias)
        grouped.state_layer.weight.data[type_index].copy_(source.state_mlp[2].weight)
        grouped.state_layer.bias.data[type_index].copy_(source.state_mlp[2].bias)
        grouped.pre_head.weight.data[type_index].copy_(source.pre_head.weight)
        grouped.pre_head.bias.data[type_index].copy_(source.pre_head.bias)


def _copy_post_heads(
    legacy: nn.ModuleList,
    grouped: GroupedFactorPostEditHead,
) -> None:
    for type_index, source in enumerate(legacy):
        grouped.hidden.weight.data[type_index].copy_(source.net[0].weight)
        grouped.hidden.bias.data[type_index].copy_(source.net[0].bias)
        grouped.output.weight.data[type_index].copy_(source.net[2].weight)
        grouped.output.bias.data[type_index].copy_(source.net[2].bias)


def test_factor_type_scan_reads_nested_parquet_lists(tmp_path: Path) -> None:
    path = tmp_path / "df_train.parquet"
    table = pa.table({"factor_types": pa.array([[2, 0], [2], [], [5, 0]])})
    pq.write_table(table, path)
    assert scan_factor_type_ids([path]) == (0, 2, 5)


def test_model_config_validates_explicit_active_mapping() -> None:
    cfg = ModelConfig.from_mapping(
        {
            "num_factor_types": 3,
            "active_factor_type_ids": [0, 2],
            "factor_executor_impl": "per_type_grouped_v2",
            "gold_edit_embedding_mode": "compact",
        }
    )
    assert cfg.active_factor_type_ids == ACTIVE_IDS
    with pytest.raises(ValueError, match="strictly increasing"):
        ModelConfig.from_mapping(
            {"num_factor_types": 3, "active_factor_type_ids": [2, 0]}
        )
    with pytest.raises(ValueError, match="address space"):
        ModelConfig.from_mapping(
            {"num_factor_types": 3, "active_factor_type_ids": [0, 3]}
        )
    shared = ModelConfig.from_mapping(
        {
            "num_factor_types": 3,
            "active_factor_type_ids": [0, 2],
            "factor_executor_impl": "shared_adapter_v1",
            "factor_adapter_rank": 4,
            "gold_edit_embedding_mode": "compact",
        }
    )
    assert shared.factor_adapter_rank == 4
    with pytest.raises(ValueError, match="factor_adapter_rank must be positive"):
        ModelConfig.from_mapping({"factor_adapter_rank": 0})


def test_compact_model_allocates_only_active_types_and_reachable_gold_ids() -> None:
    model = _compact_model()
    assert model.factor_type_ids_compact_to_stable.tolist() == [0, 2]
    assert model.factor_type_id_to_compact.tolist() == [0, -1, 1]
    assert model._num_factor_executor_modules == 2
    assert model.gold_edit_class_ids.tolist() == [0, 2, 5, 7]
    assert model._gold_edit_embeddings.num_embeddings == 4


def test_compact_model_routes_sparse_stable_ids_and_rejects_inactive_ids() -> None:
    model = _compact_model()
    outputs = model(_factor_graph(2))
    assert outputs["factor_logits_pre"] is not None
    assert outputs["factor_logits_post_gold"] is not None
    with pytest.raises(ValueError, match="absent from active_factor_type_ids"):
        model(_factor_graph(1))


def test_grouped_executor_and_post_match_legacy_modules_on_cpu() -> None:
    torch.manual_seed(7)
    input_dim = 11
    state_dim = 6
    legacy_executors = nn.ModuleList(
        [FactorTypeExecutor(input_dim, state_dim) for _ in ACTIVE_IDS]
    )
    grouped_executor = GroupedFactorTypeExecutor(
        len(ACTIVE_IDS), input_dim, state_dim
    )
    _copy_executor(legacy_executors, grouped_executor)

    compact_types = torch.tensor([1, 0, 1, 1, 0], dtype=torch.long)
    inputs = torch.randn(compact_types.numel(), input_dim)
    dispatch = build_grouped_dispatch(compact_types, num_types=len(ACTIVE_IDS))

    expected_states = torch.empty(compact_types.numel(), state_dim)
    expected_logits = torch.empty(compact_types.numel())
    for type_index, executor in enumerate(legacy_executors):
        mask = compact_types == type_index
        state, logit = executor(inputs[mask])
        expected_states[mask] = state
        expected_logits[mask] = logit
    actual_states, actual_logits = grouped_executor(inputs, dispatch)
    torch.testing.assert_close(actual_states, expected_states)
    torch.testing.assert_close(actual_logits, expected_logits)
    assert grouped_executor.last_backend == "segmented_linear"

    legacy_post = nn.ModuleList(
        [FactorPostEditHead(state_dim, state_dim) for _ in ACTIVE_IDS]
    )
    grouped_post = GroupedFactorPostEditHead(
        len(ACTIVE_IDS), state_dim, state_dim
    )
    _copy_post_heads(legacy_post, grouped_post)
    edits = torch.randn_like(actual_states)
    expected_post = torch.empty(compact_types.numel())
    for type_index, head in enumerate(legacy_post):
        mask = compact_types == type_index
        expected_post[mask] = head(actual_states[mask], edits[mask])
    actual_post = grouped_post(actual_states, edits, dispatch)
    torch.testing.assert_close(actual_post, expected_post)


def test_shared_adapter_executor_starts_from_shared_trunk_and_routes_type_heads() -> None:
    torch.manual_seed(13)
    executor = SharedAdapterFactorTypeExecutor(
        num_types=2,
        input_dim=7,
        state_dim=5,
        adapter_rank=3,
    )
    assert torch.count_nonzero(executor.adapter_up.weight) == 0
    assert torch.count_nonzero(executor.adapter_up.bias) == 0

    repeated = torch.randn(1, 7).repeat(2, 1)
    dispatch = build_grouped_dispatch(torch.tensor([0, 1]), num_types=2)
    states, logits = executor(repeated, dispatch)
    torch.testing.assert_close(states[0], states[1])
    assert logits.shape == (2,)

    loss = states.square().mean() + logits.square().mean()
    loss.backward()
    assert executor.input_layer.weight.grad is not None
    assert executor.adapter_up.weight.grad is not None
    assert executor.pre_head.weight.grad is not None


def test_shared_adapter_post_head_is_vectorized_and_trainable() -> None:
    torch.manual_seed(17)
    post = SharedAdapterFactorPostEditHead(
        num_types=2,
        state_dim=6,
        edit_dim=6,
        adapter_rank=2,
    )
    compact_types = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    dispatch = build_grouped_dispatch(compact_types, num_types=2)
    states = torch.randn(4, 6, requires_grad=True)
    edits = torch.randn(4, 6)
    logits = post(states, edits, dispatch)
    assert logits.shape == (4,)
    logits.square().mean().backward()
    assert post.hidden.weight.grad is not None
    assert post.adapter_up.weight.grad is not None
    assert post.output.weight.grad is not None


def test_shared_adapter_model_is_smaller_than_compact_per_type_model() -> None:
    per_type = _compact_model()
    shared = RepairGINFactorPressure(
        num_input_graph_nodes=16,
        num_embedding_size=8,
        num_layers=2,
        hidden=8,
        head_hidden=8,
        dropout=0.0,
        use_node_embeddings=True,
        use_edge_attributes=False,
        entity_class_ids=(0, 2, 7),
        predicate_class_ids=(0, 5),
        num_factor_types=3,
        active_factor_type_ids=ACTIVE_IDS,
        factor_executor_impl="shared_adapter_v1",
        factor_adapter_rank=2,
        gold_edit_embedding_mode="compact",
        pressure_enabled=True,
        pressure_module_sharing="shared",
    )
    per_type_factor_parameters = sum(
        parameter.numel()
        for name, parameter in per_type.named_parameters()
        if name.startswith(("_factor_executors", "_factor_post_heads"))
    )
    shared_factor_parameters = sum(
        parameter.numel()
        for name, parameter in shared.named_parameters()
        if name.startswith(("_factor_executors", "_factor_post_heads"))
    )
    assert shared_factor_parameters < per_type_factor_parameters
    outputs = shared(_factor_graph(2))
    assert outputs["factor_logits_pre"] is not None
    assert outputs["factor_logits_post_gold"] is not None
    assert shared.factor_dispatch_backend == "segmented_linear"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_executor_uses_bf16_grouped_mm_on_sm80_cuda() -> None:
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip("grouped_mm requires SM80 or newer")
    executor = GroupedFactorTypeExecutor(2, 13, 8).cuda()
    inputs = torch.randn(9, 13, device="cuda")
    compact_types = torch.tensor([1, 0, 1, 0, 0, 1, 1, 1, 0], device="cuda")
    dispatch = build_grouped_dispatch(compact_types, num_types=2)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        states, logits = executor(inputs, dispatch)
        loss = states.float().square().mean() + logits.float().square().mean()
    loss.backward()
    assert executor.last_backend == "grouped_mm_bf16"
    assert executor.input_layer.weight.grad is not None

    shared = SharedAdapterFactorTypeExecutor(2, 13, 8, adapter_rank=4).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        shared_states, shared_logits = shared(inputs, dispatch)
        shared_loss = (
            shared_states.float().square().mean()
            + shared_logits.float().square().mean()
        )
    shared_loss.backward()
    assert shared.last_backend == "segmented_linear"
    assert shared.adapter_up.weight.grad is not None

    aligned_shared = SharedAdapterFactorTypeExecutor(
        2,
        13,
        8,
        adapter_rank=16,
    ).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        aligned_states, aligned_logits = aligned_shared(inputs, dispatch)
        aligned_loss = (
            aligned_states.float().square().mean()
            + aligned_logits.float().square().mean()
        )
    aligned_loss.backward()
    assert aligned_shared.last_backend == "grouped_mm_bf16"
    assert aligned_shared.adapter_up.weight.grad is not None


def test_compact_gold_embedding_matches_full_table_for_reachable_ids() -> None:
    torch.manual_seed(11)
    full = _compact_model(gold_mode="full")
    compact = _compact_model(gold_mode="compact")
    compact._gold_edit_embeddings.weight.data.copy_(
        full._gold_edit_embeddings.weight.data.index_select(
            0, compact.gold_edit_class_ids
        )
    )
    targets = torch.tensor(
        [
            [0, 5, 7, 2, 0, 7],
            [2, 0, 0, 7, 5, 2],
        ],
        dtype=torch.long,
    )
    graph_index = torch.tensor([0, 1, 1], dtype=torch.long)
    expected = full._gold_edit_representation(targets, graph_index)
    actual = compact._gold_edit_representation(targets, graph_index)
    assert expected is not None and actual is not None
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="compact target vocabulary"):
        compact._gold_edit_representation(
            torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )


def test_legacy_v1_state_dict_remains_strictly_loadable() -> None:
    kwargs = {
        "num_input_graph_nodes": 16,
        "num_embedding_size": 8,
        "num_layers": 2,
        "hidden": 8,
        "head_hidden": 8,
        "dropout": 0.0,
        "use_node_embeddings": True,
        "use_edge_attributes": False,
        "entity_class_ids": (0, 2, 7),
        "predicate_class_ids": (0, 5),
        "num_factor_types": 3,
        "factor_executor_impl": "per_type_v1",
        "gold_edit_embedding_mode": "full",
        "pressure_enabled": True,
    }
    original = RepairGINFactorPressure(**kwargs)
    restored = RepairGINFactorPressure(**kwargs)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert not any(
        key.startswith(("factor_type_id_to_compact", "gold_edit_class_ids"))
        for key in original.state_dict()
    )
