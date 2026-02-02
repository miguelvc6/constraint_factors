#!/usr/bin/env python3
"""
Smoke test for factor node selection under batching.
Run: python tests/test_factor_batching.py
"""

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.models import _select_factor_nodes


def _make_graph(node_ids: list[int], factor_local_ids: list[int], primary_index: int) -> Data:
    x = torch.tensor(node_ids, dtype=torch.float32).view(-1, 1)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    is_factor = torch.zeros(len(node_ids), dtype=torch.bool)
    is_factor[factor_local_ids] = True
    data.is_factor_node = is_factor
    data.factor_node_index = torch.tensor(factor_local_ids, dtype=torch.long)
    data.factor_constraint_ids = torch.arange(len(factor_local_ids), dtype=torch.long)
    data.factor_satisfied_pre = torch.ones(len(factor_local_ids), dtype=torch.long)
    data.factor_checkable_pre = torch.ones(len(factor_local_ids), dtype=torch.bool)
    data.primary_factor_index = int(primary_index)
    return data


def run_smoke_test() -> None:
    g1 = _make_graph([10, 11, 12, 13], [1, 3], primary_index=0)
    g2 = _make_graph([20, 21, 22], [0, 2], primary_index=1)
    batch = Batch.from_data_list([g1, g2])

    node_emb = batch.x
    factor_emb, factor_graph_index = _select_factor_nodes(node_emb, batch)
    assert factor_emb is not None
    assert factor_graph_index is not None

    expected_global = []
    ptr = batch.ptr.tolist()
    for graph_id, graph in enumerate([g1, g2]):
        for local_id in graph.factor_node_index.tolist():
            expected_global.append(local_id + ptr[graph_id])
    expected_values = batch.x[torch.tensor(expected_global)]

    assert factor_emb.shape == expected_values.shape, "factor embedding shape mismatch"
    assert torch.allclose(factor_emb, expected_values), "factor ordering does not match labels"

    expected_graph_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    assert torch.equal(factor_graph_index.cpu(), expected_graph_index), "factor graph index mismatch"


if __name__ == "__main__":
    run_smoke_test()
    print("factor batching smoke test passed")
