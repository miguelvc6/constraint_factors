from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_09_primary_query_test", ROOT / "src" / "09_eval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load src/09_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_metrics_support_prefers_eval_factor_fields() -> None:
    eval_module = _load_eval_module()
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long),
    )
    graph.constraint_type = "single"
    graph.factor_constraint_ids = torch.tensor([202], dtype=torch.long)
    graph.factor_satisfied_pre = torch.tensor([1], dtype=torch.long)
    graph.factor_checkable_pre = torch.tensor([True], dtype=torch.bool)
    graph.primary_factor_index = -1
    graph.eval_factor_constraint_ids = torch.tensor([101, 202], dtype=torch.long)
    graph.eval_factor_satisfied_pre = torch.tensor([0, 1], dtype=torch.long)
    graph.eval_factor_checkable_pre = torch.tensor([True, True], dtype=torch.bool)
    graph.eval_primary_factor_index = 0

    calls: list[dict[str, object]] = []

    class DummyEvaluator:
        def evaluate_full(self, row, *, candidate_slots, primary_factor_index=None, factor_constraint_ids=None):
            calls.append(
                {
                    "primary_factor_index": primary_factor_index,
                    "factor_constraint_ids": list(torch.as_tensor(factor_constraint_ids).view(-1).tolist()),
                }
            )
            return {
                "local_constraint_ids": [101, 202],
                "primary_factor_index": int(primary_factor_index),
                "pre_checkable": [True, True],
                "pre_satisfied": [0, 1],
                "post_checkable": [True, True],
                "post_satisfied": [1, 1],
            }

    support = eval_module.GlobalMetricsSupport(rows=[object()], evaluator=DummyEvaluator())
    postprocess, state = support.build_postprocess([graph])
    predictions = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
    targets = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
    postprocess(predictions, targets, ["single"])

    assert calls == [{"primary_factor_index": 0, "factor_constraint_ids": [101, 202]}]
    assert "overall" in state["global_metrics"]


if __name__ == "__main__":
    test_global_metrics_support_prefers_eval_factor_fields()
    print("primary query eval support tests passed")
