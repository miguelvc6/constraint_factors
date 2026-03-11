from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.data_encoders import GraphStreamDataset, graph_dataset_filename
from modules.training_utils import load_graph_dataset


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_graph(label: int, constraint_type: str) -> Data:
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([[label, 0, 0, 0, 0, 0]], dtype=torch.long),
    )
    graph.constraint_type = constraint_type
    return graph


def _write_torch_shards(base_path: Path, shards: list[list[Data]]) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(shards):
        shard_path = base_path.with_name(f"{base_path.stem}-shard{idx:03d}.pt")
        torch.save(shard, shard_path)
    manifest_path = base_path.with_suffix(base_path.suffix + ".manifest.json")
    manifest_path.write_text(
        f'{{"graph_count": {sum(len(shard) for shard in shards)}, "sharded": true}}',
        encoding="utf-8",
    )


def test_load_graph_dataset_streams_torch_shards() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base_path = root / graph_dataset_filename("train", "node_id")
        _write_torch_shards(
            base_path,
            [
                [_make_graph(1, "single"), _make_graph(2, "single")],
                [_make_graph(3, "conflictWith")],
            ],
        )

        dataset = load_graph_dataset(base_path)

        assert isinstance(dataset, GraphStreamDataset)
        context_indices = [int(getattr(graph, "context_index")) for graph in dataset]
        labels = [int(graph.y[0, 0].item()) for graph in dataset]

    assert context_indices == [0, 1, 2]
    assert labels == [1, 2, 3]


def test_eval_accepts_sharded_test_split_without_monolith() -> None:
    eval_module = _load_module(ROOT / "src" / "09_eval.py", "eval_09_shard_test")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        processed_dir = root / "data" / "processed" / "full_minocc100"
        base_path = processed_dir / graph_dataset_filename("test", "node_id")
        graphs = [
            _make_graph(1, "single"),
            _make_graph(2, "single"),
            _make_graph(3, "conflictWith"),
        ]
        _write_torch_shards(base_path, [graphs[:2], graphs[2:]])

        dataset = eval_module.load_split(processed_dir, "node_id", "test")
        predictions = torch.cat([graph.y for graph in graphs], dim=0)
        metrics = eval_module.eval(
            None,
            dataset,
            precomputed_predictions=predictions,
        )

    assert "micro_f1" in metrics
    assert metrics["support_per_constraint_type"] == {"single": 2, "conflictWith": 1}


if __name__ == "__main__":
    test_load_graph_dataset_streams_torch_shards()
    test_eval_accepts_sharded_test_split_without_monolith()
    print("sharded graph ingestion tests passed")
