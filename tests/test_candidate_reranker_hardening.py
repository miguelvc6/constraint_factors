from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.candidates import CandidateConfig, build_inference_candidates
from modules.repair_eval import CandidateRepairs, ViolationContext


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_module(ROOT / "src" / "09_eval.py", "eval_09_candidate_hardening")
RERANKER = _load_module(ROOT / "src" / "08_train_reranker.py", "train_08_candidate_hardening")


class EmptyHeuristics:
    placeholder_ids: dict[str, int] = {}

    def candidates_for(self, context: ViolationContext) -> CandidateRepairs:
        del context
        return CandidateRepairs()


def _context() -> ViolationContext:
    return ViolationContext(
        constraint_type="single",
        constraint_id=10,
        subject=2,
        predicate=3,
        object=4,
        other_subject=0,
        other_predicate=0,
        other_object=0,
        constraint_predicates=(),
        constraint_objects=(),
    )


def _ambiguous_logits() -> torch.Tensor:
    logits = torch.zeros((6, 2), dtype=torch.float32)
    logits[:, 1] = 10.0
    return logits


def _fallback_cfg() -> CandidateConfig:
    return CandidateConfig(
        topk_candidates=1,
        topk_per_slot=1,
        heuristic_max_candidates=1,
        max_candidates_total=4,
    )


def test_empty_inference_pool_returns_only_fallback_noop() -> None:
    candidates = build_inference_candidates(
        context=_context(),
        heuristics=EmptyHeuristics(),
        proposal_logits=_ambiguous_logits(),
        cfg=_fallback_cfg(),
        placeholder_ids=set(),
        num_target_ids=2,
        identity_to_feature=[0, 1, 1],
    )

    assert len(candidates) == 1
    assert candidates[0].identity_slots == (0, 0, 0, 0, 0, 0)
    assert candidates[0].feature_slots == (0, 0, 0, 0, 0, 0)
    assert candidates[0].source == "fallback_noop"


class FallbackSelectionModel(nn.Module):
    num_target_ids = 2

    def forward(self, data):
        batch_size = int(data.num_graphs)
        logits = _ambiguous_logits().unsqueeze(0).expand(batch_size, -1, -1).clone()
        policy_logits = torch.zeros((batch_size, 6), dtype=torch.float32)
        policy_logits[:, 0] = 1.0
        return {
            "edit_logits": logits,
            "graph_emb": torch.zeros((batch_size, 1), dtype=torch.float32),
            "policy_logits": policy_logits,
        }

    def score_candidates(self, graph_emb, candidates):
        del graph_emb
        return torch.zeros(candidates.size(0), dtype=torch.float32)


def _eval_graph() -> Data:
    graph = Data(
        x=torch.tensor([1], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.zeros((1, 6), dtype=torch.long),
    )
    graph.y_identity = graph.y.clone()
    graph.y_feature = graph.y.clone()
    graph.target_representable_mask = torch.ones((1, 6), dtype=torch.bool)
    graph.constraint_type = "single"
    graph.context_index = 0
    return graph


@pytest.mark.parametrize("mode", ["chooser", "direct_safety", "policy"])
def test_evaluation_modes_select_fallback_noop_and_count_it(mode: str) -> None:
    common = {
        "contexts": [_context()],
        "heuristics": EmptyHeuristics(),
        "candidate_cfg": _fallback_cfg(),
        "identity_to_feature": [0, 1, 1],
    }
    kwargs = {}
    if mode == "chooser":
        kwargs["chooser_support"] = EVAL.ChooserSupport(**common)
    elif mode == "direct_safety":
        kwargs["direct_safety_support"] = EVAL.DirectSafetySupport(**common)
    else:
        kwargs["policy_support"] = EVAL.PolicySupport(**common)

    metrics = EVAL.eval(
        FallbackSelectionModel(),
        [_eval_graph()],
        batch_size=1,
        device="cpu",
        identity_to_feature=[0, 1, 1],
        **kwargs,
    )

    assert metrics["global_counts"] == {"tp": 0, "fp": 0, "fn": 0}
    assert metrics["fallback_noop_count"] == 1
    assert metrics["candidate_inference_rows"] == 1
    assert metrics["fallback_noop_rate"] == 1.0


class ProposalModel(nn.Module):
    def __init__(self, vocab_size: int = 5, top_id: int = 1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.top_id = top_id

    def forward(self, data):
        logits = torch.zeros((int(data.num_graphs), 6, self.vocab_size), device=data.x.device)
        logits[:, :, self.top_id] = 5.0
        return {"edit_logits": logits}


class TinyReranker(nn.Module):
    def __init__(self, num_target_ids: int = 5) -> None:
        super().__init__()
        self.num_target_ids = num_target_ids
        self.weight = nn.Parameter(torch.tensor(0.25))

    def encode_graphs(self, batch):
        return self.weight.expand(int(batch.num_graphs), 1)

    def score_candidates(self, graph_emb, candidates):
        scale = torch.arange(
            1,
            candidates.size(0) + 1,
            dtype=graph_emb.dtype,
            device=graph_emb.device,
        )
        return graph_emb.view(-1)[0] * scale


class RecordingEvaluator:
    def __init__(self) -> None:
        self.candidate_sets: list[list[tuple[int, ...]]] = []

    def evaluate_candidate_metrics(self, row, *, candidates, primary_factor_index):
        del row, primary_factor_index
        self.candidate_sets.append([tuple(candidate) for candidate in candidates])
        return [
            SimpleNamespace(
                primary_satisfied=int(index == 0),
                global_satisfied_fraction=1.0 - index / max(len(candidates), 1),
                secondary_regressions=index,
                srr=float(index > 0),
            )
            for index, _ in enumerate(candidates)
        ]


def _reranker_graph(index: int, gold: tuple[int, ...]) -> Data:
    graph = Data(
        x=torch.tensor([1], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([gold], dtype=torch.long),
    )
    graph.y_identity = graph.y.clone()
    graph.context_index = index
    graph.primary_factor_index = 0
    return graph


def _training_cfg(*, objective: str = "main"):
    return RERANKER.RerankerTrainingConfig(
        batch_size=2,
        objective=objective,
        topk_candidates=1,
        topk_per_slot=1,
        heuristic_max_candidates=1,
        max_candidates_total=4,
        include_gold=True,
    )


def test_reranker_training_forces_gold_but_validation_uses_inference_candidates(monkeypatch) -> None:
    calls: list[str] = []
    original_training = RERANKER.build_training_candidates
    original_inference = RERANKER.build_inference_candidates

    def track_training(**kwargs):
        calls.append("training")
        return original_training(**kwargs)

    def track_inference(**kwargs):
        calls.append("inference")
        return original_inference(**kwargs)

    monkeypatch.setattr(RERANKER, "build_training_candidates", track_training)
    monkeypatch.setattr(RERANKER, "build_inference_candidates", track_inference)
    model = TinyReranker()
    evaluator = RecordingEvaluator()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_graph = _reranker_graph(0, (2, 2, 2, 0, 0, 0))
    RERANKER._run_epoch(
        model=model,
        proposal_model=ProposalModel(),
        loader=DataLoader([train_graph], batch_size=1),
        contexts=[_context()],
        rows=[object()],
        heuristics=EmptyHeuristics(),
        evaluator=evaluator,
        device=torch.device("cpu"),
        cfg=_training_cfg(),
        identity_to_feature=None,
        optimizer=optimizer,
    )
    assert "training" in calls
    assert (2, 2, 2, 0, 0, 0) in evaluator.candidate_sets[0]

    calls.clear()
    evaluator = RecordingEvaluator()
    val_graphs = [
        _reranker_graph(0, (1, 1, 1, 0, 0, 0)),
        _reranker_graph(1, (2, 2, 2, 0, 0, 0)),
    ]
    _, metrics = RERANKER._run_epoch(
        model=model,
        proposal_model=ProposalModel(),
        loader=DataLoader(val_graphs, batch_size=2),
        contexts=[_context(), _context()],
        rows=[object(), object()],
        heuristics=EmptyHeuristics(),
        evaluator=evaluator,
        device=torch.device("cpu"),
        cfg=_training_cfg(),
        identity_to_feature=None,
    )
    assert calls == ["inference", "inference"]
    assert all((2, 2, 2, 0, 0, 0) not in candidates for candidates in evaluator.candidate_sets)
    assert metrics["gold_candidate_coverage"] == pytest.approx(0.5)
    assert metrics["loss_row_count"] == 1.0
    assert metrics["row_count"] == 2.0


def test_global_fix_validation_scores_every_inference_row() -> None:
    graphs = [
        _reranker_graph(0, (4, 4, 4, 0, 0, 0)),
        _reranker_graph(1, (3, 3, 3, 0, 0, 0)),
    ]
    evaluator = RecordingEvaluator()
    loss, metrics = RERANKER._run_epoch(
        model=TinyReranker(),
        proposal_model=ProposalModel(),
        loader=DataLoader(graphs, batch_size=2),
        contexts=[_context(), _context()],
        rows=[object(), object()],
        heuristics=EmptyHeuristics(),
        evaluator=evaluator,
        device=torch.device("cpu"),
        cfg=_training_cfg(objective="global_fix"),
        identity_to_feature=None,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert len(evaluator.candidate_sets) == 2
    assert metrics["row_count"] == 2.0
    assert metrics["loss_row_count"] == 2.0
    assert metrics["loss_row_coverage"] == 1.0


def test_main_validation_fails_when_natural_gold_coverage_is_zero() -> None:
    graph = _reranker_graph(0, (4, 4, 4, 0, 0, 0))
    with pytest.raises(RuntimeError, match="No validation row"):
        RERANKER._run_epoch(
            model=TinyReranker(),
            proposal_model=ProposalModel(),
            loader=DataLoader([graph], batch_size=1),
            contexts=[_context()],
            rows=[object()],
            heuristics=EmptyHeuristics(),
            evaluator=RecordingEvaluator(),
            device=torch.device("cpu"),
            cfg=_training_cfg(),
            identity_to_feature=None,
        )


def test_reranker_inference_selects_fallback_noop() -> None:
    graph = _reranker_graph(0, (4, 4, 4, 0, 0, 0))
    model = TinyReranker(num_target_ids=2)
    predictions, diagnostics = RERANKER._predict_reranker_edits(
        model=model,
        proposal_model=ProposalModel(vocab_size=2),
        data=[graph],
        contexts=[_context()],
        rows=[object()],
        heuristics=EmptyHeuristics(),
        evaluator=RecordingEvaluator(),
        device=torch.device("cpu"),
        cfg=RERANKER.RerankerTrainingConfig(
            batch_size=1,
            topk_candidates=1,
            topk_per_slot=1,
            heuristic_max_candidates=1,
            max_candidates_total=4,
        ),
        identity_to_feature=[0, 1, 1],
    )

    assert predictions == [{"add": [0, 0, 0], "del": [0, 0, 0]}]
    assert diagnostics == {
        "candidate_inference_rows": 1,
        "fallback_noop_count": 1,
        "fallback_noop_rate": 1.0,
    }
