from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.baselines import BaselineAdapter, DeleteFocusBaseline
from modules.data_encoders import dataset_variant_name
from modules.repair_eval import load_global_eval_rows


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


EVAL_MODULE = _load_module(ROOT / "src" / "09_eval.py", "eval_09_for_test")
TRAIN_MODULE = _load_module(ROOT / "src" / "07_train.py", "train_07_for_test")
_resolve_baseline_interim_paths = EVAL_MODULE._resolve_baseline_interim_paths
load_baseline_split_from_parquet = EVAL_MODULE.load_baseline_split_from_parquet
_strict_global_requires_factor_fields = EVAL_MODULE._strict_global_requires_factor_fields
GlobalMetricsSupport = EVAL_MODULE.GlobalMetricsSupport
evaluate = EVAL_MODULE.eval
_load_identity_to_feature_mapping = TRAIN_MODULE._load_identity_to_feature_mapping


def test_baseline_resolution_prefers_labeled_and_loads_factor_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        variant = dataset_variant_name("full", 100)
        base_dir = root / "data" / "interim" / variant
        labeled_dir = root / "data" / "interim" / f"{variant}_labeled"
        base_dir.mkdir(parents=True, exist_ok=True)
        labeled_dir.mkdir(parents=True, exist_ok=True)

        row = {
            "add_subject": 1,
            "add_predicate": 2,
            "add_object": 3,
            "del_subject": 4,
            "del_predicate": 5,
            "del_object": 6,
            "add_subject_feature": 11,
            "add_predicate_feature": 12,
            "add_object_feature": 99,
            "del_subject_feature": 14,
            "del_predicate_feature": 15,
            "del_object_feature": 16,
            "subject": 7,
            "predicate": 8,
            "object": 9,
            "constraint_id": 17,
            "constraint_type": "single_value",
            "factor_constraint_ids": [17, 21],
            "factor_types": [0, 3],
            "factor_checkable_pre": [True, True],
            "factor_satisfied_pre": [0, 1],
            "factor_checkable_post_gold": [True, False],
            "factor_satisfied_post_gold": [1, 0],
            "primary_factor_index": 0,
        }
        pd.DataFrame([row]).to_parquet(labeled_dir / "df_train.parquet", index=False)
        pd.DataFrame([row]).to_parquet(labeled_dir / "df_test.parquet", index=False)

        _write_text(base_dir / "globalintencoder.txt", "subject\t1\n")
        _write_text(labeled_dir / "globalintencoder.txt", "subject\t11\n")

        with _pushd(root):
            data_path, encoder_path = _resolve_baseline_interim_paths("full", 100)
            assert data_path.resolve() == labeled_dir.resolve()
            assert encoder_path.resolve() == (labeled_dir / "globalintencoder.txt").resolve()

            graphs, max_index = load_baseline_split_from_parquet(
                data_path,
                "test",
                unknown_feature_id=99,
            )

        assert len(graphs) == 1
        graph = graphs[0]
        assert graph.factor_constraint_ids.tolist() == [17, 21]
        assert graph.factor_types.tolist() == [0, 3]
        assert graph.factor_checkable_pre.tolist() == [True, True]
        assert graph.factor_satisfied_pre.tolist() == [0, 1]
        assert graph.factor_checkable_post_gold.tolist() == [True, False]
        assert graph.factor_satisfied_post_gold.tolist() == [1, 0]
        assert graph.primary_factor_index == 0
        assert graph.y_identity.tolist() == [[1, 2, 3, 4, 5, 6]]
        assert graph.y_feature.tolist() == [[11, 12, 99, 14, 15, 16]]
        assert graph.target_representable_mask.tolist() == [[True, True, False, True, True, True]]
        assert max_index == 9


def test_training_loads_identity_to_feature_mapping(tmp_path: Path) -> None:
    expected = np.asarray([0, 3, 7, 2], dtype=np.int64)
    np.save(tmp_path / "identity_to_feature.npy", expected)

    loaded = _load_identity_to_feature_mapping(tmp_path)

    assert loaded is not None
    assert np.array_equal(loaded, expected)
    assert _load_identity_to_feature_mapping(tmp_path / "missing") is None


def test_baseline_resolution_prefers_factor_labeled_base_and_identity_encoder() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        variant = dataset_variant_name("full", 100)
        base_dir = root / "data" / "interim" / variant
        labeled_dir = root / "data" / "interim" / f"{variant}_labeled"
        base_dir.mkdir(parents=True, exist_ok=True)
        labeled_dir.mkdir(parents=True, exist_ok=True)

        base_row = {
            "factor_constraint_ids": [17],
            "factor_checkable_pre": [True],
            "factor_satisfied_pre": [False],
        }
        stale_row = {"factor_constraint_ids": [99]}
        for split in ("train", "test"):
            pd.DataFrame([base_row]).to_parquet(base_dir / f"df_{split}.parquet", index=False)
            pd.DataFrame([stale_row, stale_row]).to_parquet(
                labeled_dir / f"df_{split}.parquet", index=False
            )

        _write_text(base_dir / "globalintencoder.txt", "subject\t1\n")
        _write_text(base_dir / "identity_encoder.txt", "subject\t2\n")

        with _pushd(root):
            data_path, resolved_encoder_path = _resolve_baseline_interim_paths("full", 100)
            assert data_path.resolve() == base_dir.resolve()
            assert resolved_encoder_path.resolve() == (base_dir / "identity_encoder.txt").resolve()


def test_baseline_resolution_falls_back_to_unlabeled() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        variant = dataset_variant_name("full", 100)
        base_dir = root / "data" / "interim" / variant
        base_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "add_subject": 0,
                    "add_predicate": 0,
                    "add_object": 0,
                    "del_subject": 0,
                    "del_predicate": 0,
                    "del_object": 0,
                    "subject": 0,
                    "predicate": 0,
                    "object": 0,
                    "constraint_type": "conflict_with",
                }
            ]
        ).to_parquet(base_dir / "df_train.parquet", index=False)
        _write_text(base_dir / "globalintencoder.txt", "subject\t1\n")

        with _pushd(root):
            data_path, encoder_path = _resolve_baseline_interim_paths("full", 100)
            assert data_path.resolve() == base_dir.resolve()
            assert encoder_path.resolve() == (base_dir / "globalintencoder.txt").resolve()


def test_baseline_adapter_returns_direct_identity_indices_and_feature_metrics() -> None:
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([[0, 0, 0, 7, 8, 9]], dtype=torch.long),
    )
    graph.y_identity = graph.y.clone()
    graph.y_feature = torch.tensor([[0, 0, 0, 2, 3, 4]], dtype=torch.long)
    graph.focus_triple = torch.tensor([7, 8, 9], dtype=torch.long)
    graph.constraint_type = "single"

    model = BaselineAdapter(DeleteFocusBaseline(num_graph_nodes=10_000_000))
    direct = model([graph])

    assert tuple(direct.shape) == (1, 6)
    assert direct.tolist() == [[0, 0, 0, 7, 8, 9]]
    assert sum(parameter.numel() for parameter in model.parameters()) == 0

    metrics = evaluate(
        model,
        [graph],
        batch_size=1,
        device="cpu",
        identity_to_feature=[0, 1, 1, 1, 1, 1, 1, 2, 3, 4],
    )
    assert metrics["micro_f1"] == 1.0
    assert metrics["feature_space_micro"]["f1"] == 1.0


def test_baseline_main_keeps_encoder_resolver_callable(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "interim"
    data_path.mkdir()
    identity_encoder_path = data_path / "identity_encoder.txt"

    monkeypatch.setattr(
        EVAL_MODULE,
        "_resolve_baseline_interim_paths",
        lambda dataset, min_occurrence: (data_path, identity_encoder_path),
    )
    monkeypatch.setattr(EVAL_MODULE, "_load_identity_to_feature", lambda path: None)
    monkeypatch.setattr(EVAL_MODULE, "_load_encoder_token_id", lambda path, token: None)
    monkeypatch.setattr(
        EVAL_MODULE,
        "load_baseline_split_from_parquet",
        lambda path, split, unknown_feature_id=None: ([], 0),
    )
    monkeypatch.setattr(EVAL_MODULE, "load_placeholder_ids", lambda path: {})
    monkeypatch.setattr(EVAL_MODULE, "_maybe_prepare_repair_support", lambda *args, **kwargs: None)
    monkeypatch.setattr(EVAL_MODULE, "evaluate_baselines", lambda **kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["09_eval.py", "--run-baselines", "--dataset", "full", "--no-global-metrics"],
    )

    EVAL_MODULE.main()


def test_strict_global_factor_field_requirement_is_representation_aware() -> None:
    passive_cfg = EVAL_MODULE.ModelConfig.from_mapping({"constraint_representation": "eswc_passive"})
    factorized_cfg = EVAL_MODULE.ModelConfig.from_mapping({"constraint_representation": "factorized"})

    assert _strict_global_requires_factor_fields(passive_cfg) is False
    assert _strict_global_requires_factor_fields(factorized_cfg) is True


def test_global_support_ignores_passive_graph_factor_ids_without_labels() -> None:
    graph = Data(x=torch.zeros((1,), dtype=torch.long))
    graph.factor_constraint_ids = torch.tensor([123], dtype=torch.long)
    graph.primary_factor_index = 0

    calls: list[dict[str, object]] = []

    class DummyEvaluator:
        def evaluate_full(self, row, *, candidate_slots, primary_factor_index=None, factor_constraint_ids=None):
            calls.append(
                {
                    "primary_factor_index": primary_factor_index,
                    "factor_constraint_ids": factor_constraint_ids,
                }
            )
            return {
                "local_constraint_ids": [123, 456],
                "primary_factor_index": 0,
                "pre_checkable": [True, True],
                "pre_satisfied": [1, 1],
                "post_checkable": [True, True],
                "post_satisfied": [1, 0],
            }

    support = GlobalMetricsSupport(rows=[object()], evaluator=DummyEvaluator())
    postprocess, state = support.build_postprocess([graph])
    predictions = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
    targets = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)

    postprocess(predictions, targets, ["single"])

    assert calls == [{"primary_factor_index": None, "factor_constraint_ids": None}]
    overall = state["global_metrics"]["overall"]
    assert overall["srr_denom_total"] == 1
    assert overall["srr_total"] == 1


def test_global_eval_rows_preserve_declared_correction_transition(
    tmp_path: Path,
) -> None:
    row = {
        "constraint_id": 17,
        "constraint_type": "single",
        "add_subject": 1,
        "add_predicate": 2,
        "add_object": 3,
        "del_subject": 4,
        "del_predicate": 5,
        "del_object": 6,
    }
    pd.DataFrame([row]).to_parquet(tmp_path / "df_test.parquet", index=False)

    loaded = load_global_eval_rows(tmp_path, "test")

    assert len(loaded) == 1
    assert (
        loaded[0].add_subject,
        loaded[0].add_predicate,
        loaded[0].add_object,
        loaded[0].del_subject,
        loaded[0].del_predicate,
        loaded[0].del_object,
    ) == (1, 2, 3, 4, 5, 6)

    pd.DataFrame([{key: value for key, value in row.items() if key != "add_object"}]).to_parquet(
        tmp_path / "df_test.parquet",
        index=False,
    )
    with pytest.raises(ValueError, match="correction-transition columns"):
        load_global_eval_rows(tmp_path, "test")


def test_strict_global_support_uses_parquet_pre_labels_for_passive_graphs() -> None:
    graph = Data(x=torch.zeros((1,), dtype=torch.long))
    graph.factor_constraint_ids = torch.tensor([123, 456], dtype=torch.long)
    graph.primary_factor_index = 0
    row = SimpleNamespace(
        factor_constraint_ids=[123, 456],
        factor_checkable_pre=[True, True],
        factor_satisfied_pre=[0, 1],
        primary_factor_index=0,
    )

    calls: list[dict[str, object]] = []

    class DummyEvaluator:
        def evaluate_full(
            self,
            row,
            *,
            candidate_slots,
            primary_factor_index=None,
            factor_constraint_ids=None,
        ):
            del row, candidate_slots
            calls.append(
                {
                    "primary_factor_index": primary_factor_index,
                    "factor_constraint_ids": factor_constraint_ids,
                }
            )
            return {
                "local_constraint_ids": [123, 456],
                "primary_factor_index": 0,
                "pre_checkable": [True, True],
                "pre_satisfied": [0, 1],
                "post_checkable": [True, True],
                "post_satisfied": [1, 1],
            }

    support = GlobalMetricsSupport(
        rows=[row],
        evaluator=DummyEvaluator(),
        require_pre_state_labels=True,
    )
    postprocess, state = support.build_postprocess([graph])
    predictions = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
    postprocess(predictions, predictions, ["single"])

    assert calls == [
        {
            "primary_factor_index": 0,
            "factor_constraint_ids": [123, 456],
        }
    ]
    global_metrics = state["global_metrics"]
    assert global_metrics["overall"]["primary_fix_denom_total"] == 1
    assert global_metrics["pre_state_validation"] == {
        "checked_rows": 1,
        "mismatch_count": 0,
        "source_counts": {"parquet": 1},
    }


def test_global_support_rejects_pre_state_semantic_mismatch() -> None:
    graph = Data(x=torch.zeros((1,), dtype=torch.long))
    row = SimpleNamespace(
        factor_constraint_ids=[123],
        factor_checkable_pre=[True],
        factor_satisfied_pre=[0],
        primary_factor_index=0,
    )

    class DummyEvaluator:
        def evaluate_full(self, *args, **kwargs):
            del args, kwargs
            return {
                "local_constraint_ids": [123],
                "primary_factor_index": 0,
                "pre_checkable": [True],
                "pre_satisfied": [1],
                "post_checkable": [True],
                "post_satisfied": [1],
            }

    support = GlobalMetricsSupport(
        rows=[row],
        evaluator=DummyEvaluator(),
        require_pre_state_labels=True,
    )
    postprocess, _ = support.build_postprocess([graph])
    predictions = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)

    with pytest.raises(AssertionError, match="PRE-state semantic mismatch"):
        postprocess(predictions, predictions, ["single"])


def test_make_experiment_configs_empty_processed_root_message() -> None:
    module = _load_module(ROOT / "scripts" / "make_experiment_configs.py", "make_experiment_configs_for_test")
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_root = Path(tmpdir) / "processed"
        processed_root.mkdir(parents=True, exist_ok=True)

        argv_backup = list(sys.argv)
        sys.argv = [
            "make_experiment_configs.py",
            "--processed-root",
            str(processed_root),
        ]
        try:
            try:
                module.main()
            except SystemExit as exc:
                message = str(exc)
            else:
                raise AssertionError("Expected SystemExit for empty processed root")
        finally:
            sys.argv = argv_backup

    assert "No graph artifacts found under" in message
    assert "src/02b_stratified_benchmark_sampler.py --source-dataset full --output-dataset full_strat1m --min-occurrence 100" in message
    assert "src/05_constraint_labeler.py --dataset full_strat1m --min-occurrence 100 --constraint-scope local --registry-dataset full --factor-family-policy supported_only" in message
    assert "src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-scope local --constraint-representation factorized --registry-dataset full" in message
    assert "src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive --registry-dataset full" in message


if __name__ == "__main__":
    test_baseline_resolution_prefers_labeled_and_loads_factor_fields()
    test_baseline_resolution_falls_back_to_unlabeled()
    test_strict_global_factor_field_requirement_is_representation_aware()
    test_global_support_ignores_passive_graph_factor_ids_without_labels()
    test_make_experiment_configs_empty_processed_root_message()
    print("paper run readiness tests passed")
