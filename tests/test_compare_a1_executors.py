from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_a1_executors",
    ROOT / "scripts" / "compare_a1_executors.py",
)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def _config(executor: str) -> dict:
    return {
        "model_config": {
            "dataset_variant": "toy",
            "encoding": "node_id",
            "model": "GIN_PRESSURE",
            "factor_executor_impl": executor,
            "factor_adapter_rank": 16,
        },
        "training_config": {"num_epochs": 15},
    }


def _manifest() -> dict:
    return {
        "seed": 42,
        "dataset_manifest": {"path": "dataset.json", "sha256": "dataset-hash"},
        "graph_manifests": [
            {"path": "train.manifest.json", "sha256": "train-hash"},
            {"path": "val.manifest.json", "sha256": "val-hash"},
        ],
        "parameters": {"total": 100, "trainable": 90},
    }


def _write_run(path: Path, *, executor: str, metric: float, parameters: int) -> None:
    path.mkdir(parents=True)
    manifest = _manifest()
    manifest["parameters"]["total"] = parameters
    manifest["parameters"]["trainable"] = parameters
    (path / "config.json").write_text(json.dumps(_config(executor)), encoding="utf-8")
    (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "training_history.json").write_text(
        json.dumps(
            {
                "val_loss": [0.8, 0.5],
                "epoch_seconds": [10.0, 12.0],
                "peak_cuda_allocated_bytes": parameters * 2,
                "factor_dispatch_backend": "segmented_linear",
            }
        ),
        encoding="utf-8",
    )
    evaluations = path / "evaluations"
    h2 = evaluations / "h2"
    h2.mkdir(parents=True)
    (evaluations / "model.json").write_text(
        json.dumps(
            {
                "micro_f1": metric,
                "model_selection": {"primary_fix_rate": metric},
                "per_constraint_type": {
                    "single": {"micro": {"f1": metric}},
                },
                "global_metrics_per_constraint_type": {
                    "single": {"primary_fix_rate": metric},
                },
                "global_metrics": {"overall": {"primary_fix_rate": metric}},
            }
        ),
        encoding="utf-8",
    )
    (h2 / "h2_report.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "overall": [{"variant": "normal", "micro_f1": metric}],
                "factor_semantics": [
                    {
                        "state": "pre",
                        "factor_family": "single",
                        "factor_type": "single",
                        "f1": metric,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_comparison_validates_contract_and_writes_deltas(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _write_run(
        reference,
        executor="per_type_grouped_v2",
        metric=0.6,
        parameters=100,
    )
    _write_run(
        candidate,
        executor="shared_adapter_v1",
        metric=0.7,
        parameters=70,
    )

    report = COMPARE.compare_runs(reference, candidate, output)

    assert report["evaluation_deltas"]["micro_f1"]["delta"] == pytest.approx(0.1)
    assert report["resource_deltas"]["parameters_total"]["delta"] == -30.0
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.md").is_file()
    assert (output / "per_constraint_deltas.csv").is_file()
    assert (output / "h2_factor_semantics_deltas.csv").is_file()


def test_comparison_rejects_non_executor_config_difference() -> None:
    reference = _config("per_type_grouped_v2")
    candidate = _config("shared_adapter_v1")
    candidate["training_config"]["num_epochs"] = 14
    with pytest.raises(ValueError, match="differ outside"):
        COMPARE.validate_comparison_contract(
            reference,
            candidate,
            _manifest(),
            _manifest(),
        )
