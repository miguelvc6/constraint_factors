from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_train_module():
    module_path = SRC / "07_train.py"
    spec = importlib.util.spec_from_file_location("train_07_safeguards_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = _load_train_module()


def test_edit_logit_diagnostic_ignores_invalid_class_mask() -> None:
    logits = torch.tensor([-1e9, -3.5, 7.25], dtype=torch.float32)
    assert TRAIN._max_abs_unmasked_value(logits) == pytest.approx(7.25)

    bf16_logits = torch.tensor([-1e9, -4.0, 6.0], dtype=torch.bfloat16)
    assert TRAIN._max_abs_unmasked_value(bf16_logits) == pytest.approx(6.0)


def test_validation_subset_must_cover_all_trained_constraint_types() -> None:
    train_metrics = {"single": {}, "type": {}, "valueType": {}}
    partial_val_metrics = {"single": {}, "type": {}}

    with pytest.raises(ValueError, match="valueType"):
        TRAIN._assert_validation_subset_constraint_coverage(
            train_metrics,
            partial_val_metrics,
            validation_subset_size=25_000,
        )

    TRAIN._assert_validation_subset_constraint_coverage(
        train_metrics,
        partial_val_metrics,
        validation_subset_size=None,
    )
    TRAIN._assert_validation_subset_constraint_coverage(
        train_metrics,
        train_metrics,
        validation_subset_size=25_000,
    )
