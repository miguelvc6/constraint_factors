from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scheduler", ROOT / "src" / "10_scheduler.py")
assert SPEC is not None and SPEC.loader is not None
SCHEDULER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEDULER)


def test_exact_selection_uses_complete_directory_names() -> None:
    selected = (
        "a1_factorized_imitation_per_type_compact__full_strat1m_minocc100__node_id",
        "a1_factorized_imitation_shared_adapter__full_strat1m_minocc100__node_id",
    )
    assert SCHEDULER.matches_selection(
        selected[0],
        substring=None,
        exact_names=selected,
    )
    assert not SCHEDULER.matches_selection(
        "a1_factorized_imitation__full_strat1m_minocc100__node_id",
        substring=None,
        exact_names=selected,
    )
    assert not SCHEDULER.matches_selection(
        selected[0] + "_suffix",
        substring=None,
        exact_names=selected,
    )


def test_substring_selection_remains_backward_compatible() -> None:
    name = "a1_factorized_imitation__full_strat1m_minocc100__node_id"
    assert SCHEDULER.matches_selection(
        name,
        substring="a1_factorized",
        exact_names=(),
    )
    assert not SCHEDULER.matches_selection(
        name,
        substring="m1c_safe",
        exact_names=(),
    )
