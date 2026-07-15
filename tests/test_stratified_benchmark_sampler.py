from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_sampler():
    module_path = ROOT / "src" / "02b_stratified_benchmark_sampler.py"
    spec = importlib.util.spec_from_file_location("stratified_benchmark_sampler", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source_variant(interim_root: Path) -> None:
    source = interim_root / "full_minocc100"
    source.mkdir(parents=True, exist_ok=True)
    (source / "globalintencoder.txt").write_text("encoder-placeholder", encoding="utf-8")

    base_columns = {
        "constraint_id": 1,
        "subject": 2,
        "predicate": 3,
        "object": 4,
        "other_subject": 0,
        "other_predicate": 0,
        "other_object": 0,
        "add_subject": 0,
        "add_predicate": 0,
        "add_object": 0,
        "del_subject": 0,
        "del_predicate": 0,
        "del_object": 0,
        "primary_validation_reason": "valid",
        "primary_gold_repair_status": "verified",
    }

    def row(constraint_type: str, attached_count: int, row_id: int) -> dict:
        payload = dict(base_columns)
        payload.update(
            {
                "row_id": row_id,
                "constraint_type": constraint_type,
                "local_constraint_ids": list(range(attached_count)),
                "local_constraint_ids_focus": list(range(min(attached_count, 3))),
            }
        )
        return payload

    pd.DataFrame(
        [
            row("typeA", 1, 0),
            row("typeA", 2, 1),
            row("typeA", 108, 2),
            row("typeB", 33, 3),
            row("typeB", 34, 4),
        ]
    ).to_parquet(source / "df_train.parquet", index=False)
    pd.DataFrame([row("typeA", 1, 10), row("typeA", 2, 11)]).to_parquet(
        source / "df_val.parquet",
        index=False,
    )
    pd.DataFrame([row("typeC", 300, 20)]).to_parquet(source / "df_test.parquet", index=False)


def _run_sampler(interim_root: Path, output_dataset: str) -> Path:
    sampler = _load_sampler()
    argv_backup = list(sys.argv)
    sys.argv = [
        "02b_stratified_benchmark_sampler.py",
        "--source-dataset",
        "full",
        "--output-dataset",
        output_dataset,
        "--min-occurrence",
        "100",
        "--sample-fraction",
        "0.5",
        "--seed",
        "42",
        "--scope",
        "local",
        "--interim-root",
        str(interim_root),
        "--batch-size",
        "2",
    ]
    try:
        sampler.main()
    finally:
        sys.argv = argv_backup
    return interim_root / f"{output_dataset}_minocc100"


def test_stratified_sampler_is_deterministic_and_preserves_reports(tmp_path: Path) -> None:
    interim_root = tmp_path / "data" / "interim"
    _write_source_variant(interim_root)

    first = _run_sampler(interim_root, "full_strat1m_a")
    second = _run_sampler(interim_root, "full_strat1m_b")

    for output in (first, second):
        assert (output / "globalintencoder.txt").read_text(encoding="utf-8") == "encoder-placeholder"
        assert (output / "sampling_report.csv").exists()
        assert (output / "sampling_report.md").exists()
        assert (output / "sampling_metadata.json").exists()
        assert (output / "hist_local_constraint_ids.csv").exists()
        assert (output / "hist_local_constraint_ids_by_split.csv").exists()
        assert (output / "sample_primary_validation_audit_by_constraint.csv").exists()
        assert (output / "sample_gold_repair_audit_by_constraint.csv").exists()

    first_train = pd.read_parquet(first / "df_train.parquet")
    second_train = pd.read_parquet(second / "df_train.parquet")
    assert first_train["row_id"].tolist() == second_train["row_id"].tolist()

    # train strata: typeA/1-32 has 2 -> 1 sampled; typeA/108 has 1 -> 1;
    # typeB/33-64 has 2 -> 1. Rare non-empty val/test strata keep one row.
    assert len(first_train) == 3
    assert len(pd.read_parquet(first / "df_val.parquet")) == 1
    assert len(pd.read_parquet(first / "df_test.parquet")) == 1

    report = pd.read_csv(first / "sampling_report.csv")
    train_report = report[report["split"] == "train"]
    assert sorted(train_report["sampled_count"].tolist()) == [1, 1, 1]
    assert set(train_report["attached_constraint_bin"]) == {"1-32", "108", "33-64"}

    metadata = json.loads((first / "sampling_metadata.json").read_text(encoding="utf-8"))
    assert metadata["row_order"]["method"] == "splitmix64_source_index_v1"
    assert metadata["row_order"]["seed"] == 42


def test_stratified_sampler_mixes_family_block_order(tmp_path: Path) -> None:
    interim_root = tmp_path / "data" / "interim"
    _write_source_variant(interim_root)
    source_path = interim_root / "full_minocc100" / "df_train.parquet"
    template = pd.read_parquet(source_path).iloc[0].to_dict()
    rows = []
    for family_index, family in enumerate(("typeA", "typeB", "typeC")):
        for offset in range(40):
            payload = dict(template)
            payload.update(
                {
                    "row_id": family_index * 1000 + offset,
                    "constraint_type": family,
                    "local_constraint_ids": [offset],
                    "local_constraint_ids_focus": [offset],
                }
            )
            rows.append(payload)
    pd.DataFrame(rows).to_parquet(source_path, index=False)

    first = _run_sampler(interim_root, "mixed_a")
    second = _run_sampler(interim_root, "mixed_b")
    first_train = pd.read_parquet(first / "df_train.parquet")
    second_train = pd.read_parquet(second / "df_train.parquet")

    assert first_train["row_id"].tolist() == second_train["row_id"].tolist()
    assert first_train["constraint_type"].value_counts().to_dict() == {
        "typeA": 20,
        "typeB": 20,
        "typeC": 20,
    }
    family_order = first_train["constraint_type"].tolist()
    family_transitions = sum(
        current != previous
        for previous, current in zip(family_order, family_order[1:])
    )
    assert family_transitions > 10


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_stratified_sampler_is_deterministic_and_preserves_reports(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_stratified_sampler_mixes_family_block_order(Path(tmpdir))
    print("stratified benchmark sampler tests passed")
