#!/usr/bin/env python3
"""Create a fixed stratified benchmark slice from interim parquet splits."""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from modules.data_encoders import dataset_variant_name


SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 32, "1-32"),
    (33, 64, "33-64"),
    (65, 83, "65-83"),
    (84, 107, "84-107"),
    (108, 108, "108"),
    (109, 160, "109-160"),
    (161, 267, "161-267"),
    (268, None, "268+"),
)


def _sequence_column_for_scope(scope: str) -> str:
    return "local_constraint_ids_focus" if scope == "focus" else "local_constraint_ids"


def _lengths_for_array(array: pa.Array) -> np.ndarray:
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        lengths = pc.list_value_length(array)
        lengths = pc.fill_null(lengths, 0)
        return lengths.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    values = array.to_pylist()
    return np.fromiter(
        (0 if value is None else len(value) if isinstance(value, (list, tuple)) else 1 for value in values),
        dtype=np.int64,
        count=len(values),
    )


def _bin_label(length: int) -> str:
    for lower, upper, label in DEFAULT_BINS:
        if length >= lower and (upper is None or length <= upper):
            return label
    return "0"


def _iter_split_paths(root: Path) -> Iterable[tuple[str, Path]]:
    for split in SPLITS:
        path = root / f"df_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet split: {path}")
        yield split, path


def _collect_strata(
    source_root: Path,
    *,
    column: str,
    batch_size: int,
) -> tuple[dict[tuple[str, str, str], list[int]], Counter[tuple[str, int]], int]:
    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    source_hist_by_split: Counter[tuple[str, int]] = Counter()
    total_rows = 0

    for split, parquet_path in _iter_split_paths(source_root):
        parquet_file = pq.ParquetFile(parquet_path)
        required = {"constraint_type", column}
        missing = required - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")

        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["constraint_type", column]):
            constraint_types = batch.column(0).to_pylist()
            lengths = _lengths_for_array(batch.column(1))
            for local_idx, (constraint_type, length) in enumerate(zip(constraint_types, lengths)):
                int_length = int(length)
                key = (split, str(constraint_type), _bin_label(int_length))
                strata[key].append(row_offset + local_idx)
                source_hist_by_split[(split, int_length)] += 1
            row_offset += batch.num_rows
            total_rows += batch.num_rows

    return dict(strata), source_hist_by_split, total_rows


def _sample_indices(
    strata: dict[tuple[str, str, str], list[int]],
    *,
    sample_fraction: float,
    seed: int,
) -> tuple[dict[str, set[int]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    selected_by_split: dict[str, set[int]] = {split: set() for split in SPLITS}
    report_rows: list[dict[str, object]] = []

    for split, constraint_type, attached_bin in sorted(strata):
        indices = strata[(split, constraint_type, attached_bin)]
        source_count = len(indices)
        target_count = max(1, int(round(source_count * sample_fraction)))
        target_count = min(target_count, source_count)
        if target_count == source_count:
            sampled = indices
        else:
            sampled_positions = rng.choice(source_count, size=target_count, replace=False)
            sampled = [indices[int(pos)] for pos in sampled_positions]
        selected_by_split[split].update(sampled)
        report_rows.append(
            {
                "split": split,
                "constraint_type": constraint_type,
                "attached_constraint_bin": attached_bin,
                "source_count": source_count,
                "sampled_count": target_count,
                "sample_fraction": target_count / source_count if source_count else 0.0,
            }
        )

    return selected_by_split, report_rows


def _write_sampled_splits(
    source_root: Path,
    output_root: Path,
    *,
    selected_by_split: dict[str, set[int]],
    column: str,
    batch_size: int,
) -> tuple[Counter[int], dict[str, Counter[int]], dict[str, int]]:
    combined_hist: Counter[int] = Counter()
    split_hist: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    sampled_counts: dict[str, int] = {}

    for split, parquet_path in _iter_split_paths(source_root):
        selected = selected_by_split.get(split, set())
        output_path = output_root / f"df_{split}.parquet"
        parquet_file = pq.ParquetFile(parquet_path)
        writer: pq.ParquetWriter | None = None
        row_offset = 0
        split_written = 0
        try:
            writer = pq.ParquetWriter(output_path, parquet_file.schema_arrow)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                mask_values = [
                    (row_offset + local_idx) in selected
                    for local_idx in range(batch.num_rows)
                ]
                row_offset += batch.num_rows
                if not any(mask_values):
                    continue
                table = pa.Table.from_batches([batch])
                filtered = table.filter(pa.array(mask_values))
                writer.write_table(filtered)
                split_written += filtered.num_rows

                if column in filtered.column_names:
                    lengths = _lengths_for_array(filtered[column].combine_chunks())
                    values, counts = np.unique(lengths, return_counts=True)
                    for value, count in zip(values, counts):
                        int_value = int(value)
                        int_count = int(count)
                        combined_hist[int_value] += int_count
                        split_hist[split][int_value] += int_count
        finally:
            if writer is not None:
                writer.close()
        sampled_counts[split] = split_written

    return combined_hist, split_hist, sampled_counts


def _write_histogram_csv(histogram: Counter[int], output: Path) -> None:
    rows = [
        {"num_attached_constraints": key, "count": histogram[key]}
        for key in sorted(histogram)
    ]
    pd.DataFrame(rows).to_csv(output, index=False)


def _write_split_histogram_csv(split_hist: dict[str, Counter[int]], output: Path) -> None:
    rows: list[dict[str, int | str]] = []
    for split in SPLITS:
        for key in sorted(split_hist[split]):
            rows.append({"split": split, "num_attached_constraints": key, "count": split_hist[split][key]})
    pd.DataFrame(rows).to_csv(output, index=False)


def _write_reports(
    output_root: Path,
    *,
    report_rows: list[dict[str, object]],
    source_variant: str,
    output_variant: str,
    source_total_rows: int,
    sampled_counts: dict[str, int],
    sample_fraction: float,
    seed: int,
    column: str,
) -> None:
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_root / "sampling_report.csv", index=False)

    sampled_total = sum(sampled_counts.values())
    by_split = ", ".join(f"{split}={sampled_counts.get(split, 0):,}" for split in SPLITS)
    markdown = "\n".join(
        [
            f"# Stratified Benchmark Sampling Report",
            "",
            f"- source_variant: `{source_variant}`",
            f"- output_variant: `{output_variant}`",
            f"- sequence_column: `{column}`",
            f"- sample_fraction: `{sample_fraction}`",
            f"- seed: `{seed}`",
            f"- source_rows: `{source_total_rows:,}`",
            f"- sampled_rows: `{sampled_total:,}`",
            f"- sampled_by_split: {by_split}",
            "",
            "Strata are `(split, constraint_type, attached_constraint_bin)`.",
            "The attached constraint count is `len(local_constraint_ids)` for local scope or "
            "`len(local_constraint_ids_focus)` for focus scope.",
            "",
            "Default bins: `1-32`, `33-64`, `65-83`, `84-107`, `108`, `109-160`, `161-267`, `268+`.",
            "",
        ]
    )
    (output_root / "sampling_report.md").write_text(markdown, encoding="utf-8")

    metadata = {
        "source_variant": source_variant,
        "output_variant": output_variant,
        "column": column,
        "sample_fraction": sample_fraction,
        "seed": seed,
        "source_total_rows": source_total_rows,
        "sampled_total_rows": sampled_total,
        "sampled_counts": sampled_counts,
        "bins": [
            {"lower": lower, "upper": upper, "label": label}
            for lower, upper, label in DEFAULT_BINS
        ],
    }
    (output_root / "sampling_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified benchmark variant from interim parquet splits.")
    parser.add_argument("--source-dataset", default="full", help="Source interim dataset name.")
    parser.add_argument("--output-dataset", default="full_strat1m", help="Derived output dataset name.")
    parser.add_argument("--min-occurrence", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", choices=["local", "focus"], default="local")
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output variant directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("--sample-fraction must be in (0, 1].")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    source_variant = dataset_variant_name(args.source_dataset, max(1, args.min_occurrence))
    output_variant = dataset_variant_name(args.output_dataset, max(1, args.min_occurrence))
    source_root = args.interim_root / source_variant
    output_root = args.interim_root / output_variant
    column = _sequence_column_for_scope(args.scope)

    if source_root == output_root:
        raise ValueError("Source and output variants resolve to the same directory.")
    if not source_root.exists():
        raise FileNotFoundError(f"Source interim dataset not found: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    encoder_path = source_root / "globalintencoder.txt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing source encoder: {encoder_path}")
    shutil.copy2(encoder_path, output_root / "globalintencoder.txt")

    strata, _source_hist, source_total_rows = _collect_strata(source_root, column=column, batch_size=args.batch_size)
    selected_by_split, report_rows = _sample_indices(
        strata,
        sample_fraction=float(args.sample_fraction),
        seed=int(args.seed),
    )
    combined_hist, split_hist, sampled_counts = _write_sampled_splits(
        source_root,
        output_root,
        selected_by_split=selected_by_split,
        column=column,
        batch_size=args.batch_size,
    )

    _write_histogram_csv(combined_hist, output_root / f"hist_{column}.csv")
    _write_split_histogram_csv(split_hist, output_root / f"hist_{column}_by_split.csv")
    _write_reports(
        output_root,
        report_rows=report_rows,
        source_variant=source_variant,
        output_variant=output_variant,
        source_total_rows=source_total_rows,
        sampled_counts=sampled_counts,
        sample_fraction=float(args.sample_fraction),
        seed=int(args.seed),
        column=column,
    )

    sampled_total = sum(sampled_counts.values())
    print(f"Source rows: {source_total_rows:,}")
    print(f"Sampled rows: {sampled_total:,}")
    print(f"Wrote sampled benchmark variant to {output_root}")


if __name__ == "__main__":
    main()
