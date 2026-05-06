#!/usr/bin/env python3
"""Build a histogram of attached constraints per violation instance.

The script streams parquet batches so the full dataframe never has to be
loaded into memory. By default it computes:

    num_attached_constraints = len(local_constraint_ids)
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from modules.data_encoders import dataset_variant_name, discover_min_occurrence


SPLITS: tuple[str, ...] = ("train", "val", "test")


def _default_outputs(interim_root: Path, column: str) -> tuple[Path, Path]:
    stem = f"hist_{column}"
    return interim_root / f"{stem}.csv", interim_root / f"{stem}.png"


def _lengths_for_array(array: pa.Array) -> np.ndarray:
    """Return list lengths for an Arrow list-like array without materializing values."""
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        lengths = pc.list_value_length(array)
        lengths = pc.fill_null(lengths, 0)
        return lengths.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    # Fallback for unexpected object encodings. This materializes only one batch.
    values = array.to_pylist()
    return np.fromiter(
        (0 if value is None else len(value) if isinstance(value, (list, tuple)) else 1 for value in values),
        dtype=np.int64,
        count=len(values),
    )


def _update_histogram_from_parquet(
    parquet_path: Path,
    *,
    column: str,
    batch_size: int,
    histogram: Counter[int],
    split_counts: dict[str, Counter[int]] | None,
    split: str,
) -> int:
    parquet_file = pq.ParquetFile(parquet_path)
    if column not in parquet_file.schema_arrow.names:
        raise ValueError(f"Column {column!r} not found in {parquet_path}")

    rows = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[column]):
        lengths = _lengths_for_array(batch.column(0))
        if lengths.size == 0:
            continue
        values, counts = np.unique(lengths, return_counts=True)
        for value, count in zip(values, counts):
            int_value = int(value)
            int_count = int(count)
            histogram[int_value] += int_count
            if split_counts is not None:
                split_counts[split][int_value] += int_count
        rows += int(lengths.size)
    return rows


def _write_histogram_csv(histogram: Counter[int], output: Path) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "num_attached_constraints": sorted(histogram),
            "count": [histogram[key] for key in sorted(histogram)],
        }
    )
    df.to_csv(output, index=False)
    return df


def _write_split_csv(split_counts: dict[str, Counter[int]], output: Path) -> None:
    rows: list[dict[str, int | str]] = []
    for split, histogram in split_counts.items():
        for value in sorted(histogram):
            rows.append(
                {
                    "split": split,
                    "num_attached_constraints": value,
                    "count": histogram[value],
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def _write_histogram_plot(df: pd.DataFrame, output: Path, *, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(8.0, min(22.0, len(df) * 0.28))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(df["num_attached_constraints"], df["count"], width=0.9, color="#2f6f8f")
    ax.set_xlabel("num_attached_constraints = len(local_constraint_ids)")
    ax.set_ylabel("Number of instances")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _print_summary(histogram: Counter[int], total_rows: int, csv_path: Path, png_path: Path | None) -> None:
    if total_rows == 0:
        print("No rows scanned.")
        return

    values = np.fromiter(histogram.keys(), dtype=np.int64)
    counts = np.fromiter((histogram[key] for key in histogram.keys()), dtype=np.int64)
    mean = float(np.average(values, weights=counts))
    cumulative = np.cumsum(counts[np.argsort(values)])
    sorted_values = np.sort(values)

    def percentile(p: float) -> int:
        index = int(np.searchsorted(cumulative, total_rows * p, side="left"))
        return int(sorted_values[min(index, len(sorted_values) - 1)])

    print(f"Rows scanned: {total_rows:,}")
    print(f"Min/median/mean/p95/max: {int(values.min())} / {percentile(0.50)} / {mean:.2f} / {percentile(0.95)} / {int(values.max())}")
    print(f"Wrote CSV: {csv_path}")
    if png_path is not None:
        print(f"Wrote plot: {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream parquet splits and histogram len(local_constraint_ids)."
    )
    parser.add_argument("--dataset", choices=["sample", "full"], default="full")
    parser.add_argument(
        "--min-occurrence",
        type=int,
        default=None,
        help="Dataset variant threshold. Defaults to auto-discovery for the dataset.",
    )
    parser.add_argument(
        "--interim-root",
        type=Path,
        default=Path("data/interim"),
        help="Root containing <dataset_variant>/df_*.parquet.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to include in the combined histogram.",
    )
    parser.add_argument(
        "--scope",
        choices=["local", "focus"],
        default="local",
        help="Use local_constraint_ids or local_constraint_ids_focus.",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Override the sequence column to histogram.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Parquet rows per streamed batch.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Combined histogram CSV output path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Histogram PNG output path. Defaults next to the CSV.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write the CSV histogram.",
    )
    parser.add_argument(
        "--by-split",
        action="store_true",
        help="Also write a split-level CSV next to the combined CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_occurrence = args.min_occurrence
    if min_occurrence is None:
        min_occurrence = discover_min_occurrence(args.dataset)

    dataset_variant = dataset_variant_name(args.dataset, max(1, min_occurrence))
    dataframe_root = args.interim_root / dataset_variant
    if not dataframe_root.exists():
        raise FileNotFoundError(f"Interim dataframe directory not found: {dataframe_root}")

    column = args.column or ("local_constraint_ids_focus" if args.scope == "focus" else "local_constraint_ids")
    default_csv, default_png = _default_outputs(dataframe_root, column)
    output_csv = args.output_csv or default_csv
    output_png = None if args.no_plot else (args.output_png or default_png)

    histogram: Counter[int] = Counter()
    split_counts = {split: Counter() for split in args.splits} if args.by_split else None
    total_rows = 0

    for split in args.splits:
        parquet_path = dataframe_root / f"df_{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet split: {parquet_path}")
        rows = _update_histogram_from_parquet(
            parquet_path,
            column=column,
            batch_size=args.batch_size,
            histogram=histogram,
            split_counts=split_counts,
            split=split,
        )
        total_rows += rows
        print(f"Scanned {split}: {rows:,} rows")

    df = _write_histogram_csv(histogram, output_csv)
    if split_counts is not None:
        split_csv = output_csv.with_name(f"{output_csv.stem}_by_split{output_csv.suffix}")
        _write_split_csv(split_counts, split_csv)
        print(f"Wrote split CSV: {split_csv}")

    if output_png is not None:
        _write_histogram_plot(
            df,
            output_png,
            title=f"Attached constraints per instance ({dataset_variant}, {column})",
        )

    _print_summary(histogram, total_rows, output_csv, output_png)


if __name__ == "__main__":
    main()
