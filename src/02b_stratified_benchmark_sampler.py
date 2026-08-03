#!/usr/bin/env python3
"""Create a fixed stratified benchmark slice from interim parquet splits."""

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from modules.class_hierarchy import (
    CLASS_HIERARCHY_FILENAME,
    CLASS_HIERARCHY_MANIFEST_FILENAME,
)
from modules.data_encoders import (
    DATASET_SCHEMA_VERSION,
    FEATURE_ENCODER_FILENAME,
    IDENTITY_ENCODER_FILENAME,
    IDENTITY_TO_FEATURE_FILENAME,
    LEGACY_ENCODER_FILENAME,
    dataset_variant_name,
)
from modules.sampling_contract import (
    ROW_ORDER_BUCKET_BITS,
    ROW_ORDER_BUCKET_COUNT,
    ROW_ORDER_KEY_COLUMN,
    ROW_ORDER_METHOD,
)


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
    sample_fraction: float | None,
    seed: int,
    target_rows: int | None = None,
) -> tuple[dict[str, set[int]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    selected_by_split: dict[str, set[int]] = {split: set() for split in SPLITS}
    report_rows: list[dict[str, object]] = []

    total_rows = sum(len(indices) for indices in strata.values())
    if target_rows is not None:
        if target_rows <= 0:
            raise ValueError("--target-rows must be positive.")
        if target_rows > total_rows:
            raise ValueError(
                f"Requested {target_rows:,} rows but source contains only {total_rows:,}."
            )
        if target_rows < len(strata):
            raise ValueError(
                f"--target-rows={target_rows} is smaller than the {len(strata)} non-empty strata."
            )
        ideals = {
            key: (len(indices) * target_rows / total_rows)
            for key, indices in strata.items()
        }
        targets = {
            key: min(len(strata[key]), max(1, int(math.floor(value))))
            for key, value in ideals.items()
        }
        current = sum(targets.values())
        if current < target_rows:
            order = sorted(
                strata,
                key=lambda key: (ideals[key] - math.floor(ideals[key]), len(strata[key]), key),
                reverse=True,
            )
            while current < target_rows:
                changed = False
                for key in order:
                    if targets[key] >= len(strata[key]):
                        continue
                    targets[key] += 1
                    current += 1
                    changed = True
                    if current == target_rows:
                        break
                if not changed:
                    raise RuntimeError("Unable to allocate the requested target rows across strata.")
        elif current > target_rows:
            order = sorted(
                strata,
                key=lambda key: (ideals[key] - math.floor(ideals[key]), len(strata[key]), key),
            )
            while current > target_rows:
                changed = False
                for key in order:
                    if targets[key] <= 1:
                        continue
                    targets[key] -= 1
                    current -= 1
                    changed = True
                    if current == target_rows:
                        break
                if not changed:
                    raise RuntimeError("Unable to reduce allocated stratum counts to target rows.")
    else:
        if sample_fraction is None or not 0.0 < sample_fraction <= 1.0:
            raise ValueError("--sample-fraction must be in (0, 1].")
        targets = {
            key: min(len(indices), max(1, int(round(len(indices) * sample_fraction))))
            for key, indices in strata.items()
        }

    for split, constraint_type, attached_bin in sorted(strata):
        indices = strata[(split, constraint_type, attached_bin)]
        source_count = len(indices)
        target_count = targets[(split, constraint_type, attached_bin)]
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


def _deterministic_row_order_keys(
    source_indices: np.ndarray,
    *,
    split: str,
    seed: int,
) -> np.ndarray:
    """Return a deterministic pseudo-random permutation key per source row.

    SplitMix64 is a bijection over uint64 values.  Adding a stable split/seed
    salt before applying it therefore gives every source index a unique key,
    without holding an in-memory permutation of the sampled split.
    """

    salt_bytes = hashlib.sha256(
        f"{ROW_ORDER_METHOD}:{seed}:{split}".encode("utf-8")
    ).digest()[:8]
    salt = np.uint64(int.from_bytes(salt_bytes, byteorder="little", signed=False))
    values = np.asarray(source_indices, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = values + salt + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed = mixed ^ (mixed >> np.uint64(31))
    return mixed


def _write_sampled_splits(
    source_root: Path,
    output_root: Path,
    *,
    selected_by_split: dict[str, set[int]],
    column: str,
    batch_size: int,
    seed: int,
) -> tuple[Counter[int], dict[str, Counter[int]], dict[str, int]]:
    combined_hist: Counter[int] = Counter()
    split_hist: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    sampled_counts: dict[str, int] = {}

    for split, parquet_path in _iter_split_paths(source_root):
        selected = selected_by_split.get(split, set())
        output_path = output_root / f"df_{split}.parquet"
        incomplete_path = output_root / f".{output_path.name}.incomplete"
        shuffle_root = output_root / f".{split}_row_order"
        parquet_file = pq.ParquetFile(parquet_path)
        source_schema = parquet_file.schema_arrow
        if ROW_ORDER_KEY_COLUMN in source_schema.names:
            raise ValueError(
                f"Reserved sampler column {ROW_ORDER_KEY_COLUMN!r} already exists in {parquet_path}"
            )
        shuffle_schema = source_schema.append(pa.field(ROW_ORDER_KEY_COLUMN, pa.uint64()))
        bucket_writers: dict[int, pq.ParquetWriter] = {}
        row_offset = 0
        split_written = 0
        shuffle_root.mkdir(parents=True, exist_ok=False)
        try:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                batch_start = row_offset
                mask_values = np.fromiter(
                    (
                        (row_offset + local_idx) in selected
                        for local_idx in range(batch.num_rows)
                    ),
                    dtype=np.bool_,
                    count=batch.num_rows,
                )
                row_offset += batch.num_rows
                if not bool(mask_values.any()):
                    continue
                table = pa.Table.from_batches([batch])
                filtered = table.filter(pa.array(mask_values, type=pa.bool_()))
                source_indices = np.arange(
                    batch_start,
                    batch_start + batch.num_rows,
                    dtype=np.uint64,
                )[mask_values]
                order_keys = _deterministic_row_order_keys(
                    source_indices,
                    split=split,
                    seed=seed,
                )
                shuffled = filtered.append_column(
                    ROW_ORDER_KEY_COLUMN,
                    pa.array(order_keys, type=pa.uint64()),
                )
                bucket_ids = (
                    order_keys >> np.uint64(64 - ROW_ORDER_BUCKET_BITS)
                ).astype(np.uint8, copy=False)
                for bucket_id in np.unique(bucket_ids).tolist():
                    bucket = int(bucket_id)
                    positions = np.flatnonzero(bucket_ids == bucket)
                    bucket_table = shuffled.take(pa.array(positions, type=pa.int64()))
                    writer = bucket_writers.get(bucket)
                    if writer is None:
                        bucket_path = shuffle_root / f"bucket-{bucket:03d}.parquet"
                        writer = pq.ParquetWriter(bucket_path, shuffle_schema)
                        bucket_writers[bucket] = writer
                    writer.write_table(bucket_table)
                split_written += filtered.num_rows

                if column in filtered.column_names:
                    lengths = _lengths_for_array(filtered[column].combine_chunks())
                    values, counts = np.unique(lengths, return_counts=True)
                    for value, count in zip(values, counts):
                        int_value = int(value)
                        int_count = int(count)
                        combined_hist[int_value] += int_count
                        split_hist[split][int_value] += int_count
            for writer in bucket_writers.values():
                writer.close()
            bucket_writers.clear()

            if split_written != len(selected):
                raise RuntimeError(
                    f"Selected/written row mismatch for {split}: "
                    f"selected={len(selected):,}, written={split_written:,}"
                )

            final_writer = pq.ParquetWriter(incomplete_path, source_schema)
            try:
                for bucket in range(ROW_ORDER_BUCKET_COUNT):
                    bucket_path = shuffle_root / f"bucket-{bucket:03d}.parquet"
                    if not bucket_path.exists():
                        continue
                    bucket_table = pq.read_table(bucket_path)
                    ordered = bucket_table.sort_by([(ROW_ORDER_KEY_COLUMN, "ascending")])
                    final_writer.write_table(ordered.drop_columns([ROW_ORDER_KEY_COLUMN]))
            finally:
                final_writer.close()
            incomplete_path.replace(output_path)
        finally:
            for writer in bucket_writers.values():
                writer.close()
            if incomplete_path.exists():
                incomplete_path.unlink()
            shutil.rmtree(shuffle_root, ignore_errors=True)
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


def _write_sample_semantic_audits(
    output_root: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    validation_counts: Counter[str] = Counter()
    gold_repair_counts: Counter[str] = Counter()
    validation_rows: list[dict[str, object]] = []
    gold_repair_rows: list[dict[str, object]] = []

    for split, parquet_path in _iter_split_paths(output_root):
        required = {
            "constraint_type",
            "primary_validation_reason",
            "primary_gold_repair_status",
        }
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        missing = required - schema_names
        if missing:
            raise ValueError(
                f"{parquet_path} is missing sampled semantic-audit columns: {sorted(missing)}"
            )
        frame = pd.read_parquet(parquet_path, columns=sorted(required))
        for (constraint_type, reason), count in frame.groupby(
            ["constraint_type", "primary_validation_reason"],
            dropna=False,
        ).size().items():
            key = f"{split}::{constraint_type}::{reason}"
            validation_counts[key] = int(count)
            validation_rows.append(
                {
                    "split": split,
                    "constraint_type": constraint_type,
                    "reason": reason,
                    "count": int(count),
                }
            )
        for (constraint_type, status), count in frame.groupby(
            ["constraint_type", "primary_gold_repair_status"],
            dropna=False,
        ).size().items():
            key = f"{split}::{constraint_type}::{status}"
            gold_repair_counts[key] = int(count)
            gold_repair_rows.append(
                {
                    "split": split,
                    "constraint_type": constraint_type,
                    "status": status,
                    "count": int(count),
                }
            )

    pd.DataFrame(validation_rows).to_csv(
        output_root / "sample_primary_validation_audit_by_constraint.csv",
        index=False,
    )
    pd.DataFrame(gold_repair_rows).to_csv(
        output_root / "sample_gold_repair_audit_by_constraint.csv",
        index=False,
    )
    return dict(sorted(validation_counts.items())), dict(sorted(gold_repair_counts.items()))


def _write_reports(
    output_root: Path,
    *,
    report_rows: list[dict[str, object]],
    source_variant: str,
    output_variant: str,
    source_total_rows: int,
    sampled_counts: dict[str, int],
    sample_fraction: float | None,
    target_rows: int | None,
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
            f"- sample_fraction: `{sample_fraction if sample_fraction is not None else 'allocated'}`",
            f"- target_rows: `{target_rows if target_rows is not None else 'not set'}`",
            f"- seed: `{seed}`",
            f"- row_order: `{ROW_ORDER_METHOD}`",
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
        "target_rows": target_rows,
        "seed": seed,
        "row_order": {
            "method": ROW_ORDER_METHOD,
            "seed": seed,
            "external_sort_buckets": ROW_ORDER_BUCKET_COUNT,
        },
        "source_total_rows": source_total_rows,
        "sampled_total_rows": sampled_total,
        "sampled_counts": sampled_counts,
        "bins": [
            {"lower": lower, "upper": upper, "label": label}
            for lower, upper, label in DEFAULT_BINS
        ],
    }
    (output_root / "sampling_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_dataset_contract(source_root: Path, output_root: Path) -> None:
    for filename in (
        IDENTITY_ENCODER_FILENAME,
        FEATURE_ENCODER_FILENAME,
        LEGACY_ENCODER_FILENAME,
        IDENTITY_TO_FEATURE_FILENAME,
        CLASS_HIERARCHY_FILENAME,
        CLASS_HIERARCHY_MANIFEST_FILENAME,
    ):
        source = source_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)


def _write_dataset_manifest(
    *,
    source_root: Path,
    output_root: Path,
    sampled_counts: dict[str, int],
    seed: int,
    target_rows: int | None,
    sample_fraction: float | None,
    sample_validation_by_constraint: dict[str, int],
    sample_gold_repair_by_constraint: dict[str, int],
) -> None:
    source_manifest_path = source_root / "dataset_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    payload = {
        **source_manifest,
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_variant": output_root.name,
        "parent_dataset_variant": source_root.name,
        "parent_manifest_sha256": _sha256_file(source_manifest_path)
        if source_manifest_path.exists()
        else None,
        "rows": dict(sorted(sampled_counts.items())),
        "sampling": {
            "seed": int(seed),
            "target_rows": target_rows,
            "sample_fraction": sample_fraction,
            "row_order": {
                "method": ROW_ORDER_METHOD,
                "seed": int(seed),
                "external_sort_buckets": ROW_ORDER_BUCKET_COUNT,
            },
            "primary_validation_by_constraint": sample_validation_by_constraint,
            "gold_repair_by_constraint": sample_gold_repair_by_constraint,
        },
        "outputs": {
            path.name: _sha256_file(path)
            for path in sorted(
                [*output_root.glob("df_*.parquet")]
                + [
                    output_root / filename
                    for filename in (
                        IDENTITY_ENCODER_FILENAME,
                        FEATURE_ENCODER_FILENAME,
                        LEGACY_ENCODER_FILENAME,
                        IDENTITY_TO_FEATURE_FILENAME,
                        CLASS_HIERARCHY_FILENAME,
                        CLASS_HIERARCHY_MANIFEST_FILENAME,
                        "sample_primary_validation_audit_by_constraint.csv",
                        "sample_gold_repair_audit_by_constraint.csv",
                        "sampling_metadata.json",
                    )
                    if (output_root / filename).exists()
                ]
            )
        },
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified benchmark variant from interim parquet splits.")
    parser.add_argument("--source-dataset", default="full", help="Source interim dataset name.")
    parser.add_argument("--output-dataset", default="full_strat1m", help="Derived output dataset name.")
    parser.add_argument("--min-occurrence", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help="Exact total row count allocated proportionally across non-empty strata.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", choices=["local", "focus"], default="local")
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output variant directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_rows is not None and args.sample_fraction is not None:
        raise ValueError("Use either --target-rows or --sample-fraction, not both.")
    if args.target_rows is None and args.sample_fraction is None:
        args.sample_fraction = 0.5
    if args.sample_fraction is not None and not 0.0 < args.sample_fraction <= 1.0:
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

    encoder_candidates = [
        source_root / FEATURE_ENCODER_FILENAME,
        source_root / LEGACY_ENCODER_FILENAME,
    ]
    if not any(path.exists() for path in encoder_candidates):
        raise FileNotFoundError(f"Missing source feature encoder under {source_root}")
    _copy_dataset_contract(source_root, output_root)

    strata, _source_hist, source_total_rows = _collect_strata(source_root, column=column, batch_size=args.batch_size)
    selected_by_split, report_rows = _sample_indices(
        strata,
        sample_fraction=float(args.sample_fraction) if args.sample_fraction is not None else None,
        target_rows=int(args.target_rows) if args.target_rows is not None else None,
        seed=int(args.seed),
    )
    combined_hist, split_hist, sampled_counts = _write_sampled_splits(
        source_root,
        output_root,
        selected_by_split=selected_by_split,
        column=column,
        batch_size=args.batch_size,
        seed=int(args.seed),
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
        sample_fraction=float(args.sample_fraction) if args.sample_fraction is not None else None,
        target_rows=int(args.target_rows) if args.target_rows is not None else None,
        seed=int(args.seed),
        column=column,
    )
    sample_validation_by_constraint, sample_gold_repair_by_constraint = (
        _write_sample_semantic_audits(output_root)
    )
    _write_dataset_manifest(
        source_root=source_root,
        output_root=output_root,
        sampled_counts=sampled_counts,
        seed=int(args.seed),
        target_rows=int(args.target_rows) if args.target_rows is not None else None,
        sample_fraction=float(args.sample_fraction) if args.sample_fraction is not None else None,
        sample_validation_by_constraint=sample_validation_by_constraint,
        sample_gold_repair_by_constraint=sample_gold_repair_by_constraint,
    )

    sampled_total = sum(sampled_counts.values())
    print(f"Source rows: {source_total_rows:,}")
    print(f"Sampled rows: {sampled_total:,}")
    print(f"Wrote sampled benchmark variant to {output_root}")


if __name__ == "__main__":
    main()
