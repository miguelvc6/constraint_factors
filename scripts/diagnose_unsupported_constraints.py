#!/usr/bin/env python3
"""Diagnose unsupported attached constraints in interim parquet variants.

The constraint labeler reports unsupported families at factor-occurrence level.
This script makes that table actionable by separating:

- row-level prevalence of unsupported attached constraints,
- factor-level supported vs unsupported counts,
- primary constraint support,
- estimated factor-node reduction from filtering unsupported families.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from modules.data_encoders import GlobalIntEncoder, base_dataset_name, dataset_variant_name, discover_min_occurrence


SPLITS: tuple[str, ...] = ("train", "val", "test")


def _sequence_column_for_scope(scope: str) -> str:
    return "local_constraint_ids_focus" if scope == "focus" else "local_constraint_ids"


def _coerce_sequence(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _load_encoder(path: Path) -> GlobalIntEncoder:
    encoder = GlobalIntEncoder()
    encoder.load(path)
    return encoder


def _resolve_registry_id(raw_id: str, encoder: GlobalIntEncoder) -> int:
    raw = str(raw_id).strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1].strip()
    if raw.startswith("http://www.wikidata.org/prop/direct/"):
        raw = raw.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")

    candidates: list[str] = []
    if raw.startswith("http://") or raw.startswith("https://"):
        candidates.extend([raw, f"<{raw}>"])
    else:
        candidates.append(raw)
        if raw and raw[0] in ("P", "Q") and raw[1:].isdigit():
            uri = f"http://www.wikidata.org/entity/{raw}"
            candidates.extend([uri, f"<{uri}>"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        encoded = encoder.encode(candidate, add_new=False)
        if encoded:
            return int(encoded)
    return 0


def _load_registry(path: Path, *, encoder: GlobalIntEncoder) -> dict[int, dict[str, Any]]:
    registry_df = pd.read_parquet(path)
    if "registry_json" not in registry_df.columns or registry_df.empty:
        raise ValueError(f"Registry file does not contain registry_json: {path}")
    registry_json = registry_df["registry_json"].iloc[0]
    registry = json.loads(registry_json) if isinstance(registry_json, str) else registry_json

    parsed: dict[int, dict[str, Any]] = {}
    for raw_constraint_id, entry in registry.items():
        constraint_id = _resolve_registry_id(str(raw_constraint_id), encoder)
        if constraint_id == 0:
            continue
        family = str(entry.get("constraint_family") or entry.get("constraint_type_name") or "")
        supported = entry.get("constraint_family_supported")
        if supported is None:
            supported = entry.get("constraint_type_supported", False)
        parsed[constraint_id] = {
            "constraint_family": family,
            "constraint_type_item": str(entry.get("constraint_type_item") or ""),
            "constraint_label": str(entry.get("constraint_label") or ""),
            "constraint_family_supported": bool(supported),
        }
    return parsed


def _registry_path_candidates(
    interim_root: Path,
    *,
    dataset: str,
    min_occurrence: int,
    registry_dataset: str | None,
) -> list[Path]:
    candidates: list[str] = []
    if registry_dataset:
        candidates.append(registry_dataset)
    variant = dataset_variant_name(dataset, min_occurrence)
    candidates.extend([variant, base_dataset_name(variant), dataset])
    if "_strat" in dataset:
        candidates.append(dataset.split("_strat", 1)[0])

    paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        paths.append(interim_root / f"constraint_registry_{candidate}.parquet")
    return paths


def _resolve_registry_path(
    interim_root: Path,
    *,
    dataset: str,
    min_occurrence: int,
    registry_dataset: str | None,
) -> Path:
    candidates = _registry_path_candidates(
        interim_root,
        dataset=dataset,
        min_occurrence=min_occurrence,
        registry_dataset=registry_dataset,
    )
    for path in candidates:
        if path.exists():
            return path
    joined = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No constraint registry found. Tried:\n  {joined}")


def _iter_split_paths(root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for split in splits:
        path = root / f"df_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet split: {path}")
        paths.append((split, path))
    return paths


def _write_csv(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    sort_by: list[str] | None = None,
    ascending: bool | list[bool] = True,
) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if sort_by and not df.empty:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    df.to_csv(output, index=False)
    return df


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def diagnose(
    *,
    dataframe_root: Path,
    registry: dict[int, dict[str, Any]],
    column: str,
    splits: list[str],
    batch_size: int,
) -> dict[str, Any]:
    supported_ids = {constraint_id for constraint_id, entry in registry.items() if entry["constraint_family_supported"]}

    split_stats: dict[str, Counter[str]] = {split: Counter() for split in splits}
    primary_type_counts: Counter[tuple[str, bool]] = Counter()
    primary_family_counts: Counter[tuple[str, str, bool]] = Counter()
    unsupported_family_occurrences: Counter[str] = Counter()
    unsupported_family_rows: Counter[str] = Counter()
    supported_family_occurrences: Counter[str] = Counter()
    row_unsupported_hist: Counter[int] = Counter()
    row_total_hist: Counter[int] = Counter()
    row_supported_hist: Counter[int] = Counter()
    missing_constraint_ids: Counter[int] = Counter()

    for split, parquet_path in _iter_split_paths(dataframe_root, splits):
        parquet_file = pq.ParquetFile(parquet_path)
        required_columns = {"constraint_type", "constraint_id", column}
        missing = required_columns - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")

        stats = split_stats[split]
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["constraint_type", "constraint_id", column],
        ):
            table = pa.Table.from_batches([batch])
            constraint_types = table["constraint_type"].to_pylist()
            primary_ids = table["constraint_id"].to_pylist()
            attached_lists = table[column].to_pylist()

            for constraint_type, primary_id_raw, attached_raw in zip(constraint_types, primary_ids, attached_lists):
                stats["rows"] += 1
                primary_id = int(primary_id_raw)
                primary_entry = registry.get(primary_id)
                primary_supported = bool(primary_entry and primary_entry["constraint_family_supported"])
                primary_family = str(primary_entry["constraint_family"]) if primary_entry else "__missing_registry_entry__"
                primary_type_counts[(str(constraint_type), primary_supported)] += 1
                primary_family_counts[(str(constraint_type), primary_family, primary_supported)] += 1
                if primary_entry is None:
                    stats["primary_registry_missing_rows"] += 1
                if primary_family != str(constraint_type):
                    stats["primary_family_mismatch_rows"] += 1
                if primary_supported:
                    stats["primary_supported_rows"] += 1
                else:
                    stats["primary_unsupported_rows"] += 1

                attached_ids = _coerce_sequence(attached_raw)
                supported_count = 0
                unsupported_count = 0
                row_unsupported_families: set[str] = set()

                for constraint_id in attached_ids:
                    entry = registry.get(constraint_id)
                    if entry is None:
                        missing_constraint_ids[constraint_id] += 1
                        unsupported_count += 1
                        row_unsupported_families.add("__missing_registry_entry__")
                        unsupported_family_occurrences["__missing_registry_entry__"] += 1
                        continue

                    family = str(entry["constraint_family"])
                    if constraint_id in supported_ids:
                        supported_count += 1
                        supported_family_occurrences[family] += 1
                    else:
                        unsupported_count += 1
                        row_unsupported_families.add(family)
                        unsupported_family_occurrences[family] += 1

                total_count = supported_count + unsupported_count
                stats["attached_factor_count"] += total_count
                stats["supported_factor_count"] += supported_count
                stats["unsupported_factor_count"] += unsupported_count
                row_total_hist[total_count] += 1
                row_supported_hist[supported_count] += 1
                row_unsupported_hist[unsupported_count] += 1

                if unsupported_count:
                    stats["rows_with_unsupported"] += 1
                else:
                    stats["rows_without_unsupported"] += 1
                if supported_count:
                    stats["rows_with_supported"] += 1
                else:
                    stats["rows_without_supported"] += 1
                if unsupported_count and not supported_count:
                    stats["rows_only_unsupported"] += 1

                for family in row_unsupported_families:
                    unsupported_family_rows[family] += 1

        print(f"Scanned {split}: {split_stats[split]['rows']:,} rows")

    combined = Counter()
    for stats in split_stats.values():
        combined.update(stats)

    return {
        "split_stats": split_stats,
        "combined_stats": combined,
        "primary_type_counts": primary_type_counts,
        "primary_family_counts": primary_family_counts,
        "unsupported_family_occurrences": unsupported_family_occurrences,
        "unsupported_family_rows": unsupported_family_rows,
        "supported_family_occurrences": supported_family_occurrences,
        "row_unsupported_hist": row_unsupported_hist,
        "row_total_hist": row_total_hist,
        "row_supported_hist": row_supported_hist,
        "missing_constraint_ids": missing_constraint_ids,
    }


def _family_metadata(registry: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    registry_counts: Counter[str] = Counter()
    for entry in registry.values():
        family = str(entry["constraint_family"])
        registry_counts[family] += 1
        if family not in metadata:
            metadata[family] = {
                "constraint_family": family,
                "constraint_type_item": entry["constraint_type_item"],
                "constraint_label": entry["constraint_label"],
                "constraint_family_supported": bool(entry["constraint_family_supported"]),
            }
    for family, count in registry_counts.items():
        metadata[family]["registry_entries"] = count
    return metadata


def _rows_from_split_stats(split_stats: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, stats in split_stats.items():
        row_count = int(stats["rows"])
        attached = int(stats["attached_factor_count"])
        unsupported = int(stats["unsupported_factor_count"])
        supported = int(stats["supported_factor_count"])
        rows.append(
            {
                "split": split,
                "rows": row_count,
                "primary_supported_rows": int(stats["primary_supported_rows"]),
                "primary_unsupported_rows": int(stats["primary_unsupported_rows"]),
                "primary_registry_missing_rows": int(stats["primary_registry_missing_rows"]),
                "primary_family_mismatch_rows": int(stats["primary_family_mismatch_rows"]),
                "rows_with_supported": int(stats["rows_with_supported"]),
                "rows_without_supported": int(stats["rows_without_supported"]),
                "rows_with_unsupported": int(stats["rows_with_unsupported"]),
                "rows_without_unsupported": int(stats["rows_without_unsupported"]),
                "rows_only_unsupported": int(stats["rows_only_unsupported"]),
                "attached_factor_count": attached,
                "supported_factor_count": supported,
                "unsupported_factor_count": unsupported,
                "rows_with_unsupported_rate": _rate(int(stats["rows_with_unsupported"]), row_count),
                "unsupported_factor_rate": _rate(unsupported, attached),
                "mean_attached_factors_per_row": _rate(attached, row_count),
                "mean_supported_factors_per_row": _rate(supported, row_count),
                "mean_unsupported_factors_per_row": _rate(unsupported, row_count),
                "estimated_factor_reduction_supported_only": _rate(unsupported, attached),
            }
        )
    return rows


def _rows_from_family_counts(
    *,
    occurrences: Counter[str],
    row_counts: Counter[str],
    metadata: dict[str, dict[str, Any]],
    total_rows: int,
    total_factors: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, count in occurrences.most_common():
        entry = metadata.get(family, {})
        rows.append(
            {
                "constraint_family": family,
                "constraint_type_item": entry.get("constraint_type_item", ""),
                "constraint_label": entry.get("constraint_label", ""),
                "constraint_family_supported": bool(entry.get("constraint_family_supported", False)),
                "registry_entries": int(entry.get("registry_entries", 0)),
                "attached_factor_occurrences": int(count),
                "rows_with_family": int(row_counts.get(family, 0)),
                "occurrence_share_of_all_factors": _rate(int(count), total_factors),
                "row_share": _rate(int(row_counts.get(family, 0)), total_rows),
            }
        )
    return rows


def _hist_rows(counter: Counter[int], value_column: str) -> list[dict[str, int]]:
    return [{value_column: int(value), "rows": int(counter[value])} for value in sorted(counter)]


def _write_markdown(
    output: Path,
    *,
    dataset_variant: str,
    registry_path: Path,
    column: str,
    combined: Counter[str],
    top_unsupported: list[dict[str, Any]],
    missing_count: int,
) -> None:
    rows = int(combined["rows"])
    attached = int(combined["attached_factor_count"])
    supported = int(combined["supported_factor_count"])
    unsupported = int(combined["unsupported_factor_count"])
    lines = [
        "# Unsupported Constraint Diagnostics",
        "",
        f"- dataset_variant: `{dataset_variant}`",
        f"- registry: `{registry_path}`",
        f"- attached_constraint_column: `{column}`",
        f"- rows: {rows:,}",
        f"- attached factor occurrences: {attached:,}",
        f"- supported factor occurrences: {supported:,} ({_format_pct(_rate(supported, attached))})",
        f"- unsupported factor occurrences: {unsupported:,} ({_format_pct(_rate(unsupported, attached))})",
        f"- rows with at least one unsupported factor: {int(combined['rows_with_unsupported']):,} ({_format_pct(_rate(int(combined['rows_with_unsupported']), rows))})",
        f"- primary unsupported rows: {int(combined['primary_unsupported_rows']):,} ({_format_pct(_rate(int(combined['primary_unsupported_rows']), rows))})",
        f"- primary family mismatch rows: {int(combined['primary_family_mismatch_rows']):,} ({_format_pct(_rate(int(combined['primary_family_mismatch_rows']), rows))})",
        f"- rows with only unsupported attached factors: {int(combined['rows_only_unsupported']):,} ({_format_pct(_rate(int(combined['rows_only_unsupported']), rows))})",
        f"- estimated factor-node reduction from supported-only filtering: {_format_pct(_rate(unsupported, attached))}",
    ]
    if missing_count:
        lines.append(f"- missing registry factor occurrences: {missing_count:,}")
    lines.extend(["", "## Top Unsupported Families", ""])
    lines.append("| family | label | factor occurrences | rows with family |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in top_unsupported[:15]:
        lines.append(
            "| {family} | {label} | {occ:,} | {rows:,} |".format(
                family=row["constraint_family"],
                label=row["constraint_label"] or row["constraint_type_item"],
                occ=int(row["attached_factor_occurrences"]),
                rows=int(row["rows_with_family"]),
            )
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose unsupported attached constraints and supported-only filtering impact."
    )
    parser.add_argument("--dataset", default="full_strat1m", help="Dataset variant to scan.")
    parser.add_argument(
        "--min-occurrence",
        type=int,
        default=None,
        help="Dataset variant threshold. Defaults to auto-discovery for the dataset.",
    )
    parser.add_argument(
        "--registry-dataset",
        default=None,
        help="Dataset name for constraint_registry_<dataset>.parquet. Defaults to automatic fallback.",
    )
    parser.add_argument(
        "--interim-root",
        type=Path,
        default=Path("data/interim"),
        help="Root containing interim parquet variants and constraint registries.",
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
        help="Override the attached constraint ID column.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Parquet splits to scan.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Parquet rows per streamed batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostics outputs. Defaults to data/interim/<variant>.",
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

    registry_path = _resolve_registry_path(
        args.interim_root,
        dataset=args.dataset,
        min_occurrence=max(1, min_occurrence),
        registry_dataset=args.registry_dataset,
    )
    encoder_path = dataframe_root / "globalintencoder.txt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder required to map registry IDs to parquet IDs: {encoder_path}")
    encoder = _load_encoder(encoder_path)
    registry = _load_registry(registry_path, encoder=encoder)
    column = args.column or _sequence_column_for_scope(args.scope)
    output_dir = args.output_dir or dataframe_root
    output_dir.mkdir(parents=True, exist_ok=True)

    result = diagnose(
        dataframe_root=dataframe_root,
        registry=registry,
        column=column,
        splits=list(args.splits),
        batch_size=args.batch_size,
    )

    combined = result["combined_stats"]
    metadata = _family_metadata(registry)
    split_rows = _rows_from_split_stats(result["split_stats"])
    total_rows = int(combined["rows"])
    total_factors = int(combined["attached_factor_count"])

    unsupported_rows = _rows_from_family_counts(
        occurrences=result["unsupported_family_occurrences"],
        row_counts=result["unsupported_family_rows"],
        metadata=metadata,
        total_rows=total_rows,
        total_factors=total_factors,
    )
    supported_rows = _rows_from_family_counts(
        occurrences=result["supported_family_occurrences"],
        row_counts=Counter(),
        metadata=metadata,
        total_rows=total_rows,
        total_factors=total_factors,
    )
    primary_rows = [
        {
            "constraint_type": constraint_type,
            "primary_supported": supported,
            "rows": count,
            "row_share": _rate(count, total_rows),
        }
        for (constraint_type, supported), count in result["primary_type_counts"].most_common()
    ]
    primary_family_rows = [
        {
            "constraint_type": constraint_type,
            "registry_constraint_family": registry_family,
            "primary_supported": supported,
            "rows": count,
            "row_share": _rate(count, total_rows),
        }
        for (constraint_type, registry_family, supported), count in result["primary_family_counts"].most_common()
    ]

    split_csv = output_dir / "unsupported_constraint_diagnostics_by_split.csv"
    family_csv = output_dir / "unsupported_constraint_families.csv"
    supported_family_csv = output_dir / "supported_constraint_families.csv"
    primary_csv = output_dir / "primary_constraint_support.csv"
    primary_family_csv = output_dir / "primary_constraint_registry_families.csv"
    row_hist_csv = output_dir / "unsupported_constraints_per_row.csv"
    supported_hist_csv = output_dir / "supported_constraints_per_row.csv"
    total_hist_csv = output_dir / "attached_constraints_per_row.csv"
    missing_csv = output_dir / "missing_registry_constraint_ids.csv"
    markdown_path = output_dir / "unsupported_constraint_diagnostics.md"

    _write_csv(split_rows, split_csv, sort_by=["split"])
    _write_csv(unsupported_rows, family_csv, sort_by=["attached_factor_occurrences"], ascending=False)
    _write_csv(supported_rows, supported_family_csv, sort_by=["attached_factor_occurrences"], ascending=False)
    _write_csv(primary_rows, primary_csv, sort_by=["constraint_type", "primary_supported"])
    _write_csv(
        primary_family_rows,
        primary_family_csv,
        sort_by=["constraint_type", "rows"],
        ascending=[True, False],
    )
    _write_csv(_hist_rows(result["row_unsupported_hist"], "unsupported_constraints"), row_hist_csv)
    _write_csv(_hist_rows(result["row_supported_hist"], "supported_constraints"), supported_hist_csv)
    _write_csv(_hist_rows(result["row_total_hist"], "attached_constraints"), total_hist_csv)
    _write_csv(
        [
            {"constraint_id": constraint_id, "attached_factor_occurrences": count}
            for constraint_id, count in result["missing_constraint_ids"].most_common()
        ],
        missing_csv,
    )
    _write_markdown(
        markdown_path,
        dataset_variant=dataset_variant,
        registry_path=registry_path,
        column=column,
        combined=combined,
        top_unsupported=unsupported_rows,
        missing_count=sum(result["missing_constraint_ids"].values()),
    )

    attached = int(combined["attached_factor_count"])
    unsupported = int(combined["unsupported_factor_count"])
    print(f"Rows scanned: {total_rows:,}")
    print(f"Attached factor occurrences: {attached:,}")
    print(f"Unsupported factor occurrences: {unsupported:,} ({_format_pct(_rate(unsupported, attached))})")
    print(
        "Rows with unsupported factors: "
        f"{int(combined['rows_with_unsupported']):,} "
        f"({_format_pct(_rate(int(combined['rows_with_unsupported']), total_rows))})"
    )
    print(
        "Primary unsupported rows: "
        f"{int(combined['primary_unsupported_rows']):,} "
        f"({_format_pct(_rate(int(combined['primary_unsupported_rows']), total_rows))})"
    )
    print(
        "Primary family mismatch rows: "
        f"{int(combined['primary_family_mismatch_rows']):,} "
        f"({_format_pct(_rate(int(combined['primary_family_mismatch_rows']), total_rows))})"
    )
    print(f"Wrote Markdown: {markdown_path}")
    print(f"Wrote split CSV: {split_csv}")
    print(f"Wrote unsupported family CSV: {family_csv}")


if __name__ == "__main__":
    main()
