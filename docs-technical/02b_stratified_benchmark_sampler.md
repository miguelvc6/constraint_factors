# 02b_stratified_benchmark_sampler.py

## Objective
- Create a fixed derived benchmark variant from interim parquet splits without changing the raw corpus or encoder.
- Reduce the paper dataset size by deterministic stratified sampling while preserving split boundaries, primary constraint-family coverage, and local constraint-density coverage.
- Emit each sampled split in a deterministic mixed order so streamed graph training does not consume one constraint-family block at a time.

## Inputs & Outputs
**Inputs**
- Source parquet splits under `data/interim/<source_variant>/`.
- The v2 encoder contract and dataset manifest from the source variant.
- CLI flags for source/output dataset names, `min_occurrence`, an exact target or sample fraction, seed, and constraint scope.

**Outputs**
- Sampled parquet splits under `data/interim/<output_variant>/`.
- Copied identity/feature encoders and `identity_to_feature.npy`.
- An updated `dataset_manifest.json` linked to its parent manifest.
- `sampling_report.csv`, `sampling_report.md`, and `sampling_metadata.json`.
- `hist_local_constraint_ids.csv` and `hist_local_constraint_ids_by_split.csv` for the sampled variant.

## Sampling Policy
The paper default is:
```bash
uv run src/02b_stratified_benchmark_sampler.py \
  --source-dataset full \
  --output-dataset full_strat1m \
  --min-occurrence 100 \
  --target-rows 1000000 \
  --seed 42 \
  --scope local
```

This reads from `data/interim/full_minocc100/` and writes to
`data/interim/full_strat1m_minocc100/`.

Rows are stratified by:
- split (`train`, `val`, `test`)
- primary constraint family (`constraint_type`)
- attached-constraint bin from `len(local_constraint_ids)`

Default bins:
- `1-32`
- `33-64`
- `65-83`
- `84-107`
- `108`
- `109-160`
- `161-267`
- `268+`

`--target-rows` allocates the requested total proportionally across non-empty
strata using deterministic largest-remainder allocation, subject to one row per
represented stratum. The output total is exact. `--sample-fraction` remains
available but is mutually exclusive with `--target-rows`.

Row membership and order are deterministic for a fixed seed. After selection,
the sampler orders rows by a seeded SplitMix64 key derived from the source row
index. It performs this as a 64-bucket external sort, so the one-million-row
benchmark does not need to fit in memory. The source family-block order is
intentionally not preserved: streamed graph datasets cannot use DataLoader
shuffling, and preserving those blocks caused each epoch to train on long runs
of a single family. `sampling_metadata.json` and `dataset_manifest.json` record
the exact row-order method, seed, and positive bucket count. The dataset
manifest also lists the full SHA-256 of `sampling_metadata.json`; sampled-data
integrity validation requires both copies to agree. Unsampled schema-v2 variants
remain valid without a `sampling.row_order` block.

## Pipeline Position
For paper-facing schema-v2 data, run this stage after `05_constraint_labeler.py`
has filtered invalid primary rows into a standalone source variant. Sampling the
validated source makes the final benchmark size exactly `--target-rows` instead
of sampling first and then shrinking unpredictably during validation. Sampling
before labeling remains acceptable only for development subsets.

The constraint registry remains the raw-source registry, for example
`data/interim/constraint_registry_full.parquet`; the sampled variant is a derived dataframe artifact and does not own a new `constraints.tsv`.
