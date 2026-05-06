# 02b_stratified_benchmark_sampler.py

## Objective
- Create a fixed derived benchmark variant from interim parquet splits without changing the raw corpus or encoder.
- Reduce the paper dataset size by deterministic stratified sampling while preserving split boundaries, primary constraint-family coverage, and local constraint-density coverage.

## Inputs & Outputs
**Inputs**
- Source parquet splits under `data/interim/<source_variant>/`.
- `globalintencoder.txt` from the source variant.
- CLI flags for source/output dataset names, `min_occurrence`, sample fraction, seed, and constraint scope.

**Outputs**
- Sampled parquet splits under `data/interim/<output_variant>/`.
- Copied `globalintencoder.txt`.
- `sampling_report.csv`, `sampling_report.md`, and `sampling_metadata.json`.
- `hist_local_constraint_ids.csv` and `hist_local_constraint_ids_by_split.csv` for the sampled variant.

## Sampling Policy
The paper default is:
```bash
uv run src/02b_stratified_benchmark_sampler.py \
  --source-dataset full \
  --output-dataset full_strat1m \
  --min-occurrence 100 \
  --sample-fraction 0.5 \
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

For every non-empty stratum, the sampler keeps:
```python
max(1, round(source_count * sample_fraction))
```

The sampled row order is preserved within each split, and row selection is deterministic for a fixed seed.

## Pipeline Position
Run this stage after `02_dataframe_builder.py` and before `05_constraint_labeler.py`.
This avoids labeling and graph-building work for rows that are not part of the paper benchmark.

The constraint registry remains the raw-source registry, for example
`data/interim/constraint_registry_full.parquet`; the sampled variant is a derived dataframe artifact and does not own a new `constraints.tsv`.
