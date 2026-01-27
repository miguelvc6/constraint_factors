# 02a_constraint_registry.py

## Objective
- Build a standalone registry of constraint definitions from `constraints.tsv` so graph construction can look up constraint metadata by constraint id without duplicating it per dataframe row.
- Persist the registry at `data/interim/<variant>/constraint_registry.parquet` for downstream stages.

## Inputs & Outputs
- **Inputs:** `data/raw/<dataset>/constraints.tsv` plus CLI flags for dataset/min-occurrence.
- **Outputs:** `data/interim/<variant>/constraint_registry.parquet` containing a single JSON-serialized registry map.

## Workflow
1. Parse CLI arguments (`--dataset`, `--min-occurrence`) and derive the interim variant via `dataset_variant_name()`.
2. Load `constraints.tsv` using the shared `load_constraint_data()` logic from `02_dataframe_builder.py`.
3. For each constraint id, extract:
   - `constraint_type` (from `P2302`),
   - `constrained_property`,
   - parameter predicates and parameter objects.
4. Validate:
   - every constraint id appears exactly once,
   - each constraint has exactly one constrained property,
   - parameters are non-null strings,
   - entry count matches the parsed constraints.
5. Serialize the registry with sorted keys into a single parquet row (`registry_json`).

## Output Schema
The registry maps each constraint id to:
- `constraint_type`: string (e.g., `Q21503247`)
- `constrained_property`: string (e.g., `P31`)
- `param_predicates`: list of strings
- `param_objects`: list of strings

## Common Pitfalls / Gotchas
- The script expects `constraints.tsv` under `data/raw/<dataset>/`; verify the dataset name matches the directory.
- `constraint_registry.parquet` is required by `02b_wikidata_retriever.py`; run this script first when building a fresh pipeline.
