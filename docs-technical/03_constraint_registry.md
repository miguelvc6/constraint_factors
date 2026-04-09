# 03_constraint_registry.py

## Objective
- Build a standalone registry of constraint metadata from `constraints.tsv` so downstream stages can look up constraint semantics by id without repeating the parsing per row.
- Enrich each constraint with a canonical family/type label using the static catalog.
- Persist the registry at `data/interim/constraint_registry_<dataset>.parquet`.

## Inputs & Outputs
- **Inputs:** `data/raw/<dataset>/constraints.tsv`, the `--dataset` CLI flag, and optionally `data/static/constraint_type_catalog.json`.
- **Outputs:** `data/interim/constraint_registry_<dataset>.parquet` containing a single JSON-serialized registry map.

## Workflow
1. Parse the `--dataset` CLI argument and locate `data/raw/<dataset>/constraints.tsv`.
2. Load `constraints.tsv` using the shared `load_constraint_data()` logic from `02_dataframe_builder.py`.
3. If `data/static/constraint_type_catalog.json` is missing, bootstrap it by querying Wikidata for the constraint-type QIDs found in the current dataset's `constraints.tsv`.
4. For each constraint id, extract:
   - `constraint_type` (raw object from `P2302`),
   - `constraint_type_item` (normalized QID),
   - `constraint_family` + `constraint_family_supported` (via `canonicalize_constraint_type()` and `modules.constraint_checkers`),
   - `constraint_label` (from the static catalog, when available),
   - `constrained_property`,
   - parameter predicates and parameter objects.
5. Normalize registry tokens to match the encoder’s canonical strings (preserving `^` prefixes and angle brackets, while normalizing `prop/direct/` to `entity/`).
6. Validate:
   - every constraint id appears exactly once,
   - each constraint has exactly one constrained property,
   - parameters are non-null strings,
   - entry count matches the parsed constraints.
7. Serialize the registry with sorted keys into a single parquet row (`registry_json`).

## Output Schema
The registry maps each constraint id to:
- `constraint_type`: raw string (URI or QID object stored in `constraints.tsv`)
- `constraint_type_item`: normalized QID (e.g., `Q21503247`)
- `constraint_family`: canonical family name (e.g., `conflictWith`)
- `constraint_label`: human-readable label from the static catalog (may be empty)
- `constraint_family_supported`: boolean, True if a checker exists
- `constrained_property`: string (e.g., `P31`)
- `param_predicates`: list of strings (raw predicate URIs normalized for the encoder)
- `param_objects`: list of strings (raw objects normalized for the encoder)

## Common Pitfalls / Gotchas
- The script expects `constraints.tsv` under `data/raw/<dataset>/`; verify the dataset name matches the directory.
- Automatic catalog bootstrap requires outbound network access to Wikidata on the first run in a fresh clone.
- `constraint_registry_<dataset>.parquet` is required by `04_wikidata_retriever.py`; run this script first when building a fresh pipeline.
