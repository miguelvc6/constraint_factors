# Data Artifact Specification

This document specifies the data artifacts produced by the constraint-factors
pipeline. It follows the executable pipeline in `src/` and the conceptual model
in [docs-conceptual/00-constraint_factors.md](/home/mvazquez/constraint_factors/docs-conceptual/00-constraint_factors.md).

## 1) Dataset Variants
Variants are named with:
- `<dataset>` when `min_occurrence <= 1`
- `<dataset>_minocc<k>` when `min_occurrence > 1`

This naming is produced by `dataset_variant_name()` in `src/modules/data_encoders.py`.

## 2) Raw Data (`01_data_downloader.py`)
**Location:** `data/raw/<dataset>/`

**Layout**
- `constraints.tsv`
- `constraint-corrections-*.tsv.gz.full.{train,dev,test}.tsv.gz` (full dataset)
- For the sample dataset, the downloader mirrors the same layout by extracting
  `constraints.tsv` and `constraint-corrections/` from the GitHub archive.

The downloader also ensures `data/.gitignore` exists to keep large artifacts untracked.

## 3) Interim Parquet Splits (`02_dataframe_builder.py`)
**Location:** `data/interim/<variant>/`

**Artifacts**
- `df_train.parquet`, `df_val.parquet`, `df_test.parquet`
- `identity_encoder.txt` (lossless semantic vocabulary)
- `feature_encoder.txt` (training-frequency-filtered vocabulary)
- `identity_to_feature.npy`
- `globalintencoder.txt` (compatibility alias of the feature encoder)
- `dataset_manifest.json` (schema v2, split/row/source/output hashes)

### 3.1 Parquet Schema
Each dataframe row represents one constraint violation instance with the fields below.

**Scalar integer features (int64)**
- `constraint_id`
- `subject`, `predicate`, `object`
- `other_subject`, `other_predicate`, `other_object`
- `add_subject`, `add_predicate`, `add_object`
- `del_subject`, `del_predicate`, `del_object`

Every listed scalar also has a `<name>_feature` companion. The original column
is an identity ID; the companion is a model feature ID.

**Sequence features (object arrays of int lists)**
- `constraint_predicates`, `constraint_objects`
- `subject_predicates`, `subject_objects`
- `object_predicates`, `object_objects`
- `other_entity_predicates`, `other_entity_objects`
- `local_constraint_ids`
- `local_constraint_ids_focus`

The first eight model-bearing sequence pairs also have `_feature` companions.
Local constraint ID lists remain semantic identities.

**Text / string features**
- `constraint_type` (string, e.g. `conflictWith`)
- `object_text` (literal text or empty string)
- `other_object_text` (literal text or empty string)

### 3.2 Semantics
- Identity IDs are never frequency filtered. Feature IDs are computed from
  training-only occurrence counts; rare identities map to feature `unknown`.
- Literal objects are recorded in `*_text` fields and retain a distinct raw-token
  identity ID. Rare literals may share the feature-space `unknown` ID only.
- `local_constraint_ids` is the union of:
  - the row’s `constraint_id`, and
  - every constraint attached to predicates in the local neighborhood (`P_local`).
- `local_constraint_ids_focus` is a narrower scope:
  - the row’s `constraint_id`, and
  - constraints attached to the focus predicate(s), plus the constrained property
    of the violated constraint.

## 4) Constraint Registry (`03_constraint_registry.py`)
**Location:** `data/interim/constraint_registry_<dataset>.parquet`

**Schema**
The parquet file contains a single column `registry_json` with a JSON object
mapping each constraint id to:
- `constraint_type` (raw object from `P2302`)
- `constraint_type_item` (normalized QID)
- `constraint_family` (canonical family name)
- `constraint_label` (catalog label, if available)
- `constraint_family_supported` (bool)
- `constrained_property`
- `param_predicates`
- `param_objects`

## 4b) Stratified Benchmark Variant (`02b_stratified_benchmark_sampler.py`)
**Location:** `data/interim/<derived_variant>/`

The paper-facing derived benchmark is `full_strat1m_minocc100`, produced from a
validated/primary-filtered full variant by deterministic exact-size stratified sampling.

**Artifacts**
- `df_train.parquet`, `df_val.parquet`, `df_test.parquet`
- the complete identity/feature encoder contract copied unchanged
- `class_hierarchy.parquet` and `class_hierarchy_manifest.json`, copied
  unchanged from the validated source
- parent-linked `dataset_manifest.json`
- `sampling_report.csv`, `sampling_report.md`, `sampling_metadata.json`
- `sample_primary_validation_audit_by_constraint.csv`
- `sample_gold_repair_audit_by_constraint.csv`
- `hist_local_constraint_ids.csv`
- `hist_local_constraint_ids_by_split.csv`

Sampling strata are `(split, constraint_type, attached_constraint_bin)`, where:
```python
num_attached_constraints = len(local_constraint_ids)
```

Default bins are `1-32`, `33-64`, `65-83`, `84-107`, `108`,
`109-160`, `161-267`, and `268+`. The paper run uses
`--target-rows 1000000 --seed 42`; proportional allocation yields exactly one
million retained rows while keeping at least one row per represented stratum.
The two sampled semantic audits report the realized family composition and
POST_GOLD verification status; their counts are also embedded under `sampling`
in the dataset manifest.

## 5) Wikidata Text Cache (`04_wikidata_retriever.py`)
**Location:** `data/interim/wikidata_text.parquet`

**Schema**
- `key` (string): canonical URI, placeholder token, or literal text
- `kind` (string): `uri`, `placeholder`, or `literal`
- `global_id` (Int64): encoder id for `uri`/`placeholder`, null for literals
- `text` (string): resolved label or literal text
- `embedding` (list[float16]): dense embedding vector

The cache is shared across dataset variants and is incrementally updated when
re-run.

## 6) Graph Artifacts (`06_graph.py`)
**Location:** `data/processed/<variant>/`

**Files**
- Factorized files: `{split}_graph-<encoding>.pkl`
- Passive files: `{split}_graph_repr-eswc_passive-<encoding>.pkl`
- Sharded variants: the same base names with `-shardNNN.pt` or `.pkl`
- Per-split manifest: `<graph_filename>.manifest.json`
- Incomplete-build marker: `<graph_filename>.incomplete` (present only until a build or conversion publishes a valid manifest)
- `target_vocabs.json` (class-id vocabularies for labels)
- Optional visuals: `graph_visualization.png`, `graph_visualization-non_flattened.png`

**Data object fields**
- `x`: node features (float embeddings or int IDs)
- `node_identity_id`: semantic identity for each local node
- `edge_index`: flattened edges `(subject -> predicate -> object)`
- `edge_type`: integer edge types for base vs factor wiring
- `edge_index_non_flattened`, `edge_attr_non_flattened`: subject→object edges + predicate attributes
- `y`: `(1, 6)` feature-space training target
- `y_identity`: `(1, 6)` strict semantic target
- `target_representable_mask`: whether each identity target survived feature filtering
- `x_names`: optional node name list (used when debugging)
- `role_flags`: bitmask for focus subject/predicate/object nodes
- `focus_triple`: global IDs of the focus triple `(s, p, o)`
- `focus_triple_feature`: feature IDs for the same focus triple
- `shape_id`: the encoded `constraint_id`
- `constraint_type`: string (e.g., `conflictWith`)
- `constraint_representation`: `factorized` or `eswc_passive`
- `primary_constraint_mode`: `executable_factor`, `query_definition`, `query_family`, `passive_node`, or `none`
- `factor_constraint_ids`: list of executable constraint IDs included as model factors
- `factor_node_index`: local node indices of executable factor nodes
- `primary_factor_index`: index of the violated constraint in `factor_constraint_ids`; `-1` when the primary is not executable
- `eval_factor_constraint_ids`: full local constraint ID list used for strict symbolic evaluation
- `eval_factor_types`, `eval_factor_checkable_*`, `eval_factor_satisfied_*`: eval-side mirrors aligned with `eval_factor_constraint_ids`
- `eval_primary_factor_index`: index of the violated constraint in `eval_factor_constraint_ids`
- `primary_constraint_id`, `primary_constraint_type_id`, `primary_constrained_property_id`: primary task-query metadata
- `primary_param_predicate_ids`, `primary_param_object_ids`, `primary_param_count`: variable-length primary definition metadata used by query modes
- `passive_primary_node_index`: local node index of the passive primary node when `primary_constraint_mode=passive_node`
- `is_factor_node`: boolean mask over local nodes
- `factor_constraint_types`: list of constraint family labels (debug)
- `factor_wiring_debug` (optional): wiring diagnostics when `--debug-factor-wiring` is enabled
- `context_index` (optional): integer index into violation contexts, attached later by training/evaluation code for context-aligned objectives; it is not written by `06_graph.py`

**Persistence profiles**
- `research_safe` (default): drops debug-only fields `x_names`, `factor_constraint_types`, `factor_wiring_debug`.
- `full`: retains all fields including debug-only attributes.

`target_vocabs.json` contains:
- `entity_class_ids`
- `predicate_class_ids`
- `per_split` (per-split versions of the above)

**Notes**
- Factor label tensors (`factor_checkable_*`, `factor_satisfied_*`, `factor_types`) are 1-D and must align with
  `factor_constraint_ids` length. In non-default primary modes, these executable factors exclude the primary.
- Eval label tensors (`eval_factor_checkable_*`, `eval_factor_satisfied_*`, `eval_factor_types`) are 1-D and must align with `eval_factor_constraint_ids`; strict global metrics prefer these fields and fall back to `factor_*` for old artifacts.

**Graph manifest schema v3**

Each split manifest records the dataset-manifest path/full SHA-256, split
parquet path/full SHA-256/source row count, exact graph mode fields, explicit
build limit, graph count, and one artifact record per payload. Artifact records
contain `path`, `bytes`, `object_count`, and a full `sha256`. Converted modes
also contain a `derivation` block with the source graph-manifest path/hash,
source/target primary modes, method (`rewrite` or `hard_link`), and per-artifact
lineage. The manifest is immutable provenance and remains after optional
post-acceptance payload pruning.

## 7) Labeled Constraint Factors (`05_constraint_labeler.py`)
**Location:** `data/interim/<variant>_labeled/`

**Files**
- `df_train.parquet`, `df_val.parquet`, `df_test.parquet` (with extra columns)
- `coverage_<scope>.csv`, `coverage_<scope>.md`
- `filtered_factors_<scope>.csv`, `filtered_factors_<scope>.md`
- `filtered_factor_families_<scope>.csv`
- `primary_validation_audit.csv`
- `primary_validation_audit_by_constraint.csv`
- `primary_gold_repair_audit_by_constraint.csv`
- `class_hierarchy.parquet`
- `class_hierarchy_manifest.json`
- `dataset_manifest.json`

**Additional parquet columns**
- `factor_checkable_pre`, `factor_satisfied_pre`
- `factor_checkable_post_gold`, `factor_satisfied_post_gold`
- `factor_types`
- `factor_constraint_ids`
- `num_checkable_factors_pre`, `coverage_pre`
- `num_checkable_factors_post_gold`, `coverage_post_gold`
- `primary_factor_index`, `primary_checkable_pre`, `primary_satisfied_pre`
- `primary_checkable_post_gold`, `primary_satisfied_post_gold`
- `primary_validation_reason`
- `primary_gold_repair_status`, `primary_gold_repair_verified`

The labeler can operate on either `local_constraint_ids` or
`local_constraint_ids_focus`, controlled by `--constraint-scope`.
By default, `--factor-family-policy supported_only` keeps raw local-closure
columns unchanged but writes only supported executable constraints to aligned
factor arrays. Paper data additionally uses `--filter-invalid-primary`, which
excludes every primary that is exempt, out of scope, unsupported, uncheckable,
or already satisfied and records the reason. The family-level audit prevents an
entire primary task family from disappearing behind aggregate exclusion counts.
Before checking PRE labels, the gold transition is reversed against serialized
entity facts: additions are removed and deletions are restored. The manifest
records the executable constraint semantics version used for this normalization.
Semantics v4 also freezes a training-only direct `P279` graph and uses its
reflexive-transitive closure for `type` and `valueType`. The hierarchy manifest
records its training source and hashes. Primary eligibility remains a PRE-state
definition; the separate gold-repair fields report whether POST_GOLD is
checkable and satisfied under the same executable semantics.
When this directory exists, `06_graph.py` uses it automatically unless
`--use-unlabeled-interim` is passed.

## 8) Attached Constraint Histogram (`scripts/hist_attached_constraints.py`)
The attached-constraint count for a row is:
```python
num_attached_constraints = len(local_constraint_ids)
```

Use the streaming histogram script to summarize this count without loading the
full parquet dataset into RAM:
```bash
uv run scripts/hist_attached_constraints.py \
  --dataset full_strat1m \
  --min-occurrence 100
```

By default it scans `df_train.parquet`, `df_val.parquet`, and
`df_test.parquet` under `data/interim/<variant>/`, reading only
`local_constraint_ids` in Arrow batches. It writes:
- `data/interim/<variant>/hist_local_constraint_ids.csv`
- `data/interim/<variant>/hist_local_constraint_ids.png`

Useful flags:
- `--scope focus` histograms `local_constraint_ids_focus`.
- `--by-split` also writes per-split counts.
- `--batch-size` controls streamed parquet batch size.
- `--no-plot` skips PNG generation.

## 9) Unsupported Constraint Diagnostics (`scripts/diagnose_unsupported_constraints.py`)
The labeler coverage table reports unsupported constraint families at
factor-occurrence level. Use the diagnostics script to separate row-level
prevalence from factor-level graph pressure:
```bash
uv run scripts/diagnose_unsupported_constraints.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --registry-dataset full \
  --scope local
```

The script streams `constraint_type`, `constraint_id`, and the selected attached
constraint ID column from `data/interim/<variant>/df_*.parquet`, resolves IDs
through `globalintencoder.txt` and `constraint_registry_<registry>.parquet`, and
writes:
- `unsupported_constraint_diagnostics.md`
- `unsupported_constraint_diagnostics_by_split.csv`
- `unsupported_constraint_families.csv`
- `supported_constraint_families.csv`
- `primary_constraint_support.csv`
- `primary_constraint_registry_families.csv`
- `unsupported_constraints_per_row.csv`
- `supported_constraints_per_row.csv`
- `attached_constraints_per_row.csv`
- `missing_registry_constraint_ids.csv`

The main summary includes rows with unsupported attached factors, supported vs
unsupported attached factor occurrences, primary unsupported rows, primary
constraint-family mismatches, and the estimated factor-node reduction from a
supported-only factor policy.
