# Data Artifact Specification

This document specifies the data artifacts produced by the constraint-factors
pipeline. It follows the executable pipeline in `src/` and the conceptual model
in `docs/00-constraint_factors.md`.

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
- `globalintencoder.txt` (pickled `GlobalIntEncoder`)

### 3.1 Parquet Schema
Each dataframe row represents one constraint violation instance with the fields below.

**Scalar integer features (int64)**
- `constraint_id`
- `subject`, `predicate`, `object`
- `other_subject`, `other_predicate`, `other_object`
- `add_subject`, `add_predicate`, `add_object`
- `del_subject`, `del_predicate`, `del_object`

**Sequence features (object arrays of int lists)**
- `constraint_predicates`, `constraint_objects`
- `subject_predicates`, `subject_objects`
- `object_predicates`, `object_objects`
- `other_entity_predicates`, `other_entity_objects`
- `local_constraint_ids`
- `local_constraint_ids_focus`

**Text / string features**
- `constraint_type` (string, e.g. `conflictWith`)
- `object_text` (literal text or empty string)
- `other_object_text` (literal text or empty string)

### 3.2 Semantics
- All IDs are encoded by the shared `GlobalIntEncoder`.
- Literal objects are recorded in `*_text` fields; their numeric IDs may be `0`.
- `local_constraint_ids` is the union of:
  - the row’s `constraint_id`, and
  - every constraint attached to predicates in the local neighborhood (`P_local`).
- `local_constraint_ids_focus` is a narrower scope:
  - the row’s `constraint_id`, and
  - constraints attached to the focus predicate(s), plus the constrained property
    of the violated constraint.

## 4) Constraint Registry (`02a_constraint_registry.py`)
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

## 5) Wikidata Text Cache (`02b_wikidata_retriever.py`)
**Location:** `data/interim/wikidata_text.parquet`

**Schema**
- `key` (string): canonical URI, placeholder token, or literal text
- `kind` (string): `uri`, `placeholder`, or `literal`
- `global_id` (Int64): encoder id for `uri`/`placeholder`, null for literals
- `text` (string): resolved label or literal text
- `embedding` (list[float16]): dense embedding vector

The cache is shared across dataset variants and is incrementally updated when
re-run.

## 6) Graph Artifacts (`03_graph.py`)
**Location:** `data/processed/<variant>/`

**Files**
- `{split}_graph-<encoding>.pkl` (pickle stream of `torch_geometric.data.Data`)
- Sharded variants: `{split}_graph-<encoding>-shardNNN.pt` or `.pkl`
- `target_vocabs.json` (class-id vocabularies for labels)
- Optional visuals: `graph_visualization.png`, `graph_visualization-non_flattened.png`

**Data object fields**
- `x`: node features (float embeddings or int IDs)
- `edge_index`: flattened edges `(subject -> predicate -> object)`
- `edge_type`: integer edge types for base vs factor wiring
- `edge_index_non_flattened`, `edge_attr_non_flattened`: subject→object edges + predicate attributes
- `y`: `(1, 6)` tensor `[add_s, add_p, add_o, del_s, del_p, del_o]`
- `x_names`: optional node name list (used when debugging)
- `role_flags`: bitmask for focus subject/predicate/object nodes
- `focus_triple`: global IDs of the focus triple `(s, p, o)`
- `shape_id`: the encoded `constraint_id`
- `constraint_type`: string (e.g., `conflictWith`)
- `factor_constraint_ids`: list of constraint IDs included as factors
- `primary_factor_index`: index of the violated constraint in `factor_constraint_ids`
- `factor_constraint_types`: list of constraint family labels (debug)
- `factor_wiring_debug` (optional): wiring diagnostics when `--debug-factor-wiring` is enabled

`target_vocabs.json` contains:
- `entity_class_ids`
- `predicate_class_ids`
- `per_split` (per-split versions of the above)

## 7) Labeled Constraint Factors (`04_constraint_labeler.py`)
**Location:** `data/interim/<variant>_labeled/`

**Files**
- `df_train.parquet`, `df_val.parquet`, `df_test.parquet` (with extra columns)
- `coverage_<scope>.csv`, `coverage_<scope>.md`

**Additional parquet columns**
- `factor_checkable_pre`, `factor_satisfied_pre`
- `factor_checkable_post_gold`, `factor_satisfied_post_gold`
- `factor_types`
- `factor_constraint_ids`
- `num_checkable_factors_pre`, `coverage_pre`
- `num_checkable_factors_post_gold`, `coverage_post_gold`

The labeler can operate on either `local_constraint_ids` or
`local_constraint_ids_focus`, controlled by `--constraint-scope`.
