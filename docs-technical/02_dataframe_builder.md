# 02_dataframe_builder.py

## Objective
- Transform the raw gzipped TSV corrections (downloaded by `01_data_downloader.py`) into cleaned, tokenised, train/val/test parquet splits stored under `data/interim/<variant>/` (where `<variant>` is either `<dataset>` or `<dataset>_minocc<k>`).
- Build separate semantic-identity and frequency-filtered feature vocabularies. Scientific evaluation uses identity IDs; model inputs use feature IDs.

## Inputs & Outputs
- **Inputs:** Raw files in `data/raw/<dataset>/` (`constraint-corrections-*.tsv.gz`, `constraints.tsv`), CLI flags `--dataset`, `--output-dataset`, `--split-policy`, `--min-occurrence`, and optional `--max-rows`.
- **Outputs:** `data/interim/<variant>/df_{train,val,test}.parquet`, `identity_encoder.txt`, `feature_encoder.txt`, `identity_to_feature.npy`, the compatibility alias `globalintencoder.txt`, and `dataset_manifest.json`.

## Workflow
1. **Argument handling** – `--dataset {sample,full}` chooses the raw input root, `--min-occurrence` configures how aggressively rare tokens get mapped to `unknown`, and `--max-rows` optionally caps the total rows processed across all input files.
2. **Constraint schema load** – `load_constraint_data()` reads `constraints.tsv` once, remapping the Wikidata predicate IRIs so every violation row can expand its constraint graph cheaply, and builds a property → constraint-id index for local closure.
3. **Factor token seeding** – every constraint id is pre-seeded into the encoder as `constraint_factor::<id>` so `06_graph.py` can build stable factor nodes even after the encoder is frozen.
3. **Record parsing** – `load_dataset()` iterates the gzipped TSV per constraint target (`conflictWith`, `distinct`, …):
   - `_convert_value()` encodes each string via `GlobalIntEncoder`, also swapping repeated references to the current triple (subject/predicate/object) or its “other” counterparts for reserved placeholders such as `subject` or `other_object`.
   - `_read_entity_desc()` decodes the JSON blobs describing the neighborhood (labels, other facts, HTTP fallback pages) and normalises them into predicate/object lists so they can be appended to the feature arrays.
   - Literal objects are stored in `<feature>_text` columns while entity IDs stay numeric, allowing text-only nodes later in `06_graph.py`.
   - `local_constraint_ids` are computed per row by taking `P_local` (the set of property QIDs found in the main predicate, `other_predicate`, and all neighborhood predicate lists) and then unioning every constraint in `constraints_by_property[p]` for `p ∈ P_local`, plus the row’s own `constraint_id`. The final list is unique and sorted by integer ID.
   - `local_constraint_ids_focus` captures a narrower focus scope: constraints attached to the focus predicate(s) plus the constrained property of the violated constraint.
4. **Dataset assembly** – `load()` stitches every constraint-type file into a single dictionary per split, converting Python lists to `numpy` arrays (object dtype for ragged sequences, numeric for scalars).
5. **Split handling** – The default `--split-policy preserve` maps upstream `train`, `dev`, and `test` to local `train`, `val`, and `test` without pooling or repartitioning. `--split-policy restratify` exists only for legacy reproduction. A debug `--max-rows` cap is applied independently to each upstream split.
6. **Frequency filtering** – `_compute_token_frequency()` inspects only the training split. Original IDs remain in the identity-bearing columns. Every model-bearing scalar/sequence also receives a `<column>_feature` companion in the compact feature vocabulary; rare identities map to the feature `unknown` ID without becoming semantically identical.
   - Registry-derived tokens are reserved before pruning: all constrained property IDs, constraint parameter predicates, and constraint parameter objects from `constraints.tsv` are encoded into the vocabulary and added to the reserved set so factor definitions remain representable even if their corpus frequency is below `MIN_OCCURRENCE`.
7. **Persistence** – Each final dictionary becomes a parquet split. The identity encoder, feature encoder, identity-to-feature map, raw/source hashes, split policy, row counts, and output hashes are recorded in `dataset_manifest.json` (schema v2).
8. **Optional derived benchmark sampling** – `02b_stratified_benchmark_sampler.py` can be run after this stage to create the paper-facing `full_strat1m_minocc100` variant from `full_minocc100`, before constraint labeling and graph construction.

## Common Pitfalls / Gotchas
- Memory spikes happen while concatenating large constraint targets; for the full dataset keep 20–30 GB of RAM free or use `--max-rows` for debugging subsets.
- Changing `--min-occurrence` changes model features and still invalidates graphs/models, but no longer changes semantic identity labels. Report strict identity metrics and representability coverage for every threshold.
- If `constraints.tsv` is missing or mismatched with the TSV dumps, `load_constraint_data()` will silently drop rows whose constraint IDs are unknown, shrinking the dataset.

## Implementation Details
- Reserved placeholders (`subject`, `predicate`, `object`, `other_*`, `LITERAL_OBJECT`, `unknown`) are always injected into the encoder via `_ensure_reserved_tokens()` so downstream models can rely on fixed IDs even after pruning.
- Constraint factor tokens follow the exact format `constraint_factor::<constraint_id>` and are seeded up-front, then preserved during pruning so their IDs remain stable for graph construction.
- Plain literals (for example `"Paris"@en` or a date) retain their full raw token in the identity encoder and their text in `object_text`; they therefore remain distinct graph nodes. Frequency filtering may map rare literals to the feature-space `unknown` ID without merging their semantic identities. `LITERAL_OBJECT` remains a reserved fallback for legacy/cache paths.
- Frequency remapping works on both scalar and sequence feature companions, preserving zero values while leaving the corresponding identity columns unchanged.
- Literal overlap heuristics compare subject/object labels against cached HTML snippets, inserting synthetic `pageContainsLabel` edges that graph construction later turns into nodes.
- By delaying pandas materialisation until after frequency pruning and split creation, the script keeps memory pressure manageable even for the full dataset.
- The script logs the number of base reserved tokens, registry reserved tokens, and the final vocab size after pruning to make encoder growth visible.
