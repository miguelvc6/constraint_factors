# 02_dataframe_builder.py

## Objective
- Transform the raw gzipped TSV corrections (downloaded by `01_data_downloader.py`) into cleaned, tokenised, train/val/test parquet splits stored under `data/interim/<dataset>_minocc<k>/`.
- Build and persist the `GlobalIntEncoder` so every subsequent stage shares a stable integer vocabulary for entities, predicates, literals, placeholders, and constraint metadata.

## Inputs & Outputs
- **Inputs:** Raw files in `data/raw/<dataset>/` (`constraint-corrections-*.tsv.gz`, `constraints.tsv`), CLI flags `--dataset` and `--min-occurrence`.
- **Outputs:** `data/interim/<variant>/df_{train,val,test}.parquet`, the pruned `globalintencoder.txt`, and derived metadata such as the property → constraint-id index produced by `load_constraint_data()`.

## Workflow
1. **Argument handling** – `--dataset {sample,full}` chooses the raw input root, while `--min-occurrence` configures how aggressively rare tokens get mapped to `unknown`.
2. **Constraint schema load** – `load_constraint_data()` reads `constraints.tsv` once, remapping the Wikidata predicate IRIs so every violation row can expand its constraint graph cheaply, and builds a property → constraint-id index for local closure.
3. **Record parsing** – `load_dataset()` iterates the gzipped TSV per constraint target (`conflictWith`, `distinct`, …):
   - `_convert_value()` encodes each string via `GlobalIntEncoder`, also swapping repeated references to the current triple (subject/predicate/object) or its “other” counterparts for reserved placeholders such as `subject` or `other_object`.
   - `_read_entity_desc()` decodes the JSON blobs describing the neighborhood (labels, other facts, HTTP fallback pages) and normalises them into predicate/object lists so they can be appended to the feature arrays.
   - Literal objects are stored in `<feature>_text` columns while entity IDs stay numeric, allowing text-only nodes later in `03_graph.py`.
4. **Dataset assembly** – `load()` stitches every constraint-type file into a single dictionary per split, converting Python lists to `numpy` arrays (object dtype for ragged sequences, numeric for scalars).
5. **Global split** – All raw splits are concatenated and repartitioned via `stratified_train_val_test_split()` to guarantee consistent constraint-type proportions.
6. **Frequency filtering** – `_compute_token_frequency()` inspects only the training split to decide which IDs survive the `MIN_OCCURRENCE` threshold. `_apply_frequency_filter_inplace()` replaces infrequent IDs with `UNKNOWN_TOKEN_ID`, `_prune_encoder()` and `_reindex_encoder()` compress the vocabulary, and `_remap_dataset_inplace()` updates every split accordingly.
7. **Persistence** – Each final dictionary becomes a pandas dataframe that is written as `df_train.parquet`, `df_val.parquet`, `df_test.parquet`, and the encoder is saved as `globalintencoder.txt`.

## Common Pitfalls / Gotchas
- Memory spikes happen while concatenating large constraint targets; for the full dataset keep 20–30 GB of RAM free or run per-target debugging subsets via `max_size`.
- Changing `--min-occurrence` invalidates every downstream artifact (graphs, embeddings, models) because token IDs shift. Rerun the entire pipeline when you tweak it.
- If `constraints.tsv` is missing or mismatched with the TSV dumps, `load_constraint_data()` will silently drop rows whose constraint IDs are unknown, shrinking the dataset.

## Implementation Details
- Reserved placeholders (`subject`, `predicate`, `object`, `other_*`, `LITERAL_OBJECT`, `unknown`) are always injected into the encoder via `_ensure_reserved_tokens()` so downstream models can rely on fixed IDs even after pruning.
- `LITERAL_OBJECT` is one of the reserved placeholder tokens the pipeline injects into the GlobalIntEncoder. During dataframe building, every triple slot that holds a plain literal (e.g., `"Paris"@en` or a date) can’t be mapped to a Wikidata entity ID, so the script stores `object = 0` and keeps the raw text in `object_text`. Later, when graphs are built, these literal-only nodes still need an integer class so models can point to them; `LITERAL_OBJECT` is that special token. It ensures literals share a consistent ID (and, after 02b_wikidata_retriever.py, an embedding) even though they don’t correspond to real Wikidata entities.
- `_apply_frequency_filter_inplace()` works on both scalar and sequence features, preserving zero values (used for padding) while masking only the genuinely rare identifiers.
- Literal overlap heuristics compare subject/object labels against cached HTML snippets, inserting synthetic `pageContainsLabel` edges that graph construction later turns into nodes.
- By delaying pandas materialisation until after frequency pruning and split creation, the script keeps memory pressure manageable even for the full dataset.
