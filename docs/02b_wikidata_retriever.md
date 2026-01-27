# 02b_wikidata_retriever.py

## Objective
- Precompute human-readable labels and SentenceTransformer embeddings for every identifier or literal text that appears in the interim parquet splits.
- Materialise the lookup cache at `data/interim/wikidata_text.parquet` so `03_graph.py` can turn node IDs into dense features without repeatedly calling external services.

## Inputs & Outputs
- **Inputs:** `data/interim/<variant>/df_{train,val,test}.parquet`, `data/interim/constraint_registry_{dataset}.parquet`, the matching `globalintencoder.txt`, CLI flags for dataset/min-occurrence/embed settings, and outbound network access to Wikidata + the embedding model hub.
- **Outputs:** `data/interim/wikidata_text.parquet` (or a custom `--output` path) containing keys, texts, and float16 embeddings for URIs, placeholders, and literals.

## Workflow
1. Parse CLI arguments (`--dataset`, `--min-occurrence`, `--embed-dim`, `--batch-size`, `--output`). If the frequency threshold is omitted, `discover_min_occurrence()` inspects `data/interim/` to reuse the smallest available variant.
2. Load the frozen `GlobalIntEncoder` that was saved by `02_dataframe_builder.py` and determine which parquet directory to scan via `dataset_variant_name()`.
3. `_prepare_identifier_sets()` walks `df_train/val/test.parquet`:
   - Collects every non-zero integer from the scalar columns listed in `SCALAR_FEATURES`.
   - Traverses ragged columns listed in `SEQUENCE_FEATURES`.
   - Gathers all strings from `*_text` columns.
   - Ensures placeholder tokens like `subject`, `predicate`, or `LITERAL_OBJECT` are always included so embeddings exist for synthetic nodes as well.
4. `_load_constraint_registry()` reads `constraint_registry_{dataset}.parquet` and `_collect_registry_ids()` resolves constrained properties and parameter predicate/object IDs via the frozen encoder, then unions them into the retrieval set. Unknown registry IDs are skipped with a warning instead of aborting.
5. `_load_existing_cache()` (if present) keeps previously embedded rows in memory, keyed by `(kind, key, global_id)`. This lets repeated runs skip already resolved URIs/literals.
6. `_materialise_entries()` handles the heavy lifting:
   - URIs are resolved via `WikidataUriEmbedder.embed_uris()`, which fetches labels (with HTTP batching and caching handled inside `modules.wikidata_utils`).
   - Placeholder tokens reuse cached embeddings when possible or get embedded in batches using `get_embeddings_from_texts()`.
   - Literal strings are embedded purely from text to cover values that never map to a Wikidata ID (e.g., dates or external identifiers).
7. `_persist_records()` sorts every `CacheEntry`, writes them as a parquet file with float16 embeddings, and allows other stages to reconstruct `PrecomputedWikidataCache` instances quickly.

## Common Pitfalls / Gotchas
- If you regenerate parquet splits with a different `--min-occurrence`, you must rerun this script; otherwise, `03_graph.py` will reference IDs that lack embeddings.
- The script fails fast if `constraint_registry_{dataset}.parquet` is missing, so run `02a_constraint_registry.py` first.
- Registry entries that do not exist in the frozen encoder are skipped; if you see large skip counts, regenerate the encoder or the registry to keep them in sync.
- Running without a cached SentenceTransformer model triggers a download the first time—ensure you have disk space and network access or pre-pull the model beforehand.
- Wikidata rate limiting can slow large batches; when `_materialise_entries()` hits many URIs it is safer to leave the default batch size instead of cranking it up aggressively.

## Implementation Details
- `WikidataUriEmbedder` reads `WIKIDATA_EMBEDDING_MODEL` (default `jinaai/jina-embeddings-v3`) and `WIKIDATA_EMBEDDING_FALLBACK` (default `sentence-transformers/all-MiniLM-L6-v2`) to select the embedding model. If the primary model fails to load, it falls back automatically.
- The cache distinguishes `kind` (`uri`, `placeholder`, `literal`) so `03_graph.py` can request either an embedding by integer ID (`kind=uri`, `global_id` populated) or by raw string (literals).
- When encountering multiple integer IDs that decode to the same URI, the script stores only one embedding payload and merely aliases the additional IDs, shrinking the on-disk footprint.
- Literal texts are deduplicated case-insensitively and trimmed; noisy strings like `"nan"` are ignored to avoid polluting the embedding table.
- Every embedding is stored as `np.float16` by default (configurable via `--embed-dim` and future dtype parameters), which halves storage while remaining adequate for downstream PyTorch models.
