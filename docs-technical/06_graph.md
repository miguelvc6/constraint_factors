# 06_graph.py

## Objective
- Convert the parquet splits plus the Wikidata text cache into PyTorch Geometric `Data` objects that encode every violation as a graph with rich node features.
- Persist split-specific graph collections under `data/processed/<variant>/` so training and evaluation never have to touch the original TSVs again.

For the paper-facing run, use:

- `--constraint-scope local` for factorized graphs
- `--constraint-representation factorized` for `A1`, `M1C`, `M1D`, and proposal graphs consumed by `G0`
- `--constraint-representation eswc_passive` for `B0`
- `--primary-constraint-mode executable_factor` for canonical runs
- `--primary-constraint-mode query_definition`, `query_family`, or `passive_node` only for the A1 primary-query ablations

`focus` scope remains supported as a non-paper exploratory option.

When graph materialization runs out of memory, the paper pipeline can use sharded outputs, e.g.
`--shard-size 200000 --use-torch-save`. The downstream proposal training, reranker training,
config generators, and evaluation scripts all accept shard-only graph artifacts.
For primary-query ablations, `scripts/convert_primary_query_graph_mode.py` can derive
`query_definition` and `passive_node` artifacts from an already materialized `query_family`
artifact set without rereading the raw parquet rows.

## Inputs & Outputs
- **Inputs:** Interim parquet splits from `data/interim/<variant>/` or, when present, `data/interim/<variant>_labeled/` (unless `--use-unlabeled-interim` is passed), the identity/feature encoder contract, the constraint registry (`data/interim/constraint_registry_{dataset}.parquet`), the Wikidata cache (`data/interim/wikidata_text.parquet`), and CLI flags controlling encoding/sharding options.
- **Outputs:** Graph artifacts in `data/processed/<variant>/` (`{split}_graph-<encoding>.pkl` for canonical factorized runs, `{split}_graph-<encoding>-primary_<mode>.pkl` for non-default primary-constraint modes, `{split}_graph_repr-eswc_passive-<encoding>.pkl` for passive ESWC-style runs, plus sharded `.pt/.pkl` variants), per-split manifests, `target_vocabs.json`, plus optional visualisations like `graph_visualization.png`.

## Workflow
1. Parse CLI flags to select the dataset (`--dataset`), registry source for derived variants (`--registry-dataset`), node feature encoding (`--encoding {node_id,text_embedding}`), frequency variant (`--min-occurrence`), representation regime (`--constraint-representation {factorized,eswc_passive}`), primary-constraint mode (`--primary-constraint-mode`), optional Wikidata cache override (`--wikidata-cache-path`), sharding size, persistence format (`pickle` vs `torch.save`), persistence profile (`--persistence-profile {research_safe,full}`), overwrite policy (`--overwrite {atomic,unsafe,skip}`), optional visualization, constraint scope (`--constraint-scope`), and debugging (`--debug-factor-wiring`).
2. Resolve `INTERIM_DATA_PATH` and `PROCESSED_DATA_PATH`, preferring `data/interim/<variant>_labeled/` automatically when it exists. Load the identity encoder for node identity/evaluation and the feature encoder plus identity-to-feature map for model inputs. The constraint registry is loaded once from `data/interim/constraint_registry_{dataset}.parquet`.
3. For each split (`train`, `val`, `test`):
   - Read the parquet file.
   - `pandas_to_dataset()` converts it to a Hugging Face `datasets.Dataset` without altering the row contents.
   - `compute_torch_geometric_objects()` iterates rows and calls `create_graph()` while optionally forwarding hook `on_sample` to track target vocabularies.
   - `dump_stream()` or `dump_in_shards()` writes the resulting graphs to disk (optionally with `torch.save` for faster reloads) following the chosen overwrite mode:
     - `atomic`: temp file then rename (crash-safe),
     - `unsafe`: direct write (lower peak disk, not crash-safe),
     - `skip`: reuse existing split artifacts.
   - A manifest is written for each split with graph counts, field profile, artifact sizes, and lightweight prefix checksums.
4. After each split, the script records entity/predicate targets seen in the labels (`add_*` / `del_*`) so training can precompute class vocabularies and shard metadata.
5. If `--show_graph` is enabled, `display_graph()` reads one of the stored graphs, converts it to NetworkX, and renders `graph_visualization.png` plus a non-flattened view to make edge labels inspectable.

## Derived Primary-Query Artifacts
`scripts/convert_primary_query_graph_mode.py` converts sharded factorized graph artifacts
from one primary-query mode to another while preserving graph order and shard boundaries.
The intended source is `query_family`, because that mode already stores the primary
constraint ID, family/type ID, constrained property, and parameter-definition tensors.

Example:

```bash
uv run scripts/convert_primary_query_graph_mode.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --source-mode query_family \
  --target-mode query_definition \
  --link-identical-structure \
  --overwrite
```

For `query_definition`, graph structure and query tensors are identical to `query_family`.
With `--link-identical-structure`, the converter creates linked target shard names and
updates the manifest without rewriting the large payloads; the payload-level
`primary_constraint_mode` string remains the source mode, while training uses the mode from
the experiment config. Without that flag, the converter rewrites the shards and updates the
payload-level mode string too. For `passive_node`, it appends the passive primary constraint
node and factor-definition parameter edges, sets `passive_primary_node_index`, and keeps the
passive node out of `factor_node_index`, factor supervision, and pressure edges.

## Common Pitfalls / Gotchas
- Selecting `--encoding text_embedding` without having run `04_wikidata_retriever.py` first will raise lookup errors because literal/URI embeddings are missing.
- For the paper suite, do not mix `constraint-scope` values across runs. Build the labeled parquet and factorized graphs with `local`, then keep that choice fixed throughout training and evaluation.
- Large `--shard-size` values can exhaust RAM when saving; pick smaller shards or enable `--use-torch-save` if you see pickle spills.
- `--overwrite unsafe` reduces temporary disk usage but interrupted runs can leave partial/corrupt files.
- Graphs inherit the constraint type distribution of the parquet splits—mixing mismatched dataset variants (e.g., graphs built with min-occurrence 100 but training on 50) guarantees label/encoder inconsistencies.

## Implementation Details
- `create_graph()` keys nodes by semantic identity, so two rare entities never collapse into one node. For `node_id`, `x` stores compact feature IDs while `node_identity_id` stores the original IDs. For text embeddings, identity still determines node uniqueness.
- Literal-only slots (e.g., `object_text`) remain usable because literal text is embedded directly when the object is not a Wikidata entity.
- Focus triple identities are stored in `focus_triple`, while `role_flags` marks which local nodes correspond to the main subject, predicate, or object. Models can use these flags to decide which subgraphs to attend to.
- In each violation record there’s a “focus” triple: the actual (subject, predicate, object) statement under scrutiny. When `create_graph()` finishes building the node/edge structure, it stores those three global IDs in `data_graph.focus_triple` and also flags their corresponding local nodes via `role_flags`. So “focus triple identities” just means the numeric IDs of the main triple retained on the Data object: models can always recover which nodes represent the subject/predicate/object that triggered the constraint violation, even after the graph has been expanded with neighboring statements.
- `y` is the six-slot feature-space training target. `y_identity` is the corresponding strict semantic target, and `target_representable_mask` identifies slots retained by the feature vocabulary. `focus_triple_feature` complements the identity-bearing `focus_triple`.
- Both the flattened (`edge_index`) and original predicate-aware edges (`edge_index_non_flattened` plus `edge_attr_non_flattened`) are stored so GNNs can either treat predicates as intermediate nodes or recover direct subject→object relations without recomputing.
- Graphs use `ConstraintGraphData`, whose PyG batching rules offset local node references (`edge_attr_non_flattened`, `factor_node_index`, `passive_primary_node_index`) and leave graph-local indices such as `primary_factor_index` unchanged. Model forward paths fail fast if an endpoint or predicate crosses graph boundaries.
- Constraint factors are modeled explicitly: labeled instances use `factor_constraint_ids` from `05_constraint_labeler.py`; unlabeled instances fall back to `local_constraint_ids` or `local_constraint_ids_focus` depending on `--constraint-scope`. Each factor is represented by a node `constraint_factor::<id>` using the pre-seeded global ID from the encoder. For every factor, the registry supplies `param_predicates`/`param_objects`, yielding factor → predicate → object branches.
- In the default `primary_constraint_mode=executable_factor`, `factor_*` fields retain the canonical behavior: the primary violated constraint is one executable factor and `primary_factor_index` points to it.
- In `query_definition`, `query_family`, and `passive_node` modes, `factor_*` fields contain executable secondary/context factors only. The full local factor set is mirrored in `eval_factor_*` fields so strict symbolic evaluation still includes the primary constraint. In these modes `primary_factor_index` is `-1` and `eval_primary_factor_index` points to the primary inside `eval_factor_constraint_ids`.
- Query modes store primary task metadata as tensors: `primary_constraint_id`, `primary_constraint_type_id`, `primary_constrained_property_id`, `primary_param_predicate_ids`, `primary_param_object_ids`, and `primary_param_count`.
- `passive_node` mode adds `passive_primary_node_index`. That node receives only constraint-definition parameter edges and is excluded from `factor_node_index`, factor supervision, and factor-to-local pressure edges.
- Factors also connect to the local data graph they constrain: for a factor’s `constrained_property`, every predicate node created for matching triples in the instance is linked from the factor, and the factor also links to the matching triples’ subject/object nodes. This uses per-triple predicate nodes (created with `force_create=True`) so factors “observe” the local statements they govern.
- `edge_type` is emitted alongside `edge_index` to distinguish base statement edges, factor-definition edges, and factor-to-local-statement edges during message passing.
- Factor metadata is stored on each graph: `factor_constraint_ids` (order preserved), `primary_factor_index` (which entry matches the row’s `constraint_id`), and optional debug fields (`factor_constraint_types`, `factor_wiring_debug`) when using `--persistence-profile full`. Factor wiring uses the registry's canonical `constraint_family`; `symmetric` constraints share the inverse-style mirror-property wiring.
- `--persistence-profile research_safe` (default) drops debug-only fields (`x_names`, `factor_constraint_types`, `factor_wiring_debug`) while preserving all fields required by training/evaluation objectives.
- `target_vocabs.json` records feature-space class IDs referenced by training labels. Per-split manifests use graph schema v2 and list the identity audit fields.
- `collect_sample_for_check()` and `check_data_graph()` assist in QA by loading a tiny subset of the serialized graphs and verifying they batch cleanly before committing to long training jobs.
