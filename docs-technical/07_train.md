# 07_train.py

## Objective
- Train a graph neural network (configured via JSON) on the graphs exported by `06_graph.py`, optimising the six action slots (`add/del × subject/predicate/object`) with per-slot cross-entropy while tracking per-constraint performance.
- Persist the best checkpoint, resolved configuration, full training history, and a cryptographic run-provenance manifest under the experiment directory.

## Inputs & Outputs
- **Inputs:** Processed graph files, experiment config JSONs, the feature encoder for model dimensions, and the identity encoder/map for symbolic candidate evaluation.
- **Outputs:** The model directory containing `checkpoint.pth`, `config.json`, `training_history.json`, `run_manifest.json`, plots, and later evaluation artifacts.

## Workflow
1. **Configuration intake** – The script requires `--experiment-config path/to/config.json`. The file contains `model_config` (dataset variant, encoding, architecture name, hyperparameters) and `training_config` (batch size, epochs, scheduler knobs, constraint weighting).
2. **Run directory setup** – `ensure_run_dir()` creates or reuses a deterministic slugged folder based on dataset variant, encoding, model name, and config tag, while `config_copy_path()` determines where the config snapshot will live beside the checkpoint.
3. **Data loading** – `dataset_variant_name()` selects the processed root (`data/processed/<variant>/`), `load_graph_dataset()` discovers either monolithic files or shard collections (`*-shardNNN.{pkl,pt}`), returning an in-memory list or a lazy `GraphStreamDataset`/sharded stream as appropriate. `infer_node_feature_spec()` inspects samples to decide whether node features are embeddings or categorical IDs (including optional role flags).
4. **Target vocabularies** – The model predicts six categorical slots. `load_precomputed_target_vocabs()` reuses cached entity/predicate class IDs when available, otherwise `derive_target_class_ids()` scans the loaded graphs. These IDs are passed into the model so entity and predicate heads can be expanded/masked into a shared `num_target_ids` space.
5. **Factor type setup** – For `constraint_representation="factorized"`, if `model_config.num_factor_types > 0`, training uses that value directly unless the constraint registry reports a larger compact `constraint_type_index` range, in which case the registry value wins and the resolved config is updated. If the config leaves the field at `0`, the trainer prefers the constraint registry count and only falls back to a dataset scan when no registry-derived count is available. For `constraint_representation="eswc_passive"`, the trainer clears `num_factor_types` in the resolved config because passive graphs may include passive constraint nodes but do not execute per-type factor heads.
6. **Encoder + model build** – `feature_encoder.txt` defines model input/output dimensions; `identity_encoder.txt` is used only for semantic heuristics and evaluator calls. For `GIN_PRESSURE`, `model_config.pressure_module_sharing` controls pressure blocks. `shared` is the canonical A1-family setting; `per_type` is now an ablation only.
7. **Training loop (`train()`):**
   - Wrap datasets in split-specific `DataLoader`s, shuffling the in-memory train split while leaving streaming datasets in artifact order. Paper sampled artifacts are therefore written in a deterministic mixed order by `02b_stratified_benchmark_sampler.py`. For streamed datasets the trainer disables `pin_memory`, reduces `prefetch_factor` to `1`, and keeps `persistent_workers=False` so train/validation worker pools do not overlap and exhaust shared memory at epoch boundaries.
   - Forward pass returns logits of shape `(batch, 6, num_target_ids)` where entity/predicate slots are masked to the per-split vocabularies. Each slot is compared against the gold IDs via `CrossEntropyLoss(reduction="none")`, producing a `(batch, 6)` loss matrix.
   - Per-graph loss is computed as the mean over the six slots (`graph_loss = loss_matrix.mean(dim=1)`), then optionally:
     - `FixProbabilityScheduler` adds a repair-aware penalty when violation contexts are available.
     - `DynamicConstraintWeighter` rescales each per-graph loss based on constraint types (`extract_constraint_types()` reads `data.constraint_type`).
   - Accuracy is tracked both per-slot (percentage of correctly predicted IDs) and as “all-6 correct” (all slots match simultaneously).
   - `ConstraintMetricsAccumulator` aggregates loss/accuracy per constraint type so reports can highlight which shapes dominate or lag.
   - If chooser training is enabled, candidate sets are built per graph and scored by the chooser head. Training uses an optimized path:
     - training candidates may force-include gold and carry both feature and identity tuples;
     - validation/test inference uses a separate API that cannot accept `graph.y` or any gold tuple;
     - candidate scoring is done in a packed/batched call (`score_candidates_packed`) rather than one scorer call per graph,
     - `fix1`-style chooser losses use `evaluate_candidates_loss_terms()` to compute only the required terms (no full diagnostic payload),
     - top-k candidate extraction can be restricted to valid entity/predicate class IDs per slot.
   - `torch.optim.Adam` drives the updates, `ReduceLROnPlateau` reduces LR when validation loss stalls, gradient clipping is optional, and early stopping is triggered after `training_config.early_stopping_rounds` epochs without improvement.
   - The trainer records stability diagnostics every epoch: learning rate, unclipped gradient norm mean/max, parameter norm/max absolute parameter value, edit-logit max magnitude, factor-logit max magnitude, and chooser-score max magnitude. Edit-logit magnitude excludes the fixed `-1e9` invalid-class mask; this keeps the diagnostic about learned logits rather than the masking constant.
8. **Validation** – Mirrors the training pass sans gradient steps, feeding results into the same metric accumulators for apples-to-apples comparisons. If `training_config.validation_subset_size` is set, each epoch validates on only the first N validation graphs. Streamed validation subsets force the validation loader to `num_workers=0` so the subset is one global prefix, not one prefix per worker. After the first epoch, training fails if that prefix omits any constraint family observed during training, because scheduler and early-stopping decisions would otherwise use a family-incomplete signal.
9. **Artifacts** – The checkpoint embeds resolved configs, seed, and config hashes. `run_manifest.json` records the full checkpoint hash, effective config hashes, dataset/graph manifest hashes, source commit/dirty-state hashes, package versions, command, seed, and parameter counts by top-level component.

## Common Pitfalls / Gotchas
- The `model_config.dataset_variant` and `model_config.encoding` must match the graphs on disk; mismatches surface as missing-file errors or shape mismatches deep in PyG.
- When using `GraphStreamDataset`, `len(dataset)` is undefined, so progress bars may look odd—this is expected and doesn’t mean data is missing.
- Early stopping patience is enforced even if validation batches fail intermittently; run with a stable validation split and monitor logs before trusting the saved checkpoint.
- Non-finite weighted loss, validation loss, gradient norm, or parameter norm now raises a `FloatingPointError` with epoch/batch context rather than silently producing a corrupt checkpoint.
- Subset validation changes model selection because scheduler and early stopping use the subset loss. Use full validation for final paper-facing runs.
- Generated paper configs now default to `learning_rate=1e-4`, `grad_clip=0.5`, `num_epochs=15`, `early_stopping_rounds=2`, `scheduler_patience=0`, `num_workers=2`, `pin_memory=false`, and full validation (`validation_subset_size: null`). `num_epochs` is a ceiling: representative full-validation early stopping may still end a run earlier when loss genuinely stops improving.
- Candidate inference is total: if heuristics and uniquely resolvable proposal candidates are both empty, the only candidate is the all-`NONE` no-op with source `fallback_noop`. Real candidate sets are never padded with this fallback. Evaluation JSON and reranker epoch histories report fallback count/rate.
- If CUDA is available but `num_workers` is high, pin-memory can still amplify host-memory pressure on in-memory datasets; tune `pin_memory` in the config if throughput does not justify the footprint.
- Fix-probability loss requires in-memory datasets (lists) so the script can attach `context_index` and look up contexts; streamed datasets will disable that term automatically.
- Chooser training supports streamed datasets via per-graph `context_index` assignment; contexts/parquet sidecars must align with graph ordering/counts.
- CUDA batch prefetch (`TRAIN_CUDA_PREFETCH`) is available and enabled by default; on some hardware/data combinations it may not improve throughput, so treat it as a tunable runtime flag.
- The former `h2_a1_shared_pressure` architecture is promoted to canonical A1. Its historical run is selection evidence only; paper numbers must come from a clean rerun under the v2 data/evaluation schemas.
- The H2 gold-scalar pressure ablation uses `model_config.pressure_oracle_input="gold_pre_scalar"` and requires `factor_satisfied_pre` / `factor_checkable_pre` graph fields. It is an oracle appendix run, not a deployable training configuration.

## Profiling & Throughput Controls

The trainer exposes runtime environment switches for profiling and data movement overlap:

- `TRAIN_TIMING_PROFILE=1` enables per-phase timing logs in both train and validation loops.
- `TRAIN_TIMING_WARMUP_BATCHES=<int>` excludes early warmup batches from timing summaries.
- `TRAIN_TIMING_LOG_EVERY=<int>` controls timing window size/frequency.
- `TRAIN_CUDA_PREFETCH=0|1` disables/enables asynchronous batch prefetch to GPU using a side CUDA stream.

Timing logs break the batch into phases such as:
- `data`, `forward`, `chooser`, `factor`, `backward`, `optim`, `metrics`, and `total`.

This makes bottlenecks explicit (for example, chooser-heavy runs where `chooser` dominates `total`).

## Implementation Details
- The script intentionally supports streamed graphs (via `GraphStreamDataset`) so very large runs never exceed RAM even when the serialized graphs are sharded.
- Per-slot histories are nested under `history["per_slot"][slot_index]`, enabling later analysis of which action (e.g., `del_predicate`) converged slower.
- GPU monitoring hooks (`log_cuda_memory`) fire at strategic checkpoints (epoch boundaries, first batch) to simplify diagnosing OOMs or fragmentation.
- Model checkpoints store both the state dict and the resolved configuration, allowing `09_eval.py` to rebuild the architecture without guessing hyperparameters.
- `09_eval.py` rejects a checkpoint when its embedded model config differs from `config.json`; changing a run config can no longer silently relabel an old checkpoint.
- `--seed` defaults to `42` and is persisted in both checkpoint and run manifest.
- Seed setup enables deterministic algorithms in warn-only mode, fixes cuDNN
  deterministic/benchmark flags, and records those settings. Any CUDA/PyG
  operation that cannot be deterministic emits a warning that must be retained
  with the run log.
- If `training_config.validate_factor_labels` is enabled, training asserts that factor label tensors exist and align with `factor_constraint_ids` (useful for upcoming factor supervision).
- Models receive `model_config.constraint_representation` at construction time. Passive models do not allocate or execute factor heads, post-edit heads, or gold-edit embeddings even if the passive graph contains constraint/factor nodes; factorized models remain strict and require dense `factor_types` whenever per-type factor execution is reached.

## Compact Factor Execution

`per_type_grouped_v2` preserves stable factor ids in graph artifacts while
mapping them to a compact model-local vocabulary. Paper configs record
`active_factor_type_ids=[0,2,3,4,5,9,12,14,15,16]`; training scans the
train/validation parquet factor columns and rejects a mapping that is missing
or adds an id. The stable registry address-space bound remains
`num_factor_types=29`.

On BF16 CUDA devices with SM80 or newer, the compact executor and per-type
pressure banks use ragged grouped matrix multiplication. CPU, unsupported GPU,
and full-precision evaluation use a segmented linear fallback. Shared pressure
dispatches all edges for a role through its one shared module without a
per-type loop.

With `gold_edit_embedding_mode="compact"`, gold-edit embeddings are indexed
through the reachable entity/predicate target union rather than `max_id + 1`.
The mapping and class ids are checkpoint buffers and run-provenance fields.
Unknown stable factor ids and unreachable gold targets fail instead of being
silently clamped when the gold-conditioned post-edit auxiliary head is
requested. Standard `src/09_eval.py` edit inference skips that unused
gold-conditioned head, so test-only target identities cannot be read as model
inputs or invalidate fidelity evaluation merely because they are outside the
training vocabulary. Training/validation and explicit H2 post-gold diagnostics
continue to request the auxiliary head.

## Shared-Adapter Executor Comparison

`shared_adapter_v1` replaces the independent per-type executor matrices with
one shared `1603 -> 400 -> 400` trunk. Each active factor type receives a
rank-16 residual adapter and an independent scalar precondition head. The
post-edit path uses the same pattern: one shared `800 -> 400` projection,
rank-16 residual adapters, and independent scalar heads. Adapter output
projections start at zero so optimization begins from the shared trunk rather
than a random type-specific residual.

Generate the two neutral seed-42 comparison configs without overwriting the
canonical A1 config:

```bash
uv run scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 \
  --encoding node_id \
  --include-executor-comparison
```

Run both complete directory names in one scheduler pass:

```bash
uv run src/10_scheduler.py \
  --only-exact a1_factorized_imitation_per_type_compact__full_strat1m_minocc100__node_id \
  --only-exact a1_factorized_imitation_shared_adapter__full_strat1m_minocc100__node_id \
  --paper-suite \
  --seed 42 \
  --force-retrain
```

`--only-exact` is repeatable and matches complete scheduler directory names;
it cannot be combined with the legacy substring `--only` filter. After both
standard evaluations, run `src/09_eval.py --h2-eval` for each directory and
use `scripts/compare_a1_executors.py` to verify matched configuration/data
provenance and write JSON, Markdown, per-constraint, and H2 delta artifacts.

The confirmatory no-factor-loss run uses a separate compact-per-type slug so
the earlier legacy H2 run remains immutable:

```bash
uv run src/10_scheduler.py \
  --only-exact a1_factorized_imitation_per_type_compact_no_factor_loss__full_strat1m_minocc100__node_id \
  --paper-suite --seed 42 --force-retrain
```

Generate it with `scripts/make_experiment_configs.py
--include-executor-comparison`. Its model configuration is identical to
`a1_factorized_imitation_per_type_compact`; only
`training_config.factor_loss.enabled=false` differs.

## Reranker Validation Semantics

`08_train_reranker.py` force-includes gold only in optimizer-backed training
epochs. Validation constructs the same gold-free inference candidate set used
for final scoring. For `global_fix`, validation loss scores every inference
candidate for every row and computes expected global satisfaction from that
set. For the non-paper `main` objective, supervised validation loss is defined
only on rows whose gold edit occurs naturally in the inference set; inference
metrics still cover all rows. Epoch metrics include `gold_candidate_count`,
`gold_candidate_coverage`, `loss_row_count`, and fallback count/rate, and
validation fails if `main` has zero natural-gold coverage. These changes do not
alter the 15-epoch ceiling or early-stopping schedule.

## Dynamic Weighting per constraint type

`DynamicConstraintWeighter` keeps per‑constraint weights so the trainer can emphasize underperforming constraint types. Its behaviour can be specified from the configs json files: you can toggle it on/off, choose update_frequency (epoch uses validation metrics, batch reacts after every batch), decide which metrics drive “difficulty” (target_metrics defaults to loss but can include accuracies).

The weights are updated every batch or every epoch (can choose from model's configuration).

- Per batch: averages the current batch losses per constraint and treats them as “difficulty” scores.
- Per epoch: after validation it collects per-constraint metrics (loss/acc), converts the configured metrics into difficulty (loss directly, accuracies as 1 - acc/100), and updates weights once per epoch.

To calculate the weights from the difficulties it rescales difficulties relative to their mean, blends with prior weights using smoothing, clamps between min_weight/max_weight, and renormalizes so the mean weight stays ~1.

During training each batch multiplies the per-constraint loss rows by these weights before averaging/backpropagating; if the feature is disabled, it reduces to the standard uniform mean.
