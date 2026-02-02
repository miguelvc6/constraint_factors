# 07_train.py

## Objective
- Train a graph neural network (configured via JSON) on the graphs exported by `06_graph.py`, optimising the six action slots (`add/del × subject/predicate/object`) with per-slot cross-entropy while tracking per-constraint performance.
- Persist the best checkpoint, the resolved experiment configuration, and the full training history under `models/<dataset>/<encoding>/<model>/<config_tag>/`.

## Inputs & Outputs
- **Inputs:** Processed graph files in `data/processed/<variant>/train_graph-<encoding>.pkl` and `val_graph-<encoding>.pkl`, experiment config JSONs (model + training blocks), and the frozen encoder from `data/interim/<variant>/globalintencoder.txt`.
- **Outputs:** Run directory under `models/<variant>/<encoding>/<model>/<config_tag>/` containing `checkpoint.pth`, copied config, training history JSON/plots, and any logging artifacts.

## Workflow
1. **Configuration intake** – The script requires `--experiment-config path/to/config.json`. The file contains `model_config` (dataset variant, encoding, architecture name, hyperparameters) and `training_config` (batch size, epochs, scheduler knobs, constraint weighting).
2. **Run directory setup** – `ensure_run_dir()` creates or reuses a timestamped folder, while `config_copy_path()` determines where the config snapshot will live beside the checkpoint.
3. **Data loading** – `dataset_variant_name()` selects the processed root (`data/processed/<variant>/`), `load_graph_dataset()` loads `train_graph-<encoding>.pkl` and `val_graph-<encoding>.pkl` as either lists or a lazy `GraphStreamDataset`, and `infer_node_feature_spec()` inspects them to decide whether node features are embeddings or categorical IDs (including optional role flags).
4. **Target vocabularies** – The model predicts six categorical slots. `load_precomputed_target_vocabs()` reuses cached entity/predicate class IDs when available, otherwise `derive_target_class_ids()` scans the loaded graphs. These IDs are passed into the model so entity and predicate heads can be expanded/masked into a shared `num_target_ids` space.
5. **Encoder + model build** – The frozen `GlobalIntEncoder` from `data/interim/<variant>/globalintencoder.txt` defines `num_graph_nodes`. `build_model()` instantiates the chosen architecture (e.g., message-passing network with dual branches). Device selection is automatic (CUDA if available) with memory logging hooks for debugging.
6. **Training loop (`train()`):**
   - Wrap datasets in `DataLoader`s, shuffling the in-memory split while leaving streaming datasets ordered.
   - Forward pass returns logits of shape `(batch, 6, num_target_ids)` where entity/predicate slots are masked to the per-split vocabularies. Each slot is compared against the gold IDs via `CrossEntropyLoss(reduction="none")`, producing a `(batch, 6)` loss matrix.
   - Per-graph loss is computed as the mean over the six slots (`graph_loss = loss_matrix.mean(dim=1)`), then optionally:
     - `FixProbabilityScheduler` adds a repair-aware penalty when violation contexts are available.
     - `DynamicConstraintWeighter` rescales each per-graph loss based on constraint types (`extract_constraint_types()` reads `data.constraint_type`).
   - Accuracy is tracked both per-slot (percentage of correctly predicted IDs) and as “all-6 correct” (all slots match simultaneously).
   - `ConstraintMetricsAccumulator` aggregates loss/accuracy per constraint type so reports can highlight which shapes dominate or lag.
   - `torch.optim.Adam` drives the updates, `ReduceLROnPlateau` reduces LR when validation loss stalls, gradient clipping is optional, and early stopping is triggered after `training_config.early_stopping_rounds` epochs without improvement.
7. **Validation** – Mirrors the training pass sans gradient steps, feeding results into the same metric accumulators for apples-to-apples comparisons.
8. **Artifacts** – Once training finishes (or early stopping fires), the best-performing weights are saved via `torch.save()` alongside the effective `model_cfg`/`training_cfg`. `history_path()` stores the scalar curves, and `plot_training_history()` renders PNG charts for quick inspection.

## Common Pitfalls / Gotchas
- The `model_config.dataset_variant` and `model_config.encoding` must match the graphs on disk; mismatches surface as missing-file errors or shape mismatches deep in PyG.
- When using `GraphStreamDataset`, `len(dataset)` is undefined, so progress bars may look odd—this is expected and doesn’t mean data is missing.
- Early stopping patience is enforced even if validation batches fail intermittently; run with a stable validation split and monitor logs before trusting the saved checkpoint.
- If CUDA is available but `num_workers` is high, pin-memory defaults to `True`; on systems with constrained RAM this can lead to OS-level swapping—tune `pin_memory` in the config if needed.
- Fix-probability loss requires in-memory datasets (lists) so the script can attach `context_index` and look up contexts; streamed datasets will disable that term automatically.

## Implementation Details
- The script intentionally supports streamed graphs (via `GraphStreamDataset`) so very large runs never exceed RAM even when the serialized graphs are sharded.
- Per-slot histories are nested under `history["per_slot"][slot_index]`, enabling later analysis of which action (e.g., `del_predicate`) converged slower.
- GPU monitoring hooks (`log_cuda_memory`) fire at strategic checkpoints (epoch boundaries, first batch) to simplify diagnosing OOMs or fragmentation.
- Model checkpoints store both the state dict and the resolved configuration, allowing `05_eval.py` to rebuild the architecture without guessing hyperparameters.
- If `training_config.validate_factor_labels` is enabled, training asserts that factor label tensors exist and align with `factor_constraint_ids` (useful for upcoming factor supervision).

## Dynamic Weighting per constraint type

`DynamicConstraintWeighter` keeps per‑constraint weights so the trainer can emphasize underperforming constraint types. Its behaviour can be specified from the configs json files: you can toggle it on/off, choose update_frequency (epoch uses validation metrics, batch reacts after every batch), decide which metrics drive “difficulty” (target_metrics defaults to loss but can include accuracies).

The weights are updated every batch or every epoch (can choose from model's configuration).

- Per batch: averages the current batch losses per constraint and treats them as “difficulty” scores.
- Per epoch: after validation it collects per-constraint metrics (loss/acc), converts the configured metrics into difficulty (loss directly, accuracies as 1 - acc/100), and updates weights once per epoch.

To calculate the weights from the difficulties it rescales difficulties relative to their mean, blends with prior weights using smoothing, clamps between min_weight/max_weight, and renormalizes so the mean weight stays ~1.

During training each batch multiplies the per-constraint loss rows by these weights before averaging/backpropagating; if the feature is disabled, it reduces to the standard uniform mean.
