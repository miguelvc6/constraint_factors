# Model Config JSON Reference

This document enumerates every field accepted by experiment config JSON files used by `07_train.py` (proposal models) and `08_train_reranker.py` (rerankers). The source of truth is the config dataclasses in `src/modules/config.py`, `src/modules/reranker.py`, and `src/08_train_reranker.py`.

## 1) Top-Level Keys

Experiment config JSON files are JSON objects that can include the keys below.

| Key | Required | Description |
| --- | --- | --- |
| `model_config` | Yes | Graph model architecture and dataset wiring. Used by proposal and reranker runs. |
| `training_config` | Yes | Training hyperparameters and loss options. Shape depends on whether this is a proposal or a reranker run. |
| `reranker_config` | No | Reranker head hyperparameters (only used by `08_train_reranker.py`). |
| `proposal_config` | No | Proposal checkpoint selection for reranker runs (only used by `08_train_reranker.py`). |

## 2) `model_config` Fields (Graph Model)

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `dataset_variant` | string | `"full"` | Which dataset variant to consume. Must match the preprocessed graph files on disk. |
| `encoding` | string | `"text_embedding"` | Node feature encoding; selects graph files on disk. |
| `model` | string | `"GIN"` | Graph model identifier. Example values: `GIN`, `GIN_PRESSURE`, `RERANKER`. |
| `min_occurrence` | int | `100` | Frequency threshold used when building the dataset. |
| `num_embedding_size` | int | `128` | Width of learned embeddings for integer node ids. |
| `num_layers` | int | `2` | Message-passing layers in the backbone. |
| `hidden_channels` | int | `128` | Channel size inside the message-passing stack. Alias: `hidden` maps to this. |
| `head_hidden` | int | `128` | Hidden width shared by prediction heads. |
| `dropout` | float | `0.5` | Dropout probability for head activations. |
| `use_node_embeddings` | bool | `true` | If `true`, embed integer node ids. If `false`, pass features through. |
| `use_role_embeddings` | bool | `false` | Append learned focus-role embeddings to node features. |
| `role_embedding_dim` | int | `8` | Dimensionality of each learned role embedding. |
| `num_role_types` | int | `4` | Number of distinct role ids expected in `role_flags`. |
| `use_edge_attributes` | bool | `false` | Use edge attributes rather than treating edges as nodes. |
| `use_edge_subtraction` | bool | `false` | Uses edge subtraction; requires `use_edge_attributes=true`. |
| `entity_class_ids` | array[int] or int | `null` | Optional vocabulary subset for entity targets. Can be a single int or any iterable of ints. |
| `predicate_class_ids` | array[int] or int | `null` | Optional vocabulary subset for predicate targets. Can be a single int or any iterable of ints. |
| `num_factor_types` | int | `0` | Number of distinct factor type ids. `0` disables type conditioning. |
| `factor_type_embedding_dim` | int | `8` | Embedding dimension for factor type conditioning. |
| `pressure_enabled` | bool | `false` | Toggle factor-pressure injection during message passing. |
| `pressure_type_conditioning` | string | `"none"` | How to condition pressure messages on factor types: `none`, `concat`, `gate`. |
| `enable_policy_choice` | bool | `false` | Enable policy-choice head over graph embeddings. |
| `policy_num_classes` | int | `6` | Number of policy classes. Must be `>= 6` for the default policy set. |

## 3) `training_config` Fields (Proposal / Graph Models)

These fields are consumed by `07_train.py` and validated in `src/modules/config.py`.

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `batch_size` | int | `124` | Number of graphs per optimization step. |
| `num_epochs` | int | `5` | Max training epochs. |
| `early_stopping_rounds` | int | `5` | Patience before early stopping triggers. |
| `grad_clip` | float or null | `1.0` | Gradient norm cap; `null` disables clipping. |
| `learning_rate` | float | `0.001` | Base learning rate for Adam. |
| `weight_decay` | float | `0.0005` | Adam weight decay. |
| `scheduler_factor` | float | `0.5` | Multiplicative drop factor for LR scheduler. |
| `scheduler_patience` | int | `3` | Epochs with no improvement before LR reduction. |
| `num_workers` | int | `0` | DataLoader worker processes. |
| `pin_memory` | bool or null | `null` | Override DataLoader `pin_memory` (null keeps default). |
| `validate_factor_labels` | bool | `false` | Strict factor label assertions per batch. |
| `constraint_loss` | object | see below | Constraint loss configuration object. |
| `fix_probability_loss` | object | see below | Fix-probability loss configuration object. |
| `factor_loss` | object | see below | Factor supervision loss configuration object. |
| `chooser` | object | see below | Chooser configuration for candidate selection. |
| `policy_filter_strict` | bool | `true` | Enforce strict policy filtering when policy choice is enabled. |

### 3.1) `training_config.constraint_loss`

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `dynamic_reweighting` | object | see below | Dynamic per-constraint loss weighting. |

### 3.1.1) `training_config.constraint_loss.dynamic_reweighting`

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Toggle dynamic reweighting. |
| `target_metrics` | array[string] or string | `["loss"]` | Validation metrics used to compute difficulty. Accepts a string or list. |
| `update_frequency` | string | `"epoch"` | `epoch` or `batch`. |
| `scale` | float | `1.0` | Strength of reweighting relative to uniform weights. |
| `min_weight` | float | `0.5` | Lower clamp for generated weights. |
| `max_weight` | float | `3.0` | Upper clamp for generated weights. Must be `>= min_weight`. |
| `smoothing` | float | `0.2` | Interpolation factor toward previous weights. Must be in `[0, 1]`. |

### 3.2) `training_config.fix_probability_loss`

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Toggle the fix-aware loss term. |
| `initial_weight` | float | `0.5` | Weight at the start (after warmup). |
| `final_weight` | float | `0.05` | Asymptotic weight once decay finishes. |
| `decay_epochs` | float | `40.0` | Time constant (exponential) or span (linear). |
| `warmup_epochs` | float | `0.0` | Epochs to hold the initial weight before decay. |
| `schedule` | string | `"exponential"` | `exponential` or `linear`. |

### 3.3) `training_config.factor_loss`

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable factor supervision loss. |
| `weight_pre` | float | `0.1` | Loss weight multiplier. |
| `pos_weight` | float or null | `null` | Optional positive class weight for imbalance. |
| `only_checkable` | bool | `true` | If `true`, exclude non-checkable factors. |
| `per_graph_reduction` | string | `"mean"` | `mean` or `sum` reduction per graph. |

### 3.4) `training_config.chooser`

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable chooser loss. |
| `topk_candidates` | int | `20` | Candidates per slot to consider. |
| `max_candidates_total` | int | `80` | Cap on total candidates per graph. |
| `beta_no_regression` | float | `0.5` | Weight for no-regression term. |
| `gamma_primary` | float | `0.0` | Weight for primary constraint satisfaction. |
| `loss_mode` | string | `"fix1"` | `fix1`, `primary_only`, or `global_fix`. |

## 4) `training_config` Fields (Reranker Runs)

Reranker configs use a different `training_config` schema defined in `RerankerTrainingConfig` (`src/08_train_reranker.py`).

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `batch_size` | int | `32` | Number of graphs per optimization step. |
| `num_epochs` | int | `5` | Max training epochs. |
| `early_stopping_rounds` | int | `3` | Patience before early stopping triggers. |
| `grad_clip` | float or null | `1.0` | Gradient norm cap; `null` disables clipping. |
| `learning_rate` | float | `0.0001` | Base learning rate. |
| `weight_decay` | float | `0.0001` | Weight decay. |
| `scheduler_factor` | float | `0.5` | Multiplicative drop factor for LR scheduler. |
| `scheduler_patience` | int | `2` | Epochs with no improvement before LR reduction. |
| `num_workers` | int | `0` | DataLoader worker processes. |
| `pin_memory` | bool | `false` | DataLoader `pin_memory`. |
| `objective` | string | `"main"` | `main` or `global_fix`. |
| `regression_weight` | float | `0.5` | Weight for regression penalty (alias: `beta`). |
| `topk_candidates` | int | `20` | Candidate list size used for reranking. |
| `topk_per_slot` | int | `5` | Candidates per slot for heuristic generation. |
| `heuristic_max_candidates` | int | `30` | Cap for heuristic candidate enumeration. |
| `heuristic_max_values` | int | `3` | Max values per slot in heuristic enumeration. |
| `include_gold` | bool | `true` | Whether to include the gold candidate in reranker inputs. |
| `max_candidates_total` | int | `80` | Cap on total candidates per graph. |
| `assume_complete_entity_facts` | bool | `true` | If `true`, assume full entity facts are present when building candidates. |
| `constraint_scope` | string | `"local"` | `local` or `focus`. |

## 5) `reranker_config` Fields

| Field | Type | Default | Description / Notes |
| --- | --- | --- | --- |
| `candidate_embedding_dim` | int | `64` | Embedding width for candidate ids. |
| `candidate_hidden_dim` | int | `128` | Hidden width for candidate MLP. |
| `dropout` | float | `0.1` | Dropout in reranker MLPs. |

## 6) `proposal_config` Fields (Reranker Runs)

This block selects which proposal checkpoint the reranker should use.

| Field | Type | Required | Description / Notes |
| --- | --- | --- | --- |
| `checkpoint_path` | string | No | Path to proposal checkpoint. |
| `run_dir` | string | No | Path to proposal run directory (checkpoint assumed at `checkpoint.pth`). |
| `model` | string | No | Proposal model name if looking up by `config_tag`. |
| `config_tag` | string | No | Proposal config tag to resolve under the proposal model runs. |

## 7) Validation Notes

The config loaders are strict and will raise on unknown keys. Additional behaviors to be aware of:

- `model_config.hidden` is accepted as an alias for `model_config.hidden_channels`.
- `training_config.dynamic_reweighting` is accepted as a top-level fallback for `training_config.constraint_loss.dynamic_reweighting`.
- `training_config.regression_weight` in rerankers also accepts the legacy key `beta`.
- `pressure_type_conditioning` must be one of `none`, `concat`, `gate`.
- `chooser.loss_mode` must be `fix1`, `primary_only`, or `global_fix`.
- `fix_probability_loss.schedule` must be `exponential` or `linear`.
