# Model Config Reference

Date: 2026-03-11

This document lists the paper-relevant config fields accepted by `src/07_train.py` and `src/08_train_reranker.py`.

## Top-level keys

- `model_config`
- `training_config`
- `reranker_config` for reranker runs
- `proposal_config` for reranker runs

## `model_config`

Core fields:

- `dataset_variant`
- `encoding`
- `model`
- `min_occurrence`
- `num_layers`
- `hidden_channels`
- `head_hidden`
- `dropout`
- `use_edge_attributes`
- `use_edge_subtraction`
- `use_role_embeddings`
- `role_embedding_dim`
- `num_role_types`
- `entity_class_ids`
- `predicate_class_ids`
- `num_factor_types`
- `active_factor_type_ids`
- `factor_type_embedding_dim`
- `factor_executor_impl`
- `factor_adapter_rank`
- `gold_edit_embedding_mode`
- `pressure_enabled`
- `pressure_type_conditioning`
- `pressure_module_sharing`
- `pressure_residual_scale`
- `pressure_oracle_input`
- `enable_policy_choice`
- `policy_num_classes`

Paper-facing additions:

- `constraint_representation`
  - allowed values: `factorized`, `eswc_passive`
  - `B0` should use `eswc_passive`
  - `A1`, `M1C`, `M1D`, and proposal sources for `G0` should use `factorized`

- `primary_constraint_mode`
  - allowed values: `executable_factor`, `query_definition`, `query_family`, `passive_node`, `none`
  - default: `executable_factor`
  - canonical `B0`, `A1`, `M1C`, `M1D`, `G0`, and H2 runs should keep `executable_factor`
  - A1 primary-query ablations use `query_definition`, `query_family`, or `passive_node`; their graph artifacts must be built with the same mode so the filename suffix matches the config

- `pressure_module_sharing`
  - allowed values: `per_type`, `shared`
  - paper-generator default: `shared`
  - `shared` is canonical for `A1`, `M1C`, `M1D`, their primary-query variants, and new H2 controls
  - `per_type` is retained only for the parameterization ablation and legacy checkpoint reproduction

- `pressure_oracle_input`
  - allowed values: `none`, `gold_pre_scalar`
  - default: `none`
  - `gold_pre_scalar` appends the gold pre-repair factor violation scalar to per-type pressure messages; use this only for the H2 gold-scalar oracle ablation

- `active_factor_type_ids`
  - sorted stable registry ids allocated by compact factor executors
  - paper dataset value: `[0, 2, 3, 4, 5, 9, 12, 14, 15, 16]`
  - `num_factor_types` remains the stable registry address-space bound (`29`); it is not the compact module count

- `factor_executor_impl`
  - `per_type_v1`: legacy dense `ModuleList` layout retained for old checkpoints
  - `per_type_grouped_v2`: architecture-equivalent compact per-type executor using grouped BF16 CUDA dispatch with a segmented fallback
  - `shared_adapter_v1`: shared two-layer executor/post-edit trunks plus compact type-specific low-rank residual adapters and scalar heads
  - `legacy_shared`: historical shared executor ablation

- `factor_adapter_rank`
  - positive adapter bottleneck width for `shared_adapter_v1`
  - locked comparison value: `16`
  - ignored by executor implementations that do not allocate adapters

- `gold_edit_embedding_mode`
  - `full`: legacy raw-id-sized embedding table
  - `compact`: embedding rows are the sorted union of reachable entity and predicate target ids, including `NONE=0`

## `training_config` for proposal runs

Core optimization fields:

- `batch_size`
- `num_epochs`
- `early_stopping_rounds`
- `grad_clip`
- `learning_rate`
- `weight_decay`
- `scheduler_factor`
- `scheduler_patience`
- `num_workers`
- `pin_memory`
- `validate_factor_labels`
- `train_subset_size`
- `validation_subset_size`

Generator defaults for the paper-facing proposal configs:

- `batch_size: 256`
- `num_epochs: 15`
- `early_stopping_rounds: 2`
- `grad_clip: 0.5`
- `learning_rate: 1e-4`
- `scheduler_factor: 0.5`
- `scheduler_patience: 0`
- `num_workers: 2`
- `pin_memory: false`

These defaults are intentionally conservative for the large streamed graph artifacts under `data/processed/`. Fifteen epochs is the maximum; early stopping still restores the best checkpoint when representative validation loss stops improving.

The paper-facing reranker generator uses the same schedule (`num_epochs: 15`, `early_stopping_rounds: 2`, `learning_rate: 1e-4`, `grad_clip: 0.5`, `scheduler_patience: 0`) with its own reranker batch size.

`make_experiment_configs.py` normally preserves existing optional H2 and primary-query config files. Pass `--overwrite-existing` during an intentional full rerun so those files are refreshed to the current schedule instead of retaining an older policy.

Paper-facing generated configs set `validation_subset_size: null` so scheduler and early-stopping decisions use the complete validation split. Set it to a positive integer only for development runs that should validate on the first N validation graphs each epoch. For streamed graph artifacts, subset validation uses a single validation worker so the stream produces one global prefix rather than one prefix per worker, and training rejects a prefix that omits a constraint family observed in training.

Set `train_subset_size` to a positive integer for bounded execution runs that should train on only the first N training graphs each epoch. Leave it unset or `null` for full training. For streamed graph artifacts, subset training also uses a single worker so each epoch consumes one deterministic global prefix.

For `num_factor_types`, the paper-facing generators prefer the compact factor-type count derived from the constraint registry rather than inferring from a single graph sample.

Nested blocks:

- `constraint_loss.dynamic_reweighting`
- `fix_probability_loss`
- `factor_loss`
- `chooser`
- `direct_safety`

### `chooser`

- `enabled`
- `topk_candidates`
- `max_candidates_total`
- `beta_no_regression`
- `gamma_primary`
- `loss_weight`
- `loss_mode`

Paper use:

- `M1C`: enabled
- `A1`, `B0`, `M1D`: disabled

Generator defaults for chooser-enabled proposal runs are `loss_weight: 0.25`, `beta_no_regression <= 0.5`, and `gamma_primary: 0.0` unless a targeted stress-test config explicitly overrides `gamma_primary`.

### `direct_safety`

- `enabled`
- `alpha_primary`
- `beta_secondary`
- `topk_candidates`
- `max_candidates_total`

Paper use:

- `M1D`: enabled
- `A1`, `B0`, `M1C`: disabled

### `factor_loss`

This remains supported, but it is not part of the default paper-facing suite.

## `training_config` for reranker runs

Reranker configs use the schema in `src/08_train_reranker.py`.

Paper-relevant fields:

- `validation_subset_size`
- `objective`
  - `main`
  - `global_fix`
- `topk_candidates`
- `topk_per_slot`
- `max_candidates_total`
- `regression_weight`
- `constraint_scope`

Paper use:

- `G0`: `objective="global_fix"`

Gold inclusion applies only to optimizer-backed training epochs. Reranker
validation always uses inference candidates. `global_fix` scores every
validation row; `main` reports natural `gold_candidate_coverage` and omits
supervised loss only for uncovered rows, failing when coverage is zero.

## Validation notes

- Config loading is strict: unknown keys raise an error.
- `pressure_type_conditioning` must be one of `none`, `concat`, `gate`.
- `pressure_module_sharing` must be one of `per_type`, `shared`.
- `pressure_oracle_input` must be one of `none`, `gold_pre_scalar`.
- `constraint_representation` must be one of `factorized`, `eswc_passive`.
- `chooser` and `direct_safety` should not both be enabled in the same proposal config.
