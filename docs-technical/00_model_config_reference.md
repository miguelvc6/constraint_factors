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
- `factor_type_embedding_dim`
- `pressure_enabled`
- `pressure_type_conditioning`
- `pressure_residual_scale`
- `enable_policy_choice`
- `policy_num_classes`

Paper-facing additions:

- `constraint_representation`
  - allowed values: `factorized`, `eswc_passive`
  - `B0` should use `eswc_passive`
  - `A1`, `M1C`, `M1D`, and proposal sources for `G0` should use `factorized`

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

Generator defaults for the paper-facing proposal configs:

- `batch_size: 256`
- `num_workers: 2`
- `pin_memory: false`

These defaults are intentionally conservative for the large streamed graph artifacts under `data/processed/`.

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

## Validation notes

- Config loading is strict: unknown keys raise an error.
- `pressure_type_conditioning` must be one of `none`, `concat`, `gate`.
- `constraint_representation` must be one of `factorized`, `eswc_passive`.
- `chooser` and `direct_safety` should not both be enabled in the same proposal config.
