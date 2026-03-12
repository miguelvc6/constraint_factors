# Training and Evaluation Execution Plan

Date: 2026-03-11

This document defines the recommended execution order for training and evaluating the baselines and learned models in [docs-technical/00_models_and_evaluation_matrix.md](/home/mvazquez/constraint_factors/docs-technical/00_models_and_evaluation_matrix.md).

The plan is optimized for:

- paper-facing reproducibility
- short hyperparameter search
- minimal wasted long-running training
- one consistent dataset/encoding/backbone policy

Conceptual-to-technical mapping for this paper line:

- conceptual `M0 ESWC model` -> technical `B0`
- conceptual `M1 Main model` -> technical `A1`, `M1C`, and `M1D`
- conceptual `M2 Global Fix model` -> technical `G0`
- heuristic results -> `DFB`, `AMB`, `CSM`

`A1` is the representation-only step inside the broader conceptual `M1` story, while `M1C` and `M1D` are the two decision-level safe-factor realizations of that same model family.

## 1. Fixed run policy

Before running anything, freeze these decisions for the entire paper run:

- dataset variant: one paper dataset only, typically `full`
- `min_occurrence`: one value only, typically `100`
- encoding: one paper encoding only
- proposal random seed: `42`
- reranker random seed: `42`
- constraint neighborhood for the paper line: `local`

Reproducibility notes:

- `src/07_train.py` currently hardcodes `set_seed(42)`, so all proposal runs are already locked to seed `42`.
- `src/08_train_reranker.py` accepts `--seed`; use `--seed 42` for all reranker runs.
- heuristic baselines are deterministic.
- do not mix encodings or dataset variants inside the same paper table.
- do not edit configs once training starts; generate configs once, then only make one explicit “locked” hyperparameter update after the short search.

Recommended run ledger:

- record `git rev-parse HEAD`
- keep the generated config JSONs under `models/`
- keep scheduler logs under `logs/`

Strict-metrics precondition:

- proposal and reranker evaluations in `--paper-suite` mode require graph artifacts that already contain factor-label fields
- in the current pipeline, that means running `05_constraint_labeler.py` before `06_graph.py`; once `data/interim/<variant>_labeled/` exists, `06_graph.py` will use it automatically unless `--use-unlabeled-interim` is passed

## 2. Overall execution order

Run in this order:

1. Prepare paper artifacts.
2. Generate the canonical paper configs.
3. Run heuristic baselines first.
4. Run a brief `M1C` hyperparameter search only.
5. Select one winning `M1C` configuration.
6. Propagate the winning factorized-model settings to `A1`, `M1C`, and `M1D`, and reuse only the compatible training schedule for `B0`.
7. Train and evaluate the canonical learned suite in order:
   - `B0`
   - `A1`
   - final `M1C`
   - `M1D`
   - `G0`
8. Freeze tables and figures from those final runs only.

This order avoids spending time on secondary model families before the main model has a stable configuration.

Result coverage relative to the conceptual docs:

- heuristic references: `DFB`, `AMB`, `CSM`
- prior passive-context baseline: `B0`
- representation effect inside the main executable-factor story: `A1`
- main safe-factor results: `M1C`, `M1D`
- global-fix upper-bound reference: `G0`

The policy-choice model is intentionally out of scope for this paper line and is therefore not scheduled here.
`G0` is the repository's global-fix reference implementation for conceptual `M2`.

## 3. Step-by-step plan

### Step 0. Prepare paper artifacts

The paper line must materialize the artifact stack before config generation:

1. optional text cache for `text_embedding`
2. labeled interim parquet with `constraint-scope=local`
3. factorized processed graphs
4. passive processed graphs

**Optional: build the text cache**

Only needed when the paper encoding is `text_embedding`.

```bash
uv run src/04_wikidata_retriever.py \
  --dataset full \
  --min-occurrence 100
```

**Build labeled interim parquet for the paper scope**

```bash
uv run src/05_constraint_labeler.py \
  --dataset full \
  --min-occurrence 100 \
  --constraint-scope local
```

**Build factorized executable-factor graphs**

For `node_id`:

```bash
uv run src/06_graph.py \
  --dataset full \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-scope local \
  --constraint-representation factorized
```

If monolithic graph writes run out of memory, add:

```bash
  --shard-size 200000 \
  --use-torch-save
```

For `text_embedding`:

```bash
uv run src/06_graph.py \
  --dataset full \
  --min-occurrence 100 \
  --encoding text_embedding \
  --constraint-scope local \
  --constraint-representation factorized
```

The training, reranker, config-generation, and evaluation paths ingest these shard artifacts transparently.

**Build passive ESWC-style graphs**

For `node_id`:

```bash
uv run src/06_graph.py \
  --dataset full \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-representation eswc_passive
```

The same optional shard flags may be used here if needed:

```bash
  --shard-size 200000 \
  --use-torch-save
```

For `text_embedding`:

```bash
uv run src/06_graph.py \
  --dataset full \
  --min-occurrence 100 \
  --encoding text_embedding \
  --constraint-representation eswc_passive
```

Paper readiness check:

- factorized train/test artifacts exist under `data/processed/full_minocc100/`
- passive train/test artifacts exist under `data/processed/full_minocc100/`
- labeled parquet exists under `data/interim/full_minocc100_labeled/`
- coverage reports exist:
  - `coverage_local.csv`
  - `coverage_local.md`
- baseline evaluation now emits global metrics when labeled parquet is available

### Step 1. Generate canonical configs

Use the canonical paper config generator:

```bash
uv run scripts/make_experiment_configs.py --models-root models
```

This emits only:

- `b0_eswc_reproduction`
- `a1_factorized_imitation`
- `m1c_safe_factor_chooser`
- `m1d_safe_factor_direct`
- `g0_globalfix_reference`

If appendix runs are needed later, generate them separately with `--include-experimental`.

### Step 2. Run heuristic baselines first

Run baselines before any neural training so the paper already has stable reference numbers:

```bash
uv run src/09_eval.py \
  --run-baselines \
  --dataset full \
  --min-occurrence 100 \
  --per-constraint-csv
```

Outputs are written under:

- `models/baselines/full/parquet/`

This gives the reference results for:

- `DFB`
- `AMB`
- `CSM`

If `AMB` is not used in the final main table, keep it as appendix support.

Code note:

- `src/09_eval.py --run-baselines` now prefers `data/interim/<variant>_labeled/` when it exists and falls back to `data/interim/<variant>/` otherwise.
- This is the paper-safe path for getting heuristic `GFR`, `SRR`, `SIR`, and disruption metrics from the labeled parquet files.

### Step 3. Run a brief `M1C` hyperparameter search

Search only on the paper’s main practical model: `M1C`.

Reason:

- `M1C` is the main paper model.
- `A1` and `M1D` should inherit its backbone/optimizer settings.
- `B0` should reuse the same training schedule where possible, but not get its own expensive search.
- `G0` should not trigger a separate reranker search in phase 1.

Generate the short search set:

```bash
uv run scripts/make_hparam_search_configs_m1.py \
  --processed-root data/processed \
  --models-root models \
  --dataset-variant full \
  --min-occurrence 100 \
  --encoding <paper_encoding> \
  --num-configs 5 \
  --seed 42
```

Recommended search budget:

- default: `5` configs max
- if compute is very tight: rerun with `--num-configs 3`
- one seed only
- no repeated sweeps

Run the short search with the scheduler:

```bash
uv run src/10_scheduler.py \
  --only hp_m1c_ \
  --paper-suite \
  --keep-going
```

Why this is acceptable even if training is long:

- the search is capped at a very small number of configs
- early stopping is enabled in the generated configs
- evaluation is automatic and strict

### Step 4. Select one winning `M1C` config

Choose the winner using the evaluation JSONs from the search runs.

Primary selection criteria:

1. primary-fix behavior
2. lower `SRR`
3. higher `GFR`
4. fidelity as a tie-breaker
5. lower disruption as final tie-breaker

Practical rule:

- use the `model_selection` block in each copied `eval.json` as a quick ranking aid
- do not accept a config that improves fidelity by noticeably worsening `SRR`

Search outputs to inspect:

- `models/hp_m1c_*/eval.json`

If you need the per-constraint breakdown, inspect the resolved run directory under `models/`; the scheduler does not copy `per_constraint.csv` back into the config directory.

### Step 5. Lock the paper hyperparameters

Once one `M1C` run wins, copy these settings into the canonical proposal configs:

Backbone and optimizer:

- `num_layers`
- `hidden_channels`
- `head_hidden`
- `dropout`
- `learning_rate`
- `weight_decay`
- `batch_size`
- `scheduler_factor`
- `scheduler_patience`
- `early_stopping_rounds`

Factorized-model settings to lock alongside the backbone:

- `pressure_type_conditioning`
- `pressure_residual_scale`
- `factor_executor_impl`
- `factor_loss.weight_pre`
- `factor_loss.weight_post_gold`
- chooser settings selected by the sweep:
  - `chooser.topk_candidates`
  - `chooser.max_candidates_total`
  - `chooser.beta_no_regression`
  - `chooser.gamma_primary`
  - `chooser.loss_weight`

Apply the full locked factorized setting bundle to:

- `a1_factorized_imitation`
- `m1c_safe_factor_chooser`
- `m1d_safe_factor_direct`

For `b0_eswc_reproduction`, reuse only the compatible shared training schedule:

- `learning_rate`
- `weight_decay`
- `batch_size`
- `scheduler_factor`
- `scheduler_patience`
- `early_stopping_rounds`

Do not copy factorized-only settings such as pressure or factor-loss weights into `B0`, because `B0` uses `constraint_representation="eswc_passive"`.

Do not run a second search on:

- `A1`
- `M1D`
- `B0`
- `G0`

### Step 6. Train and evaluate the final learned suite

Run the canonical learned models after the hyperparameters are locked.

Recommended order:

1. `B0`
2. `A1`
3. final `M1C`
4. `M1D`
5. `G0`

Reason:

- `B0` establishes the prior-work baseline
- `A1` establishes the representation-only step
- `M1C` is the main practical model
- `M1D` is the direct-loss counterpart
- `G0` depends on the factorized proposal stack and is the most downstream model

Recommended execution via scheduler:

```bash
uv run src/10_scheduler.py \
  --paper-suite \
  --keep-going
```

`--paper-suite` does not enforce the exact `B0 -> A1 -> M1C -> M1D -> G0` order by itself; the scheduler runs proposal configs before rerankers and otherwise follows directory-name ordering. Use the manual `--only` commands below when order matters.

If you want to enforce the order manually, use substring filters:

```bash
uv run src/10_scheduler.py --only b0_eswc_reproduction --paper-suite
uv run src/10_scheduler.py --only a1_factorized_imitation --paper-suite
uv run src/10_scheduler.py --only m1c_safe_factor_chooser --paper-suite
uv run src/10_scheduler.py --only m1d_safe_factor_direct --paper-suite
uv run src/10_scheduler.py --only g0_globalfix_reference --paper-suite
```

The scheduler will:

- train the run
- copy `checkpoint.pth`, `training_history.json`, and `eval.json` back into the config directory
- evaluate with strict global metrics
- automatically add `--use-chooser` for chooser runs
- automatically evaluate reranker runs from `reranker_predictions.json`

## 4. Manual evaluation commands

Use these only if you do not use the scheduler.

### Proposal models

`B0`, `A1`, and `M1D`:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --strict-global-metrics \
  --per-constraint-csv
```

`M1C`:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --use-chooser \
  --strict-global-metrics \
  --per-constraint-csv
```

### Reranker model

Train with:

```bash
uv run src/08_train_reranker.py \
  --experiment-config models/g0_globalfix_reference__<variant>__<encoding>/config.json \
  --seed 42
```

Then evaluate with:

```bash
uv run src/09_eval.py \
  --run-directory models/<g0_run_dir> \
  --reranker-predictions models/<g0_run_dir>/reranker_predictions.json \
  --strict-global-metrics \
  --per-constraint-csv
```

## 5. Final reporting set

Only the following runs should feed the paper tables:

- heuristic baselines: `DFB`, `CSM`, optional `AMB`
- learned models:
  - `B0`
  - `A1`
  - final `M1C`
  - `M1D`
  - `G0`

The hyperparameter search runs must not appear in the final paper tables.

These runs are sufficient to support the conceptual paper claims:

- `B0` gives the passive-context ESWC-style comparison point
- `A1` tests whether executable-factor structure helps before safety-aware decision logic
- `M1C` and `M1D` are the main safe-factor results
- `G0` serves as the global-satisfaction reference / upper-bound style comparison
- `DFB`, `AMB`, and `CSM` provide heuristic anchors

## 6. Minimal reproducibility checklist

Before declaring the suite complete, verify:

- the same dataset variant, `min_occurrence`, and encoding were used everywhere
- all proposal runs used the fixed seed behavior in `src/07_train.py`
- all reranker runs used `--seed 42`
- each generated config directory contains:
  - `config.json`
  - `checkpoint.pth`
  - `training_history.json`
  - `eval.json`
- baselines were written under `models/baselines/full/parquet/`
- the resolved runtime directories under `models/` contain `evaluations/model.json` and `evaluations/per_constraint.csv` for the final paper runs

## 7. Recommended stop rule

Stop after:

- one brief `M1C` sweep
- one locked final training pass for the canonical suite

Do not expand into:

- per-model searches
- multi-seed sweeps
- separate reranker searches
- appendix-model training

unless the final paper suite fails in a way that blocks the main claims.
