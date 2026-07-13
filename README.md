# Constraint Factors

Executable constraint factors for neuro-symbolic knowledge graph repair.

This repository studies the fidelity-safety gap in local KG repair: executable constraint factors improve historical repair imitation, but symbolic repair safety and non-vacuous evidence preservation must be evaluated as separate objectives. The codebase includes the full research pipeline: dataset preparation, constraint metadata construction, factor labeling, graph materialization, baseline evaluation, learned models, and paper-facing experiment scheduling.

## Research Scope

The project focuses on local repair in collaborative knowledge graphs, with Wikidata constraint-correction data as the main experimental setting.

Core questions:

- Can executable local constraint factors improve historical repair imitation beyond passive constraint context?
- Does better curator-edit imitation imply safer post-edit symbolic graph state?
- How do factorized proposal models compare with chooser-based, direct-loss, global-satisfaction, and delete-focus references?

The canonical paper-facing model surface is:

- `B0`: ESWC-style passive baseline
- `A1`: factorized imitation model with shared role-pressure blocks
- `M1C`: safe factor chooser model
- `M1D`: safe factor direct-loss model
- `G0`: global-fix reranker reference

All stored pre-schema-v2 paper metrics require a clean rerun. Start with the
[scientific integrity remediation and commands](docs-technical/09_scientific_integrity_remediation.md)
and the [paper narrative reset](docs-conceptual/09_paper_narrative_after_integrity_remediation.md).

## Setup

### Requirements

- `uv` for environment and dependency management
- Python `3.12`
- Linux/macOS shell environment
- Optional CUDA-capable GPU for training; CPU execution is fine for preprocessing and small smoke runs

Notes:

- The `sample` dataset is appropriate for debugging and smoke tests.
- The `full` dataset is the paper-facing run surface and is substantially larger.
- `text_embedding` runs require outbound network access for Wikidata lookup and model downloads.
- The first `src/03_constraint_registry.py` run may also use outbound network access to bootstrap `data/static/constraint_type_catalog.json` on a fresh clone.

### Environment Installation

```bash
uv sync --group dev --python 3.12
```

Optional shell activation:

```bash
source .venv/bin/activate
```

## Minimal Quickstart

This path builds a small local run on the `sample` dataset using `node_id` graphs.

### 1. Download the sample corpus

```bash
uv run src/01_data_downloader.py --dataset sample
```

### 2. Build parquet splits

```bash
uv run src/02_dataframe_builder.py \
  --dataset sample \
  --min-occurrence 100 \
  --split-policy preserve \
  --max-rows 300 \
  --overwrite
```

### 3. Build the constraint registry

```bash
uv run src/03_constraint_registry.py --dataset sample
```

On a fresh clone, this step bootstraps `data/static/constraint_type_catalog.json`
from the selected dataset's `constraints.tsv` if the catalog is missing.

### 4. Label local constraint factors

```bash
uv run src/05_constraint_labeler.py \
  --dataset sample \
  --min-occurrence 100 \
  --constraint-scope local \
  --factor-family-policy supported_only \
  --filter-invalid-primary \
  --max-rows 300 \
  --overwrite
```

### 5. Materialize factorized graphs

```bash
uv run src/06_graph.py \
  --dataset sample \
  --min-occurrence 100 \
  --constraint-scope local \
  --constraint-representation factorized \
  --encoding node_id \
  --registry-dataset sample \
  --max-instances 300 \
  --overwrite atomic
```

### 6. Run baseline evaluation

```bash
uv run src/09_eval.py \
  --run-baselines \
  --dataset sample \
  --min-occurrence 100
```

At this point you should have:

- raw data under `data/raw/sample/`
- parquet splits under `data/interim/sample_minocc100/`
- labeled parquet under `data/interim/sample_minocc100_labeled/`
- processed graphs under `data/processed/sample_minocc100/`

## Full Research Pipeline

The default paper-oriented run uses the `full` dataset, `min-occurrence 100`, `constraint-scope local`, and a fixed encoding choice for the entire experiment table.

### Data and artifact preparation

```bash
uv run src/01_data_downloader.py --dataset full

uv run src/02_dataframe_builder.py --dataset full --min-occurrence 100 --split-policy preserve --overwrite

uv run src/03_constraint_registry.py --dataset full

uv run src/05_constraint_labeler.py \
  --dataset full \
  --output-dataset full_valid \
  --min-occurrence 100 \
  --registry-dataset full \
  --constraint-scope local \
  --factor-family-policy supported_only \
  --filter-invalid-primary \
  --overwrite

uv run src/02b_stratified_benchmark_sampler.py \
  --source-dataset full_valid \
  --output-dataset full_strat1m \
  --min-occurrence 100 \
  --target-rows 1000000 \
  --seed 42 \
  --overwrite
```

For `node_id` factorized graphs:

```bash
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-scope local \
  --constraint-representation factorized \
  --registry-dataset full
```

For the passive baseline graphs:

```bash
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-representation eswc_passive \
  --registry-dataset full
```

If you want `text_embedding` graphs, build the text cache first:

```bash
uv run src/04_wikidata_retriever.py \
  --dataset full \
  --min-occurrence 100
```

Then run `src/06_graph.py` with `--encoding text_embedding`.

### Generate experiment configs

```bash
uv run scripts/make_experiment_configs.py \
  --models-root models \
  --variant full_strat1m_minocc100 \
  --encoding node_id
```

### Run the canonical paper suite

```bash
uv run src/10_scheduler.py --paper-suite --seed 42 --force-retrain
```

For a stricter ordered run, execute the model families one at a time:

```bash
uv run src/10_scheduler.py --only b0_eswc_reproduction --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_factorized_imitation --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only m1c_safe_factor_chooser --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only m1d_safe_factor_direct --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only g0_globalfix_reference --paper-suite --seed 42 --force-retrain
```

## Repository Layout

- `src/`: primary pipeline stages and training/evaluation entrypoints
- `src/modules/`: reusable modeling, evaluation, and utility components
- `scripts/`: config generation and support utilities
- `tests/`: smoke tests and paper-surface regression checks
- `data/`: raw, interim, processed, and static artifacts
- `models/`: generated configs, checkpoints, and experiment outputs
- `logs/`: scheduler and run logs

## Documentation Map

Documentation is intentionally split into two areas:

- [docs-conceptual/](docs-conceptual): research framing, hypotheses, benchmark intent, and evaluation philosophy
- [docs-technical/](docs-technical): implementation details, pipeline behavior, artifacts, and operational notes

Recommended entry points:

- [docs-conceptual/00-constraint_factors.md](docs-conceptual/00-constraint_factors.md)
- [docs-conceptual/constraint_factors_research_overview.html](docs-conceptual/constraint_factors_research_overview.html)
- [docs-conceptual/constraint_types.md](docs-conceptual/constraint_types.md)
- [docs-technical/00_models_and_evaluation_matrix.md](docs-technical/00_models_and_evaluation_matrix.md)
- [docs-technical/00_training_and_evaluation_execution_plan.md](docs-technical/00_training_and_evaluation_execution_plan.md)
- [docs-technical/09_scientific_integrity_remediation.md](docs-technical/09_scientific_integrity_remediation.md)
- [docs/README.md](docs/README.md)

## Validation and Smoke Tests

Useful checks during development:

```bash
uv run pytest -q
uv run scripts/validate_scientific_integrity.py --help
uv run python tests/smoke_pipeline_local_closure.py
```

The last command is an end-to-end smoke test and may download sample data if it is not already present.
