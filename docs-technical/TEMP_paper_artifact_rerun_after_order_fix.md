# Temporary Paper-Artifact Rerun Commands

Delete this file after the successful rerun has been transferred to the paper
run ledger. These commands assume `data/interim/full_valid_minocc100/` and
`data/interim/constraint_registry_full.parquet` already passed the semantic
remediation. The sampler-order change invalidates the sampled parquet hashes,
all derived graphs, configs, checkpoints, evaluations, and diagnostics; it does
not require rebuilding the main dataframe or relabeling `full_valid`.

Run every command from the repository root. Keep `--force-retrain`: an old
checkpoint is incompatible with the refreshed data provenance and passive-model
parameterization.

## 1. Record and test the implementation

```bash
set -euo pipefail
mkdir -p models/paper_diagnostics
git rev-parse HEAD | tee models/paper_diagnostics/order_fix_rerun_commit.txt

uv run pytest -q \
  tests/test_stratified_benchmark_sampler.py \
  tests/test_factor_executor_v1.py \
  tests/test_training_safeguards.py \
  tests/test_scientific_integrity_remediation.py \
  tests/test_h2_eval.py
```

## 2. Recreate the main sampled benchmark

```bash
uv run src/02b_stratified_benchmark_sampler.py \
  --source-dataset full_valid \
  --output-dataset full_strat1m \
  --min-occurrence 100 \
  --target-rows 1000000 \
  --seed 42 \
  --scope local \
  --overwrite

uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --stage interim \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_preflight_interim.json
```

## 3. Rebuild every main and appendix graph mode

```bash
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-scope local \
  --constraint-representation factorized \
  --primary-constraint-mode executable_factor \
  --registry-dataset full \
  --shard-size 200000 \
  --use-torch-save \
  --overwrite atomic

uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-scope local \
  --constraint-representation eswc_passive \
  --registry-dataset full \
  --shard-size 200000 \
  --use-torch-save \
  --overwrite atomic

for mode in query_definition query_family passive_node; do
  uv run src/06_graph.py \
    --dataset full_strat1m \
    --min-occurrence 100 \
    --encoding node_id \
    --constraint-scope local \
    --constraint-representation factorized \
    --primary-constraint-mode "$mode" \
    --registry-dataset full \
    --shard-size 200000 \
    --use-torch-save \
    --overwrite atomic
done

uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
  --primary-constraint-mode executable_factor \
  --stage data \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_preflight_factorized.json

uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation eswc_passive \
  --stage data \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_preflight_passive.json

for mode in query_definition query_family passive_node; do
  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant full_strat1m_minocc100 \
    --encoding node_id \
    --constraint-representation factorized \
    --primary-constraint-mode "$mode" \
    --stage data \
    --verify-hashes \
    --output "models/paper_diagnostics/integrity_preflight_${mode}.json"
done
```

## 4. Refresh every paper config

```bash
uv run scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 \
  --encoding node_id \
  --overwrite-existing

uv run scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 \
  --encoding node_id \
  --include-h2-ablations \
  --overwrite-existing

uv run scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 \
  --encoding node_id \
  --include-primary-query-ablations \
  --overwrite-existing

uv run python - <<'PY'
import json
from pathlib import Path

variant = "full_strat1m_minocc100__node_id"
names = (
    "b0_eswc_reproduction",
    "a1_factorized_imitation",
    "m1c_safe_factor_chooser",
    "m1d_safe_factor_direct",
    "g0_globalfix_reference",
    "h2_a1_per_type_pressure",
    "h2_a1_no_factor_loss",
    "h2_a1_legacy_shared_executor",
    "h2_a1_gold_scalar_pressure",
    "a1_qdef_secondary_factors",
    "a1_qfamily_secondary_factors",
    "a1_primary_passive_secondary_factors",
)
for name in names:
    path = Path("models") / f"{name}__{variant}" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    training = config["training_config"]
    assert training["num_epochs"] == 15, path
    assert training["validation_subset_size"] is None, path
print(f"validated {len(names)} refreshed configs")
PY
```

## 5. Recreate baselines and the canonical learned suite

```bash
uv run src/09_eval.py \
  --run-baselines \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --registry-dataset full \
  --strict-global-metrics \
  --per-constraint-csv

uv run src/10_scheduler.py --only b0_eswc_reproduction --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_factorized_imitation --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only m1c_safe_factor_chooser --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only m1d_safe_factor_direct --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only g0_globalfix_reference --paper-suite --seed 42 --force-retrain
```

Run G0 only after A1 finishes and passes its scheduler evaluation.

## 6. Recreate manuscript appendix runs

```bash
uv run src/10_scheduler.py --only h2_a1_per_type_pressure --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_no_factor_loss --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_legacy_shared_executor --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_gold_scalar_pressure --paper-suite --seed 42 --force-retrain

uv run src/10_scheduler.py --only a1_qdef_secondary_factors --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_qfamily_secondary_factors --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_primary_passive_secondary_factors --paper-suite --seed 42 --force-retrain
```

## 7. Recreate paper diagnostics

```bash
uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/a1_factorized_imitation__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --registry-dataset full

uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/m1c_safe_factor_chooser__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --registry-dataset full

uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/m1d_safe_factor_direct__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --registry-dataset full

uv run scripts/analyze_deletion_degeneracy.py \
  --g0-run-directory models/g0_globalfix_reference__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --registry-dataset full

uv run src/09_eval.py \
  --run-directory models/a1_factorized_imitation__full_strat1m_minocc100__node_id \
  --h2-eval \
  --strict-global-metrics \
  --registry-dataset full
```

## 8. Run the main acceptance gate

```bash
uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --run-directory models/b0_eswc_reproduction__full_strat1m_minocc100__node_id \
  --run-directory models/a1_factorized_imitation__full_strat1m_minocc100__node_id \
  --run-directory models/m1c_safe_factor_chooser__full_strat1m_minocc100__node_id \
  --run-directory models/m1d_safe_factor_direct__full_strat1m_minocc100__node_id \
  --run-directory models/g0_globalfix_reference__full_strat1m_minocc100__node_id \
  --baseline-directory models/baselines/full_strat1m_minocc100/parquet \
  --stage all \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_results.json
```

## 9. Build the frequency-filtering sensitivity artifacts

The threshold-1 and threshold-10 directories are not currently present, so
their commands begin at dataframe construction. The paper sensitivity uses B0,
A1, and all deterministic baselines at each threshold.

```bash
for k in 1 10; do
  base="full_freq${k}"
  valid="full_valid_freq${k}"
  bench="full_strat1m_freq${k}"
  if [ "$k" -eq 1 ]; then
    variant="$bench"
  else
    variant="${bench}_minocc${k}"
  fi

  uv run src/02_dataframe_builder.py \
    --dataset full \
    --output-dataset "$base" \
    --min-occurrence "$k" \
    --split-policy preserve \
    --overwrite

  uv run src/05_constraint_labeler.py \
    --dataset "$base" \
    --output-dataset "$valid" \
    --min-occurrence "$k" \
    --registry-dataset full \
    --constraint-scope local \
    --factor-family-policy supported_only \
    --filter-invalid-primary \
    --overwrite

  uv run src/02b_stratified_benchmark_sampler.py \
    --source-dataset "$valid" \
    --output-dataset "$bench" \
    --min-occurrence "$k" \
    --target-rows 1000000 \
    --seed 42 \
    --scope local \
    --overwrite

  uv run src/06_graph.py \
    --dataset "$bench" \
    --min-occurrence "$k" \
    --encoding node_id \
    --constraint-scope local \
    --constraint-representation factorized \
    --primary-constraint-mode executable_factor \
    --registry-dataset full \
    --shard-size 200000 \
    --use-torch-save \
    --overwrite atomic

  uv run src/06_graph.py \
    --dataset "$bench" \
    --min-occurrence "$k" \
    --encoding node_id \
    --constraint-scope local \
    --constraint-representation eswc_passive \
    --registry-dataset full \
    --shard-size 200000 \
    --use-torch-save \
    --overwrite atomic

  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant "$variant" \
    --encoding node_id \
    --constraint-representation factorized \
    --stage data \
    --verify-hashes \
    --output "models/paper_diagnostics/integrity_preflight_${variant}_factorized.json"

  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant "$variant" \
    --encoding node_id \
    --constraint-representation eswc_passive \
    --stage data \
    --verify-hashes \
    --output "models/paper_diagnostics/integrity_preflight_${variant}_passive.json"

  uv run scripts/make_experiment_configs.py \
    --variant "$variant" \
    --encoding node_id \
    --overwrite-existing

  uv run src/09_eval.py \
    --run-baselines \
    --dataset "$bench" \
    --min-occurrence "$k" \
    --registry-dataset full \
    --strict-global-metrics \
    --per-constraint-csv

  uv run src/10_scheduler.py \
    --only "b0_eswc_reproduction__${variant}__node_id" \
    --paper-suite \
    --seed 42 \
    --force-retrain

  uv run src/10_scheduler.py \
    --only "a1_factorized_imitation__${variant}__node_id" \
    --paper-suite \
    --seed 42 \
    --force-retrain

  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant "$variant" \
    --encoding node_id \
    --run-directory "models/b0_eswc_reproduction__${variant}__node_id" \
    --run-directory "models/a1_factorized_imitation__${variant}__node_id" \
    --baseline-directory "models/baselines/${variant}/parquet" \
    --stage all \
    --verify-hashes \
    --output "models/paper_diagnostics/integrity_results_${variant}.json"
done
```

The rerun is complete only when the main and both sensitivity acceptance gates
exit zero. Preserve their JSON outputs, scheduler logs, generated configs, run
manifests, training histories, evaluation directories, and diagnostic reports
with the paper artifact bundle.
