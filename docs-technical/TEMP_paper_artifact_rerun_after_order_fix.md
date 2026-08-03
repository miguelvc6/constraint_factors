# Temporary Low-Storage Paper-Artifact Rerun

Delete this file after the successful rerun is copied into the paper run
ledger. Run every block from the repository root. The procedure treats graph
payloads as reproducible intermediates: build one compatible storage group,
validate it with full hashes, produce every dependent artifact, pass the
acceptance gate, then unlink only the shard names recorded by the graph
manifests.

Never delete a graph manifest. Preserve dataset and graph manifests, acceptance
reports, pruning receipts, configs, checkpoints, run manifests, histories,
evaluations, scheduler logs, and diagnostics. After a group is pruned,
`validate_scientific_integrity.py --stage data` is expected to fail for that
group because the payloads are intentionally absent.

The commands assume `data/interim/full_valid_minocc100/` and
`data/interim/constraint_registry_full.parquet` already passed semantic
remediation. Keep `--force-retrain`: the sampled row order and graph-manifest
lineage invalidate older checkpoints. The 15-epoch maximum and early stopping
remain unchanged.

## 1. Record and test the implementation

```bash
set -euo pipefail

mkdir -p models/paper_diagnostics
git rev-parse HEAD | tee models/paper_diagnostics/order_fix_rerun_commit.txt
uv run pytest -q
```

## 2. Recreate and validate the sampled parquet

The interim gate must pass before any graph construction. It checks the exact
SplitMix64 row-order method, the shared seed, the positive external-sort bucket
count, and the hash of `sampling_metadata.json`.

```bash
set -euo pipefail

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

## 3. Canonical factorized/passive group: build and generate configs

`GRAPH_MIN_FREE_GIB` is a local safety floor, not an estimate of total build
size. Raise it for the host if the remaining two representations plus temporary
files need more headroom.

```bash
set -euo pipefail

mkdir -p data/processed/full_strat1m_minocc100
check_disk() {
  min_gib="$1"
  free_kib="$(df -Pk data/processed | awk 'NR == 2 {print $4}')"
  required_kib="$((min_gib * 1024 * 1024))"
  if [ "$free_kib" -lt "$required_kib" ]; then
    echo "insufficient graph-build space: need ${min_gib} GiB free" >&2
    exit 1
  fi
}

check_disk "${GRAPH_MIN_FREE_GIB:-50}"
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

check_disk "${GRAPH_MIN_FREE_GIB:-50}"
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-representation eswc_passive \
  --registry-dataset full \
  --shard-size 200000 \
  --use-torch-save \
  --overwrite atomic

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
```

Check the locked epoch ceiling before training. This block uses full scheduler
slugs so a similarly named directory cannot be selected accidentally.

```bash
set -euo pipefail

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
print(f"validated {len(names)} configs")
PY
```

## 4. Canonical factorized/passive group: produce all dependants

Run G0 only after A1 completes and its scheduler evaluation succeeds.

```bash
set -euo pipefail

uv run src/09_eval.py \
  --run-baselines \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --registry-dataset full \
  --strict-global-metrics \
  --per-constraint-csv

uv run src/10_scheduler.py --only b0_eswc_reproduction__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_factorized_imitation__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only m1c_safe_factor_chooser__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only m1d_safe_factor_direct__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only g0_globalfix_reference__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain

uv run src/10_scheduler.py --only h2_a1_per_type_pressure__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_no_factor_loss__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_legacy_shared_executor__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_gold_scalar_pressure__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
```

Produce every canonical diagnostic while the graph payloads are present.

```bash
set -euo pipefail

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

## 5. Canonical factorized/passive group: accept, then prune

Both final reports include all canonical and H2 runs. Each report also commits
to the three manifests for the representation it authorizes pruning.
Before either report can pass, every strict model and baseline evaluation must
record a zero-mismatch PRE-state audit covering the full test support, and its
primary-fix denominator and per-family denominator sum must equal that support.

```bash
set -euo pipefail

common_args=(
  --dataset-variant full_strat1m_minocc100
  --encoding node_id
  --run-directory models/b0_eswc_reproduction__full_strat1m_minocc100__node_id
  --run-directory models/a1_factorized_imitation__full_strat1m_minocc100__node_id
  --run-directory models/m1c_safe_factor_chooser__full_strat1m_minocc100__node_id
  --run-directory models/m1d_safe_factor_direct__full_strat1m_minocc100__node_id
  --run-directory models/g0_globalfix_reference__full_strat1m_minocc100__node_id
  --run-directory models/h2_a1_per_type_pressure__full_strat1m_minocc100__node_id
  --run-directory models/h2_a1_no_factor_loss__full_strat1m_minocc100__node_id
  --run-directory models/h2_a1_legacy_shared_executor__full_strat1m_minocc100__node_id
  --run-directory models/h2_a1_gold_scalar_pressure__full_strat1m_minocc100__node_id
  --baseline-directory models/baselines/full_strat1m_minocc100/parquet
  --stage all
  --verify-hashes
)

uv run scripts/validate_scientific_integrity.py \
  "${common_args[@]}" \
  --constraint-representation factorized \
  --primary-constraint-mode executable_factor \
  --output models/paper_diagnostics/integrity_results_canonical_factorized.json
uv run scripts/validate_scientific_integrity.py \
  "${common_args[@]}" \
  --constraint-representation eswc_passive \
  --primary-constraint-mode executable_factor \
  --output models/paper_diagnostics/integrity_results_canonical_passive.json
```

The dry runs prove the cleanup plan. The delete calls cannot run unless the
reports are `ok: true`, contain the matching current manifest hashes, and state
that payload hashes were verified.

```bash
set -euo pipefail

factor_manifests=(
  data/processed/full_strat1m_minocc100/train_graph-node_id.pkl.manifest.json
  data/processed/full_strat1m_minocc100/val_graph-node_id.pkl.manifest.json
  data/processed/full_strat1m_minocc100/test_graph-node_id.pkl.manifest.json
)
passive_manifests=(
  data/processed/full_strat1m_minocc100/train_graph_repr-eswc_passive-node_id.pkl.manifest.json
  data/processed/full_strat1m_minocc100/val_graph_repr-eswc_passive-node_id.pkl.manifest.json
  data/processed/full_strat1m_minocc100/test_graph_repr-eswc_passive-node_id.pkl.manifest.json
)

uv run scripts/prune_graph_artifacts.py \
  --graph-manifest "${factor_manifests[@]}" \
  --integrity-report models/paper_diagnostics/integrity_results_canonical_factorized.json \
  --receipt models/paper_diagnostics/prune_canonical_factorized_dry_run.json \
  --dry-run
uv run scripts/prune_graph_artifacts.py \
  --graph-manifest "${passive_manifests[@]}" \
  --integrity-report models/paper_diagnostics/integrity_results_canonical_passive.json \
  --receipt models/paper_diagnostics/prune_canonical_passive_dry_run.json \
  --dry-run
uv run scripts/prune_graph_artifacts.py \
  --graph-manifest "${factor_manifests[@]}" \
  --integrity-report models/paper_diagnostics/integrity_results_canonical_factorized.json \
  --receipt models/paper_diagnostics/prune_canonical_factorized.json \
  --delete
uv run scripts/prune_graph_artifacts.py \
  --graph-manifest "${passive_manifests[@]}" \
  --integrity-report models/paper_diagnostics/integrity_results_canonical_passive.json \
  --receipt models/paper_diagnostics/prune_canonical_passive.json \
  --delete
```

## 6. Query-family/query-definition group

Build `query_family` from parquet, then create the structurally identical
`query_definition` names as hard links. Validate and train both before unlinking
the linked `query_definition` names first.

```bash
set -euo pipefail

mkdir -p data/processed/full_strat1m_minocc100
check_disk() {
  min_gib="$1"
  free_kib="$(df -Pk data/processed | awk 'NR == 2 {print $4}')"
  [ "$free_kib" -ge "$((min_gib * 1024 * 1024))" ] || {
    echo "insufficient graph-build space: need ${min_gib} GiB free" >&2
    exit 1
  }
}

check_disk "${GRAPH_MIN_FREE_GIB:-35}"
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-scope local \
  --constraint-representation factorized \
  --primary-constraint-mode query_family \
  --registry-dataset full \
  --shard-size 200000 \
  --use-torch-save \
  --overwrite atomic
uv run scripts/convert_primary_query_graph_mode.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --source-mode query_family \
  --target-mode query_definition \
  --link-identical-structure \
  --overwrite

uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
  --primary-constraint-mode query_family \
  --stage data \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_preflight_query_family.json
uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
  --primary-constraint-mode query_definition \
  --stage data \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_preflight_query_definition.json

uv run src/10_scheduler.py --only a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_qdef_secondary_factors__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
```

```bash
set -euo pipefail

uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
  --primary-constraint-mode query_family \
  --run-directory models/a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id \
  --stage all \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_results_query_family.json
uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
  --primary-constraint-mode query_definition \
  --run-directory models/a1_qdef_secondary_factors__full_strat1m_minocc100__node_id \
  --stage all \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_results_query_definition.json

qdef_manifests=(
  data/processed/full_strat1m_minocc100/train_graph-node_id-primary_query_definition.pkl.manifest.json
  data/processed/full_strat1m_minocc100/val_graph-node_id-primary_query_definition.pkl.manifest.json
  data/processed/full_strat1m_minocc100/test_graph-node_id-primary_query_definition.pkl.manifest.json
)
qfamily_manifests=(
  data/processed/full_strat1m_minocc100/train_graph-node_id-primary_query_family.pkl.manifest.json
  data/processed/full_strat1m_minocc100/val_graph-node_id-primary_query_family.pkl.manifest.json
  data/processed/full_strat1m_minocc100/test_graph-node_id-primary_query_family.pkl.manifest.json
)
uv run scripts/prune_graph_artifacts.py \
  --graph-manifest "${qdef_manifests[@]}" \
  --integrity-report models/paper_diagnostics/integrity_results_query_definition.json \
  --receipt models/paper_diagnostics/prune_query_definition.json \
  --delete
uv run scripts/prune_graph_artifacts.py \
  --graph-manifest "${qfamily_manifests[@]}" \
  --integrity-report models/paper_diagnostics/integrity_results_query_family.json \
  --receipt models/paper_diagnostics/prune_query_family.json \
  --delete
```

## 7. Passive-primary-node group

Build this representation directly from parquet only after query-family storage
has been reclaimed.

```bash
set -euo pipefail

mkdir -p data/processed/full_strat1m_minocc100
free_kib="$(df -Pk data/processed | awk 'NR == 2 {print $4}')"
min_gib="${GRAPH_MIN_FREE_GIB:-35}"
[ "$free_kib" -ge "$((min_gib * 1024 * 1024))" ] || {
  echo "insufficient graph-build space: need ${min_gib} GiB free" >&2
  exit 1
}

uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-scope local \
  --constraint-representation factorized \
  --primary-constraint-mode passive_node \
  --registry-dataset full \
  --shard-size 200000 \
  --use-torch-save \
  --overwrite atomic
uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
  --primary-constraint-mode passive_node \
  --stage data \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_preflight_passive_node.json
uv run src/10_scheduler.py --only a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id --paper-suite --seed 42 --force-retrain
uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
  --primary-constraint-mode passive_node \
  --run-directory models/a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id \
  --stage all \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_results_passive_node.json

passive_node_manifests=(
  data/processed/full_strat1m_minocc100/train_graph-node_id-primary_passive_node.pkl.manifest.json
  data/processed/full_strat1m_minocc100/val_graph-node_id-primary_passive_node.pkl.manifest.json
  data/processed/full_strat1m_minocc100/test_graph-node_id-primary_passive_node.pkl.manifest.json
)
uv run scripts/prune_graph_artifacts.py \
  --graph-manifest "${passive_node_manifests[@]}" \
  --integrity-report models/paper_diagnostics/integrity_results_passive_node.json \
  --receipt models/paper_diagnostics/prune_passive_node.json \
  --delete
```

## 8. Frequency sensitivities, one threshold at a time

The loop holds only one threshold's two graph representations at once. For each
threshold it produces B0, A1, all baselines, both full-hash representation
preflights, both final acceptance reports, and both pruning receipts before
continuing.

```bash
set -euo pipefail

mkdir -p models/paper_diagnostics data/processed
check_disk() {
  min_gib="$1"
  free_kib="$(df -Pk data/processed | awk 'NR == 2 {print $4}')"
  [ "$free_kib" -ge "$((min_gib * 1024 * 1024))" ] || {
    echo "insufficient graph-build space: need ${min_gib} GiB free" >&2
    exit 1
  }
}

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
  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant "$variant" \
    --stage interim \
    --verify-hashes \
    --output "models/paper_diagnostics/integrity_preflight_${variant}_interim.json"

  check_disk "${GRAPH_MIN_FREE_GIB:-50}"
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
  check_disk "${GRAPH_MIN_FREE_GIB:-50}"
  uv run src/06_graph.py \
    --dataset "$bench" \
    --min-occurrence "$k" \
    --encoding node_id \
    --constraint-representation eswc_passive \
    --registry-dataset full \
    --shard-size 200000 \
    --use-torch-save \
    --overwrite atomic

  factor_preflight="models/paper_diagnostics/integrity_preflight_${variant}_factorized.json"
  passive_preflight="models/paper_diagnostics/integrity_preflight_${variant}_passive.json"
  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant "$variant" \
    --encoding node_id \
    --constraint-representation factorized \
    --primary-constraint-mode executable_factor \
    --stage data \
    --verify-hashes \
    --output "$factor_preflight"
  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant "$variant" \
    --encoding node_id \
    --constraint-representation eswc_passive \
    --primary-constraint-mode executable_factor \
    --stage data \
    --verify-hashes \
    --output "$passive_preflight"

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

  factor_acceptance="models/paper_diagnostics/integrity_results_${variant}_factorized.json"
  passive_acceptance="models/paper_diagnostics/integrity_results_${variant}_passive.json"
  common_args=(
    --dataset-variant "$variant"
    --encoding node_id
    --run-directory "models/b0_eswc_reproduction__${variant}__node_id"
    --run-directory "models/a1_factorized_imitation__${variant}__node_id"
    --baseline-directory "models/baselines/${variant}/parquet"
    --stage all
    --verify-hashes
  )
  uv run scripts/validate_scientific_integrity.py \
    "${common_args[@]}" \
    --constraint-representation factorized \
    --primary-constraint-mode executable_factor \
    --output "$factor_acceptance"
  uv run scripts/validate_scientific_integrity.py \
    "${common_args[@]}" \
    --constraint-representation eswc_passive \
    --primary-constraint-mode executable_factor \
    --output "$passive_acceptance"

  factor_manifests=(
    "data/processed/${variant}/train_graph-node_id.pkl.manifest.json"
    "data/processed/${variant}/val_graph-node_id.pkl.manifest.json"
    "data/processed/${variant}/test_graph-node_id.pkl.manifest.json"
  )
  passive_manifests=(
    "data/processed/${variant}/train_graph_repr-eswc_passive-node_id.pkl.manifest.json"
    "data/processed/${variant}/val_graph_repr-eswc_passive-node_id.pkl.manifest.json"
    "data/processed/${variant}/test_graph_repr-eswc_passive-node_id.pkl.manifest.json"
  )
  uv run scripts/prune_graph_artifacts.py \
    --graph-manifest "${factor_manifests[@]}" \
    --integrity-report "$factor_acceptance" \
    --receipt "models/paper_diagnostics/prune_${variant}_factorized.json" \
    --delete
  uv run scripts/prune_graph_artifacts.py \
    --graph-manifest "${passive_manifests[@]}" \
    --integrity-report "$passive_acceptance" \
    --receipt "models/paper_diagnostics/prune_${variant}_passive.json" \
    --delete
done
```

## 9. Completion condition

The rerun is complete only when every final acceptance gate and pruning command
exits zero. Archive all immutable manifests and reports plus the retained model
and diagnostic artifacts. Do not run a post-pruning `--stage data` gate and do
not recreate graph payloads merely to make that gate pass.
