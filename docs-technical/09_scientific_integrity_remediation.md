# Scientific Integrity Remediation

Date: 2026-07-13

This is the implementation and rerun record for the repository audit. It is
paired with the [paper-narrative update](../docs-conceptual/09_paper_narrative_after_integrity_remediation.md).

## Validity Boundary

All stored paper metrics produced before evaluation schema v2 or constraint
semantics `wikidata-main-v4` are invalid for final reporting. They may be
retained as historical diagnostics, but must not be mixed with rerun values.
The reset is required because the data split, semantic label space, constraint
evaluator, graph batching, candidate inference, and metric definitions all
changed.

The historical `h2_a1_shared_pressure` result remains evidence for choosing a
more parsimonious architecture. Its old test metrics are not final A1 results.

## Implemented Changes

| Area | Integrity problem | Implemented contract |
| --- | --- | --- |
| Split policy | Raw train/dev/test rows were pooled and repartitioned. | `02_dataframe_builder.py` defaults to `--split-policy preserve`; dev maps to val. Legacy restratification is explicit. |
| Frequency filtering | Rare semantic identities were replaced by one ID before graph construction and evaluation. | Identity columns/encoder remain lossless. `<column>_feature`, `feature_encoder.txt`, and `identity_to_feature.npy` provide the filtered model space. |
| Filtering interpretation | `min_occurrence` was entangled with label correctness. | Filtering is now only a modeling decision. Strict identity metrics, representability coverage, and no-filter/intermediate sensitivity runs expose its effect. |
| Sampling | Fractional per-stratum rounding did not guarantee the named benchmark size. | `--target-rows` performs deterministic proportional allocation with an exact global total. Sampling follows primary validation. |
| Constraint semantics | Labeler and evaluator had duplicated parsers and inconsistent parameter use; `P2309` mode items could be mistaken for statement predicates, and type checks ignored subclass paths. | `modules/constraint_semantics.py` is shared by labeling and evaluation. Family-specific `P2306`, `P2305`, `P2303`, `P2308`, `P2309`, and `P4680` semantics are centralized. `P2309` maps its three allowed Q-items to `P31`, `P279`, or both. `type`/`valueType` use a hashed training-only `P279*` closure shared by labeling and evaluation. |
| Evidence state | Focus/comparison statements could be absent, a later entity snapshot could already contain a gold addition or omit a gold deletion, and shared-subject comparison rows could attach the third entity description to the wrong entity. | Focus/comparison statements are seeded, the third description is assigned to `other_object` when the subject is shared and to `other_subject` otherwise, then the declared transition is reversed to construct PRE. POST_GOLD reapplies the edit. |
| Primary-row validity | Rows could enter training/evaluation with an exempt, unsupported, uncheckable, or already-satisfied primary; aggregate counts could hide total loss of one family. | Labeling emits primary pre/post fields and reason codes. `--filter-invalid-primary` excludes invalid rows and writes aggregate and per-family primary-validation audits. |
| Historical edit validity | An eligible PRE violation was implicitly treated as proof that the observed edit fully repaired it. | PRE eligibility and POST_GOLD repair verification are separate. `primary_gold_repair_status` and a per-family audit report verified, post-uncheckable, and post-unsatisfied observed edits. |
| PyG batching | Local node references could be offset incorrectly or cross graph boundaries. | `ConstraintGraphData.__inc__` defines field-specific offsets; model forward validates source, target, and predicate graph membership. |
| Graph identity | Feature filtering could merge distinct nodes. | Node maps are keyed by identity; `x` carries features and `node_identity_id` carries identity. Targets include `y`, `y_identity`, and `target_representable_mask`. |
| Gold leakage | Candidate inference and oracle analysis used a builder that could inspect `graph.y`. | Training and inference APIs are separate. `build_inference_candidates` has no graph/gold parameter; all evaluation/oracle consumers use it. |
| Fidelity | Edit F1 could credit equality in the compressed feature space. | Headline precision/recall/F1 use strict identity triples. Feature-space F1 and representable-only F1 are labeled diagnostics. Ambiguous feature predictions remain unresolved, never assigned an arbitrary identity. |
| Primary success | Model selection used heuristic action-pattern matching as “primary fix.” | Primary fix is the eligible pre-violation to post-checkable/post-satisfied transition. Action matching is `repair_action_match_rate` and is diagnostic only. |
| SRR/SIR | Undefined zero-denominator samples were averaged as zeros. | Headline SRR/SIR are pooled numerator/denominator ratios. `*_macro_defined` and defined supports are reported separately. |
| A1 size | Per-type role-pressure blocks dominated the architecture. | Canonical A1-family configs now use `pressure_module_sharing="shared"`. `_pressure_role_modules` falls from 41,829,600 to 1,442,400 parameters, a 96.6% subsystem reduction. |
| Artifact provenance | Filenames did not prove config/data/code compatibility. | Checkpoints embed config hashes and seed. `run_manifest.json` records checkpoint/config/data/graph hashes, command, source state, deterministic-runtime flags, environment, and component parameter counts. Each `model.json` records the run-manifest/checkpoint hashes; evaluation rejects config/checkpoint mismatch. |
| Acceptance gate | Stale results could be reused without a machine check. | `scripts/validate_scientific_integrity.py` validates schema-v2 data, primary invariants, graph batching, A1 sharing, checkpoint compatibility/provenance, strict identity metrics, and pooled ratios. |

## A1 Promotion

`a1_factorized_imitation` remains the canonical artifact tag, but its generated
architecture now uses shared role-pressure blocks. `M1C`, `M1D`, G0's A1
proposal, and A1-derived ablations inherit the same setting so comparisons do
not confound safety objectives with a 40.39M-parameter pressure difference.

Do not relabel the old A1 checkpoint. Its embedded config says `per_type`, while
the updated canonical config says `shared`; evaluation now rejects that mismatch.
A clean A1 checkpoint and provenance manifest are mandatory.

The 96.6% figure applies to `_pressure_role_modules`, not to the entire model.
Final total/trainable counts must be copied from the new `run_manifest.json`.

## Mandatory Reruns

| Artifact/experiment | Required action | Reason |
| --- | --- | --- |
| Full interim data | Rebuild | Preserved splits and dual ID spaces. |
| Constraint registry | Rebuild | Freeze the registry paired with the rerun. |
| Validated full data | Rebuild | Unified semantics and primary exclusion audit. |
| Exact 1M benchmark | Rebuild | Sample only eligible primary violations; exact total. |
| Factorized and passive graphs | Rebuild | Identity-keyed nodes, schema-v2 targets, corrected batching. |
| DFB/AMB/CFM/CDM baselines | Rerun | Changed rows, identities, and global evaluator. |
| B0, A1, M1C, M1D | Retrain and reevaluate | Changed split/features/graphs/metrics; shared-pressure architecture matching. |
| G0 | Retrain and regenerate predictions | New A1 proposal and target-free inference candidates. |
| Candidate-oracle and deletion-degeneracy analyses | Rerun | Candidate set and semantic evaluator changed. |
| H2 and primary-query appendix runs used by the paper | Retrain/reevaluate | They share invalidated data, graphs, and metric definitions. |
| `min_occurrence` sensitivity | Run B0 and A1 at 1, 10, and 100 | Frequency filtering is a declared modeling choice. |

## Clean Main Pipeline

Run from the repository root. These commands intentionally replace the
canonical data artifacts; archive any historical local outputs first if needed.

### 1. Data, semantics, and exact benchmark

```bash
uv run src/02_dataframe_builder.py \
  --dataset full \
  --output-dataset full \
  --min-occurrence 100 \
  --split-policy preserve \
  --overwrite

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
  --scope local \
  --overwrite
```

Record the exclusion counts from
`data/interim/full_valid_minocc100/primary_validation_audit.csv`, verify family
retention in `primary_validation_audit_by_constraint.csv`, record observed-edit
success from `primary_gold_repair_audit_by_constraint.csv`, and record the
hierarchy edge count/hash from `class_hierarchy_manifest.json`. Record final
split counts from `data/interim/full_strat1m_minocc100/dataset_manifest.json`
and the realized sampled-family outcomes from
`sample_primary_validation_audit_by_constraint.csv` and
`sample_gold_repair_audit_by_constraint.csv`.

Before constructing graphs, run the interim-only gate:

```bash
uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --stage interim \
  --verify-hashes \
  --output models/paper_diagnostics/integrity_preflight_interim.json
```

Do not proceed unless it exits zero. Output showing semantics older than
`wikidata-main-v4`, missing hierarchy files, loss of any primary family, or a
sample total other than 1,000,000 is stale.

### Section 1 rerun record (2026-07-13)

The canonical rerun completed and the interim gate passed 65 checks with no
errors or warnings.

| Artifact | Recorded result |
| --- | --- |
| Constraint semantics | `wikidata-main-v4` |
| Frozen training hierarchy | 259,206 direct `P279` edges; 186,989 child classes |
| Hierarchy SHA-256 | `2128504411ad579d71d12478275ad3096710d6a8d0eab9ce1704c5c35714d971` |
| Eligible full pool | 1,630,370 rows: train 1,304,339; val 163,052; test 162,979 |
| Exact benchmark | 1,000,000 rows: train 800,025; val 100,010; test 99,965 |
| Verified observed edits | 734,462 / 1,000,000 (73.45%) |
| Interim integrity gate | `ok: true`; 65 checks; 0 errors; 0 warnings |

The correction corpus has nine source primary families. `symmetric` is still an
executable attached-factor family, but there is no primary symmetric correction
file. The sampled POST_GOLD verification rates are:

| Primary family | Sample rows | Verified POST_GOLD |
| --- | ---: | ---: |
| `conflictWith` | 121,845 | 77.31% |
| `distinct` | 279,861 | 57.93% |
| `inverse` | 118,081 | 100.00% |
| `itemRequiresStatement` | 100,976 | 99.61% |
| `oneOf` | 8,408 | 100.00% |
| `single` | 135,431 | 60.78% |
| `type` | 87,560 | 67.65% |
| `valueRequiresStatement` | 63,683 | 100.00% |
| `valueType` | 84,155 | 54.46% |

These are dataset diagnostics, not model results. The 265,538
post-unsatisfied rows remain eligible observed violations for correction
imitation, but the manuscript must not call their historical edits verified
repairs.

### Resume after a semantics-v4 code update

If `full_minocc100` already has schema v2, preserved splits, and matching
encoders, and `constraint_registry_full.parquet` already exists, the expensive
dataframe and registry commands do not need to be repeated. Re-run the labeler,
sampler, and interim gate above:

```bash
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
  --scope local \
  --overwrite
```

### 2. Graphs

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
  --constraint-representation eswc_passive \
  --registry-dataset full \
  --shard-size 200000 \
  --use-torch-save \
  --overwrite atomic
```

### 3. Preflight validation

```bash
uv run scripts/validate_scientific_integrity.py \
  --dataset-variant full_strat1m_minocc100 \
  --encoding node_id \
  --constraint-representation factorized \
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
```

Both commands must exit zero before training.

### 4. Configs

Generate the canonical suite first so existing configs are updated, then add
optional ablations without overwriting it.

```bash
uv run scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 \
  --encoding node_id

uv run scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 \
  --encoding node_id \
  --include-h2-ablations
```

### 5. Baselines and learned suite

```bash
uv run src/09_eval.py \
  --run-baselines \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --registry-dataset full \
  --strict-global-metrics \
  --per-constraint-csv

uv run src/10_scheduler.py --only b0_eswc_reproduction --paper-suite --seed 42
uv run src/10_scheduler.py --only a1_factorized_imitation --paper-suite --seed 42
uv run src/10_scheduler.py --only m1c_safe_factor_chooser --paper-suite --seed 42
uv run src/10_scheduler.py --only m1d_safe_factor_direct --paper-suite --seed 42
uv run src/10_scheduler.py --only g0_globalfix_reference --paper-suite --seed 42
```

Run G0 only after the promoted A1 checkpoint has passed run validation.

### 6. Diagnostics used by the paper

```bash
uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/a1_factorized_imitation__full_strat1m_minocc100__node_id \
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

For candidate-oracle tables that include the safety models, also run:

```bash
uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/m1c_safe_factor_chooser__full_strat1m_minocc100__node_id \
  --strict-global-metrics --registry-dataset full

uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/m1d_safe_factor_direct__full_strat1m_minocc100__node_id \
  --strict-global-metrics --registry-dataset full
```

### 7. Appendix experiments used in the manuscript

The historical `h2_a1_shared_pressure` row is retired as an independent
ablation because that architecture is now A1. The reverse parameterization
control is generated as `h2_a1_per_type_pressure`.

```bash
uv run src/10_scheduler.py --only h2_a1_per_type_pressure --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_no_factor_loss --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_legacy_shared_executor --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only h2_a1_gold_scalar_pressure --paper-suite --seed 42 --force-retrain
```

If the primary-query experiments remain in the manuscript, rebuild their graph
modes, generate configs, and rerun all three:

```bash
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

uv run scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 \
  --encoding node_id \
  --include-primary-query-ablations

uv run src/10_scheduler.py --only a1_qdef_secondary_factors --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_qfamily_secondary_factors --paper-suite --seed 42 --force-retrain
uv run src/10_scheduler.py --only a1_primary_passive_secondary_factors --paper-suite --seed 42 --force-retrain
```

### 8. Result acceptance

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

No number enters a paper table unless this command exits zero.

## Frequency-Filtering Sensitivity

Repeat the validated-data, exact-sampling, graph, config, B0, and A1 steps for
`min_occurrence=1` and `10`. Keep raw split policy, target rows, seed, graph
settings, and optimization settings fixed. Use explicitly named variants so the
no-filter run cannot collide with another `min_occurrence=1` artifact:

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
    --dataset full --output-dataset "$base" --min-occurrence "$k" \
    --split-policy preserve --overwrite

  uv run src/05_constraint_labeler.py \
    --dataset "$base" --output-dataset "$valid" --min-occurrence "$k" \
    --registry-dataset full --constraint-scope local \
    --factor-family-policy supported_only --filter-invalid-primary --overwrite

  uv run src/02b_stratified_benchmark_sampler.py \
    --source-dataset "$valid" --output-dataset "$bench" \
    --min-occurrence "$k" --target-rows 1000000 --seed 42 --overwrite

  uv run src/06_graph.py \
    --dataset "$bench" --min-occurrence "$k" --encoding node_id \
    --constraint-scope local --constraint-representation factorized \
    --registry-dataset full --shard-size 200000 --use-torch-save --overwrite atomic

  uv run src/06_graph.py \
    --dataset "$bench" --min-occurrence "$k" --encoding node_id \
    --constraint-representation eswc_passive --registry-dataset full \
    --shard-size 200000 --use-torch-save --overwrite atomic

  uv run scripts/validate_scientific_integrity.py \
    --dataset-variant "$variant" --encoding node_id \
    --constraint-representation factorized --stage data --verify-hashes

  uv run scripts/make_experiment_configs.py --variant "$variant" --encoding node_id
  uv run src/09_eval.py --run-baselines --dataset "$bench" \
    --min-occurrence "$k" --registry-dataset full --strict-global-metrics
  uv run src/10_scheduler.py --only "b0_eswc_reproduction__${variant}__node_id" \
    --paper-suite --seed 42 --force-retrain
  uv run src/10_scheduler.py --only "a1_factorized_imitation__${variant}__node_id" \
    --paper-suite --seed 42 --force-retrain
done
```

At each threshold report:

- identity and feature vocabulary sizes;
- active-target-slot and fully-representable-row coverage;
- strict identity Micro-F1;
- representable-only and feature-space Micro-F1;
- primary fix, pooled SRR/SIR, GFR, and disruption.

If the main A1-vs-B0 conclusion changes across thresholds, filtering is a
moderator and the paper must say so. If it is stable, threshold 100 remains a
computational choice supported by sensitivity analysis.

## Acceptance Checklist

- Upstream splits are preserved and manifest hashes match.
- Every retained primary is pre-checkable and pre-violated.
- Final sampled row total is exactly 1,000,000; split totals are reported.
- Graph manifests are schema v2 and the batching check passes.
- Canonical A1 uses shared pressure and its run manifest reports no more than
  1.5M `_pressure_role_modules` parameters.
- Every checkpoint config matches its effective `config.json`.
- Inference and oracle candidate generation declare no gold access.
- Headline F1 is strict identity F1.
- Primary fix is transition-based; action match is diagnostic.
- SRR/SIR equal pooled totals; macro-defined diagnostics include support.
- Final tables use only evaluation schema v2 results.
