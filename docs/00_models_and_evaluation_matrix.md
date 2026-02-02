# Models & Evaluation Matrix

Date: 2026-02-02

This document specifies the implemented model variants, how they are trained/evaluated, and how to run the paper‑relevant suites with the current configuration system.

---

## 1) Model Catalog (Implemented Variants)

Legend for components:
- **Factor head**: predicts factor satisfaction logits (pre‑state).
- **Factor loss**: BCE on factor labels (pre‑state), masked by checkable.
- **Typed pressure**: factor→variable pressure messages conditioned on factor types.
- **Chooser**: candidate scoring head (proposal‑side Fix‑1).
- **Policy head**: predicts discrete policy class.
- **Reranker**: separate candidate scorer (08_train_reranker).

| Name (paper short) | Training script | Components enabled | Objective / loss terms | Inference decision rule | Expected behavior |
| --- | --- | --- | --- | --- | --- |
| **P0 Proposal Imitation** | `07_train.py` | Factor head: optional; Factor loss: OFF; Typed pressure: OFF; Chooser: OFF; Policy head: OFF; Reranker: NO | `L_edit` (6‑slot CE) | Argmax per slot | Curator‑like edits; no explicit regression control. |
| **P1 Proposal + Factor Loss** | `07_train.py` | Factor head: ON; Factor loss: ON; Typed pressure: OFF; Chooser: OFF; Policy head: OFF; Reranker: NO | `L_edit + λ·L_factor_pre` | Argmax per slot | Adds factor‑label supervision; no explicit Fix‑1. |
| **P2 Proposal + Typed Pressure** | `07_train.py` | Factor head: optional; Factor loss: OFF; Typed pressure: ON; Chooser: OFF; Policy head: OFF; Reranker: NO | `L_edit` | Argmax per slot | Pressure‑augmented message passing; still imitation‑only. |
| **P3 Proposal + Chooser (Fix‑1)** | `07_train.py` | Factor head: ON; Factor loss: ON; Typed pressure: OFF; Chooser: ON; Policy head: OFF; Reranker: NO | `L_edit + λ·L_factor_pre + L_chooser` where `L_chooser = CE(gold) + β·E[no‑regression] (+γ·primary)` | Chooser selects best candidate | Improves secondary regressions without reranker. |
| **P4 Proposal + Chooser + Typed Pressure** | `07_train.py` | Factor head: ON; Factor loss: ON; Typed pressure: ON; Chooser: ON; Policy head: OFF; Reranker: NO | `L_edit + λ·L_factor_pre + L_chooser` | Chooser selects best candidate | Same as P3 but with pressure messages. |
| **P5 Policy Choice** | `07_train.py` | Factor head: ON; Factor loss: ON; Typed pressure: ON; Chooser: OFF; Policy head: ON; Reranker: NO | `L_edit + λ·L_factor_pre + L_policy` | Policy‑filtered candidates, heuristic tie‑break | Strategy‑level decisions; restricted edit class. |
| **P6 Policy Choice + Chooser** | `07_train.py` | Factor head: ON; Factor loss: ON; Typed pressure: ON; Chooser: ON; Policy head: ON; Reranker: NO | `L_edit + λ·L_factor_pre + L_policy + L_chooser` | Policy‑filtered chooser selection | Strategy + candidate‑level scoring. |
| **R1 Reranker Fix‑1** | `08_train_reranker.py` | Reranker: ON (proposal model used for candidates) | `CE(gold) + β·E[no‑regression vs gold]` | Reranker chooses best candidate | Strong Fix‑1 behavior; not bound to slot argmax. |
| **R2 Reranker GlobalFix** | `08_train_reranker.py` | Reranker: ON | `-E[global satisfaction]` | Reranker chooses best candidate | Maximizes GFR; may reduce fidelity. |
| **Baselines** | `09_eval.py --run-baselines` | DFB / AMB / CSM heuristics | N/A | Heuristic repair | Sanity checks; not learned. |

Policy class set (documented in code, `src/modules/policy.py`):
- P0 NOOP
- P1 DELETE_FOCUS_TRIPLE
- P2 DELETE_CONFLICT_TRIPLE
- P3 ADD_VALUE_TO_FOCUS_PREDICATE
- P4 CHANGE_PREDICATE
- P5 OTHER

---

## 2) Evaluation Modes

### Scripts
- **`src/09_eval.py`**: Evaluates trained proposal models, reranker predictions, or baselines.
- **`src/08_train_reranker.py`**: Trains reranker and can emit reranker predictions for eval.

### Flags / modes
- `--strict-global-metrics`: fail fast unless GFR/SRR/SIR/disruption can be computed.
- `--no-global-metrics`: disable global metrics (ignored in strict mode).
- `--per-constraint-csv`: write per‑constraint CSV; forced on in strict mode.
- `--use-chooser`: evaluate proposals using chooser head instead of slot argmax.
- `--use-policy-choice`: apply policy‑filtered candidates before selection.
- `--reranker-predictions <path>`: evaluate saved reranker outputs without model forward pass.

### Output artifacts
- `models/<run>/evaluations/model.json` contains:
  - Standard fidelity metrics (precision/recall/F1)
  - `global_metrics_computed: true|false`
  - Optional `global_metrics` with `overall` and `per_constraint_type` keys
  - If present, `overall_gfr`, `overall_srr`, `overall_sir`, and disruption means
- `models/<run>/evaluations/per_constraint.csv` contains per‑constraint breakdown:
  - `constraint_type`, `support`, `fidelity_micro_f1`
  - `primary_fix_rate`, `primary_exact_rate`, `primary_alternative_rate`, `primary_total`
  - `gfr`, `srr`, `sir`
  - disruption means (`disruption_add_mean`, `disruption_del_mean`, `disruption_total_ops_mean`, `disruption_changed_mean`)

---

## 3) Experiment Suite Mapping

### Starter paper suite (minimal recommended runs)
Use these to cover the key figures/tables:

1) **Fidelity vs SRR frontier**
- `p0_imitation` (proposal argmax)
- `p3_chooser_fix1` (proposal + chooser)
- `r1_fix1_reranker` (reranker Fix‑1)
- `r2_global_fix_reranker` (global satisfaction upper bound)

2) **Typed pressure ablation**
- `p3_chooser_fix1` (no typed pressure)
- `p4_chooser_fix1_typed_pressure` (typed pressure)

3) **Chooser / Fix‑1 ablation**
- `p1_factor_loss` (no chooser)
- `p3_chooser_fix1` (chooser)

4) **Reranker vs integrated chooser**
- `p3_chooser_fix1` (proposal + chooser)
- `r1_fix1_reranker` (reranker Fix‑1)

5) **Policy choice comparisons**
- `p5_policy_choice` (policy only)
- `p6_policy_choice_with_chooser` (policy + chooser)

---

## 4) Exact Config Keys (config.json)

### Model config keys (`model_config`)
- `model`: model class name (`GIN`, `GIN_PRESSURE`, or `RERANKER`)
- `pressure_enabled`: bool
- `pressure_type_conditioning`: `none|concat|gate`
- `enable_policy_choice`: bool
- `policy_num_classes`: int (default policy set requires >= 6)
- `num_factor_types`: int

### Training config keys (`training_config`)
- `factor_loss.enabled`: bool
- `factor_loss.weight_pre`: float
- `fix_probability_loss.enabled`: bool
- `chooser.enabled`: bool
- `chooser.topk_candidates`: int
- `chooser.max_candidates_total`: int
- `chooser.beta_no_regression`: float
- `chooser.gamma_primary`: float
- `chooser.loss_mode`: `fix1|primary_only|global_fix`
- `policy_filter_strict`: bool

### Reranker config keys (`training_config` in reranker configs)
- `objective`: `main|global_fix`
- `topk_candidates`, `topk_per_slot`, `max_candidates_total`, `include_gold`, `regression_weight`

---

## 5) Example Config Snippets

### Proposal + Chooser + Typed Pressure
```json
{
  "model_config": {
    "model": "GIN_PRESSURE",
    "pressure_enabled": true,
    "pressure_type_conditioning": "concat",
    "enable_policy_choice": false
  },
  "training_config": {
    "factor_loss": { "enabled": true, "weight_pre": 0.1 },
    "chooser": {
      "enabled": true,
      "loss_mode": "fix1",
      "beta_no_regression": 0.5,
      "gamma_primary": 0.0,
      "topk_candidates": 20,
      "max_candidates_total": 80
    }
  }
}
```

### Reranker Fix‑1
```json
{
  "model_config": { "model": "RERANKER" },
  "training_config": {
    "objective": "main",
    "topk_candidates": 20,
    "topk_per_slot": 5,
    "max_candidates_total": 80,
    "include_gold": true,
    "regression_weight": 0.5
  },
  "proposal_config": {
    "model": "GIN_PRESSURE",
    "config_tag": "m1_main_fix1"
  }
}
```

### Policy Choice
```json
{
  "model_config": {
    "model": "GIN_PRESSURE",
    "enable_policy_choice": true,
    "policy_num_classes": 6
  },
  "training_config": {
    "policy_filter_strict": true
  }
}
```

---

## 6) Cross‑Check: Runnable vs Disabled

Runnable from scheduler with current configs:
- Proposal variants: `p0_imitation`, `p1_factor_loss`, `p2_typed_pressure`, `p3_chooser_fix1`, `p4_chooser_fix1_typed_pressure`, `p5_policy_choice`, `p6_policy_choice_with_chooser`.
- Rerankers: `m1_fix1_reranker`, `m2_global_fix_reranker`.
- Baselines: `--run-baselines` in `09_eval.py`.

Disabled / not implemented:
- `m3_policy_choice_reranker` in generator remains **disabled** (explicitly set `disabled: true`).

---

## 7) Scheduler Behavior (Paper Suite)

- `src/10_scheduler.py --paper-suite`:
  - Enables `--strict-global-metrics` during eval.
  - Automatically passes `--use-chooser` if `training_config.chooser.enabled`.
  - Automatically passes `--use-policy-choice` if `model_config.enable_policy_choice`.

---

## 8) Notes for New Collaborators

- Candidate generation is shared between proposal and reranker via `src/modules/candidates.py`.
- Policy filtering is deterministic and documented in `src/modules/policy.py`.
- If `chooser` or `policy` is enabled, datasets must be in‑memory lists (not streaming), because contexts and parquet rows must align by index.
- Global metrics require registry + encoder + factor fields; strict mode will fail fast if missing.

