# Paper-Facing Models and Evaluation Matrix

Date: 2026-03-11

This document defines the canonical paper-facing suite implemented in the repository. The default experiment surface should describe only these models.

## Canonical learned suite

| ID | Name | Representation | Training path | Objective | Inference role |
| --- | --- | --- | --- | --- | --- |
| `B0` | `B0 ESWC-Reproduction` | `eswc_passive` | `src/07_train.py` | `L_edit` | slot argmax |
| `A1` | `A1 Factorized Imitation` | `factorized` | `src/07_train.py` | `L_edit` | slot argmax |
| `M1C` | `M1C Safe Factor Chooser` | `factorized` | `src/07_train.py` | `L_edit + L_chooser` | chooser over symbolic candidates |
| `M1D` | `M1D Safe Factor Direct` | `factorized` | `src/07_train.py` | `L_edit + alpha L_primary + beta L_secondary` | candidate argmax from proposal logits |
| `G0` | `G0 GlobalFix Reference` | `factorized` proposal + reranker | `src/08_train_reranker.py` | `L_global` | reranker over symbolic candidates |

## Definitions

### B0 ESWC-Reproduction
- Uses `constraint_representation="eswc_passive"`.
- Disables local executable factor expansion, local factor-scope wiring, typed pressure, chooser loss, and direct safety loss.
- Exists to reproduce the prior-paper regime as a baseline, not as a factor-model ablation.

### A1 Factorized Imitation
- Uses `constraint_representation="factorized"`.
- Keeps the factorized graph, per-type factor executors, and per-role pressure, but trains only with edit imitation plus auxiliary factor supervision.
- Answers whether factorized local constraint context helps before any safety-aware decision objective.

### M1C Safe Factor Chooser
- Builds on `A1`.
- Uses chooser scoring over the same symbolic candidate set used by reranking.
- Retains the same per-type factor executor backbone and auxiliary factor supervision as `A1`.
- Default paper configs should keep a non-zero primary term (`gamma_primary > 0`) so primary-fix preference is explicit.

### M1D Safe Factor Direct
- Builds on `A1`.
- Reuses the same candidate builder and symbolic evaluator contract as `M1C` and `G0`.
- Retains the same per-type factor executor backbone and auxiliary factor supervision as `A1`.
- Computes candidate scores directly from proposal slot logits, then optimizes expected primary-failure and secondary-regression penalties.

### G0 GlobalFix Reference
- Uses the factorized proposal regime as candidate source.
- Trains a reranker for expected global satisfaction.
- It is a frontier reference, not the default practical system.

## Fixed paper defaults

The default paper-facing generator should enforce these settings for `A1`, `M1C`, and `M1D`:

- backbone: `GIN_PRESSURE`
- graph path: multi-relational / edge-attribute regime
- encoding: whichever dataset artifact is selected, typically frozen text embeddings when available
- role embeddings: enabled
- dynamic per-type reweighting: enabled
- fix-probability loss: disabled
- factor-loss-only training: disabled

`B0` is the exception: it keeps the passive representation and disables typed pressure.

## Default experiment bundle

The default config generator should emit only:

- `b0_eswc_reproduction`
- `a1_factorized_imitation`
- `m1c_safe_factor_chooser`
- `m1d_safe_factor_direct`
- `g0_globalfix_reference`

Appendix or exploratory variants such as policy-choice, factor-loss-only, untyped-pressure, and non-paper rerankers should be gated behind `--include-experimental`.

## Main evaluation suites

### Main results
- `DFB`
- `CSM`
- `B0`
- `A1`
- `M1C`
- `M1D`
- `G0`

### Minimal ablation view
- `B0 -> A1`: representation effect
- `A1 -> M1C`: chooser-based safe selection
- `A1 -> M1D`: direct-loss safe selection

### Required metrics
- historical fidelity: precision / recall / micro-F1
- primary fix rate
- global fix rate (`GFR`)
- secondary regression rate (`SRR`)
- secondary improvement rate (`SIR`)
- disruption / edit-minimality metrics

### Evaluation rule
- `M1C`, `M1D`, and `G0` must share the same candidate-level symbolic evaluator contract so reported safety metrics are definitionally aligned.
