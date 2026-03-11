# Residual Code vs Research Direction Audit

Date: 2026-03-11

This document tracks the remaining gaps after the phase-1 paper-alignment refactor.

## Landed in phase 1

- canonical paper-facing suite: `B0`, `A1`, `M1C`, `M1D`, `G0`
- explicit graph representation split:
  - `factorized`
  - `eswc_passive`
- default config generator limited to the canonical suite, with extras behind `--include-experimental`
- chooser and direct-safety paths separated as first-class alternatives
- shared candidate-level symbolic evaluator reused across chooser, direct-loss, reranker, and evaluation
- evaluation support for direct-safety inference

## Remaining gaps

### 1. Factor execution is still generic-plus-typed, not per-type neural programs

Current code:

- uses factor nodes
- uses typed conditioning
- uses generic factor scoring / pressure mechanisms

Still missing:

- distinct executable factor modules `f_t`
- distinct message functions `g_{t,r}`

This is the main remaining conceptual gap relative to the strongest research-ideal formulation.

### 2. Proposal training still relies on symbolic candidate evaluation for safety terms

`M1D` now implements the direct-loss story, but the safety quantities are still candidate-level symbolic measurements rather than direct post-edit neural factor predictions inside the proposal model.

That is acceptable for phase 1, but it should be described honestly.

### 3. Experimental side paths still exist in the codebase

They are no longer on the default paper surface, but the code still supports:

- policy-choice
- factor-loss-only runs
- appendix rerankers and sweep utilities

That is fine as long as they remain clearly marked experimental.

### 4. Checker coverage remains bounded by implemented symbolic families

The shared evaluator is only as broad as the supported symbolic checker set. Unsupported constraint families still limit how much of the local constraint set is fully checkable.

## Priority order for future work

1. Per-type executable factor modules `f_t`
2. Per-type / per-role message modules `g_{t,r}`
3. Direct post-edit neural factor prediction inside the proposal model
4. Expanded symbolic checker coverage
