# Residual Code vs Research Direction Audit

Date: 2026-03-11

This document tracks the remaining gaps after the phase-1 paper-alignment refactor and the per-type factor-executor upgrade.

## Landed in phase 1

- canonical paper-facing suite: `B0`, `A1`, `M1C`, `M1D`, `G0`
- explicit graph representation split:
  - `factorized`
  - `eswc_passive`
- default config generator limited to the canonical suite, with extras behind `--include-experimental`
- chooser and direct-safety paths separated as first-class alternatives
- shared candidate-level symbolic evaluator reused across chooser, direct-loss, reranker, and evaluation
- evaluation support for direct-safety inference
- per-type executable factor modules `f_t` for factorized proposal models
- per-type / per-role pressure modules `g_{t,r}` for factorized pressure updates
- direct post-edit neural factor prediction for the historical gold edit
- factor-loss support for both pre-state and gold-post factor supervision

## Remaining gaps

### 1. Experimental side paths still exist in the codebase

They are no longer on the default paper surface, but the code still supports:

- policy-choice
- factor-loss-only runs
- appendix rerankers and sweep utilities

That is fine as long as they remain clearly marked experimental.

### 2. Checker coverage remains bounded by implemented symbolic families

The shared evaluator is only as broad as the supported symbolic checker set. Unsupported constraint families still limit how much of the local constraint set is fully checkable.

## Deliberate scope boundary

The repository still does **not** implement:

- predicted-edit neural rollouts through the factor stack
- direct neural estimation of candidate-level primary / secondary outcomes for arbitrary proposed edits

For the current paper direction, these are treated as optional long-range extensions rather than priority gaps.

The intended paper contribution is the executable-factor, pressure-aware, neuro-symbolic repair stack with:

- per-type factor execution
- per-role pressure during message passing
- chooser and direct-loss safe-factor variants
- shared symbolic candidate evaluation for decision-time safety metrics

That scope is already coherent and publishable without replacing symbolic candidate evaluation with a fully neural candidate-level safety estimator.

## Priority order for future work

1. Expanded symbolic checker coverage
2. Cleanup or retirement of long-tail experimental paths once they stop being useful
3. Optional exploratory work on neural candidate-level post-edit modeling, only if a later paper needs it
