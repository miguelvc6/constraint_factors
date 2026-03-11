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

### 1. Candidate-level safety terms are still symbolic at decision time

Current code:

- uses per-type neural factor executors for pre-state factor scoring
- uses per-type / per-role neural pressure messages during factorized message passing
- predicts post-edit factor satisfaction for the historical gold edit as an auxiliary proposal loss
- still evaluates candidate edits with the shared symbolic checker/evaluator stack for `M1C`, `M1D`, `G0`, and final metrics

Still missing:

- predicted-edit neural rollouts through the factor stack
- direct neural estimation of candidate-level primary / secondary outcomes for arbitrary proposed edits

This is now the main remaining conceptual gap relative to the strongest end-to-end neural formulation.

### 2. Experimental side paths still exist in the codebase

They are no longer on the default paper surface, but the code still supports:

- policy-choice
- factor-loss-only runs
- appendix rerankers and sweep utilities

That is fine as long as they remain clearly marked experimental.

### 3. Checker coverage remains bounded by implemented symbolic families

The shared evaluator is only as broad as the supported symbolic checker set. Unsupported constraint families still limit how much of the local constraint set is fully checkable.

## Priority order for future work

1. Neural candidate-level post-edit factor prediction beyond the gold edit
2. Predicted-edit neural rollouts or neural candidate-level safety estimation
3. Expanded symbolic checker coverage
4. Cleanup or retirement of long-tail experimental paths once they stop being useful
