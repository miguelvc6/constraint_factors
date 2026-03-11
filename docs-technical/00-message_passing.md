# Message Passing and Constraint Execution

Date: 2026-03-11

This document separates the current repository implementation from the longer-range research ideal.

## Current v1 implementation

The repository currently implements a factor-aware message-passing model with these properties:

- factor nodes are present in `factorized` graphs
- factor nodes carry type ids and primary-factor indexing
- factor scope is extracted explicitly from factor-to-variable role edges
- each factor type dispatches to its own executable factor module `f_t`
- each factor type and role dispatches to its own message module `g_{t,r}`
- the proposal model predicts both pre-state factor satisfaction and gold-post factor satisfaction

This is best described as:

- per-type executable neural factor modules over local factor scopes
- per-type / per-role pressure messages over local factor scopes
- shared symbolic evaluation over candidate edits for chooser, direct-safety, reranking, and reported metrics

In notation, the implemented factorized path now follows the intended local-scope structure:

\[
z_c = f_{t(c)}(\mathrm{AGG}_{pred}(c), \mathrm{AGG}_{subj}(c), \mathrm{AGG}_{obj}(c), h_c)
\]

\[
m_{c \to v} = g_{t(c), r(v,c)}(z_c, h_v)
\]

The repository still does not perform learned graph rewriting or learned candidate rollouts for arbitrary predicted edits.

## Current implementation contract

At the code level, phase 1 assumes:

- `B0` uses `constraint_representation="eswc_passive"`
- `A1`, `M1C`, `M1D`, and proposal sources for `G0` use `constraint_representation="factorized"`
- `M1C`, `M1D`, and `G0` all consume the same candidate-level symbolic evaluator outputs:
  - primary satisfied flag
  - secondary regression rate
  - global satisfied fraction
  - disruption fields used by evaluation

That shared evaluator contract is the mechanism that keeps safety metrics aligned across chooser, direct-loss, reranker, and final evaluation.

## Current boundary

The current implementation already includes:

- one executable factor module per constraint type, `f_t`
- per-type / per-role message functions, `g_{t,r}`
- explicit scope aggregation and message emission specialized by factor role
- direct post-edit neural factor prediction for the historical gold edit used in training

What remains deferred is the stronger end-to-end neural version:

- learned post-edit factor prediction for arbitrary predicted candidates rather than only the gold edit
- learned candidate-level safety estimation replacing symbolic candidate evaluation
- internal neural edit rollout or graph rewriting rather than symbolic candidate checking

## Repository wording rule

When describing the current codebase:

- say "per-type executable factor modules with per-role pressure messages"
- say that post-edit neural factor prediction is available for the gold edit as auxiliary supervision
- say that candidate-level safety evaluation remains symbolic

When describing future work:

- refer to predicted-edit neural rollouts and neural candidate-level safety estimation as the deferred extension

This distinction keeps the docs honest and prevents overclaiming relative to the implementation.
