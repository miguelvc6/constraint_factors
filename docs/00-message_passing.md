# Message Passing and Constraint Execution

Date: 2026-03-11

This document separates the current repository implementation from the longer-range research ideal.

## Current v1 implementation

The repository currently implements a factor-aware message-passing model with these properties:

- factor nodes are present in `factorized` graphs
- factor nodes carry type ids and primary-factor indexing
- message passing can inject typed-conditioned pressure from factor nodes
- factor scoring is shared across factor instances, with optional type conditioning

This is best described as:

- a generic factor-message mechanism
- plus typed conditioning
- plus shared symbolic evaluation over candidate edits

It is not yet a full library of per-type executable neural subprograms.

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

## Deferred research-ideal design

The stronger research formulation remains deferred:

- one executable factor module per constraint type, `f_t`
- optional per-type or per-role message functions, `g_{t,r}`
- explicit scope aggregation and message emission specialized by factor family
- potentially direct post-edit neural factor prediction inside the proposal model

In notation, that deferred version would look like:

\[
z_c = \mathrm{AGG}(\{h_v : v \in S(c)\})
\]

\[
\hat{s}_c = f_{t(c)}(z_c)
\]

\[
m_{c \to v} = g_{t(c), r(v,c)}(z_c, h_v)
\]

That is still a valid future direction, but it is not what the repository should currently claim to implement.

## Repository wording rule

When describing the current codebase:

- say "typed-conditioned generic factor pressure/messages"
- do not say "true per-type executable factor programs" unless that refactor has actually landed

When describing future work:

- refer to the per-type `f_t` / `g_{t,r}` architecture as a deferred research-ideal extension

This distinction keeps the docs honest and prevents overclaiming relative to the implementation.
