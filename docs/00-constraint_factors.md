# Executable Constraint Factors

Date: 2026-03-11

This document describes the factorized repair direction and, in phase 1 of the repository, the `M1D` direct-loss variant specifically.

## Research idea

Move from passive constraint context to executable local constraint factors:

- represent locally applicable constraints as first-class factor nodes
- let factor-aware message passing influence repair proposals
- evaluate predicted edits against the local constraint set
- optimize for safer local repair, not global closure at any cost

The repository now treats this as a two-track safe-factor family:

- `M1C`: chooser-based safe selection
- `M1D`: direct-loss safe selection

This file is about `M1D`.

## Graph regime

`M1D` uses `constraint_representation="factorized"`.

That means:

- local executable factor nodes are included in the graph
- factor metadata is attached to the graph object
- typed pressure can be enabled in message passing
- the symbolic evaluator can assess primary and secondary effects of candidate edits

`B0` is the non-factorized exception and uses `eswc_passive`.

## Prediction target

The proposal model still predicts the six-slot add/delete edit:

\[
\hat{\Delta} = (\hat{y}_{s+},\hat{y}_{p+},\hat{y}_{o+},\hat{y}_{s-},\hat{y}_{p-},\hat{y}_{o-})
\]

The training signal remains anchored in curator edits via `L_edit`.

## Candidate-based direct safety objective

`M1D` does not add a chooser head.

Instead it:

1. builds the same symbolic candidate set used by chooser and reranker paths
2. scores each candidate directly from proposal slot logits
3. converts those candidate scores into a soft distribution
4. evaluates the candidate set with the shared symbolic evaluator
5. computes expected safety penalties from that distribution

The phase-1 objective is:

\[
\mathcal{L}_{M1D} = \mathcal{L}_{edit} + \alpha \mathcal{L}_{primary} + \beta \mathcal{L}_{secondary}
\]

where:

- `L_primary` is expected primary-failure penalty over the candidate distribution
- `L_secondary` is expected secondary-regression rate over the candidate distribution

This keeps the direct-loss story real without requiring a second incompatible execution stack.

## Relation to M1C

`M1C` and `M1D` share:

- the same factorized graph regime
- the same typed-pressure-capable proposal backbone
- the same symbolic candidate builder
- the same symbolic evaluator contract

They differ only in the decision objective:

- `M1C`: `L_edit + L_chooser`
- `M1D`: `L_edit + alpha L_primary + beta L_secondary`

This resolves the earlier repo-level ambiguity where `M1` referred to two different training formulations.

## Current implementation boundary

The current implementation is intentionally modest:

- factor nodes and typed conditioning exist
- primary and secondary safety terms are candidate-level symbolic quantities
- safety losses are integrated into proposal training

What is not claimed here:

- true per-type executable neural factor programs `f_t`
- separate per-type or per-role message executors `g_{t,r}`
- direct post-edit neural factor supervision inside the proposal backbone

Those remain later-phase research directions and are documented in `docs/00-message_passing.md`.
