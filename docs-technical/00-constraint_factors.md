# Executable Constraint Factors

This document states the conceptual research direction behind executable constraint factors for local KG repair. It is about the model idea and its intended scientific role, not about the current repository implementation.

Implementation details, current status, and configuration choices belong in `docs-technical/`.

## Research idea

Move from passive constraint context to executable local constraint factors:

- represent locally applicable constraints as first-class factor nodes
- let factor-aware message passing influence repair proposals
- evaluate predicted edits against the local constraint set
- optimize for safer local repair, not global closure at any cost

The repository now treats this as a two-track safe-factor family:

- fix the primary violation,
- avoid unnecessary secondary damage,
- preserve compatibility with historical curator behavior.

## 2. Core Hypothesis

The main hypothesis is that explicit executable constraint factors improve repair quality because they let the model reason about **constraint interaction** inside the local subgraph rather than only at evaluation time.

More specifically:

- a factorized repair model should reduce secondary regressions relative to a passive-context baseline,
- while preserving or nearly preserving historical fidelity and primary-fix performance,
- because secondary constraints are represented as active local pressures rather than ignored side conditions.

## 3. Conceptual Graph Upgrade

Each violation instance is represented as a heterogeneous local graph with two kinds of nodes.

### Variable nodes

These represent the local data state:

- entities
- predicates
- literals or values
- optionally role-specific local graph elements

### Constraint factor nodes

These represent locally applicable constraint instances.

Each factor:

- has a constraint type `t(c)`,
- is attached to a local scope of variables,
- behaves like a reusable executable module rather than a passive annotation.

Conceptually, this turns the per-violation subgraph into a local factor graph in which multiple constraints can act on the same variables at once.

## 4. Constraint Factors as Executable Semantics

Each constraint factor is meant to approximate the semantics of a constraint family in differentiable form.

At a high level, a factor should:

1. read the embeddings of the variables in its scope,
2. build a local factor state,
3. estimate whether the current local configuration is compatible with the constraint,
4. emit feedback to the variables involved in that configuration.

This does not require exact symbolic imitation. The conceptual goal is a learned approximation that is:

- type-specific,
- reusable across instances of the same constraint family,
- able to express local tension between competing repair options.

The detailed message-passing view of this idea is specified in [docs-conceptual/00-message_passing.md](/home/mvazquez/constraint_factors/docs-conceptual/00-message_passing.md).

## 5. Safe-Factor Model Family

The executable-factor idea gives rise to a family of safe local repair models rather than a single objective.

### `A1`: Factorized imitation

This is the representation-only step:

- factorized local constraint structure is present,
- but the model is still optimized mainly to imitate historical edits.

Its role is to test whether executable local structure helps even before explicit safety-aware decision logic is added.

### `M1C`: Chooser-based safe factor model

This variant keeps the factorized proposal model, then selects among explicit candidate repairs with a chooser objective.

Conceptually, it asks:

- can factorized local reasoning produce better candidate sets and better local choices,
- while keeping the final decision layer explicit and interpretable?

### `M1D`: Direct-loss safe factor model

This variant keeps the factorized proposal model but trains it with direct safety-aware penalties:

\[
\mathcal{L}_{M1D} = \mathcal{L}_{edit} + \alpha \mathcal{L}_{primary} + \beta \mathcal{L}_{secondary}
\]

Conceptually:

- `L_primary` enforces repair of the triggering violation,
- `L_secondary` discourages regressions on other locally applicable constraints,
- `L_edit` preserves fidelity to historical repair behavior.

`M1C` and `M1D` are therefore two decision-level realizations of the same broader executable-factor direction.

## 6. Desired Behavior

The conceptual target is not “satisfy every local constraint after every edit.”

The target is more conservative:

- repair the primary violation,
- avoid worsening other local constraints unless necessary,
- remain close enough to historical edits to stay curator-aligned.

This matters because collaborative KG repair is not theorem proving. Editors often make one locally sensible correction without globally normalizing the entire neighborhood.

## 7. Evaluation Logic

The conceptual model family should be judged on a trade-off surface, not on fidelity alone.

The key dimensions are:

- historical fidelity
- primary fix rate
- secondary regression rate
- secondary improvement rate
- global fix rate
- edit minimality / disruption

The central scientific question is whether executable constraint factors improve the safety side of this trade-off without collapsing fidelity.

## 8. Role of This Document

This file defines the **research framing** of executable constraint factors.

It should answer:

- why this direction exists,
- what the model family is trying to achieve,
- how `A1`, `M1C`, and `M1D` relate conceptually.

It should not be used as the source of truth for:

- exact code paths,
- current implementation boundaries,
- active config defaults,
- experiment scheduling.

Those belong in `docs-technical/`.
