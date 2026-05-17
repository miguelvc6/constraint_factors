# Results-Aligned Narrative for Executable Constraint Factors

This document reframes the executable-constraint-factor paper narrative around the trained and evaluated paper-suite results. It complements the original hypothesis document in [00-constraint_factors.md](/home/mvazquez/constraint_factors/docs-conceptual/00-constraint_factors.md), which records the initial research framing. Implementation details, run directories, and scheduler behavior belong in [docs-technical/](/home/mvazquez/constraint_factors/docs-technical/).

## 1) Revised Paper Thesis

### Core finding

Executable constraint factors substantially improve historical repair imitation, but better imitation does not automatically produce safer post-edit graphs.

The current results show a clear separation between two goals:

- **fidelity to historical curator repairs**, measured by precision, recall, and micro-F1;
- **symbolic repair safety**, measured by primary fix rate, global fix rate (`GFR`), secondary regression rate (`SRR`), secondary improvement rate (`SIR`), and edit disruption.

The factorized representation improves the first goal. It does not, by itself, improve the second.

### Revised thesis statement

Historical edit fidelity and symbolic constraint safety are partially misaligned in Wikidata repair. Executable constraint factors help neural models imitate curator edits more accurately, but representation-level constraint awareness is insufficient to guarantee lower secondary regressions or higher global constraint satisfaction. Explicit candidate-level safety selection or reranking is needed to move along the fidelity-safety frontier.

### Scientific contribution

1. **Representation result:** factorized executable-constraint context improves historical repair prediction over a passive ESWC-style baseline.
2. **Evaluation result:** higher historical fidelity does not imply safer post-edit constraint state.
3. **Diagnostic result:** factor heads learn useful satisfaction signals and factor pressure causally affects predictions, but the pressure mostly supports imitation and primary repair behavior rather than secondary no-regression.
4. **Trade-off result:** a global-fix reranker can improve global satisfaction, but it sacrifices historical fidelity and primary-fix behavior.

The paper should therefore be presented as an empirical study of the fidelity-safety trade-off, not as a claim that the current practical safe-factor objectives dominate the baseline on all safety metrics.

---

## 2) Model Roles Under the Revised Narrative

### `B0`: Passive ESWC-style baseline

`B0` is the prior-work-style learned baseline. It uses passive constraint context and slot-argmax repair prediction. In the current results it is weaker at historical imitation but unexpectedly strong on global safety metrics.

Interpretation:

- `B0` remains the correct baseline for historical repair modeling.
- Its safety strength suggests that simple or conservative edit behavior can produce low measured secondary regression.
- This makes `B0` a stronger and more interesting comparator than originally expected.

### `A1`: Factorized imitation model

`A1` isolates the representation effect. It uses factorized executable-constraint context and auxiliary factor supervision, but no candidate-level safety objective.

Current result:

- primary fix is similar to `B0`;
- micro-F1 is much better than `B0`;
- `GFR`, `SRR`, and `SIR` are worse than `B0`.

Interpretation:

- `A1` validates that executable factors improve curator-edit imitation.
- `A1` refutes the stronger claim that factorized representation alone improves repair safety.
- The model learns to use constraint context in the service of predicting what curators did, not necessarily in the service of preserving all locally applicable constraints.

### `M1C`: Safe factor chooser

`M1C` is intended to test whether a symbolic candidate chooser can improve the safety side of the trade-off while retaining the factorized proposal model.

Current result:

- the completed follow-up run uses the intended non-zero primary term (`gamma_primary = 0.2`);
- micro-F1 is lower than `A1` and the earlier `gamma_primary = 0.0` chooser run;
- `SRR` is slightly lower than the earlier chooser run, but still worse than `B0` and `M1D`;
- primary fix, `GFR`, `SIR`, and disruption do not improve over `B0`.

Interpretation:

- The current `M1C` result should not be framed as the main positive safety result.
- The non-zero-primary follow-up resolves the configuration mismatch but does not change the conclusion: chooser supervision, as currently configured, does not produce a clear no-regression or global-safety win.
- The small `SRR` improvement over the earlier chooser run comes with lower fidelity and lower primary-fix behavior, so it should be treated as a trade-off result rather than a recovered safety result.

### `M1D`: Direct-loss safe factor model

`M1D` tests a direct candidate-level safety objective over proposal logits.

Current result:

- micro-F1 is close to `A1`;
- primary fix is close to `B0` and `A1`;
- `SRR` is better than `A1` and `M1C`, but not enough to clearly beat `B0`;
- dense-context H2 slices suggest `M1D` can move safety in the intended direction in some regimes.

Interpretation:

- `M1D` is the strongest practical candidate for a safety-aware factorized model.
- Its current evidence supports a nuanced claim: direct safety pressure can improve some safety slices, but the aggregate result is not yet a decisive baseline win.

### `G0`: Global-fix reranker reference

`G0` is the global-consistency reference. It uses factorized proposals and reranks symbolic candidates for global satisfaction.

Current result:

- highest `GFR`;
- highest `SIR`;
- very low historical fidelity;
- weaker primary-fix behavior than expected.

Interpretation:

- `G0` cleanly demonstrates the high-consistency / low-fidelity end of the frontier.
- It shows that better global satisfaction is reachable with the candidate/evaluator setup.
- It also shows that optimizing global satisfaction directly can diverge sharply from curator-like repairs.

---

## 3) Result Pattern and Paper Claims

### Main empirical pattern

The paper-facing result table currently supports the following claim:

> Factorized executable constraints are valuable for representation learning and repair imitation, but safe local repair remains a separate decision problem.

It does not support the stronger original claim:

> Factorized executable constraints, with the current practical objectives, reduce secondary regressions while preserving fidelity.

### Defensible claims

The paper can claim:

- `A1` demonstrates that executable constraint context improves historical repair prediction.
- Improved historical repair prediction does not imply improved symbolic graph safety.
- `M1C` and `M1D` expose the difficulty of turning local factor signals into safe candidate choices.
- `G0` shows that global consistency can be improved, but with a large fidelity cost.
- H2 diagnostics show that factor pressure is causally important, but mostly for imitation and primary repair behavior.

### Claims to avoid unless new experiments change the table

Avoid claiming:

- the current `M1C` is the main successful safety model;
- `A1` improves no-regression behavior over `B0`;
- factor pressure alone reduces secondary regressions;
- `G0` is an upper bound that preserves primary-fix quality.

---

## 4) H2 Diagnostic Interpretation

H2 evaluations explain why the main result diverges from the original expectation.

### Factor semantics are learned

The factor heads achieve high weighted F1 on pre-repair and post-gold factor satisfaction. This means the factor modules are not failing simply because they cannot approximate symbolic satisfaction.

However, several factor families are highly imbalanced. For families where almost all factors are satisfied, F1 can be high even when AUROC is weak. This matters for interpreting semantic success: the factor heads are useful, but not uniformly discriminative across rare violation states.

### Factor pressure is causally used

Removing factor pressure causes large drops in micro-F1 and primary fix rate for `A1`, `M1C`, and `M1D`. This shows that executable factors affect the proposal model's decisions.

The key finding is that this effect is mainly on fidelity and primary repair behavior, not on secondary safety.

### Secondary pressure has weak influence

The `secondary_only_pressure` masking variant changes very few predictions and produces negligible metric changes. This suggests that the architecture currently routes most useful pressure through primary/focal repair signals rather than secondary no-regression signals.

This is the central explanation for the surprising result:

> the model learns and uses factor pressure, but the learned pressure is not aligned strongly enough with secondary constraint preservation.

### Density slices refine the story

Most test instances are in dense local contexts. In those slices, `M1D` shows the clearest movement toward better `GFR`/`SRR`, while `M1C` tends to preserve stronger fidelity. This suggests the final paper should include density-sliced analysis rather than relying only on aggregate metrics.

---

## 5) Revised Research Questions

The paper can be organized around these questions:

### RQ1: Do executable constraint factors improve historical repair modeling?

Expected answer from current results: yes.

Evidence:

- `A1` improves micro-F1 over `B0`.
- H2 shows factor heads learn satisfaction signals.
- Pressure-masking ablations show factor pressure causally affects predictions.

### RQ2: Does better repair imitation imply safer post-edit graphs?

Expected answer from current results: no.

Evidence:

- `A1` improves micro-F1 but worsens aggregate `GFR`, `SRR`, and `SIR` versus `B0`.
- This demonstrates a fidelity-safety mismatch.

### RQ3: Can local safety objectives recover the safety side of the frontier?

Expected answer from current results: partially, but not conclusively.

Evidence:

- `M1D` improves over `A1`/`M1C` on some safety metrics and dense-context slices.
- `M1C` improves fidelity but does not produce a clear safety win.
- More targeted safety-objective experiments are needed before claiming a practical local objective succeeds.

### RQ4: What happens when global symbolic consistency is optimized directly?

Expected answer from current results: consistency improves, fidelity collapses.

Evidence:

- `G0` obtains the best `GFR` and `SIR`.
- `G0` has much lower micro-F1 and weaker primary-fix behavior.

---

## 6) Recommended Paper Structure

### Introduction

Frame the problem as repair under two competing desiderata:

- match historical curator behavior;
- avoid worsening the local symbolic constraint state.

Motivate executable constraint factors as a way to make constraint state available to neural repair models, but make clear that the paper studies whether this representation actually translates into safer repairs.

### Method

Present:

- `B0` as passive historical imitation;
- `A1` as factorized historical imitation;
- `M1C` and `M1D` as two attempts to add local safety-aware decision pressure;
- `G0` as global symbolic reranking reference.

### Evaluation

Separate the metrics into:

- historical fidelity;
- primary repair success;
- global/local symbolic safety;
- disruption/edit minimality;
- H2 diagnostics.

### Results

Lead with the main trade-off:

- factorized representation improves fidelity;
- aggregate safety does not automatically improve;
- global reranking improves consistency at a fidelity cost.

Then use H2 to explain mechanism:

- factor semantics are learned;
- pressure matters;
- secondary pressure is weak;
- dense contexts are where safety-aware variants are most informative.

### Discussion

Emphasize that this is not a failure of the factor representation. It is evidence that repair imitation and symbolic no-regression are different optimization targets.

The discussion should position safe candidate selection as the main open challenge.

---

## 7) Next Experiments

The next experiments should be targeted rather than broad. The goal is to determine whether the safety side of the current trade-off narrative can be strengthened, while prioritizing mechanism-focused ablations over expensive broad sweeps.

### E1: Completed updated `M1C` chooser run

The updated `M1C` chooser run completed with `gamma_primary = 0.2`. It should be used as the final non-zero-primary chooser result, while the earlier `gamma_primary = 0.0` run remains useful as a diagnostic comparison.

Outcome:

- it lowers `SRR` slightly relative to the earlier chooser run;
- it does not improve `GFR` over `B0`, `A1`, or `M1D`;
- it reduces fidelity and primary-fix behavior relative to the earlier chooser run and `A1`;
- it therefore does not justify reframing `M1C` as the main successful safety model.

Why this matters:

- `M1C` is the practical chooser-based path for local safe selection;
- the completed non-zero-primary follow-up shows that stronger chooser-side primary pressure does not, by itself, recover the safety side of the trade-off;
- the remaining question is whether candidate-oracle analysis shows a selection-learning bottleneck or a candidate-generation bottleneck.

### E2: Spend compute on H2 supporting ablations

The initial compute priority should be ablations rather than a broad `M1D` safety-weight grid. The current narrative depends on explaining why executable factors help fidelity but do not automatically improve safety.

Train the planned H2 ablations:

- no auxiliary factor loss;
- shared pressure modules;
- legacy shared executor.

Compare against `A1` on:

- factor semantic AUROC/AUPRC, not only F1;
- pressure-masking deltas;
- primary vs secondary pressure influence;
- density slices.

Decision rule:

- if no-factor-loss keeps fidelity but weakens H2 semantic metrics, auxiliary factor supervision is justified;
- if shared pressure behaves similarly to per-type pressure, the type-specific executor claim should be softened;
- if secondary pressure remains weak across all ablations, the paper should identify secondary routing as the main architectural limitation.

Why this matters:

- ablations are the most compute-efficient way to support the revised mechanism story;
- they clarify whether the observed behavior is due to factor supervision, typed pressure, or the executor design.

### E3: Candidate oracle analysis

For each evaluated instance, determine whether the candidate set contains at least one candidate that:

- fixes the primary constraint;
- does not regress secondary constraints;
- improves or preserves `GFR`;
- has acceptable disruption.

Then compare:

- oracle candidate performance;
- model-selected candidate performance;
- `G0` reranker performance.

Decision rule:

- if the oracle is strong but `M1C`/`M1D` are weak, the problem is selection learning;
- if the oracle is weak, the problem is candidate generation.

Why this matters:

- it identifies whether future work should focus on better candidate generation or better scoring/reranking.

### E4: Density- and family-specific reporting

Report aggregate metrics and sliced metrics by:

- local factor-density bucket;
- primary constraint family;
- high-risk families with weak aggregate safety;
- secondary exposure bucket.

Decision rule:

- if safety wins exist only in dense or family-specific regimes, frame the paper around those regimes rather than broad aggregate dominance.

Why this matters:

- H2 indicates dense contexts carry most examples and reveal differences among `A1`, `M1C`, and `M1D`;
- aggregate metrics may hide the regime where executable factors are most relevant.

### E5: Optional `M1D` safety-weight grid

A larger `M1D` grid over secondary and primary weights may be useful later, but it is not the initial compute priority because full runs are expensive.

Possible grid, if compute becomes available:

- `beta_secondary = 0.75, 1.0, 2.0`
- `alpha_primary = 1.0, 1.5, 2.0`

Decision rule:

- primary fix must remain close to `B0`;
- `SRR` should improve over `A1` and ideally over `B0`;
- disruption must not grow enough to turn the model into an over-editing heuristic.

### E6: Calibrated factor semantics

Evaluate factor semantic calibration more directly for imbalanced families.

Focus on:

- `valueRequiresStatement`;
- `oneOf`;
- `single`;
- `itemRequiresStatement`;
- `symmetric`.

Decision rule:

- if F1 is high but AUROC/calibration are weak, report that factor heads learn the dominant satisfaction prior but not always robust violation discrimination.

Why this matters:

- it prevents overclaiming from high factor-semantic F1.

---

## 8) Revised Success Criteria

The original success criterion was broad aggregate safety improvement over `B0`.

The revised criteria are:

1. `A1` should improve historical fidelity over `B0`.
2. H2 should show that factor semantics and pressure are actually learned.
3. `M1C`/`M1D` should be evaluated as attempts to move along the safety frontier, not assumed winners.
4. `G0` should define the high-consistency / low-fidelity reference.
5. Any recovered safety claim must be tied to aggregate wins, density-slice wins, or family-specific wins with clear scope.

This keeps the paper empirically honest while preserving a meaningful contribution.
