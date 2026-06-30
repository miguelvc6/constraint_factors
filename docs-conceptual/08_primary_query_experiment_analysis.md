# Primary-Query Factor Experiments: Uncommitted Analysis

This note analyzes the new primary-query factor experiments that are present in
the working tree but have not yet been committed to git. It is intentionally
diagnostic rather than canonical. The runs should not replace the current paper
table in [00-constraint_factors.md](/home/mvazquez/constraint_factors/docs-conceptual/00-constraint_factors.md)
without a controlled retraining pass.

## Uncommitted Experiment Set

The uncommitted model artifacts are:

- `models/a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id/`
- `models/a1_qdef_secondary_factors__full_strat1m_minocc100__node_id/`
- `models/a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id/`

They test variants of how the primary constraint is represented relative to
secondary factors:

| Run | Primary constraint mode | Epochs | Role in analysis |
| --- | --- | ---: | --- |
| `A1` | canonical factorized baseline | 10 | Current paper-facing representation result. |
| `QFamily` | `query_family` | 2 | Query-like primary constraint represented at family granularity. |
| `QDef` | `query_definition` | 2 | Query-like primary constraint represented at definition granularity. |
| `PrimaryPassive` | `passive_node` | 2 | Primary represented passively while secondary factors remain executable. |

The epoch mismatch is a major confound. The three new runs trained for two
epochs, while the canonical `A1` result trained for ten. The new results are
therefore useful for directional diagnosis, not for final model ranking.

## Aggregate Results

| Run | Primary Fix | Micro-F1 | Local Sat. | SIR | SRR | Disrupt. | Focus Del. | Non-Vac PF | Vac. Sat. Improve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A1` | 0.8589 | 0.6785 | 0.8638 | 0.1397 | 0.0128 | 1.0330 | n/a | n/a | n/a |
| `QFamily` | 0.6529 | 0.2352 | 0.8733 | 0.1867 | 0.0034 | 0.7547 | 0.2332 | 0.1071 | 0.2254 |
| `QDef` | 0.6609 | 0.2512 | 0.8743 | 0.1875 | 0.0039 | 0.8151 | 0.2426 | 0.1027 | 0.2326 |
| `PrimaryPassive` | 0.6398 | 0.2206 | 0.8659 | 0.1863 | 0.0085 | 0.8137 | 0.2293 | 0.1094 | 0.2210 |

The new runs sharply reduce historical edit fidelity. `QDef` is the strongest
of the three on Micro-F1, but its 0.2512 score is far below `A1` at 0.6785 and
even below the delete-focus/global-satisfaction diagnostic endpoint from the
current paper table. This means the query-style primary representation, at
least in these two-epoch runs, disrupts the learned imitation behavior that
motivates executable factors as a representation result.

The symbolic metrics move in the opposite direction. All three new runs improve
local satisfaction and SIR relative to `A1`, and all reduce SRR substantially:

- `QFamily`: local satisfaction +0.0095, SIR +0.0470, SRR -0.0094.
- `QDef`: local satisfaction +0.0105, SIR +0.0478, SRR -0.0088.
- `PrimaryPassive`: local satisfaction +0.0021, SIR +0.0466, SRR -0.0043.

This is exactly the kind of trade-off the paper narrative needs to foreground:
the system can be pushed toward more favorable local symbolic metrics, but the
cost to historical repair fidelity is severe.

## Candidate-Oracle Pattern

| Run | Safe Avail. | Selected Safe | Non-Vac Safe Avail. | Selected Non-Vac Safe | Oracle PF | Selected PF | Oracle Local Sat. | Selected Local Sat. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A1` | 0.2992 | 0.2468 | 0.0961 | 0.0596 | 0.3101 | 0.2730 | 0.9036 | 0.8638 |
| `QFamily` | 0.9331 | 0.7787 | 0.1753 | 0.1073 | 0.9878 | 0.7944 | 0.9099 | 0.8733 |
| `QDef` | 0.9331 | 0.7845 | 0.1754 | 0.0913 | 0.9878 | 0.8062 | 0.9099 | 0.8742 |
| `PrimaryPassive` | 0.9331 | 0.7281 | 0.1755 | 0.1082 | 0.9878 | 0.7662 | 0.9099 | 0.8658 |

This is the most important new evidence. The new primary-query graph regime
appears to expose a much richer safety-oriented candidate space: safe candidate
availability jumps from about 30% for `A1` to about 93% for all three new runs,
and oracle primary-fix availability jumps from 31% to almost 99%.

That does not mean the models have solved safe repair. Non-vacuous safe
availability remains much lower, around 17.5%, and selected non-vacuous safe
rates remain near 9-11%. The gap between "safe candidate exists" and
"non-vacuous safe candidate exists" supports the current paper's claim that
local satisfaction must be separated from useful repair.

The selected edits also still show evidence-removal pressure. The new runs have
focus deletion rates around 23-24% and vacuous satisfaction-improvement rates
around 22-23%. This is lower than the full delete-focus degeneracy but high
enough that the non-vacuity issue remains central.

## Factor Semantics

| Run | Mean Factor F1 | Mean AUROC | Mean AUPRC | Mean ECE |
| --- | ---: | ---: | ---: | ---: |
| `A1` | 0.9809 | 0.8975 | 0.9893 | 0.1038 |
| `QFamily` | 0.8948 | 0.6824 | 0.9738 | 0.1584 |
| `QDef` | 0.8903 | 0.6807 | 0.9755 | 0.1647 |
| `PrimaryPassive` | 0.8645 | 0.7102 | 0.9733 | 0.1400 |

The factor heads are much less semantically sharp in the new runs. Mean factor
F1 remains superficially high, but AUROC drops from 0.8975 in `A1` to roughly
0.68-0.71. Several families fall below 0.8 AUROC, including `single`,
`itemRequiresStatement`, `valueRequiresStatement`, and `symmetric`; many of
these are highly imbalanced, so the high AUPRC should not be read as robust
discrimination.

This weakens any interpretation that the new primary-query variants are better
constraint reasoners. Their symbolic gains appear more consistent with changed
candidate/edit behavior and lower disruption than with stronger learned factor
semantics.

## Per-Family Pattern

The aggregate symbolic gains are not uniform. The largest SIR contribution is
still concentrated in `distinct`, where the new runs reach SIR around 0.76.
For many other families, SIR remains near zero. Primary fix also collapses for
families that require adding or selecting specific supporting statements:

- `itemRequiresStatement` F1 is near zero in all three new runs.
- `valueRequiresStatement` F1 is near zero in `QFamily` and `QDef`.
- `inverse`, `valueType`, and `type` have very low Micro-F1 in the query runs.

This means the new setup is not learning a generally better repair policy. It
is shifting the edit distribution toward lower-disruption and often deletion-
compatible choices that improve some local aggregate metrics while losing the
fine-grained historical edit structure.

## Interpretation for the Paper Narrative

These uncommitted experiments strengthen the fidelity-safety gap narrative, but
they should not be presented as a new positive model result yet.

The positive reading is that primary-query graph construction may expose a
candidate space where symbolic safety is much more available. That is valuable:
it suggests the earlier candidate bottleneck was partly representational, not
only an objective-selection problem.

The negative reading is equally important. The current primary-query variants
damage historical imitation, weaken factor semantics, and still leave most
non-vacuous safe repair unrealized. The symbolic gains are therefore not enough
to overturn the paper's current thesis. They reinforce the claim that repair
quality is multi-axis: local satisfaction, primary fix, non-vacuity, historical
fidelity, and secondary no-regression can move independently.

## Recommended Paper Position

Do not move these runs into the main result table in their current form. Treat
them as an appendix diagnostic or future-work bridge:

1. Primary-query construction is promising for candidate coverage and symbolic
   availability.
2. It is not yet a viable replacement for `A1` because it sacrifices the main
   representation result: high historical repair fidelity.
3. The next controlled experiment should train the query variants for the same
   number of epochs as `A1` and report both ordinary metrics and non-vacuity
   metrics.
4. Any future safety objective built on this regime must explicitly distinguish
   non-vacuous repair from local satisfaction by deletion.

The clean narrative is: executable factors improve imitation in the canonical
setup; query-style primary factorization may improve safety opportunity; neither
alone solves non-vacuous safe repair.
