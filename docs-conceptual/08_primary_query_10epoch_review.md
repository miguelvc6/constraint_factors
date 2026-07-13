# Primary-Query Factor Experiments: 10-Epoch Review

> Historical diagnostic only. These runs predate schema-v2 identity, semantics,
> candidate, and metric fixes. Their numbers and conclusions require a clean
> rerun; see [09_paper_narrative_after_integrity_remediation.md](09_paper_narrative_after_integrity_remediation.md).

This note reviews the primary-query factor experiments.
The completed 10-epoch runs are:

- `models/a1_qfamily_secondary_factors_10ep__full_strat1m_minocc100__node_id/`
- `models/a1_qdef_secondary_factors_10ep__full_strat1m_minocc100__node_id/`
- `models/a1_primary_passive_secondary_factors_10ep__full_strat1m_minocc100__node_id/`

They have standard `src/09_eval.py` outputs: `model.json` and
`per_constraint.csv`. They do not currently have the oracle or H2 diagnostic
outputs that were available for the earlier two-epoch runs, so this review is
limited to historical fidelity, primary fix, local satisfaction, secondary
improvement/regression, disruption, and available non-vacuity fields.

## Aggregate Results

| Run | Mode | Primary Fix | Micro-F1 | Local Sat. | SIR | SRR | Disrupt. | Focus Del. | Non-Vac PF | Vac. Sat. Improve |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | canonical passive | 0.8603 | 0.5434 | 0.8705 | 0.1925 | 0.0097 | 1.0036 | n/a | n/a | n/a |
| `A1` | canonical executable factor | 0.8589 | 0.6785 | 0.8638 | 0.1397 | 0.0128 | 1.0330 | n/a | n/a | n/a |
| `QFamily-10ep` | `query_family` | 0.7481 | 0.3798 | 0.8663 | 0.1332 | 0.0090 | 0.9783 | 0.1966 | 0.0991 | 0.1844 |
| `QDef-10ep` | `query_definition` | 0.7557 | 0.3928 | 0.8679 | 0.1604 | 0.0087 | 0.9850 | 0.2204 | 0.1015 | 0.2083 |
| `PrimaryPassive-10ep` | `passive_node` | 0.7711 | 0.4071 | 0.8695 | 0.1899 | 0.0092 | 0.9951 | 0.2482 | 0.1037 | 0.2369 |


These models are not competitive with canonical
`A1` on the paper's main representation claim. The best primary-query
variant, `PrimaryPassive-10ep`, reaches 0.4071 Micro-F1, which remains 0.2714
below `A1` and 0.1363 below `B0`. The primary-query variants therefore still
fail as historical repair imitation models.

## Safety Trade-Off

The results no support the claim that primary-query construction broadly improves symbolic safety. The picture is
more modest:

- All three variants improve local satisfaction over `A1`, by 0.0025 to
  0.0057.
- All three reduce SRR relative to `A1`, by about 0.0036 to 0.0041.
- `QDef` and `PrimaryPassive` improve SIR over `A1`, while `QFamily` is lower.
- All three reduce disruption relative to `A1`.
- All three lose substantial primary fix relative to `A1`.

The closest safety-oriented variant is `PrimaryPassive-10ep`: it has the best
primary fix among the three, the best Micro-F1 among the three, the best local
satisfaction, and SIR close to `B0`. But it still falls well short of `A1` on
fidelity and primary fix, and it has the highest focus deletion and vacuous
satisfaction-improvement rates among the primary-query runs.

This is not a decisive safe-repair result. It is a clearer example of the
fidelity-safety trade-off: these graph regimes can improve some secondary and
local satisfaction metrics while losing the historical repair signal that makes
`A1` valuable.

## Per-Constraint Pattern

| Type | A1 F1 | QFamily F1 | QDef F1 | PrimaryPassive F1 | A1 Primary Fix | QFamily Primary Fix | QDef Primary Fix | PrimaryPassive Primary Fix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `conflictWith` | 0.7030 | 0.4646 | 0.4573 | 0.4586 | 0.8316 | 0.7886 | 0.7564 | 0.7370 |
| `single` | 0.5707 | 0.4386 | 0.4387 | 0.4743 | 0.8787 | 0.8395 | 0.8324 | 0.8440 |
| `type` | 0.4264 | 0.1876 | 0.2018 | 0.2466 | 0.6981 | 0.5715 | 0.5791 | 0.6021 |
| `distinct` | 0.9183 | 0.7116 | 0.6812 | 0.5600 | 0.9722 | 0.9619 | 0.9641 | 0.9615 |
| `valueType` | 0.4624 | 0.0818 | 0.0942 | 0.1460 | 0.7218 | 0.5291 | 0.5356 | 0.5617 |
| `itemRequiresStatement` | 0.4987 | 0.0642 | 0.1269 | 0.1716 | 0.8185 | 0.5672 | 0.6167 | 0.6378 |
| `inverse` | 0.8009 | 0.4342 | 0.4500 | 0.4730 | 0.9013 | 0.8336 | 0.8278 | 0.8237 |
| `valueRequiresStatement` | 0.7433 | 0.1788 | 0.2898 | 0.5181 | 0.9120 | 0.6277 | 0.6818 | 0.7839 |
| `oneOf` | 0.6936 | 0.4132 | 0.3598 | 0.1900 | 0.6848 | 0.3517 | 0.3123 | 0.1668 |

The per-family breakdown shows that the failure is broad. Every 10-epoch
primary-query variant is below `A1` on Micro-F1 for every listed constraint
family. The largest fidelity losses are in families that require more specific
structural edits, especially `valueType`, `itemRequiresStatement`,
`valueRequiresStatement`, and `inverse`.

`PrimaryPassive` is generally the strongest of the three for fidelity, but it
is not uniformly better. It is notably weaker on `distinct` and `oneOf`, while
better on `valueRequiresStatement`. This looks like a changed inductive bias,
not a uniformly better primary constraint representation.

## Implication for the Paper Narrative

The 10-epoch runs strengthen the existing paper narrative rather than replacing
it.

The main positive result remains canonical `A1`: executable factors improve
historical repair imitation over passive constraint context. The primary-query
variants do not preserve that result. Even after 10 epochs, they are closer to
weak repair-prior behavior than to the canonical factorized model.

The primary-query variants do provide useful diagnostic evidence. They show that
changing how the primary constraint is represented can shift the model toward
lower SRR, lower disruption, and slightly better local satisfaction. But those
gains come with a large loss in primary repair and historical fidelity. This is
another instance of the fidelity-safety gap, not a solution to it.

## Recommendation

Do not add these primary-query variants to the main paper table as candidate
models. If included, put them in an appendix as a negative or diagnostic
ablation:

1. `PrimaryPassive-10ep` is the best of the three and is the only one worth
   carrying forward.
2. The query-family and query-definition variants should not be treated as
   successful replacements for executable primary factors.
3. Before making any candidate-space claim, rerun the oracle and H2 diagnostics
   for the 10-epoch runs.
4. The paper should continue to present primary-query factorization as future
   design space, not as part of the core contribution.

The concise interpretation is: longer training narrows the gap but does not
rescue primary-query factorization. The canonical executable-factor `A1` remains
the representation result; the new runs are useful evidence that safety-oriented
metrics can improve while imitation and primary repair degrade.
