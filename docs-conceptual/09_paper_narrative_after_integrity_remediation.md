# Paper Narrative After Integrity Remediation

Date: 2026-07-13

This document records how the audit changes the research claims and paper
structure. Implementation details and exact rerun commands are in the
[technical remediation report](../docs-technical/09_scientific_integrity_remediation.md).

## Executive Decision

The paper should enter a result-reset state. The hypotheses and study design
remain usable, but all numerical conclusions based on pre-schema-v2 artifacts
must be withdrawn until the clean rerun is complete.

The architecture previously tagged `h2_a1_shared_pressure` becomes canonical
`A1`. This is primarily a parsimony decision: its role-pressure component has
1,442,400 parameters instead of 41,829,600, a 96.6% subsystem reduction, while
the historical run showed no performance penalty and slightly better metrics.
The performance observation is selection evidence, not an unbiased final test
result.

## Claim Status

| Previous statement | Status now | Paper treatment |
| --- | --- | --- |
| A1 improves fidelity over B0. | Unresolved pending rerun. | Keep as a hypothesis; replace all numbers after schema-v2 evaluation. |
| Factorized representation alone does not improve symbolic safety. | Unresolved pending rerun. | Reassess using transition-based primary fix and pooled SRR/SIR. |
| M1C/M1D improve particular safety trade-offs. | Unresolved pending rerun. | Retrain with architecture-matched shared pressure and target-free inference. |
| G0 is a global-fix frontier/reference. | Unresolved pending rerun. | Regenerate its candidates/predictions and rerun deletion-degeneracy analysis. |
| Frequency threshold 100 defines the task vocabulary. | Retracted. | Threshold 100 is a model-capacity decision; semantic identities are never filtered. |
| Historical Micro-F1 is exact repair fidelity. | Retracted. | Headline fidelity is strict identity triple F1; feature-space F1 is diagnostic. |
| Heuristic alternative-action match is primary fix. | Retracted. | Primary fix is an observed violated-to-satisfied constraint transition. |
| Every observed correction is a verified primary repair. | Retracted. | Eligibility is defined from PRE. Report separately whether POST_GOLD is checkable and satisfies the executable primary constraint. |
| Per-sample mean SRR/SIR is the aggregate safety rate. | Retracted. | Use pooled eligible-event ratios; show defined-sample macros only as diagnostics. |
| Shared-pressure A1 is 96.6% smaller. | Narrow and retain. | Say the **role-pressure subsystem** is 96.6% smaller, not the whole model. Report total counts from the new run manifest. |

## Revised Research Story

### Dataset and task definition

The repair task is defined in semantic identity space. Raw upstream train,
development, and test boundaries are preserved. A deterministic exact-size
benchmark is sampled only after rows have passed a primary-violation validity
check. The paper should report:

- raw and retained rows by split;
- exclusion counts by `primary_validation_reason`;
- the exact sampling target and realized split allocation;
- identity and feature vocabulary sizes;
- target representability coverage.
- the per-family rate at which the observed gold edit is a verified primary repair.

This replaces language implying that frequency pruning changes which entities
or literals count as the same answer.

### Frequency filtering

Frequency filtering remains a legitimate modeling decision: it controls input
and output capacity, computational cost, and rare-feature generalization. It no
longer changes graph identity or ground truth. The main run may retain
`min_occurrence=100`, but B0 and A1 must also be evaluated at 1 and 10 with all
other decisions fixed.

Interpretation rule:

- stable A1-vs-B0 conclusions support threshold 100 as a computational choice;
- materially changing conclusions make filtering a moderator that belongs in
  the main results/discussion;
- a large gap between feature-space and strict identity F1 indicates that prior
  apparent performance came partly from identity collapse.

### Constraint semantics and eligible population

The paper should describe one shared executable semantics layer used for label
construction, candidate assessment, and global evaluation. It should state that
parameter predicates are interpreted by constraint family, exceptions and
main-value scope are honored, and insufficient evidence yields “uncheckable”
rather than a guessed truth value.

For `type` and `valueType`, relation-mode items select `P31`, `P279`, or both,
and class membership is tested through `P279*`. The available corpus does not
contain a complete historical Wikidata taxonomy, so the executable benchmark
freezes the union of direct `P279` facts found in training-split entity
descriptions and uses its reflexive-transitive closure for every split.
Validation and test context cannot add hierarchy edges. This is a reproducible,
training-only operationalization, not complete-Wikidata entailment; hierarchy
edge count, hash, and coverage belong in Methods and Limitations.

The estimand changes from “all correction rows” to “rows whose primary
constraint is supported, applicable, checkable, and violated in the pre-state.”
Exclusions are part of the benchmark definition and must be reported, not hidden
as preprocessing loss.

The upstream corpus supplies observed correction events, not a guarantee that
each event is a complete repair under this executable semantics. Retain eligible
PRE violations for the correction-imitation task, but stratify the observed
edits as verified, post-uncheckable, or post-unsatisfied. Do not describe the
one-million-row sample as one million successful repairs.

The completed Section 1 audit makes this distinction material: 73.45% of the
sampled historical edits verify as complete primary repairs, while 26.55% leave
the executable primary constraint unsatisfied. The paper must therefore frame
the supervised target as an **observed curator correction**, and reserve
“repair success” for the transition-based semantic metric. Family-level counts
and rerun provenance are recorded in the
[technical remediation report](../docs-technical/09_scientific_integrity_remediation.md#section-1-rerun-record-2026-07-13).

### Canonical A1

Canonical A1 now uses:

- factorized local constraint graphs;
- per-type factor executors;
- shared role-pressure blocks;
- edit imitation plus auxiliary factor supervision;
- no candidate-level safety objective.

This preserves A1's conceptual role as the representation-only comparison while
removing a large parameterization that did not improve the historical run.
M1C, M1D, and G0's proposal must use the same shared-pressure base so their
differences isolate decision objectives.

The architecture was selected after observing the historical H2 result. To
avoid post-selection overclaiming:

1. justify promotion primarily by parsimony and architecture matching;
2. do not quote the old test improvement as confirmatory evidence;
3. lock the architecture before the refreshed test evaluation;
4. report the historical comparison as exploratory model selection;
5. use the refreshed schema-v2 test only once for final reporting.

### Metrics

The main table should use:

- strict identity Micro-F1, precision, and recall;
- active-target-slot and fully-representable-row coverage;
- primary fix transition rate and eligible denominator;
- POST_GOLD verified-repair rate by primary family as a dataset diagnostic;
- GFR;
- pooled SRR and pooled SIR with their total numerators/denominators;
- disruption/edit-minimality;
- non-vacuous primary fix and focus-deletion diagnostics where relevant.

Secondary table or appendix diagnostics may include:

- feature-space Micro-F1;
- representable-only Micro-F1;
- `repair_action_match_rate`;
- `srr_macro_defined` and `sir_macro_defined` with defined supports;
- candidate-oracle headroom.

The feature-space score and heuristic action match must never be used for model
selection under names suggesting semantic correctness.

## Section-Level Paper Changes

### Abstract

Remove all current numerical improvements until reruns pass the integrity gate.
The abstract may still frame the study as testing whether executable constraint
factors improve curator-edit imitation and symbolic repair safety. Add final
numbers only from evaluation schema v2.

### Methods

Add subsections for:

1. preserved source splits and exact validated sampling;
2. semantic identity versus filtered feature vocabularies;
3. primary eligibility and exclusion audit;
4. centralized Wikidata constraint semantics;
5. target-free candidate inference;
6. strict fidelity and pooled global metrics;
7. artifact provenance and integrity validation.

### Model description

Replace per-type role-pressure language for A1/M1C/M1D with shared role-pressure
blocks. Distinguish the 96.6% pressure-subsystem reduction from total model size.
Include total/trainable and component counts from `run_manifest.json`.

### Results

Delete or mark obsolete every table populated from existing `evaluations/model.json`
files. Rebuild tables only after all rows pass the validator. Include confidence
intervals or multi-seed variation if used in the journal submission; a single
seed should be identified as such rather than implying uncertainty was measured.

### Discussion

Explicitly separate three questions:

- Does executable factorization improve strict curator-edit imitation?
- Does it improve semantic safety under constraint transitions?
- Do safety-aware decision objectives improve the fidelity-safety frontier over
  architecture-matched A1?

Discuss filtering sensitivity, eligibility coverage, and unresolved rare targets
as limitations. Discuss incomplete training-derived taxonomy coverage and
historical edits that do not verify as complete repairs. Do not infer
full-Wikidata validity from locally checkable rows.

## Rerun Interpretation Matrix

| Schema-v2 outcome | Defensible conclusion |
| --- | --- |
| A1 improves strict identity F1 over B0 across thresholds. | Executable factor context improves repair imitation robustly. |
| Improvement appears only in feature-space F1 or only at threshold 100. | The effect is partly vocabulary compression; narrow the representation claim. |
| A1 improves fidelity but worsens pooled SRR/GFR. | Factorization helps imitation but is not sufficient for safe repair. |
| M1C/M1D improve pooled safety at comparable strict F1. | Safety-aware decision objectives improve the architecture-matched frontier. |
| Improvements disappear after removing gold candidate access. | Earlier candidate-system gains were leakage artifacts and must be retracted. |
| High action match but low primary transition rate. | Heuristics mimic plausible edit forms without reliably resolving constraints. |
| Large post-uncheckable/focus-deletion rates. | Apparent satisfaction is deletion/vacuity driven; foreground non-vacuous results. |
| Shared A1 matches per-type A1. | Prefer shared A1 by parsimony; report the pressure-subsystem reduction. |
| Shared A1 underperforms materially on the refreshed validation set. | Reconsider promotion before test evaluation; do not switch after viewing refreshed test results. |

## Reporting Guardrails

- Do not combine schema-v1 and schema-v2 metrics in one quantitative table.
- Do not call feature-space equality exact fidelity.
- Do not call action-pattern agreement primary repair success.
- Do not average undefined SRR/SIR samples as zeros.
- Do not claim the full model is 96.6% smaller.
- Do not describe the promoted A1's old test advantage as confirmatory.
- Do not omit primary-exclusion or representability denominators.
- Do not equate an eligible correction row with a verified gold repair.
- Do not describe the frozen training hierarchy as complete Wikidata closure.
- Do not accept results without matching config/checkpoint/data provenance.

## Completion Condition

The paper narrative can leave reset state only when the technical validator
passes the full canonical suite and the conceptual claims above have been filled
with schema-v2 results. Until then, all result statements should use future or
hypothesis language.
