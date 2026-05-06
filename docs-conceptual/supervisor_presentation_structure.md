# Supervisor Presentation Structure

This document proposes a supervisor-facing presentation structure for the `Constraint Factors` project. It is intentionally more explicit than a normal talk outline so it can serve as both:

- the deck skeleton,
- and the content checklist for each section.

The structure below is aligned with the current repository scope:

- conceptual story: executable constraint factors for local KG repair,
- current paper-facing model surface: `B0`, `A1`, `M1C`, `M1D`, `G0`,
- current artifact pipeline: downloader -> dataframe builder -> registry -> optional text retrieval -> factor labeling -> graph materialization -> training/evaluation.

## 1. Framing Decision

For a PhD-supervisor presentation, the deck should be organized around the **research question first**, not around the code pipeline first.

Recommended order:

1. problem and scientific relevance,
2. core idea and hypotheses,
3. dataset / benchmark construction,
4. model formulation,
5. model family and baselines,
6. evaluation logic,
7. current status, limitations, and next steps.

This ordering makes it clear that the repository is supporting a research claim rather than merely implementing an engineering pipeline.

## 2. Recommended Top-Level Sections

These are the recommended main sections for the markdown presentation brief.

1. **Project Overview and Scientific Relevance**
2. **Problem Setting and Research Gap**
3. **Core Idea: Executable Constraint Factors**
4. **Research Questions, Hypotheses, and Expected Contributions**
5. **Dataset and Benchmark Construction**
6. **Graph Formulation and Learning Setup**
7. **Model Families: Current Paper Line**
8. **Reasoning Floor, Baselines, and Comparison Logic**
9. **Evaluation Plan and Success Criteria**
10. **Current Implementation Status**
11. **Planned Future Continuations**
12. **Risks, Scope Boundaries, and Discussion Points**
13. **Appendix Material**

## 3. Note on “Reasoning Floor”

`Reasoning Floor` is not currently a formal repository term. If you want to keep it in the presentation, define it explicitly as:

> the minimum level of constraint-sensitive behavior that should outperform naive repair heuristics and passive-context modeling.

For clarity, one of these section titles is preferable:

- `Reasoning Floor and Lower-Bound Baselines`
- `Minimal Competence Floor`
- `Baselines and Lower-Bound Comparison`

Recommended choice:

- `Reasoning Floor, Baselines, and Comparison Logic`

That keeps your wording while making the section interpretable to someone reading the deck cold.

## 4. Recommended Deck Shape

Suggested main deck length:

- `12-16` main slides
- `4-8` appendix slides

Suggested section-to-slide mapping:

| Section | Purpose | Likely slides |
| --- | --- | ---: |
| Project Overview and Scientific Relevance | one-slide summary of what the project is and why it matters | 1 |
| Problem Setting and Research Gap | motivate collaborative KG repair and limits of prior work | 1-2 |
| Core Idea | introduce factor nodes and local constraint reasoning | 1-2 |
| Research Questions / Hypotheses | make claims precise and testable | 1 |
| Dataset and Benchmark Construction | explain where instances come from and how the benchmark is built | 2-3 |
| Graph Formulation and Learning Setup | explain the violation-centered graph and repair objective | 2 |
| Model Families | place `B0`, `A1`, `M1C`, `M1D`, `G0` on one continuum | 1-2 |
| Reasoning Floor / Baselines | explain heuristic floors and what counts as meaningful progress | 1 |
| Evaluation | define metrics and expected trade-offs | 1-2 |
| Current Status / Future Work | separate implemented work from future extensions | 1-2 |

## 5. Detailed Section Structure

## 5.1 Project Overview and Scientific Relevance

This should answer, in the first minute:

- What is the project about?
- Why is it scientifically interesting?
- What is the main claim?

Recommended content:

- One-sentence project summary:
  - *This project studies whether local KG repair models can make safer edits by representing constraints as executable factor nodes instead of passive context.*
- Problem domain:
  - collaborative knowledge graphs,
  - soft constraints,
  - local historical repair prediction.
- Scientific significance:
  - bridges symbolic constraints and learned repair,
  - studies safety / collateral damage in graph editing,
  - reframes repair as a multi-constraint local reasoning problem.

Recommended supervisor-facing emphasis:

- This is not only an implementation project.
- The scientific contribution is the **representation and decision formulation**:
  - executable constraint factors,
  - no-regression objective,
  - evaluation beyond fidelity alone.

## 5.2 Problem Setting and Research Gap

This section should make the baseline problem precise.

Recommended subsections:

### A. Collaborative KG repair setting

- Each example starts from a violated constraint instance.
- The model predicts a local repair edit.
- Historical edits provide supervision.

### B. Why existing repair models are insufficient

- prior models condition on one violated constraint,
- they imitate the curator edit,
- they do not explicitly model secondary locally applicable constraints,
- therefore they can fix the triggering violation while breaking nearby constraints.

### C. Research gap

State the gap explicitly:

- Existing structure-aware repair models are still **myopic** with respect to local constraint interaction.
- The missing ingredient is not just more context, but **active constraint semantics inside the model**.

## 5.3 Core Idea: Executable Constraint Factors

This is the conceptual center of the talk.

Recommended points:

- Replace passive constraint nodes with **first-class executable factor nodes**.
- Each factor corresponds to a local constraint instance.
- Factors are typed by constraint family.
- Factors do two jobs:
  - estimate local satisfaction / violation pressure,
  - send feedback into message passing.

Recommended contrast slide:

| Prior passive setting | Proposed factorized setting |
| --- | --- |
| violated constraint as context node | all locally applicable constraints as factor nodes |
| semantics mostly implicit | semantics approximated by per-type factor functions |
| one-constraint view | multi-constraint local interaction |
| evaluation checks safety after prediction | model representations are shaped by safety pressure during prediction |

Message to emphasize:

- The innovation is not “constraints are present.”
- The innovation is “constraints are **executable participants** in the local graph computation.”

## 5.4 Research Questions, Hypotheses, and Expected Contributions

Recommended framing:

### Main research questions

- Can executable local constraint factors reduce collateral damage during KG repair?
- Can they do so without losing too much fidelity to historical curator behavior?
- Which decision mechanism is best: imitation only, chooser-based safe selection, or direct safety-aware loss?

### Main hypotheses

- `H1`: factorized executable constraints reduce secondary regressions relative to passive-context models.
- `H2`: per-type factor functions provide reusable local semantics across instances of the same constraint family.
- `H3`: safe decision objectives improve the fidelity-safety trade-off beyond representation-only gains.

### Expected contributions

- new local factor-graph formulation for KG repair,
- new safe local repair objective based on primary fix + secondary no-regression,
- expanded evaluation surface using `GFR`, `SRR`, and `SIR`,
- empirical trade-off analysis between curator fidelity and local consistency.

## 5.5 Dataset and Benchmark Construction

This section should be detailed because the benchmark is part of the research contribution.

Recommended subsections:

### A. Starting corpus

- Historical Wikidata constraint-repair data.
- Each row contains:
  - violating triple,
  - optional conflicting triple,
  - constraint id / type metadata,
  - local neighborhood context,
  - gold human repair in six-slot add/delete format.

### B. Pipeline overview

Use the actual repository stages:

1. `01_data_downloader.py`
2. `02_dataframe_builder.py`
3. `03_constraint_registry.py`
4. `04_wikidata_retriever.py` for `text_embedding` runs
5. `05_constraint_labeler.py`
6. `06_graph.py`

### C. Data fetching

Explain:

- `sample` vs `full` datasets,
- raw source files under `data/raw/<dataset>/`,
- same layout is normalized for downstream reproducibility.

### D. Interim dataframe construction

Explain why this stage matters:

- consistent integer vocabulary,
- standardized train/val/test parquet splits,
- preservation of local neighborhood evidence,
- computation of local candidate constraint sets:
  - `local_constraint_ids`,
  - `local_constraint_ids_focus`.

### E. Constraint registry

Explain:

- registry is built once from `constraints.tsv`,
- normalizes constraint family/type,
- stores constrained property and parameters,
- makes downstream factor construction and checking consistent.

### F. Local constraint set definition

This is important enough to get its own slide or subheading.

Recommended wording:

- The current paper line uses bounded local closure through `constraint_scope=local`.
- This should be presented as a conservative local neighborhood definition, not an unbounded global closure claim.

### G. Constraint labeling

Explain what `05_constraint_labeler.py` adds:

- pre-edit factor labels,
- post-gold-edit factor labels,
- checkable vs non-checkable status,
- coverage reports by constraint family.

This is the main bridge from “raw repair corpus” to “constraint-aware benchmark”.

### H. Graph materialization

Explain what `06_graph.py` creates:

- violation-centered graphs,
- factorized representation for `A1`, `M1C`, `M1D`, `G0`,
- passive `eswc_passive` representation for `B0`,
- node encodings:
  - `node_id`,
  - `text_embedding`.

### I. Dataset-construction limitations to state explicitly

- checker coverage is limited to implemented constraint families,
- local evidence is incomplete for some constraints,
- checkability is conservative,
- changing `min_occurrence` changes downstream IDs and artifacts,
- local closure is intentionally bounded.

## 5.6 Graph Formulation and Learning Setup

This section explains what the model sees and predicts.

Recommended subsections:

### A. Per-instance graph

- variable nodes:
  - entities,
  - predicates,
  - literals,
  - focus-role information.
- factor nodes:
  - locally applicable constraints,
  - typed by family,
  - aligned with `factor_constraint_ids`.

### B. Message passing story

- base variable-to-variable GNN passes structural context,
- factor functions evaluate local compatibility,
- factor-to-variable pressure reshapes node representations,
- multiple constraints can reinforce or compete.

If you want one memorable phrase for the talk:

- *the model should feel local tension, not just read local context.*

### C. Prediction target

- six-slot edit prediction:
  - `add_subject`, `add_predicate`, `add_object`,
  - `del_subject`, `del_predicate`, `del_object`.

### D. Objective story

Differentiate clearly:

- imitation objective: reproduce historical edits,
- primary-fix objective: repair the triggering violation,
- secondary objective: avoid making other local constraints worse.

This is the point where the talk transitions from graph representation to decision behavior.

## 5.7 Model Families: Current Paper Line

This section should present the canonical suite as a clean ladder.

Recommended order:

### Heuristic references

- `DFB`: Delete Focus Baseline
- `AMB`: Add Mirror Baseline
- `CSM`: Constraint Shape Majority

### Learned models

- `B0`: ESWC-style passive baseline
- `A1`: factorized imitation
- `M1C`: safe factor chooser
- `M1D`: safe factor direct-loss
- `G0`: global-fix reranker reference

Recommended interpretation of the ladder:

- `B0 -> A1` tests the representation effect.
- `A1 -> M1C` tests chooser-based safe selection.
- `A1 -> M1D` tests direct-loss safe selection.
- `G0` is not the practical target; it is the high-consistency reference point.

Recommended supervisor-facing message:

- This is not a pile of unrelated models.
- It is one controlled research ladder from passive imitation to safe factorized decision-making.

## 5.8 Reasoning Floor, Baselines, and Comparison Logic

This is the best place to keep your `Reasoning Floor` concept.

Recommended definition for the slide:

- The reasoning floor is the minimum level of behavior that demonstrates useful constraint sensitivity beyond naive repair.

What belongs here:

- Why `DFB` is a strong but crude floor for some constraint families.
- Why `CSM` is a meaningful memorization baseline.
- Why `B0` is the relevant learned lower bound for the paper.
- Why simply fixing the primary violation is insufficient.

Recommended comparison logic:

- If a factorized model does not beat `B0` on `SRR` or `GFR`, the executable-factor story is weak.
- If it beats `B0` on safety but collapses fidelity, the contribution becomes ambiguous.
- The desired result is a better trade-off, not a one-metric win.

## 5.9 Evaluation Plan and Success Criteria

This section should define how the project will be judged.

Recommended metric groups:

### A. Historical fidelity

- precision,
- recall,
- micro-F1.

### B. Primary task success

- primary fix rate.

### C. Local safety / collateral impact

- `SRR`: Secondary Regression Rate,
- `SIR`: Secondary Improvement Rate,
- `GFR`: Global Fix Rate over the local constraint set.

### D. Edit behavior

- disruption / edit minimality,
- delete-add tendencies,
- qualitative plausibility.

Recommended success criteria for the talk:

- `A1` should show whether factorized representation alone helps.
- `M1C` and/or `M1D` should lower `SRR` and improve `GFR` relative to `B0`.
- `G0` should illustrate the high-consistency / low-fidelity end of the trade-off frontier.

Recommended figures/tables:

1. main comparison table for `B0`, `A1`, `M1C`, `M1D`, `G0`, plus heuristics,
2. Pareto-style plot:
   - x-axis: fidelity,
   - y-axis: `GFR` or inverse `SRR`,
3. per-constraint-family breakdown,
4. qualitative case studies.

## 5.10 Current Implementation Status

This section should clearly separate what is already present in the repository from what is still planned.

Safe points to state as currently landed:

- canonical paper-facing suite exists: `B0`, `A1`, `M1C`, `M1D`, `G0`,
- graph representation split exists:
  - `factorized`,
  - `eswc_passive`,
- shared symbolic candidate evaluator is reused across chooser, direct-loss, reranker, and evaluation,
- per-type factor executors and per-role pressure modules are in scope for factorized models,
- factor labeling and coverage reporting are implemented.

Important scope boundary to say explicitly:

- the project does **not** currently depend on a fully neural predicted-edit rollout through the factor stack.
- candidate-level safety can remain symbolically evaluated in the current paper line.

That is a strength for clarity, not a weakness to hide.

## 5.11 Planned Future Continuations

This section should distinguish:

### Near-term continuations

- expand symbolic checker coverage to more constraint families,
- strengthen per-family coverage and checkability,
- run the canonical paper suite consistently on the locked artifact stack,
- finalize tables, plots, and case studies.

### Medium-term continuations

- cleaner ablations around factor typing and pressure,
- larger-scale analysis across constraint families,
- stronger discussion of when chooser-based vs direct-loss decision layers win.

### Longer-range exploratory continuations

- neural post-edit rollout through the factor stack,
- learned candidate-level safety estimation,
- policy-choice or planner-executor variants if needed for a later paper.

Recommended wording:

- These are continuations, not prerequisites for the current paper line.

## 5.12 Risks, Scope Boundaries, and Discussion Points

This section is especially useful for a supervisor meeting.

Recommended risks to surface proactively:

- symbolic checker coverage limits evaluation breadth,
- local evidence incompleteness can make some factors uncheckable,
- safety improvements may trade off against historical fidelity,
- some heuristic baselines can look surprisingly strong on narrow constraint families,
- bounded local closure may underspecify wider graph consequences.

Recommended discussion prompts for the supervisor:

- Is the current local-closure definition scientifically sufficient for the paper line?
- Should the main claim focus on safer local repair rather than stronger global consistency?
- Is the `M1C` vs `M1D` comparison central enough for the main narrative, or should one be treated as the primary model and the other as support?
- How much emphasis should the benchmark-construction contribution receive relative to the modeling contribution?

## 5.13 Appendix Material

Recommended appendix sections:

- constraint family taxonomy,
- exact pipeline stages and CLI surface,
- message-passing equations,
- data artifact schema,
- training/evaluation execution order,
- additional ablations and qualitative examples.

## 6. Recommended Final Structure for the Markdown Presentation Brief

If you want one concrete structure to commit to now, use this:

1. `Project Overview and Scientific Relevance`
2. `Problem Setting and Research Gap`
3. `Core Idea: Executable Constraint Factors`
4. `Research Questions, Hypotheses, and Expected Contributions`
5. `Dataset and Benchmark Construction`
6. `Graph Formulation and Learning Setup`
7. `Model Families: Current Paper Line`
8. `Reasoning Floor, Baselines, and Comparison Logic`
9. `Evaluation Plan and Success Criteria`
10. `Current Implementation Status`
11. `Planned Future Continuations`
12. `Risks, Scope Boundaries, and Discussion Points`
13. `Appendix`

## 7. Suggested Next Document

After agreeing on the structure, the next useful file should be a second markdown document that fills these sections with:

- deck-ready bullets,
- candidate figures,
- exact terminology,
- and short speaker notes.

That second document should stay in `docs-conceptual/` because it is mainly about research framing and presentation narrative rather than repository operation.
