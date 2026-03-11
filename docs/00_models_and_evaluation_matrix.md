# Paper-Focused Models & Evaluation Matrix

Date: 2026-03-11

This document presents a model catalog for a **paper-focused experimental plan**.
Its goal is not to enumerate every runnable variant, but to define the **smallest model set that yields a strong, coherent conference paper**.

The central objective of the paper is to test whether **Executable Constraint Factors** make KG repair **safer** by reducing collateral damage on locally applicable constraints while preserving the strengths of the previous ESWC system.

---

## 1) Paper objective and scientific claim

### Continuation-paper positioning
This paper should read as a **direct continuation** of the ESWC paper, not as a new benchmark over many loosely related architectures.

The previous paper established that a structure-aware multi-relational GIN with a six-slot decoder can learn curator-like repairs from violation subgraphs.
The new paper should establish the next step:

> **By upgrading passive constraint context into executable constraint factors and adding intent-aware candidate selection, we can reduce secondary regressions without giving up curator alignment.**

### What the paper must demonstrate
The experiments should support four linked claims:

1. **Previous-work baseline:** the ESWC-style model remains strong on historical fidelity and primary repair.
2. **Representation upgrade:** executable factorization improves the local repair state representation beyond the previous passive-constraint setup.
3. **Decision upgrade:** intent-aware candidate selection reduces secondary regressions beyond imitation-only decoding.
4. **Trade-off frontier:** a global-satisfaction reference can improve consistency further, but at the cost of curator fidelity.

Everything in the model suite should serve one of these claims.

---

## 2) Design principles for a publishable model suite

To maximize clarity and acceptance probability, the model suite should obey the following rules:

1. **One strict prior-work baseline.**
   The main baseline must be an explicit reproduction of the published ESWC model family, not a generic "old-style" objective in the new codebase.

2. **One main proposed model.**
   The paper should have a single clearly identifiable answer to the research question.

3. **Only a small number of ablations.**
   Each ablation must isolate one causal question. If a model does not answer a distinct scientific question, it should be removed from the main paper.

4. **One reference extreme.**
   A global-fix model is useful as a frontier point, but it is not the main method.

5. **Heuristics are anchors, not the story.**
   Symbolic baselines should remain in the tables for context, but the paper should not spend much narrative budget on them.

6. **Keep the backbone fixed whenever possible.**
   The previous paper already established the best encoder family. The new paper should not reopen backbone selection unless absolutely necessary.

---

## 3) Fixed backbone and fairness constraints

For fairness and narrative continuity, all learned proposal models in the new paper should share the same base recipe unless the model definition fundamentally requires otherwise.

### Fixed recipe for learned proposal models
Use the previous paper's best setting as the default backbone:

- **Graph topology:** non-flattened multi-relational representation
- **Backbone:** GIN/GINE family
- **Node features:** frozen text embeddings
- **Structural features:** role embeddings enabled
- **Decoder:** six-slot decoder
- **Training:** dynamic per-type reweighting enabled
- **Old soft fix-probability loss:** disabled

This is important for two reasons:

- It makes the continuation claim credible: the new paper improves on the **best previously validated family**, not on a weakened baseline.
- It avoids diffusion into a second architecture-search paper.

### Exception: strict ESWC baseline
The prior-work baseline should preserve the **old representation regime**:

- violated constraint included as in the ESWC system,
- no locally applicable executable factor expansion,
- no typed pressure,
- no chooser,
- pure edit-imitation decoding.

This model should be presented as a **reproduction of the previous paper's model family**, not merely as a loss ablation.

---

## 4) Final paper model set

This is the recommended model set for the main paper.

### Core learned models

| Paper name | Role in paper | Training script | Core idea | Objective | Inference | Include in main tables? |
| --- | --- | --- | --- | --- | --- | --- |
| **B0 ESWC-Reproduction** | Main baseline | `07_train.py` | Reproduce the previous published model family as faithfully as possible | `L_edit` | Slot argmax | **Yes** |
| **A1 Factorized Imitation** | Representation ablation | `07_train.py` | Add executable factor nodes and typed pressure, but keep imitation-only training | `L_edit` | Slot argmax | **Yes** |
| **M1 Safe Factor Model** | Main proposed model | `07_train.py` | Executable factors + typed pressure + candidate chooser trained to prefer primary-fixing, low-regression candidates | `L_edit + L_chooser` | Chooser selects candidate | **Yes** |
| **G0 GlobalFix Reference** | Frontier / upper-bound reference | `08_train_reranker.py` | Candidate scorer optimized for global satisfaction rather than curator fidelity | `L_global` | Reranker selects candidate | **Yes** |

### Heuristic baselines

| Paper name | Role in paper | Include in main tables? | Notes |
| --- | --- | --- | --- |
| **CSM** | Strong symbolic anchor | **Yes** | Must remain because it was the strongest symbolic baseline in the previous paper. |
| **DFB** | Low-fidelity / high-deletion anchor | **Yes** | Useful for showing that validity alone is insufficient. |
| **AMB** | Constraint-specific anchor | Optional | Keep for per-constraint appendix or for inverse-like constraints; not necessary in every global table. |

---

## 5) Model definitions

### B0) ESWC-Reproduction
**Purpose**
Provide the defensible prior-work baseline for the continuation paper.

**Definition**
A faithful reproduction of the published ESWC model family:

- previous subgraph formulation,
- passive constraint context rather than executable local factor expansion,
- multi-relational GIN/GINE backbone,
- frozen text embeddings,
- role embeddings,
- six-slot decoder,
- dynamic per-type reweighting,
- no chooser,
- no typed pressure,
- no global or no-regression candidate objective.

**Objective**
\[
\mathcal{L}_{\text{B0}} = \mathcal{L}_{edit}
\]

**Expected behavior**
- strong historical fidelity,
- strong primary fix rate,
- no explicit control over secondary regressions.

**Why it is in the paper**
This is the model the new method must beat or at least match on fidelity while improving SRR/GFR.

---

### A1) Factorized Imitation
**Purpose**
Isolate the contribution of the **representation upgrade**.

**Definition**
Same proposal architecture family as the main model, but without the chooser:

- locally applicable constraints added as executable factor nodes,
- typed factor-to-variable pressure enabled,
- same backbone and decoder family as the paper default,
- no chooser,
- no policy head,
- no reranker.

**Objective**
\[
\mathcal{L}_{\text{A1}} = \mathcal{L}_{edit}
\]

**Expected behavior**
- fidelity close to B0,
- similar or slightly better primary fix,
- some improvement in SRR/GFR if the factorized representation already helps,
- still fundamentally imitation-driven.

**Why it is in the paper**
It answers: *Are executable factors useful even before adding explicit safe-selection behavior?*

---

### M1) Safe Factor Model
**Purpose**
Serve as the paper's **main proposed model**.

**Definition**
A factorized proposal model with intent-aware candidate selection:

- executable factor nodes over the locally applicable constraint set,
- typed pressure during message passing,
- proposal model produces candidate edits,
- chooser scores candidates using primary-fix and no-regression preferences,
- no policy abstraction in the main paper.

**Recommended training view**
The model should operationalize the idea that the system remains curator-oriented, but among plausible repairs it should prefer those that do not damage secondary constraints.

**Objective (paper-level view)**
\[
\mathcal{L}_{\text{M1}} = \mathcal{L}_{edit} + \mathcal{L}_{chooser}
\]

where the chooser should favor:

- candidates that fix the **primary** violated constraint,
- candidates with lower **secondary regression**,
- optionally slight preference for lower disruption when tied.

**Expected behavior**
- small or acceptable drop in fidelity relative to B0,
- same or better primary fix,
- clearly improved SRR,
- improved GFR,
- low disruption relative to global-fix models.

**Why it is in the paper**
This is the model that directly instantiates the main hypothesis and should be the best practical trade-off point.

---

### G0) GlobalFix Reference
**Purpose**
Provide the "consistency-first" frontier point.

**Definition**
A candidate reranker trained to prefer edits that maximize global or local-set satisfaction after execution.

**Objective**
\[
\mathcal{L}_{\text{G0}} = -\mathbb{E}[\text{GlobalSatisfaction}(G^{\Delta})]
\]

**Expected behavior**
- highest or near-highest GFR,
- low SRR,
- reduced historical fidelity,
- higher disruption,
- useful as a reference extreme, not as the main system.

**Why it is in the paper**
It makes the frontier legible: the paper is not claiming global optimality, but a better fidelity–safety trade-off.

---

## 6) Models explicitly removed from the main paper

The following implemented variants should **not** appear in the main narrative unless later experiments show a compelling and unexpected result.

### Remove from the main paper
- **Factor-loss-only variants**
  These are auxiliary-training controls, not a distinct scientific contribution.

- **Policy-choice models**
  These introduce a second paper thesis (planner/executor modularity and interpretability). They are interesting, but they compete with the main executable-factor story.

- **Fix-1 reranker as a main competitor**
  This creates unnecessary duplication with the integrated chooser. Unless it substantially outperforms M1, it should remain an appendix or engineering note.

### Allowed as appendix / reserve experiments
- **M1 without typed pressure**
  Use only if you need a precise pressure ablation.

- **AMB in global tables**
  Include only if reviewers are likely to ask for consistency with the previous paper's exact baseline set.

- **Current-codebase imitation-only variant without strict ESWC reproduction**
  Useful internally, but not a substitute for B0.

---

## 7) Final experiment suite

### Suite A — Main result table (mandatory)
This is the core paper table.

Include:
- DFB
- CSM
- B0 ESWC-Reproduction
- A1 Factorized Imitation
- M1 Safe Factor Model
- G0 GlobalFix Reference

Report:
- Historical Fidelity (Precision / Recall / F1)
- Primary Fix Rate
- Global Fix Rate (GFR)
- Secondary Regression Rate (SRR)
- Secondary Improvement Rate (SIR)
- Disruption / edit-size measures

### Suite B — Ablation table (mandatory)
This table should explain *where the gain comes from*.

Include:
- B0 ESWC-Reproduction
- A1 Factorized Imitation
- M1 Safe Factor Model

Interpretation:
- **B0 → A1** isolates the benefit of executable factorization and typed pressure.
- **A1 → M1** isolates the benefit of intent-aware candidate selection.

### Suite C — Frontier figure (mandatory)
A scatter or Pareto-style figure showing the central trade-off.

Recommended axes:
- x-axis: Historical Fidelity F1
- y-axis: Secondary Regression control or Global Fix Rate

Recommended plotted points:
- DFB
- CSM
- B0
- A1
- M1
- G0

This figure should visually communicate that **M1 is the practical frontier point**, while **G0 is the consistency-maximizing extreme**.

### Suite D — Per-constraint breakdown (mandatory)
At minimum compare:
- CSM
- B0
- M1

This prevents the paper from appearing to win only through easy constraint families.

### Suite E — Typed pressure ablation (optional but recommended)
Include only if stable and informative.

Compare:
- `M1-no-pressure`
- `M1`

Use this suite only if it adds a crisp message. Do not expand into a large architecture table.

---

## 8) Evaluation protocol

### Primary metrics
Retain the previous paper's core metrics:

1. **Historical Fidelity**
   Triple-and-operation match precision, recall, and micro-F1.

2. **Primary Fix Rate**
   Fraction of instances where the triggering constraint is repaired.

### New safety metrics
These are the differentiating metrics of the new paper:

3. **Global Fix Rate (GFR)**
   Satisfaction fraction across the locally applicable constraint set after applying the predicted edit.

4. **Secondary Regression Rate (SRR)**
   Fraction of secondary constraints that were satisfied pre-edit and become violated post-edit.

5. **Secondary Improvement Rate (SIR)**
   Fraction of previously unsatisfied secondary constraints that become satisfied post-edit.

6. **Disruption / Edit Minimality**
   Number of changed triples, delete/add ratio, and total operation count.

### Required evaluation rules
- Compute global/secondary metrics in **strict mode** for all learned models in the main suite.
- Report both **overall** and **per-constraint-type** results where support is sufficient.
- Do not rely on fidelity alone.
- Do not report only GFR without SRR; the paper's claim is about **safer local repair**, not merely higher consistency.

---

## 9) Recommended configuration policy

### General rule
Do not search over many backbones or input encodings in this paper.
The ESWC paper already established that landscape.

### Recommended learned-model defaults
Use the following defaults unless a model definition requires a change:

```json
{
  "model_config": {
    "model": "GIN_PRESSURE",
    "pressure_enabled": true,
    "pressure_type_conditioning": "concat",
    "enable_policy_choice": false
  },
  "training_config": {
    "factor_loss": { "enabled": false, "weight_pre": 0.0 },
    "fix_probability_loss": { "enabled": false },
    "chooser": {
      "enabled": false,
      "loss_mode": "fix1",
      "beta_no_regression": 0.5,
      "gamma_primary": 0.0,
      "topk_candidates": 20,
      "max_candidates_total": 80
    }
  }
}