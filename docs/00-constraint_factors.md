# Executable Constraint Factors for Neuro-Symbolic KG Repair

## 1) Research Idea, Main Hypothesis, Scientific Contribution

### Core idea
Extend *structure-aware neural repair* for collaborative Knowledge Graphs (KGs) by representing **constraints as first-class executable factor nodes** inside the per-violation subgraph. These factors are not passive context nodes; they perform **typed, reusable constraint evaluation** and inject **constraint pressure signals** into message passing.

This builds on the previous ESWC work where:
- the violated constraint appeared in the subgraph as a **simple node** (mostly as a connector),
- and the model predicted **historical edit actions** (add/delete) for a single repair step.

### Main hypothesis
**H1 (Executable constraints reduce collateral damage):**  
By internalizing constraint semantics as executable factors, a repair model can preserve historical fidelity and primary-fix performance while **reducing regressions on secondary locally-applicable constraints**.

**H2 (Constraint semantics become compositional):**  
Constraint-type–specific factor functions can learn reusable validation operators that transfer across constraint instances and enable principled reasoning over multiple constraints simultaneously.

### Scientific contribution
1. **Subgraph formulation upgrade:** from passive constraint nodes to **factor-graph constraint operators**.
2. **Intent-aware training objective:** enforce primary repair while regularizing secondary constraints via **no-regression** rather than hard satisfaction.
3. **Evaluation beyond single-constraint repair:** introduce metrics for secondary regressions and global consistency to measure collateral effects.
4. **Trade-off analysis:** demonstrate a Pareto frontier between curator fidelity and global constraint satisfaction.

We do not assume that all applicable constraints should be satisfied after every repair. In collaborative knowledge graphs, constraints are soft and repairs are intent-driven: editors typically address one violation at a time, even if other constraints remain unsatisfied. Our goal is therefore not global consistency, but locally improving the graph state without introducing unnecessary regressions.

---

## 2) Paper Narrative Summary

### Problem
Collaborative KGs (e.g., Wikidata) rely on soft, evolving constraints. Existing neural repair systems learn from historical edits but are **myopic**: they optimize repairing one violated constraint at a time and can unintentionally **break other constraints** in the local context.

### Limitation of prior work
Prior structure-aware repair models:
- condition on one violated constraint instance,
- produce edits that match human behavior,
- do not explicitly model other constraints that might be impacted by the edit,
- thus do not control collateral violations.

### Proposed solution
Introduce **Executable Constraint Factors**:
- expand the per-instance subgraph to include all **locally applicable constraints** as **factor nodes**,
- learn **one shared factor function per constraint type** that evaluates constraint satisfaction,
- incorporate factor-to-variable feedback during message passing,
- train with an **intent-aware loss** that prioritizes fixing the primary constraint while discouraging secondary constraint regressions.

### Key claims to validate
- The proposed model improves **secondary regression rate** and **global satisfaction** without sacrificing primary fix rate.
- A "Global Fix Model" provides an upper bound on global satisfaction but sacrifices fidelity.
- A "Policy Choice Model" demonstrates robustness and interpretability at the strategy level.

---

## 3) Dataset Construction

### Starting point
Use the historical Wikidata constraint repair corpus (violation instances paired with human repairs).

Each instance includes:
- a violating triple `(s, p, o)` (focus triple),
- an optional conflicting triple `(s', p', o')`,
- a constraint identifier/type and definition metadata,
- a local neighborhood context around involved entities,
- the ground-truth historical repair edit `Δ*` as add/delete operations.

### Extensions needed for this work
1. **Constraint retrieval** (schema-level):
   - Query Wikidata (or cached constraint registry) for constraint instances associated with:
     - properties in the focus/conflicting triples,
     - optionally properties in the 1-hop neighborhood.
2. **Constraint validation labels**:
   - For each applicable constraint instance `c` in the subgraph, compute satisfaction on:
     - pre-repair state `G^-`,
     - post-repair state `G^+` (after applying the historical edit `Δ*`).
   - These become binary labels `s_c^-` and `s_c^+`.

### Locally applicable constraint set `C_local`
Two viable definitions (start with the conservative one):

**(A) Predicate-scope constraints (recommended v1):**  
Include constraint instances attached to the properties appearing in the focus/conflicting triples.

**(B) Local closure (broader):**  
Include constraint instances attached to any property within 1-hop of the shared entities involved in the violation.

---

## 4) Per-Instance Subgraph Model

Each violation instance yields a directed heterogeneous graph:

### Variable nodes (data graph)
- Entity nodes (items)
- Predicate nodes (properties) *(optional as nodes; needed if factors connect via predicates)*
- Literal nodes (values)
- Optional: role-specific nodes for focus triple elements

### Factor nodes (constraint graph)
For each constraint instance `c ∈ C_local`, add a constraint factor node:
- typed by constraint family `t(c)` (e.g., `conflictWith`, `single`, `valueType`, ...),
- parameterized by a shared function `f_{t(c)}`.

### Edges
**Variable → Variable edges**
- Standard KG edges from neighborhood triples, using either:
  - flattened form (`s → p → o`) or
  - multi-relational form (`s → o` with `p` as edge attribute).
  (This work prefers multi-relational representation for topology fidelity.)

**Variable → Factor edges**
- Define the **scope** of each constraint factor:
  - e.g., for `conflictWith(p,q)` connect `{p,q}` (and optionally the subject `s`),
  - for `single(p)` connect `{p, s}` (and all values connected to `(s,p,*)` in subgraph).

**Factor → Variable edges**
- Enables factor feedback (“constraint pressure”) into message passing.
- Can be represented explicitly, or implicitly applied as directed messages.

---

## 5) Per-Instance Prediction Objective and Labels

### Primary prediction task (Main model and ESWC-style models)
Predict a repair edit operation `Δ` in the **six-slot add/delete format**:
$$
\hat{\Delta} = (\hat{y}_{s+},\hat{y}_{p+},\hat{y}_{o+},\hat{y}_{s-},\hat{y}_{p-},\hat{y}_{o-})
$$
Labels:
- `Δ*` from historical edit logs.

### Auxiliary prediction task (factor satisfaction)
For each constraint factor node `c`, predict satisfaction probability:
$$
\hat{s}_c(G) \in [0,1]
$$
Labels:
- `s_c^-` from symbolic validation on `G^-`
- `s_c^+` from symbolic validation on `G^+` (historical fix applied)

**Primary constraint:** the violated one, `c*`, has `s_{c*}^- = 0` by construction.

---

## 6) Variables and Factor Nodes

### Variable nodes
Hold embeddings for:
- entities / literals,
- predicates (if modeled as nodes),
- optional role embeddings (subject/predicate/object role flags).

### Factor nodes
Each factor node is a constraint instance with:
- a **type** `t ∈ {conflictWith, single, ...}`,
- a **scope** over variable nodes,
- a learned executable evaluation function `f_t(...)` producing:
  - satisfaction estimate `\hat{s}_c`,
  - and a message `m_{c→v}` to affected variables.

**Key property:**  
All factor instances of the same type share parameters → one function per type.

**Behavioral approximiations:**
Constraint factor functions are not intended to reproduce the exact symbolic semantics of constraints. Instead, they learn a differentiable approximation of how constraint violations manifest in historical repair behavior, enabling the model to reason about constraint pressure during inference.

---

## 7) GNN Message Passing and Constraint Execution

### Variable-to-variable message passing (base encoder)
Use a GNN backbone (e.g., GIN on multi-relational topology):
$$
h_v^{(k+1)} = \mathrm{GNN}\left(h_v^{(k)}, \{(h_u^{(k)}, e_{u\to v}) : u \in \mathcal{N}(v)\}\right)
$$

### Constraint evaluation (factor execution)
For each factor node `c` with scope `S(c)`:
1. Aggregate scope embeddings:
$$
z_c^{(k)} = \mathrm{AGG}_c\left(\{h_v^{(k)} : v \in S(c)\}\right)
$$
2. Compute satisfaction probability:
$$
\hat{s}_c^{(k)} = \sigma\left(f_{t(c)}(z_c^{(k)})\right)
$$
3. Emit feedback messages to scoped variables:
$$
m_{c\to v}^{(k)} = g_{t(c)}\left(z_c^{(k)}, h_v^{(k)}\right)
$$

### Constraint-to-variable integration (factor feedback)
Variables incorporate constraint pressure:
$$
h_v^{(k+1)} \leftarrow h_v^{(k+1)} + \sum_{c: v\in S(c)} m_{c\to v}^{(k)}
$$

### Weight updates
Standard backpropagation updates:
- GNN encoder parameters,
- decoder parameters (for edit prediction),
- factor-type networks `f_t` and message networks `g_t`.

---

## 8) Loss Function (Fix 1: Intent-aware, no-regression)

### Components

#### (1) Edit imitation loss (historical fidelity)
$$
\mathcal{L}_{edit} = \frac{1}{6}\sum_{k} CE(\hat{y}_k, y_k^*)
$$

#### (2) Primary constraint satisfaction (must fix triggering constraint)
Compute predicted satisfaction of primary constraint after predicted edit:
- Either approximate via candidate-based application (recommended) or
- Use the factor output on the post-edit predicted state representation.

Loss:
$$
\mathcal{L}_{primary} = BCE(\hat{s}_{c^*}^{+}, 1)
$$

#### (3) Secondary no-regression loss (guardrails, not global satisfaction)
For secondary constraints `c ≠ c*`:
$$
\mathcal{L}_{secondary} =
\frac{1}{|\mathcal{C}_{sec}|}
\sum_{c \neq c^*}
\max\left(0, \hat{s}_c^{-} - \hat{s}_c^{+}\right)
$$
This penalizes **making secondary constraints worse** than pre-state.

### Total loss (Main model)
$$
\boxed{
\mathcal{L} = \mathcal{L}_{edit} + \alpha \mathcal{L}_{primary} + \beta \mathcal{L}_{secondary}
}
$$
with:
- `α` large (primary must be fixed),
- `β` small (avoid collateral regressions but keep curator fidelity).

---

## 9) Models and Baselines

### Learned models

#### M0) ESWC model (previous work)
- Standard graph encoding + GNN encoder
- Six-slot decoder
- Trained primarily on `L_edit`
- Constraints are passive nodes (only violated constraint included)

#### M1) Main model (Fix 1)
- Variable + factor nodes (locally applicable constraints)
- Factor execution and feedback during message passing
- Loss = `L_edit + α L_primary + β L_secondary`

#### M2) Global Fix Model (constraint-maximizing)
Goal: maximize global satisfaction, regardless of curator edits.

**Recommended training method (stable): candidate-based planner**
- Generate candidate edits `{Δ_1,...,Δ_K}` from:
  - heuristics (DFB, AMB),
  - ESWC-like model proposals,
  - small enumerations for common constraints.
- Model outputs a distribution `p(Δ_k)`.
- Reward candidate by global satisfaction after applying it:
  - `Sat(G^{Δ_k}) = mean_c s_c(G^{Δ_k})` over all `C_local`.

Loss:
$$
\mathcal{L}_{global} = -\sum_k p(Δ_k)\cdot Sat(G^{Δ_k})
$$
(No edit imitation term.)

Expected to optimize Global Fix Rate, sacrifice fidelity.

The Global Fix Model is included as a reference point rather than a practical repair system. It illustrates the trade-off between curator fidelity and global constraint satisfaction and serves as an approximate upper bound on achievable global fix rates.

#### M3) Policy Choice Model (Fix 2)
Predict a strategy class rather than exact edit:
- Classes: `{delete_focus, delete_conflict, add_mirror, add_required, abstain, ...}`
Train with cross-entropy against a strategy label derived from historical edit.
Use a deterministic executor to map policy → edit for fix-rate metrics.

### Heuristic baselines

#### H1) DeleteFocusBaseline (DFB)
Always delete the violating triple.

#### H2) AddMirrorBaseline (AMB)
For inverse/symmetric constraints, add mirrored triple.

#### H3) ConstraintShapeMajority (CSM)
Memorize the most common repair for each constraint definition.

---

## 10) Evaluation Metrics

### Primary metrics (from ESWC)
#### (A) Historical Fidelity (Micro F1)
Compare predicted edits `T_pred` vs historical edits `T_gold`:
- Precision / Recall / F1 using strict triple+operation match.

#### (B) Primary Fix Rate
Apply predicted edit to `G^-` → `G^{Δ}` and validate **primary constraint** `c*`.
Report fraction of instances where `c*` is fixed.

### New global/secondary metrics
#### (C) Global Fix Rate (GFR)
After applying predicted edit, compute satisfaction fraction over all locally applicable constraints:
$$
GFR = \frac{1}{|\mathcal{C}_{local}|}\sum_{c \in \mathcal{C}_{local}} s_c(G^{Δ})
$$

#### (D) Secondary Regression Rate (SRR)
Fraction of secondary constraints that were satisfied pre-edit but violated post-edit:
$$
SRR = \frac{|\{c \neq c^*: s_c(G^{-})=1 \land s_c(G^{Δ})=0\}|}{|\{c \neq c^*: s_c(G^{-})=1\}|}
$$

#### (E) Secondary Improvement Rate (SIR)
Fraction of secondary violated constraints that become satisfied:
$$
SIR = \frac{|\{c \neq c^*: s_c(G^{-})=0 \land s_c(G^{Δ})=1\}|}{|\{c \neq c^*: s_c(G^{-})=0\}|}
$$

#### (F) Disruption / Edit Minimality
Measures how invasive the predicted edit is:
- `#changed_triples`
- `delete/add ratio`
- optional graph-edit distance proxy.

### Policy model metrics
#### (G) Policy Accuracy
Top-1 strategy prediction accuracy vs derived policy labels.

---

## 11) Expected Behaviors and Evaluation Outcomes

| Model      |   Fidelity | Primary Fix | Global Fix |    SRR | Disruption |
| ---------- | ---------: | ----------: | ---------: | -----: | ---------: |
| ESWC       |       High |        High |     Medium | Medium |        Low |
| Main       | Slightly ↓ |      Same/↑ |          ↑ |     ↓↓ |        Low |
| Global Fix |         ↓↓ |        High |         ↑↑ |      ↓ |         ↑↑ |
| Policy     |     Medium |      Medium |     Medium | Medium |     Medium |
| DFB        |        Low | High (some) |   Unstable |   High |       High |
| CSM        |     Medium |      Medium |     Medium | Medium |        Low |

### Heuristics
#### DFB
- High Primary Fix Rate for constraints solvable by deletion
- Low Historical Fidelity (unless humans often delete focus)
- Poor Global Fix / high SRR due to blunt deletions and collateral removals

#### AMB
- Very high Fix Rate on inverse constraints
- Low coverage elsewhere → poor overall performance

#### CSM
- Moderate to high Historical Fidelity (memorizes common repairs)
- Good Primary Fix Rate when repair patterns are stable
- Limited generalization, moderate SRR

---

### Learned models

#### M0 ESWC model
**Behavior**
- Strong imitation of curator repairs
- Fixes primary constraint well
- No control over collateral constraints

**Expected metrics**
- High Fidelity
- High Primary Fix
- Medium Global Fix Rate
- Higher SRR than Main model

---

#### M1 Main model (Fix 1)
**Behavior**
- Mostly curator-aligned repairs
- When multiple repairs are plausible, prefer those with fewer secondary regressions
- Factor nodes learn validation semantics and guide decisions

**Expected metrics**
- Fidelity: slightly lower than ESWC (small drop)
- Primary Fix: same or slightly higher
- Global Fix Rate: higher than ESWC
- SRR: significantly lower than ESWC
- SIR: moderate improvement

**Narrative**
- Best trade-off point: curator-like but safer.

---

#### M2 Global Fix Model
**Behavior**
- Prioritizes maximizing satisfaction across `C_local`
- May over-edit or prefer aggressive deletions/additions
- Will diverge from curator norms

**Expected metrics**
- Fidelity: low
- Primary Fix: high
- Global Fix Rate: highest
- SRR: low (if truly optimizing)
- Disruption: high

**Narrative**
- Establishes the “global optimality” extreme; illustrates fidelity–validity frontier.

---

#### M3 Policy Choice Model
**Behavior**
- Predicts coarse repair strategy reliably
- Depends on executor quality for concrete edit
- More robust to ambiguity in specific object choice

**Expected metrics**
- Policy accuracy: high on common constraint types
- Fidelity: medium (executor loses detail)
- Primary Fix: medium-to-high depending on executor
- Global Fix: moderate
- SRR: moderate

**Narrative**
- Demonstrates that repair can be modular: planner + executor, enabling interpretable repair pipelines.

---

## 12) Recommended Presentation of Results (Narrative-First)

### Main tables/figures to include
1. **Core comparison table** (Fidelity, Primary Fix, Global Fix, SRR, Disruption)
2. **Per-constraint breakdown** (as in ESWC) + show SRR per constraint family
3. **Pareto plot**: Fidelity (x) vs Global Fix or SRR (y)
   - Show ESWC vs Main vs Global Fix
4. **Qualitative examples** of repairs where:
   - ESWC matches curator but introduces secondary violation,
   - Main deviates minimally to avoid regression,
   - Global Fix chooses a non-human but globally consistent repair.

---

## 13) Summary of the Full Specification

- Input: violation-centered subgraph `G^-` enriched with locally applicable constraint factor nodes.
- Output: repair edit `Δ` (six-slot format) and factor satisfaction signals.
- Main model learns curator-like repairs with constraint-aware guardrails:
  - fix primary constraint,
  - avoid worsening secondary constraints.
- Evaluation extends beyond single-constraint repair with global consistency and regression metrics.
- Additional models illustrate extremes:
  - Global Fix (maximize constraint satisfaction),
  - Policy Choice (strategy-level planning),
  - plus symbolic heuristics for coverage and sanity baselines.
