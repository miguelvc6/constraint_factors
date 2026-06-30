# A1 Primary-Query Ablation: Implementation and Evaluation Plan

## 0. Purpose

This document specifies an implementation-ready ablation suite for testing whether the current `A1` factorized imitation model underperforms on symbolic safety because the primary violated constraint is modeled as an executable factor node together with secondary/local constraints.

The goal is **not** to remove the primary constraint from the task. The goal is to separate two roles that are currently conflated:

1. **Repair intent:** the primary constraint `c*` defines what violation the model is asked to repair.
2. **Executable context pressure:** locally applicable non-primary constraints provide contextual safety/no-regression information.

The main proposed model keeps the primary constraint as a **typed task query** and removes it from the executable factor set. Two controls test whether the query needs definition-level information and whether passive graph-level primary information is enough.

---

## 1. Hypothesis

### Main hypothesis

The current `A1` architecture may conflate target repair intent with executable constraint pressure by treating the primary violated constraint as a normal factor node. A cleaner architecture should:

- keep the primary constraint visible as the task/query;
- prevent the primary constraint from emitting factor pressure;
- reserve executable factor pressure for non-primary local constraints;
- test whether secondary/context factors become more informative for symbolic safety once the primary factor is removed from the pressure pathway.

### Important interpretation note

Do not frame this as simply “primary pressure dominates secondary pressure.” Existing H2 masks need careful interpretation: in the current code, `primary_only_pressure` keeps only primary factor pressure, while `secondary_only_pressure` keeps non-primary factor pressure. The more precise experimental question is:

> Should the primary violated constraint be an executable factor, or should it be represented as a separate repair-intent query while executable factors represent local context?

---

## 2. Model Suite

Train exactly three new models. Use the same dataset, seed, encoding, backbone, and training schedule as the canonical `A1` run unless implementation constraints require otherwise.

### 2.1 Main model: `A1-QDef`

**Proposed run id**

```text
a1_qdef_secondary_factors__full_strat1m_minocc100__node_id
```

**Representation**

Executable factor nodes include only non-primary local constraints:

```text
C_exec(i) = C_local(i) \ {c*(i)}
```

The primary constraint is represented as a **definition-conditioned query vector** that conditions the decoder, not the pressure pathway.

**Primary query contents**

The query should encode:

1. primary constraint family/type;
2. constrained property;
3. parameter predicates;
4. parameter objects;
5. optionally a focus-role summary from the focus subject/predicate/object node states.

Minimal query encoder:

```text
q_def = MLP([
    emb_family(type(c*)),
    emb_id(constrained_property(c*)),
    mean_pool(emb_id(param_predicates(c*))),
    mean_pool(emb_id(param_objects(c*))),
    focus_summary
])
```

Decoder conditioning:

```text
shared = f_graph(graph_emb) + f_query(q_def)
```

An additive query bias is preferable for the first implementation because it preserves most of the current decoder topology.

### 2.2 Control 1: `A1-QFamily`

**Proposed run id**

```text
a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id
```

**Representation**

- Executable factors are secondary/context only.
- The primary constraint query contains only the family/type id.

```text
q_family = emb_family(type(c*))
```

**Purpose**

This tests whether coarse family information is enough to recover repair intent. If `A1-QFamily` performs close to `A1-QDef`, the repair task may be mostly family-driven. If `A1-QDef` is better, constrained property and parameter definition matter.

### 2.3 Control 2: `A1-PrimaryPassive`

**Proposed run id**

```text
a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id
```

**Representation**

- Executable factors are secondary/context only.
- The primary constraint is added as a passive node or passive metadata representation.
- The primary constraint does **not**:
  - emit factor-to-variable pressure;
  - receive factor satisfaction supervision;
  - appear in `factor_node_index` for executable factor routing;
  - appear in `factor_constraint_ids` used by factor loss or H2 factor semantics.

**Purpose**

This control tests whether it is enough to keep the primary constraint somewhere in the graph representation, as passive context, without explicit query conditioning.

---

## 3. Required Data-Schema Separation

The new implementation must separate model-executable factors from symbolic-evaluation factors.

### 3.1 Existing conflation to avoid

Current factorized graphs use fields such as:

```text
factor_constraint_ids
factor_node_index
primary_factor_index
factor_checkable_pre
factor_satisfied_pre
factor_checkable_post_gold
factor_satisfied_post_gold
factor_types
```

In existing `A1`, these fields include the primary constraint. For the new ablations this is not sufficient, because evaluation still needs the full local constraint set including the primary, while the model should execute only non-primary factors.

### 3.2 New schema convention

Use `factor_*` fields for **executable factors only**.

Use `eval_factor_*` fields for **symbolic evaluation support over the full local set**.

Recommended fields:

```text
# Executable factors: model pressure + factor loss + H2 factor semantics
factor_constraint_ids              # secondary/context constraints only
factor_node_index                  # executable factor nodes only
factor_types                       # aligned with factor_constraint_ids
factor_checkable_pre               # aligned with factor_constraint_ids
factor_satisfied_pre               # aligned with factor_constraint_ids
factor_checkable_post_gold         # aligned with factor_constraint_ids
factor_satisfied_post_gold         # aligned with factor_constraint_ids
primary_factor_index = -1          # no primary executable factor in these runs

# Full local evaluation state: used by strict symbolic metrics
eval_factor_constraint_ids         # full C_local, including primary
eval_factor_types                  # aligned with eval_factor_constraint_ids
eval_factor_checkable_pre
eval_factor_satisfied_pre
eval_factor_checkable_post_gold
eval_factor_satisfied_post_gold
eval_primary_factor_index          # index of c* inside eval_factor_constraint_ids

# Primary task query / metadata
primary_constraint_id
primary_constraint_type_id
primary_constrained_property_id
primary_param_predicate_ids
primary_param_object_ids
primary_param_count
passive_primary_node_index         # only for A1-PrimaryPassive
```

Important: `eval_factor_*` should preserve the exact factor order and labels from the labeled parquet output before filtering out the primary executable factor.

---

## 4. Implementation Plan

## 4.1 Add a primary-constraint mode to config

Add a model/data option, preferably in `ModelConfig`:

```python
primary_constraint_mode: Literal[
    "executable_factor",   # existing A1 behavior
    "query_definition",    # A1-QDef
    "query_family",        # A1-QFamily
    "passive_node",        # A1-PrimaryPassive
    "none",                # optional negative control, not required now
]
```

Default:

```python
primary_constraint_mode = "executable_factor"
```

This preserves existing paper-suite behavior unless the new mode is explicitly requested.

### Backward compatibility requirement

Existing canonical runs must remain unchanged:

- `B0`
- `A1`
- `M1C`
- `M1D`
- `G0`
- existing H2 ablations

Do not change their configs, graph semantics, or artifact paths.

---

## 4.2 Modify graph construction

Target file:

```text
src/06_graph.py
```

### Required behavior

First construct the full local factor list as currently done:

```python
all_factor_ids = _factor_ids_for_graph(graph, constraint_scope)
primary_constraint_id = int(graph["constraint_id"])
```

Then derive executable factor ids depending on mode:

```python
if primary_constraint_mode == "executable_factor":
    executable_factor_ids = all_factor_ids
else:
    executable_factor_ids = [cid for cid in all_factor_ids if int(cid) != primary_constraint_id]
```

### Preserve evaluation vectors

Before filtering, copy the full labeled arrays into `eval_factor_*` fields. Then filter the arrays for executable factors only:

```python
exec_positions = [
    pos for pos, cid in enumerate(all_factor_ids)
    if primary_constraint_mode == "executable_factor" or int(cid) != primary_constraint_id
]
```

Use `exec_positions` to create:

```text
factor_constraint_ids
factor_types
factor_checkable_pre
factor_satisfied_pre
factor_checkable_post_gold
factor_satisfied_post_gold
```

Use the full positions to create:

```text
eval_factor_constraint_ids
eval_factor_types
eval_factor_checkable_pre
eval_factor_satisfied_pre
eval_factor_checkable_post_gold
eval_factor_satisfied_post_gold
eval_primary_factor_index
```

For `primary_constraint_mode != "executable_factor"`, set:

```python
data_graph.primary_factor_index = -1
```

For existing behavior, keep the current value.

### Passive primary node mode

For `primary_constraint_mode == "passive_node"`:

- create a primary constraint node with the same global token style currently used for factor nodes;
- add definition/parameter edges from the primary passive node to its parameter predicates/objects;
- do **not** add local pressure edges from the primary node to focus/local variables;
- do **not** include the primary passive node in `factor_node_index`;
- store it separately:

```python
data_graph.passive_primary_node_index = int(primary_local_id)
```

This lets graph-level pooling see the primary constraint node, while factor execution ignores it.

### Query modes

For `query_definition` and `query_family`, the primary does not need to be a graph node. Store query metadata as tensors:

```python
data_graph.primary_constraint_id = torch.tensor([primary_constraint_id], dtype=torch.long)
data_graph.primary_constraint_type_id = torch.tensor([primary_type_id], dtype=torch.long)
data_graph.primary_constrained_property_id = torch.tensor([constrained_property_gid or 0], dtype=torch.long)
data_graph.primary_param_predicate_ids = torch.tensor([...], dtype=torch.long)
data_graph.primary_param_object_ids = torch.tensor([...], dtype=torch.long)
data_graph.primary_param_count = torch.tensor([num_params], dtype=torch.long)
```

Avoid metadata attribute names containing `index` unless PyG batching increment behavior is explicitly handled.

---

## 4.3 Modify evaluation support to use `eval_factor_*`

Target files:

```text
src/09_eval.py
src/modules/repair_eval.py
src/modules/reranker_eval.py
```

The most important change is in evaluation support that collects pre-vectors from graph artifacts. Current behavior reads graph-side fields such as:

```text
factor_constraint_ids
factor_satisfied_pre
factor_checkable_pre
primary_factor_index
```

For the new runs, these fields will contain only executable secondary/context factors. Strict symbolic metrics must prefer:

```text
eval_factor_constraint_ids
eval_factor_satisfied_pre
eval_factor_checkable_pre
eval_primary_factor_index
```

Recommended fallback logic:

```python
factor_constraint_ids = getattr(data, "eval_factor_constraint_ids", None)
if factor_constraint_ids is None:
    factor_constraint_ids = getattr(data, "factor_constraint_ids", None)

pre_satisfied = getattr(data, "eval_factor_satisfied_pre", None)
if pre_satisfied is None:
    pre_satisfied = getattr(data, "factor_satisfied_pre", None)

pre_checkable = getattr(data, "eval_factor_checkable_pre", None)
if pre_checkable is None:
    pre_checkable = getattr(data, "factor_checkable_pre", None)

primary_index = getattr(data, "eval_primary_factor_index", None)
if primary_index is None:
    primary_index = getattr(data, "primary_factor_index", None)
```

This preserves old runs and makes new runs evaluable.

---

## 4.4 Add primary-query encoder to the model

Target file:

```text
src/modules/models.py
```

### Query encoder behavior

Add optional primary query support to `BaseGraphModel` or specifically to `RepairGINFactorPressure`.

Suggested constructor arguments:

```python
primary_constraint_mode: str = "executable_factor"
primary_query_enabled: bool = False
primary_query_hidden_dim: int | None = None
primary_query_include_focus_summary: bool = True
```

Enable query encoding when:

```python
primary_constraint_mode in {"query_definition", "query_family"}
```

### Family-only query

For `query_family`:

```python
q = family_embedding(primary_constraint_type_id)
q = primary_query_projection(q)
```

### Definition query

For `query_definition`:

```python
family = family_embedding(primary_constraint_type_id)
property_vec = node_id_embedding(primary_constrained_property_id)
param_pred_vec = mean_pool(node_id_embedding(primary_param_predicate_ids), primary_param_count)
param_obj_vec = mean_pool(node_id_embedding(primary_param_object_ids), primary_param_count)
focus_vec = focus_summary(node_emb, data.batch, data.role_flags)  # optional but recommended

q = primary_query_mlp(torch.cat([
    family,
    property_vec,
    param_pred_vec,
    param_obj_vec,
    focus_vec,
], dim=-1))
```

### Pooling variable-length parameter tensors

PyG batches variable-length tensors by concatenation. Use `primary_param_count` to reconstruct graph ids:

```python
counts = data.primary_param_count.view(-1)
param_graph_index = torch.repeat_interleave(
    torch.arange(batch_size, device=device),
    counts.to(device=device),
)
```

Scatter-add parameter embeddings into a `[batch_size, hidden_dim]` tensor and divide by count. If a graph has zero parameters, use a zero vector for the parameter pool.

### Focus summary

Use existing `role_flags` if available. A simple version:

```python
focus_mask = role_flags != ROLE_NONE
focus_summary = scatter_mean(node_emb[focus_mask], batch[focus_mask], dim=0, dim_size=batch_size)
```

A more structured version can separately pool subject, predicate, and object focus-role nodes. Start simple unless implementation is easy.

### Decoder conditioning

Current models compute:

```python
shared = F.leaky_relu(self.shared_projection(graph_emb), negative_slope=0.1)
```

For query models, modify to:

```python
shared_base = self.shared_projection(graph_emb)
query_bias = self.primary_query_projection(q)
shared = F.leaky_relu(shared_base + query_bias, negative_slope=0.1)
```

This preserves the same decoder topology and makes the ablation easier to interpret.

---

## 4.5 Factor execution behavior

For all three new models:

- factor execution should run only over `factor_node_index`;
- `factor_node_index` should contain only non-primary executable factors;
- factor loss should supervise only executable factors;
- H2 factor semantics should report only executable secondary/context factors;
- strict symbolic evaluation should still use `eval_factor_*` full local labels.

No change should be needed inside the core per-type factor executor if graph construction filters `factor_node_index` and aligned `factor_*` tensors correctly.

---

## 4.6 Config generation

Target file:

```text
scripts/make_experiment_configs.py
```

Add an opt-in flag:

```bash
--include-primary-query-ablations
```

This should generate exactly these configs:

```text
models/a1_qdef_secondary_factors__full_strat1m_minocc100__node_id/config.json
models/a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id/config.json
models/a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id/config.json
```

All should inherit locked `A1` / factorized model settings:

- dataset: `full_strat1m_minocc100`
- encoding: `node_id`
- backbone: `GIN_PRESSURE`
- factor executor: `per_type_v1`
- pressure module sharing: same as locked A1/M1C settings
- pressure residual scale: same as locked A1/M1C settings
- factor auxiliary loss: same as locked A1 unless the query model becomes unstable
- seed: `42`
- no chooser
- no direct safety loss
- no policy-choice head

Only the primary constraint mode should differ.

---

## 5. Tests to Add Before Training

Add tests before launching full training. These tests should be small and use sample/smoke data where possible.

### 5.1 Graph schema tests

Target:

```text
tests/test_primary_query_graph_schema.py
```

Required assertions for `query_definition`, `query_family`, and `passive_node` modes:

```python
primary_constraint_id not in data.factor_constraint_ids
primary_constraint_id in data.eval_factor_constraint_ids
data.primary_factor_index == -1
data.eval_primary_factor_index >= 0
len(data.factor_constraint_ids) == len(data.factor_types)
len(data.factor_constraint_ids) == len(data.factor_satisfied_pre)
len(data.eval_factor_constraint_ids) == len(data.eval_factor_satisfied_pre)
```

For `executable_factor` mode, assert backward compatibility:

```python
primary_constraint_id in data.factor_constraint_ids
data.primary_factor_index >= 0
```

### 5.2 Passive node test

For `passive_node` mode:

```python
hasattr(data, "passive_primary_node_index")
data.passive_primary_node_index not in data.factor_node_index
```

Also verify that no pressure edge originates from the passive primary node:

```python
edge_type in {4, 5, 6} should not have src == passive_primary_node_index
```

### 5.3 Query batching test

Create a small batch with at least two graphs with different parameter counts.

Assert:

```python
batch.primary_constraint_type_id.shape[0] == batch_size
batch.primary_constrained_property_id.shape[0] == batch_size
batch.primary_param_count.shape[0] == batch_size
sum(batch.primary_param_count) == len(batch.primary_param_predicate_ids)
```

Run a forward pass for:

- `query_definition`
- `query_family`
- `passive_node`

Assert model output includes normal edit logits:

```python
outputs["edit_logits"].shape == (batch_size, 6, num_target_ids)
```

### 5.4 Evaluation compatibility test

Create a graph where executable factors exclude the primary but `eval_factor_*` includes it. Assert strict global evaluation computes primary fix from `eval_primary_factor_index`, not from `primary_factor_index`.

### 5.5 Backward compatibility test

Run existing paper-surface tests:

```bash
uv run python tests/test_paper_surface.py
uv run python tests/test_paper_run_readiness.py
uv run python tests/test_factor_batching.py
```

Existing canonical configs should remain unchanged.

---

## 6. Artifact Preparation

If existing processed graph artifacts cannot support the new schema dynamically, materialize new graph artifacts for each primary mode.

Recommended graph artifact naming should include the primary mode to avoid overwriting canonical A1 graphs.

Example convention:

```text
data/processed/full_strat1m_minocc100/
  train_graph-node_id-primary_query_definition-shardNNN.pt
  test_graph-node_id-primary_query_definition-shardNNN.pt
  train_graph-node_id-primary_query_family-shardNNN.pt
  test_graph-node_id-primary_query_family-shardNNN.pt
  train_graph-node_id-primary_passive_node-shardNNN.pt
  test_graph-node_id-primary_passive_node-shardNNN.pt
```

If changing `graph_dataset_filename()` is too invasive, place new artifacts under separate derived variants:

```text
full_strat1m_minocc100_qdef
full_strat1m_minocc100_qfamily
full_strat1m_minocc100_primary_passive
```

The first option is cleaner if artifact discovery can be extended safely.

---

## 7. Training Plan

Do not run a hyperparameter search initially. Use the locked `A1` settings so the experiment isolates representation changes.

Recommended order:

```bash
uv run src/10_scheduler.py \
  --only a1_qfamily_secondary_factors \
  --paper-suite

uv run src/10_scheduler.py \
  --only a1_primary_passive_secondary_factors \
  --paper-suite

uv run src/10_scheduler.py \
  --only a1_qdef_secondary_factors \
  --paper-suite
```

Rationale:

1. `A1-QFamily` is the simplest query model and should reveal major implementation errors quickly.
2. `A1-PrimaryPassive` checks graph-side compatibility.
3. `A1-QDef` is the main model and has the richest query encoder.

If compute is limited, run `A1-QFamily` first as a smoke/feasibility model, then `A1-QDef`.

---

## 8. Evaluation Plan

Run standard strict evaluation for each model:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --strict-global-metrics \
  --per-constraint-csv
```

Run H2 sidecar evaluation:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --strict-global-metrics \
  --h2-eval
```

Run candidate-oracle analysis:

```bash
uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/<run_dir> \
  --strict-global-metrics \
  --output-dir models/<run_dir>/evaluations/oracle
```

---

## 9. Main Comparison Table

Compare the three new runs against existing `A1`, `B0`, and `M1D`.

| Model | Primary mode | Executable factors | Primary Fix | Micro-F1 | GFR / Local Satisfaction | SIR | SRR | Disruption | Focus deleted | Non-vacuous primary fix |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | passive baseline | none/local passive | existing | existing | existing | existing | existing | existing | existing | existing |
| `A1` | executable factor | primary + secondary | existing | existing | existing | existing | existing | existing | existing | existing |
| `M1D` | executable factor + direct safety | primary + secondary | existing | existing | existing | existing | existing | existing | existing | existing |
| `A1-QFamily` | family query | secondary only | new | new | new | new | new | new | new | new |
| `A1-PrimaryPassive` | passive primary node | secondary only | new | new | new | new | new | new | new | new |
| `A1-QDef` | definition query | secondary only | new | new | new | new | new | new | new | new |

---

## 10. H2 Diagnostics to Report

For the new query models, H2 needs slightly different interpretation.

### 10.1 Factor semantics

Because executable factors exclude the primary, H2 factor semantics now measure only secondary/context factors.

Report:

- F1;
- AUROC;
- AUPRC;
- ECE;
- support and positive rate.

The primary-vs-secondary field should be updated or replaced, because there is no primary executable factor in these runs.

### 10.2 Pressure masking

For these models:

- `no_factor_pressure` means “remove all secondary/context pressure.”
- `primary_only_pressure` is no longer meaningful because no primary executable factor exists.
- `secondary_only_pressure` should be equivalent or near-equivalent to normal, because all executable factors are secondary/context factors.

Recommended H2 adjustment:

```text
normal
no_context_factor_pressure
```

Old names can remain for compatibility, but the report should explicitly mark cases where `primary_factor_index == -1`.

### 10.3 Success pattern in H2

The main evidence for the hypothesis would be:

- removing context factor pressure from `A1-QDef` causes a larger safety-relevant change than secondary-only masking did in A1;
- `A1-QDef` improves SRR/GFR/non-vacuity relative to A1 without collapsing fidelity;
- factor pressure deltas affect safety metrics, not only Micro-F1 and primary fix.

---

## 11. Candidate-Oracle Analysis

For each new run, report:

- candidate set non-empty rate;
- candidate count mean;
- oracle safe available rate;
- selected safe rate;
- oracle non-vacuous safe available rate;
- selected non-vacuous safe rate;
- selected focus-deleted rate;
- selected vacuous satisfaction-improvement rate;
- selected GFR/SIR/SRR.

Key comparisons:

```text
selected_non_vacuous_safe_rate(A1-QDef) vs selected_non_vacuous_safe_rate(A1)
selected_srr(A1-QDef) vs selected_srr(A1)
selected_gfr(A1-QDef) vs selected_gfr(A1)
```

If oracle availability stays similar but selected-safe improves, the new representation improves selection behavior. If oracle availability is low and selected-safe stays low, the bottleneck is likely candidate generation or the imitation objective rather than primary-factor conflation.

---

## 12. Success Criteria

`A1-QDef` supports the hypothesis if it satisfies most of the following:

1. **Fidelity preservation:** Micro-F1 remains close to A1. A drop of a few points may be acceptable if safety improves clearly.
2. **Primary repair preservation:** primary fix remains close to A1/B0.
3. **Safety improvement over A1:** SRR decreases and/or GFR/SIR improves.
4. **Non-vacuity does not degrade:** focus deletion and vacuous satisfaction improvement do not increase materially.
5. **Control separation:** `A1-QDef` outperforms `A1-QFamily` if definition-level information matters.
6. **Query advantage:** `A1-QDef` outperforms `A1-PrimaryPassive` if explicit task-query conditioning matters.
7. **Mechanistic support:** H2 shows context factor pressure has measurable causal influence after the primary factor is removed from pressure execution.

Do not require the new model to beat `B0` on every safety metric for the experiment to be useful. The first result to look for is whether `A1-QDef` improves the A1 trade-off.

---

## 13. Failure Modes and Interpretation

### Case A: `A1-QDef` loses Micro-F1 and primary fix badly

Interpretation:

- The primary executable factor was providing essential task-intent information.
- The query encoder may be too weak or injected too late.

Next step:

- strengthen query conditioning;
- include separate focus subject/predicate/object summaries;
- add direct primary-fix loss only after the representation ablation is understood.

### Case B: `A1-QDef` matches A1 fidelity but safety does not improve

Interpretation:

- Primary-factor conflation is not the main bottleneck.
- Safety failure is likely due to edit-imitation objective, candidate generation, local metric mismatch, or weak secondary-actionability.

Next step:

- focus on decision-level safety objectives and candidate oracle gaps.

### Case C: `A1-QFamily` matches `A1-QDef`

Interpretation:

- Coarse constraint family may explain most repair intent.
- Definition parameters may be underused, noisy, or unnecessary for this dataset slice.

Next step:

- report family-only as a strong simple control;
- inspect family-specific results.

### Case D: `A1-PrimaryPassive` matches or beats `A1-QDef`

Interpretation:

- Passive graph-level primary information may be sufficient.
- Query injection might not be necessary, or current query encoder is flawed.

Next step:

- inspect whether passive primary node affects graph pooling disproportionately;
- test graph pooling with and without passive primary node.

### Case E: safety improves but non-vacuity worsens

Interpretation:

- The model may be moving toward satisfaction-by-deletion.

Next step:

- do not claim safe repair improvement;
- report this as another non-vacuity failure mode.

---

## 14. Optional Follow-up: Split Non-Primary Factors

If the main experiment is promising or ambiguous, add a later diagnostic split:

1. **Target-adjacent non-primary factors**
   - factors attached to the focus predicate;
   - factors attached to the constrained property;
   - factors sharing focus subject/object scope.

2. **True collateral factors**
   - non-primary factors not directly tied to the focus property or target violation.

Then add H2 masks:

```text
normal
no_all_context_pressure
target_adjacent_only_pressure
true_collateral_only_pressure
```

This tests whether the apparent “secondary” signal is actually redundant target-repair pressure distributed across local non-primary factors.

Do not implement this before the three-model ablation unless the first results are ambiguous and worth deepening.

---

## 15. Codex `/goal` Prompt

Use the following as the high-level Codex goal.

```text
Implement a primary-query ablation suite for the constraint_factors repository.

Goal: add three A1-style models that exclude the primary violated constraint from executable factor nodes while keeping it available as repair intent.

The three modes are:
1. query_definition: primary represented as a definition-conditioned query injected into the decoder; executable factors are secondary/local constraints only.
2. query_family: primary represented only by constraint family/type query; executable factors are secondary/local constraints only.
3. passive_node: primary represented as a passive graph node/metadata, with no pressure edges and no factor loss; executable factors are secondary/local constraints only.

Implementation requirements:
- Preserve all existing canonical behavior by default with primary_constraint_mode="executable_factor".
- Modify graph construction so factor_* fields refer to executable factors only, while eval_factor_* fields preserve the full local constraint set including the primary constraint for strict symbolic evaluation.
- For query modes, add primary query metadata tensors: primary_constraint_id, primary_constraint_type_id, primary_constrained_property_id, primary_param_predicate_ids, primary_param_object_ids, primary_param_count.
- For passive_node mode, add a passive_primary_node_index and ensure the passive primary node is not in factor_node_index and emits no factor-to-local pressure edges.
- Modify evaluation to prefer eval_factor_* and eval_primary_factor_index when present, falling back to factor_* for old runs.
- Add a primary query encoder to the model. query_family uses family/type embedding only. query_definition uses family/type, constrained property, pooled parameter predicates, pooled parameter objects, and optionally focus-role summary. Inject the query into the shared decoder representation.
- Add config generation flag --include-primary-query-ablations that emits the three configs with locked A1 settings.
- Add tests for graph schema, batching, passive node pressure exclusion, evaluation compatibility, and backward compatibility.
- Do not modify existing completed run artifacts or canonical configs.

After implementation, run tests, generate the new configs, train/evaluate the three runs with strict global metrics, run H2 sidecar evaluation, and run candidate-oracle analysis for each.
```

---

## 16. Minimal Command Sequence After Implementation

```bash
# 1. Run compatibility and new tests
uv run python tests/test_factor_batching.py
uv run python tests/test_paper_surface.py
uv run python tests/test_paper_run_readiness.py
uv run python tests/test_primary_query_graph_schema.py
uv run python tests/test_primary_query_model_forward.py
uv run python tests/test_primary_query_eval_support.py

# 2. Generate configs
uv run scripts/make_experiment_configs.py \
  --models-root models \
  --include-primary-query-ablations

# 3. Materialize graph artifacts if the new graph modes require separate files.
# Exact commands depend on the final CLI names implemented by Codex.

# 4. Train/evaluate the three runs
uv run src/10_scheduler.py --only a1_qfamily_secondary_factors --paper-suite
uv run src/10_scheduler.py --only a1_primary_passive_secondary_factors --paper-suite
uv run src/10_scheduler.py --only a1_qdef_secondary_factors --paper-suite

# 5. H2 sidecar evaluations
uv run src/09_eval.py \
  --run-directory models/a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --h2-eval

uv run src/09_eval.py \
  --run-directory models/a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --h2-eval

uv run src/09_eval.py \
  --run-directory models/a1_qdef_secondary_factors__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --h2-eval

# 6. Candidate-oracle analyses
uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --output-dir models/a1_qfamily_secondary_factors__full_strat1m_minocc100__node_id/evaluations/oracle

uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --output-dir models/a1_primary_passive_secondary_factors__full_strat1m_minocc100__node_id/evaluations/oracle

uv run scripts/analyze_candidate_oracle.py \
  --run-directory models/a1_qdef_secondary_factors__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --output-dir models/a1_qdef_secondary_factors__full_strat1m_minocc100__node_id/evaluations/oracle
```

---

## 17. Final Reporting Decision Rule

After all three runs finish, decide the paper role as follows:

| Outcome | Interpretation | Paper action |
|---|---|---|
| `A1-QDef` improves SRR/GFR/non-vacuity with modest fidelity loss | primary-query separation helps | Add as new model or appendix result; update narrative around intent/context separation. |
| `A1-QDef` matches A1 but no safety gain | primary executable factor was not the main bottleneck | Report as negative diagnostic; keep current fidelity-safety narrative. |
| `A1-QFamily` approximately matches `A1-QDef` | coarse family intent is enough | Simplify query story; definition conditioning not necessary. |
| `A1-PrimaryPassive` approximately matches `A1-QDef` | passive primary context is enough | Query injection not essential; pressure removal may be the key. |
| all query/passive models degrade primary fix heavily | primary executable factor is needed for task intent | Treat as evidence that better intent encoding is needed before removing primary pressure. |
| safety improves but focus deletion/vacuity worsens | unsafe satisfaction-by-deletion reappears | Do not claim safety improvement; emphasize non-vacuity. |

The experiment is valuable even if the new model fails, because it directly tests whether the primary constraint should be an executable factor or a separate repair-intent representation.

---

## 18. Short Paper-Narrative Update if the Experiment Works

If `A1-QDef` improves the A1 trade-off, the paper can add a sharper architectural lesson:

> Executable constraints are useful, but the primary violated constraint and local context constraints should not necessarily share the same computational role. The primary constraint is best understood as repair intent, while non-primary local constraints provide executable context for safety-aware pressure.

This would fit naturally into the current fidelity-safety-gap narrative without returning to the overstrong original claim that executable factors automatically solve safe repair.
