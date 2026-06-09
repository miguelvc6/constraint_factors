# Executable Constraint Factors: Message Passing and Constraint Execution

This document specifies the conceptual message-passing equations and computational semantics of constraint factors as executable subprograms, aligned with the current factorized implementation.

It is the architecture-specification companion to [docs-conceptual/00-constraint_factors.md](/home/mvazquez/constraint_factors/docs-conceptual/00-constraint_factors.md). Code-level details are split across [docs-technical/06_graph.md](/home/mvazquez/constraint_factors/docs-technical/06_graph.md), [docs-technical/07_train.md](/home/mvazquez/constraint_factors/docs-technical/07_train.md), and [docs-technical/00_model_config_reference.md](/home/mvazquez/constraint_factors/docs-technical/00_model_config_reference.md).

The current model does not execute hard-coded symbolic constraint checkers inside the GNN forward pass. Instead, symbolic executors create per-factor satisfaction labels offline, and the factorized model learns type-specific neural executors and pressure modules from those labels.

The goal is to make explicit:
- how constraint factor scopes are represented,
- how factor satisfaction is predicted,
- how factor states generate pressure,
- how this pressure interacts with GNN message passing,
- and how repair decisions emerge without hardcoded repair heuristics.

This conceptual specification does not assume that every decision-time safety quantity must be produced by a fully neural post-edit rollout. The main research plan remains compatible with symbolic candidate evaluation at repair-selection time, as long as executable factors shape the proposal model itself.

---

## 1. Graph Structure and Notation

We work with a heterogeneous factor graph per violation instance.

### Node sets

- Variable nodes $v \in \mathcal{V}$
  - entities, predicates, literals, and role-specific predicate occurrences
- Constraint factor nodes $c \in \mathcal{C}$
  - each corresponds to a constraint instance
  - each has a dense constraint type id $t(c) \in \{0,\ldots,|\mathcal{T}|-1\}$

### Factor scope roles

The implementation does not store each factor scope as one ordered tuple with fixed arity. Instead, graph construction adds typed factor-to-variable edges:

$$
\mathcal{P}(c) = \{v : c \rightarrow v \text{ has role predicate}\}
$$

$$
\mathcal{S}(c) = \{v : c \rightarrow v \text{ has role subject}\}
$$

$$
\mathcal{O}(c) = \{v : c \rightarrow v \text{ has role object}\}
$$

The conceptual scope is therefore:

$$
\mathrm{scope}(c) =
\mathcal{P}(c) \cup \mathcal{S}(c) \cup \mathcal{O}(c)
$$

These sets can be empty, singleton, or multi-node. This is important for constraints such as `single`, where many local predicate or object occurrences can participate, and for duplicated predicate nodes, where each statement has its own predicate occurrence.

For `conflictWith`, the predicate role may include local predicate occurrences matching the constrained property and the conflicting property. The subject role is scoped around the focus subject, and object role edges are added for local triples using the constrained property. It is therefore not generally represented as a fixed tuple $(p,q,s)$.

---

## 2. Base Variable Message Passing

Let $h_v^{(k)} \in \mathbb{R}^d$ be the embedding of node $v$ after layer $k$. Initial node features are either learned node-id embeddings or precomputed text features, optionally concatenated with focus-role embeddings, then projected into the hidden dimension.

For a GIN/GINE backbone layer:

$$
\tilde{h}_v^{(k+1)} =
\mathrm{GNN}_k\Big(
h_v^{(k)},
\{(h_u^{(k)}, e_{u\to v}) : u \in \mathcal{N}(v)\}
\Big)
$$

This captures structural and semantic context from the materialized graph. In the pressure-enabled model, factor pressure is applied after each backbone message-passing layer.

---

## 3. Factor Execution: Satisfaction Prediction

Factor execution is type-specific and neural. The current default implementation is `per_type_v1`.

### 3.1 Role summaries

For each role $r \in \{P,S,O\}$, define the mean summary:

$$
\bar{h}_{c,r}^{(k)} =
\frac{1}{\max(1, n_{c,r})}
\sum_{v \in \mathcal{R}_r(c)}
h_v^{(k)}
$$

where:

$$
n_{c,r} = |\mathcal{R}_r(c)|
$$

and:

$$
\mathcal{R}_P(c)=\mathcal{P}(c),\quad
\mathcal{R}_S(c)=\mathcal{S}(c),\quad
\mathcal{R}_O(c)=\mathcal{O}(c)
$$

If a role has no scoped variables, its summary is the zero vector and its count is zero.

### 3.2 Factor input

The factor executor input is:

$$
\phi_c^{(k)} =
\Big[
h_c^{(k)}
\;\Vert\;
\bar{h}_{c,P}^{(k)}
\;\Vert\;
\bar{h}_{c,S}^{(k)}
\;\Vert\;
\bar{h}_{c,O}^{(k)}
\;\Vert\;
\log(1+n_{c,P})
\;\Vert\;
\log(1+n_{c,S})
\;\Vert\;
\log(1+n_{c,O})
\Big]
$$

Thus:

$$
\phi_c^{(k)} \in \mathbb{R}^{4d+3}
$$

This replaces the older fixed-arity sketch:

$$
z_c^{(k)} =
\big[
h_p^{(k)} \;\Vert\; h_q^{(k)} \;\Vert\; h_s^{(k)}
\big]
$$

That sketch is incomplete for the current code because it omits the factor-node embedding $h_c^{(k)}$, object-role evidence, and role counts, and it assumes there is exactly one node per role.

### 3.3 Type-specific executor

Each factor type has its own executor:

$$
(a_c^{(k)}, \ell_c^{\mathrm{pre},(k)})
=
F_{t(c)}(\phi_c^{(k)})
$$

where:
- $a_c^{(k)} \in \mathbb{R}^{d_f}$ is the learned factor state,
- $\ell_c^{\mathrm{pre},(k)} \in \mathbb{R}$ is a pre-edit satisfaction logit.

The predicted pre-edit satisfaction probability is:

$$
\hat{s}_c^{\mathrm{pre},(k)}
=
\sigma(\ell_c^{\mathrm{pre},(k)})
$$

The implementation trains this logit against `factor_satisfied_pre` with binary cross-entropy. Positive logits predict satisfaction; negative logits predict violation.

A violation score can be derived as:

$$
\widehat{\mathrm{viol}}_c^{(k)}
=
1 - \hat{s}_c^{\mathrm{pre},(k)}
$$

but the `per_type_v1` pressure path does not directly pass this scalar into the pressure modules.

### 3.4 Post-gold satisfaction head

During training and evaluation with gold edit labels, the model also predicts post-gold satisfaction. Let $e_y$ be the mean embedding of the six gold edit target ids for the graph containing factor $c$. Then:

$$
\ell_c^{\mathrm{post},(k)}
=
Q_{t(c)}\Big(a_c^{(k)}, e_y\Big)
$$

and:

$$
\hat{s}_c^{\mathrm{post},(k)}
=
\sigma(\ell_c^{\mathrm{post},(k)})
$$

This is trained against `factor_satisfied_post_gold`. It is a supervised auxiliary head, not a full neural symbolic rollout of arbitrary candidate repairs.

---

## 4. Constraint-to-Variable Feedback: Factor Pressure

Constraint pressure is enabled by `GIN_PRESSURE` with `pressure_enabled=true` and factorized graph representation.

For the current `per_type_v1` path, pressure messages are computed after the backbone layer has produced $\tilde{h}^{(k+1)}$. The model rebuilds the factor scope features from those post-GNN states:

$$
\tilde{\phi}_c^{(k+1)}
=
\Phi_c\left(\tilde{h}^{(k+1)}\right)
$$

and runs the type-specific executor:

$$
(\tilde{a}_c^{(k+1)}, \tilde{\ell}_c^{(k+1)})
=
F_{t(c)}(\tilde{\phi}_c^{(k+1)})
$$

The pressure message then uses the factor state and destination variable embedding:

$$
m_{c \rightarrow v,r}^{(k)}
=
G_{t(c),r}
\Big(
\big[
\tilde{a}_c^{(k+1)} \;\Vert\; \tilde{h}_v^{(k+1)}
\big]
\Big)
$$

where:
- $r \in \{P,S,O\}$ is the factor edge role,
- $G_{t,r}$ is a role-conditioned pressure MLP,
- by default pressure modules are separate per factor type and role,
- the H2 shared-pressure ablation shares pressure modules across factor types.

This means pressure is learned from the internal factor state, not from a hand-written negative vector such as:

$$
-\alpha \cdot \mathrm{viol}_c \cdot h_v
$$

That negative-feedback expression is only an intuition; it is not the equation implemented by the current model.

### Aggregation and residual update

For each variable node $v$, aggregate all incoming factor-pressure messages and normalize by the number of pressure edges into $v$:

$$
p_v^{(k)}
=
\frac{1}{\max(1,d_v)}
\sum_{(c,r):\, v \in \mathcal{R}_r(c)}
m_{c \rightarrow v,r}^{(k)}
$$

where:

$$
d_v =
\sum_{(c,r)}
\mathbf{1}\{v \in \mathcal{R}_r(c)\}
$$

The final layer update is:

$$
h_v^{(k+1)}
=
\tilde{h}_v^{(k+1)}
+
\lambda p_v^{(k)}
$$

where $\lambda$ is `pressure_residual_scale`.

---

## 5. Legacy Shared Executor Path

The code still supports `factor_executor_impl=legacy_shared` for ablations. That path differs from the current paper-facing factor executor:

$$
\ell_c^{\mathrm{pre}}
=
H\Big(
\mathrm{Proj}(h_c)
\;\Vert\;
\tau_{t(c)}
\Big)
$$

when factor type embeddings $\tau_{t(c)}$ are enabled.

Its pressure path can form a learned scalar from the factor-node embedding:

$$
u_c = \sigma(w^\top h_c)
$$

and feed:

$$
\big[
h_c \;\Vert\; h_v \;\Vert\; \rho_r \;\Vert\; u_c \;\Vert\; \tau_{t(c)}
\big]
$$

into a shared pressure MLP. This is retained as a legacy/ablation behavior and should not be used as the main equation for the current factorized model.

---

## 6. Decoder and Repair Behavior

After message passing and optional pressure injection, the model pools node states:

$$
h_G = \mathrm{meanpool}_{v \in G}(h_v)
$$

Then separate subject, predicate, and object branches produce six slot distributions:

$$
\hat{y}
=
\big(
\hat{y}_{add,s},
\hat{y}_{add,p},
\hat{y}_{add,o},
\hat{y}_{del,s},
\hat{y}_{del,p},
\hat{y}_{del,o}
\big)
$$

There is no explicit "delete focus" rule inside the model. Repair behavior is learned through:
- ordinary repair imitation loss on the six edit slots,
- optional factor satisfaction losses on pre and post-gold factor labels,
- optional factor pressure injected into node states,
- optional candidate chooser or direct safety objectives in the broader training pipeline.

---

## 7. Multi-Constraint Interaction

Because a graph can contain multiple factors, one variable can receive pressure from several constraints:

$$
p_v^{(k)}
\propto
\sum_{c : v \in \mathrm{scope}(c)}
m_{c \rightarrow v}^{(k)}
$$

The implemented aggregation averages these messages by pressure degree before applying the residual scale. Multiple constraints can therefore reinforce each other, compete, or produce mixed pressure signals through the learned message functions.

This interaction is the key difference between:
- flattened graphs with passive constraint context, and
- factorized graphs with supervised factor states and pressure feedback.

---

## 8. Flattened Graphs vs Factorized Pressure

| Aspect | Flattened graph | Current factorized pressure model |
|---|---|---|
| Constraint node | Passive context | Factor node with typed scope edges |
| Factor execution | None, or shared passive head | Type-specific neural executor over role summaries |
| Satisfaction signal | Not explicitly supervised | `factor_satisfied_pre` and `factor_satisfied_post_gold` logits |
| Pressure | None | Role- and type-conditioned residual messages |
| Scope arity | Implicit in graph topology | Variable-size role sets with mean summaries and counts |
| Repair logic | Pattern imitation | Proposal model shaped by factor states, pressure, and optional safety objectives |
