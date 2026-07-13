# 05_constraint_labeler.py

## Objective
- Generate per-factor constraint labels (checkable + satisfied) for the local or focus constraint neighborhood.
- Produce both **pre-edit** and **post-gold-edit** labels without rebuilding graphs.
- Track coverage and emit per-type summaries and coverage reports.

For the paper-facing run, `--constraint-scope local` is the canonical setting. `focus` remains supported for exploratory or appendix work only.

## Inputs & Outputs
**Inputs**
- Parquet split file(s) produced by `02_dataframe_builder.py` (from `data/interim/<dataset_variant>`).
- Constraint registry from `03_constraint_registry.py` (`data/interim/constraint_registry_<dataset>.parquet`).
- Identity encoder (`identity_encoder.txt`) for registry and evidence semantics.
- Training split entity descriptions, from which the labeler freezes the
  benchmark's direct `P279` class graph.

**Outputs**
- Labeled parquet files under `data/interim/<dataset_variant>_labeled/` with additional columns:
  - `factor_checkable_pre`, `factor_satisfied_pre`
  - `factor_checkable_post_gold`, `factor_satisfied_post_gold`
  - `factor_types` (constraint type ids, aligned with `factor_constraint_ids`)
  - `factor_constraint_ids` (the constraint ids evaluated for the row)
  - `num_checkable_factors_pre`, `coverage_pre`
  - `num_checkable_factors_post_gold`, `coverage_post_gold`
  - `primary_factor_index`, `primary_checkable_pre`, `primary_satisfied_pre`
  - `primary_checkable_post_gold`, `primary_satisfied_post_gold`
  - `primary_validation_reason`
  - `primary_gold_repair_status`, `primary_gold_repair_verified`
- Coverage reports in the same output folder:
  - `coverage_<scope>.csv`
  - `coverage_<scope>.md`
- Filtered factor reports in the same output folder:
  - `filtered_factors_<scope>.csv`
  - `filtered_factors_<scope>.md`
  - `filtered_factor_families_<scope>.csv`
- `primary_validation_audit.csv`, `primary_validation_audit_by_constraint.csv`
- `primary_gold_repair_audit_by_constraint.csv`
- `class_hierarchy.parquet`, `class_hierarchy_manifest.json`
- an updated schema-v2 dataset manifest

By default, `--factor-family-policy supported_only` writes only executable
supported constraints into `factor_constraint_ids` and aligned label arrays.
Unsupported secondary constraints remain in `local_constraint_ids` for
auditability but are not emitted as supervised factor nodes. For paper data,
use `--filter-invalid-primary`: exempt, out-of-scope, uncheckable, unsupported,
and already-satisfied primary rows are excluded and counted by reason.

Primary eligibility and gold-repair verification are intentionally separate.
`primary_validation_reason=valid` means PRE is an eligible observed violation.
It does not assert that the historical correction is a complete repair.
`primary_gold_repair_status` records `verified`, `post_uncheckable`, or
`post_unsatisfied` for retained rows. This preserves the exact-size correction
benchmark while exposing the semantic success rate of its observed edits.

## Evidence Model
The labeler builds a normalized evidence structure per row:
```
facts_by_entity: Dict[entity_id, Dict[predicate_id, Set[object_id]]]
```
where entity/predicate/object IDs match the representation found in the parquet.
The third entity-description block belongs to `other_object` when the
comparison statement shares the focus subject; otherwise it belongs to
`other_subject`. The labeler, graph builder, hierarchy builder, and evaluator
use the same ownership rule.

### P_local
`P_local` is the union of:
- `predicate`, `other_predicate`
- all predicate IDs appearing in `subject_predicates`, `object_predicates`, `other_entity_predicates`
- the resolved `add_predicate` and `del_predicate`, when present

Facts are restricted to `P_local` to ensure local-closure compatibility.

### Completeness Assumptions
We cannot directly observe whether all statements for an entity-property pair are present, so we use a conservative proxy:
- If `--assume-complete-entity-facts` (default), treat the entity facts blob as complete for all properties in scope.
- If `--no-assume-complete-entity-facts`, only treat properties explicitly present in the facts blob as complete.

For **single**, we additionally require:
- at least one statement for `(subject, P, *)`, and
- completeness for `(subject, P, *)`.

## Gold Edit Application
Two states are evaluated:
- **PRE**: serialized facts normalized to the transition declared by the row.
  Any gold addition is removed and any gold deletion is restored first. This
  prevents later entity snapshots from leaking the correction into PRE.
- **POST_GOLD**: apply `add_*` and `del_*` edits to the facts representation.

Edits are resolved through placeholder tokens (`subject`, `predicate`, `object`, `other_*`) when present.
If an edit references an entity/property/value outside the local evidence structure, the corresponding
constraint checks are marked **not checkable** (conservative).

## Type Hierarchy

Wikidata type constraints permit subclasses of an allowed class. Semantics v4
therefore evaluates `P31`/`P279` class values through the reflexive-transitive
closure of `P279`. The hierarchy is the sorted union of direct `P279` facts in
the upstream **training split's entity descriptions**. Validation and test
contexts never contribute edges.

This follows Wikidata's documented semantics for
[subject-type constraints](https://www.wikidata.org/wiki/Help:Property_constraints_portal/Subject_class/en)
and [value-type constraints](https://www.wikidata.org/wiki/Help:Property_constraints_portal/Value_class/en),
including the required `P2309` relation mode and subclass-path matching.

The labeler writes the direct edges to `class_hierarchy.parquet` and records
the source split, source-manifest hash, predicate identity, edge count, and
artifact hash in `class_hierarchy_manifest.json`. Derived samplers copy both
files unchanged, and global evaluation loads the same artifact. Missing paths
are absent under this frozen benchmark operationalization; it is not a claim
of complete historical or current Wikidata taxonomy coverage.

## Constraint Types Implemented (semantics v4)
Per-type checkability and satisfaction are implemented in `src/modules/constraint_checkers.py`.
Registry parsing is centralized in `src/modules/constraint_semantics.py` and is
shared with reranker/global evaluation. `P2306` is interpreted by family
(inverse property, required property, or conflicting property), `P2305` carries
allowed/qualifying items, `P2303` exceptions disable affected rows, and `P4680`
scope is honored. `P2309` relation-mode items are translated to executable
predicates: `Q21503252 -> P31`, `Q21514624 -> P279`, and
`Q30208840 -> {P31, P279}`. They are never treated as predicates themselves.
Direct class values are then matched against the frozen `P279*` closure.
Because `P2309` is mandatory for these constraint families, a missing or
unknown relation mode is uncheckable rather than guessed.
The focus and comparison triples are explicitly seeded into the evidence state
before applying edits.
Canonical constraint-family names come from the registry (`constraint_family`), generated via the
static catalog in `data/static/constraint_type_catalog.json`. On a fresh clone,
`03_constraint_registry.py` bootstraps that catalog automatically if it is missing.
- `conflictWith`
- `inverse`
- `symmetric`
- `itemRequiresStatement`
- `valueRequiresStatement`
- `oneOf`
- `single`
- `type`
- `valueType`
- `distinct`

Semantics follow the short descriptions in [docs-conceptual/constraint_types.md](/home/mvazquez/constraint_factors/docs-conceptual/constraint_types.md). When evidence is insufficient, the factor is marked **not checkable** to prioritize correctness over coverage.

## Coverage Summary
At the end of a run, the script prints a per-type summary including:
- checkable rate (pre / post)
- satisfied rate (pre / post)

Use this report to tune completeness assumptions and identify constraint types with weak coverage.
The aggregate coverage table includes every attached factor and therefore is
not a primary-task retention table. Use
`primary_validation_audit_by_constraint.csv` to verify that each source primary
family has retained checkable violations before sampling. Use
`primary_gold_repair_audit_by_constraint.csv` to report the fraction of
observed corrections that are verified primary repairs; do not infer this from
aggregate attached-factor coverage.

In the full corpus, `symmetric` can appear in aggregate attached-factor
coverage but has no primary correction file. The primary audit therefore has
nine source families while the attached-factor coverage has ten executable
families.

## CLI
Example usage:
```bash
python src/05_constraint_labeler.py \
  --dataset sample \
  --min-occurrence 100 \
  --constraint-scope local \
  --factor-family-policy supported_only \
  --filter-invalid-primary \
  --overwrite
```

Key flags:
- `--constraint-scope {local,focus}` selects `local_constraint_ids` vs `local_constraint_ids_focus`.
  The paper default is `local`.
- `--factor-family-policy {supported_only,all}` controls factor supervision.
  The paper default is `supported_only`; use `all` only to reproduce the older
  all-attached-factor behavior.
- `--registry-dataset` selects the raw dataset registry to use for derived variants such as `full_strat1m`; use `--registry-dataset full` for the paper benchmark.
- `--output-dataset` writes a standalone labeled variant; without it the legacy `<variant>_labeled` path is used.
- `--filter-invalid-primary` is mandatory for paper-facing reruns.
- `--assume-complete-entity-facts/--no-assume-complete-entity-facts` toggles completeness assumptions.
- `--max-rows` caps rows per parquet for debugging.
