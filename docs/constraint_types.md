# Wikidata Constraint Types — Brief Guide


## Typing (what kinds of things subjects/values may be)

### `type`

Constrains the **subject** that uses a property to be an instance/subclass of one or more specified classes.
*Example:* items with `date of birth (P569)` should be `human (Q5)` or similar living-being classes.

### `valueType`

Constrains the **value** of a property to be of certain classes (for item-valued properties) or datatypes (for literals).
*Example:* values of `mother (P25)` must be items of class `human (Q5)`.

---

## Dependency (what else must be stated)

### `itemRequiresStatement`

If an item has a given property, it **must also** have another specified statement.
*Example:* if an item has `country of citizenship (P27) = Austria`, it must also have `place of birth (P19)`.

### `valueRequiresStatement`

For each **target value** used with a property, that **value item itself** must carry another statement.
*Example:* if `employer (P108)` points to organization *X*, then *X* must have `instance of (P31) = organization`.

### `inverse`

If item **A** uses property **P** to point to item **B**, then **B** should have the **inverse** property **P′** pointing back to **A**.
*Example:* `parent (P40)` on child implies `child (P40)` (or `has child`) on the parent item, depending on modeling.

---

## Value domain / enumeration

### `oneOf`

Restricts the value to be **one of an explicit, finite list** of allowed items (a closed set).
*Example:* `sex or gender (P21)` value must be one of a curated list (as defined in the constraint).

---

## Cardinality & Uniqueness

### `single`

An item should have **at most one** statement for the property (no multiple values).
*Example:* `date of birth (P569)` is expected to appear once per item.

### `distinct`

All values of the property must be **globally unique across items** (no two items share the same value).
*Example:* external IDs like `VIAF ID (P214)` must be distinct.

---

## Consistency / Mutual exclusion

### `conflictWith`

Two properties (or specific value patterns) **should not co-occur** on the same item; if one is present, the other must be absent.
*Example:* `date of birth (P569)` conflicting with `year of birth missing (Q...)`-style markers, or mutually exclusive status flags.


## Repair-aware Evaluation

When `src/05_eval.py` finds the cached violation contexts under `data/interim/<variant>/`, it automatically augments
the standard precision/recall scores with a report that checks whether the predicted triple would **resolve the violation**
even when it differs from the gold target. Every sample is classified per action (`add`, `del`) as:

- `exact` – prediction matches the gold triple (including the "no-op" case where both are empty);
- `alternative_fix` – prediction differs from gold but matches one of the repair heuristics below;
- `missing_prediction` – a gold repair exists but the model emitted `NONE`;
- `no_fix` – the predicted triple neither matches gold nor a recognised repair.

The heuristics inspect the parquet metadata for the row (focus triple, "other" triple, constraint parameters)
and accept the following alternative repairs:

- **All constraint types** – deleting the focus triple; deleting the `other_*` triple when the violation
  references a conflicting statement.
- **conflictWith / single / oneOf** – additionally credit deletions of mixed triples such as
  `(subject, predicate, other_object)`, `(subject, other_predicate, object)`, or
  `(other_subject, predicate, object)` depending on whether the clash is per-value or per-property.
- **distinct** – deleting the duplicate `(other_subject, predicate, object)` triple belonging to the competing
  item.
- **oneOf additions** – adding `(subject, predicate, value)` where `value` is listed in the constraint’s
  `items (P2305)` parameter (either via the literal predicate ID or the `predicate` placeholder).
- **type additions** – adding `(subject, relation, class)` where `class` comes from `class (P2308)` and
  `relation` is either the constraint’s `relation (P2309)` parameter or the defaults
  `instance of (P31)` / `subclass of (P279)`.
- **valueType additions** – same as `type` but anchored on the object placeholder, i.e. adding statements
  about the value entity.
- **itemRequiresStatement additions** – adding `(subject, property, ANY)` whenever `property` appears in the
  constraint’s `property (P2306)` list; any value satisfies the requirement.
- **valueRequiresStatement additions** – adding `(object, property, ANY)` for every required property in
  `P2306`, thereby satisfying obligations on the target item.
- **inverse additions** – adding `(object, inverse_predicate, subject)` where `inverse_predicate` is taken
  from `inverse property (P1696)` or, if unspecified, defaults to the original predicate (covering symmetric
  properties).

The resulting JSON (`repair_metrics` inside `models/<run>/evaluations/model.json`) reports counts and rates
per action and per constraint type so you can see whether a model is at least producing a valid repair even
when it misses the exact gold edit.
