"""Resolve correction operations and reconstruct their authoritative pre-state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Triple = tuple[Any, Any, Any]


def resolve_other_entity_id(
    *,
    subject: Any,
    other_subject: Any,
    other_object: Any,
) -> Any:
    """Identify the entity described by the third serialized fact block.

    When the comparison statement shares the focus subject, the extra entity
    is its object. Otherwise the corpus builder assigns that block to the
    comparison subject.
    """

    if other_subject not in (None, "", 0):
        if other_subject == subject and other_object not in (None, "", 0):
            return other_object
        return other_subject
    if other_object not in (None, "", 0):
        return other_object
    return 0


@dataclass(frozen=True)
class ResolvedRowEdits:
    add: Triple | None
    delete: Triple | None

    @property
    def predicates(self) -> set[Any]:
        return {
            triple[1]
            for triple in (self.add, self.delete)
            if triple is not None and triple[1] not in (None, "", 0)
        }


def _resolve_value(
    value: Any,
    placeholder_map: dict[Any, Any],
    coerce_value: Callable[[Any], Any],
) -> Any:
    try:
        value = placeholder_map.get(value, value)
    except TypeError:
        pass
    return coerce_value(value)


def resolve_row_edits(
    row: Any,
    *,
    placeholder_map: dict[Any, Any],
    coerce_value: Callable[[Any], Any],
) -> ResolvedRowEdits:
    """Resolve add/delete slots, including identity-placeholder references."""

    def _resolve(kind: str) -> Triple | None:
        triple = tuple(
            _resolve_value(
                getattr(row, f"{kind}_{component}", 0),
                placeholder_map,
                coerce_value,
            )
            for component in ("subject", "predicate", "object")
        )
        if any(value in (None, "", 0) for value in triple):
            return None
        return triple

    return ResolvedRowEdits(add=_resolve("add"), delete=_resolve("del"))


def normalize_pre_edit_state(
    facts_by_entity: dict[Any, dict[Any, set[Any]]],
    predicates_present: dict[Any, set[Any]],
    edits: ResolvedRowEdits,
) -> None:
    """Make correction operations authoritative over serialized entity facts.

    Entity descriptions can reflect a later snapshot and therefore already
    contain an addition or omit a deletion. A correction row defines the
    transition: additions are absent before the edit and deletions are present.
    """

    if edits.add is not None:
        subject, predicate, obj = edits.add
        entity_facts = facts_by_entity.setdefault(subject, {})
        entity_facts.setdefault(predicate, set()).discard(obj)
        predicates_present.setdefault(subject, set()).add(predicate)

    if edits.delete is not None:
        subject, predicate, obj = edits.delete
        facts_by_entity.setdefault(subject, {}).setdefault(predicate, set()).add(obj)
        predicates_present.setdefault(subject, set()).add(predicate)
