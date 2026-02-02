from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .repair_eval import ViolationContext


POLICY_NOOP = 0
POLICY_DELETE_FOCUS = 1
POLICY_DELETE_CONFLICT = 2
POLICY_ADD_VALUE_FOCUS_PREDICATE = 3
POLICY_CHANGE_PREDICATE = 4
POLICY_OTHER = 5

POLICY_NAMES: tuple[str, ...] = (
    "NOOP",
    "DELETE_FOCUS_TRIPLE",
    "DELETE_CONFLICT_TRIPLE",
    "ADD_VALUE_TO_FOCUS_PREDICATE",
    "CHANGE_PREDICATE",
    "OTHER",
)


@dataclass(frozen=True)
class PolicyDecision:
    policy_id: int
    policy_name: str


def _is_triple(values: Sequence[int], none_class: int) -> bool:
    return all(int(v) != none_class for v in values)


def derive_policy_label(
    candidate: Sequence[int],
    context: ViolationContext,
    *,
    none_class: int = 0,
) -> PolicyDecision:
    if len(candidate) != 6:
        raise ValueError("Candidate must have 6 slots.")
    add = candidate[:3]
    delete = candidate[3:6]

    if not _is_triple(add, none_class) and not _is_triple(delete, none_class):
        return PolicyDecision(POLICY_NOOP, POLICY_NAMES[POLICY_NOOP])

    if _is_triple(delete, none_class):
        focus = (context.subject, context.predicate, context.object)
        if tuple(delete) == focus:
            return PolicyDecision(POLICY_DELETE_FOCUS, POLICY_NAMES[POLICY_DELETE_FOCUS])
        other = (context.other_subject, context.other_predicate, context.other_object)
        if context.other_predicate != none_class and tuple(delete) == other:
            return PolicyDecision(POLICY_DELETE_CONFLICT, POLICY_NAMES[POLICY_DELETE_CONFLICT])

    if _is_triple(add, none_class):
        if int(add[1]) == int(context.predicate):
            return PolicyDecision(
                POLICY_ADD_VALUE_FOCUS_PREDICATE,
                POLICY_NAMES[POLICY_ADD_VALUE_FOCUS_PREDICATE],
            )
        if int(add[0]) == int(context.subject) and int(add[1]) != int(context.predicate):
            return PolicyDecision(POLICY_CHANGE_PREDICATE, POLICY_NAMES[POLICY_CHANGE_PREDICATE])

    return PolicyDecision(POLICY_OTHER, POLICY_NAMES[POLICY_OTHER])


def candidate_matches_policy(
    candidate: Sequence[int],
    policy_id: int,
    context: ViolationContext,
    *,
    none_class: int = 0,
) -> bool:
    decision = derive_policy_label(candidate, context, none_class=none_class)
    if policy_id == POLICY_OTHER:
        return True
    return decision.policy_id == policy_id


def filter_candidates_by_policy(
    candidates: Sequence[Sequence[int]],
    policy_id: int,
    context: ViolationContext,
    *,
    strict: bool,
    none_class: int = 0,
) -> tuple[list[Sequence[int]], list[bool]]:
    mask: list[bool] = []
    filtered: list[Sequence[int]] = []
    for cand in candidates:
        match = candidate_matches_policy(cand, policy_id, context, none_class=none_class)
        mask.append(match)
        if match:
            filtered.append(cand)
    if strict:
        return (filtered if filtered else list(candidates)), mask
    return list(candidates), mask
