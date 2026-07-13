from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch_geometric.data import Data

from .repair_eval import ConstraintRepairHeuristics, ViolationContext


NUM_SLOTS = 6
NONE_CLASS_INDEX = 0


@dataclass(frozen=True)
class CandidateConfig:
    topk_candidates: int = 20
    topk_per_slot: int = 5
    heuristic_max_candidates: int = 30
    heuristic_max_values: int = 3
    force_include_gold_train: bool = True
    # Legacy config input. Inference APIs ignore this field unconditionally.
    include_gold: bool | None = None
    max_candidates_total: int = 80

    @property
    def include_gold_during_training(self) -> bool:
        if self.include_gold is not None:
            return bool(self.include_gold)
        return bool(self.force_include_gold_train)


@dataclass(frozen=True)
class RepairCandidate:
    identity_slots: tuple[int, int, int, int, int, int]
    feature_slots: tuple[int, int, int, int, int, int]
    source: str


def candidate_feature_tuples(candidates: Sequence[RepairCandidate]) -> list[tuple[int, int, int, int, int, int]]:
    return [candidate.feature_slots for candidate in candidates]


def candidate_identity_tuples(candidates: Sequence[RepairCandidate]) -> list[tuple[int, int, int, int, int, int]]:
    return [candidate.identity_slots for candidate in candidates]


def gold_candidate(graph: Data) -> tuple[int, int, int, int, int, int]:
    y = getattr(graph, "y", None)
    if y is None:
        raise ValueError("Graph missing y target tensor.")
    if y.dim() == 2:
        y = y[0]
    return tuple(int(v) for v in y.tolist())


def _coerce_gold_candidate(
    *,
    graph: Data | None,
    gold_slots: Sequence[int] | None,
) -> tuple[int, int, int, int, int, int]:
    if gold_slots is not None:
        if len(gold_slots) != NUM_SLOTS:
            raise ValueError(f"Expected gold_slots length {NUM_SLOTS}, got {len(gold_slots)}")
        return tuple(int(v) for v in gold_slots)
    if graph is None:
        raise ValueError("Either graph or gold_slots must be provided to build_candidates().")
    return gold_candidate(graph)


def candidate_from_triple(triple: tuple[int, int, int], *, action: str) -> tuple[int, int, int, int, int, int]:
    if action == "add":
        return (triple[0], triple[1], triple[2], 0, 0, 0)
    return (0, 0, 0, triple[0], triple[1], triple[2])


def _select_values(
    values: Iterable[int] | None, *, placeholder_ids: set[int], none_class: int, max_values: int
) -> list[int]:
    if not values:
        return []
    unique = []
    seen: set[int] = set()
    for value in values:
        if value in (none_class, None):
            continue
        if value in placeholder_ids:
            continue
        if value in seen:
            continue
        seen.add(int(value))
        unique.append(int(value))
        if len(unique) >= max_values:
            break
    return unique


def _instantiate_patterns(
    patterns,
    *,
    placeholder_ids: set[int],
    none_class: int,
    max_values: int,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for pattern in patterns:
        subj_vals = _select_values(pattern.subjects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        pred_vals = _select_values(pattern.predicates, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        obj_vals = _select_values(pattern.objects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        if not subj_vals or not pred_vals or not obj_vals:
            continue
        for s in subj_vals:
            for p in pred_vals:
                for o in obj_vals:
                    candidates.append((s, p, o))
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates


def _topk_triples_from_logits(
    logits: torch.Tensor,
    *,
    slots: tuple[int, int, int],
    topk_triples: int,
    topk_per_slot: int,
    slot_allowed_ids: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None,
) -> list[tuple[int, int, int]]:
    slot_vals = []
    slot_ids = []
    for local_idx, slot in enumerate(slots):
        allowed_ids = None if slot_allowed_ids is None else slot_allowed_ids[local_idx]
        if allowed_ids is None:
            k = max(1, min(topk_per_slot, logits.size(-1)))
            vals, ids = torch.topk(logits[slot], k=k)
        else:
            allowed = allowed_ids
            if allowed.device != logits.device:
                allowed = allowed.to(device=logits.device)
            if allowed.dtype != torch.long:
                allowed = allowed.to(dtype=torch.long)
            if allowed.numel() >= topk_per_slot and allowed.numel() > 0:
                restricted = logits[slot].index_select(0, allowed)
                k = max(1, min(topk_per_slot, restricted.size(0)))
                vals_local, idx_local = torch.topk(restricted, k=k)
                ids = allowed.index_select(0, idx_local)
                vals = vals_local
            else:
                # Preserve legacy behavior when the allowed-id set is too small:
                # old logic would still return top-k over full vocabulary.
                k = max(1, min(topk_per_slot, logits.size(-1)))
                vals, ids = torch.topk(logits[slot], k=k)
        slot_vals.append(vals.cpu())
        slot_ids.append(ids.cpu())
    if len(slot_vals) != 3:
        return []
    k_combos = min(len(slot_vals[0]), len(slot_vals[1]), len(slot_vals[2]))
    if k_combos <= 0:
        return []
    combos: list[tuple[float, int, int, int]] = []
    for i in range(k_combos):
        for j in range(k_combos):
            for k in range(k_combos):
                score = float(slot_vals[0][i] + slot_vals[1][j] + slot_vals[2][k])
                combos.append((score, int(slot_ids[0][i]), int(slot_ids[1][j]), int(slot_ids[2][k])))
    combos.sort(key=lambda x: x[0], reverse=True)
    return [(s, p, o) for _, s, p, o in combos[:topk_triples]]


def _identity_to_feature_value(value: int, mapping: Sequence[int] | None) -> int:
    if mapping is None:
        return int(value)
    if value < 0 or value >= len(mapping):
        return -1
    return int(mapping[value])


def _feature_to_identity_map(
    mapping: Sequence[int] | None,
    *,
    num_target_ids: int,
) -> list[int]:
    if mapping is None:
        return list(range(num_target_ids))
    inverse = [-1] * num_target_ids
    ambiguous: set[int] = set()
    for identity_id, feature_id_raw in enumerate(mapping):
        feature_id = int(feature_id_raw)
        if feature_id < 0 or feature_id >= num_target_ids:
            continue
        if inverse[feature_id] == -1:
            inverse[feature_id] = int(identity_id)
        elif inverse[feature_id] != identity_id:
            ambiguous.add(feature_id)
    for feature_id in ambiguous:
        inverse[feature_id] = -1
    inverse[NONE_CLASS_INDEX] = NONE_CLASS_INDEX
    return inverse


def _candidate_from_identity(
    identity_slots: Sequence[int],
    *,
    identity_to_feature: Sequence[int] | None,
    source: str,
) -> RepairCandidate | None:
    identity = tuple(int(value) for value in identity_slots)
    feature = tuple(_identity_to_feature_value(value, identity_to_feature) for value in identity)
    if any(value < 0 for value in feature):
        return None
    return RepairCandidate(identity_slots=identity, feature_slots=feature, source=source)  # type: ignore[arg-type]


def _candidate_from_feature(
    feature_slots: Sequence[int],
    *,
    feature_to_identity: Sequence[int],
    source: str,
) -> RepairCandidate | None:
    feature = tuple(int(value) for value in feature_slots)
    identity = tuple(
        int(feature_to_identity[value]) if 0 <= value < len(feature_to_identity) else -1
        for value in feature
    )
    if any(value < 0 for value in identity):
        return None
    return RepairCandidate(identity_slots=identity, feature_slots=feature, source=source)  # type: ignore[arg-type]


def _proposal_triples(
    *,
    proposal_logits: torch.Tensor,
    cfg: CandidateConfig,
    slot_allowed_ids: tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    | None,
    precomputed_add_topk: Sequence[tuple[int, int, int]] | None,
    precomputed_del_topk: Sequence[tuple[int, int, int]] | None,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    add_topk = (
        _topk_triples_from_logits(
            proposal_logits,
            slots=(0, 1, 2),
            topk_triples=cfg.topk_candidates,
            topk_per_slot=cfg.topk_per_slot,
            slot_allowed_ids=(slot_allowed_ids[0], slot_allowed_ids[1], slot_allowed_ids[2])
            if slot_allowed_ids is not None
            else None,
        )
        if precomputed_add_topk is None
        else [(int(s), int(p), int(o)) for s, p, o in precomputed_add_topk]
    )
    del_topk = (
        _topk_triples_from_logits(
            proposal_logits,
            slots=(3, 4, 5),
            topk_triples=cfg.topk_candidates,
            topk_per_slot=cfg.topk_per_slot,
            slot_allowed_ids=(slot_allowed_ids[3], slot_allowed_ids[4], slot_allowed_ids[5])
            if slot_allowed_ids is not None
            else None,
        )
        if precomputed_del_topk is None
        else [(int(s), int(p), int(o)) for s, p, o in precomputed_del_topk]
    )
    return add_topk, del_topk


def _build_candidate_pool(
    *,
    context: ViolationContext,
    heuristics: ConstraintRepairHeuristics,
    proposal_logits: torch.Tensor,
    cfg: CandidateConfig,
    placeholder_ids: set[int],
    num_target_ids: int,
    identity_to_feature: Sequence[int] | None,
    slot_allowed_ids: tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    | None,
    precomputed_add_topk: Sequence[tuple[int, int, int]] | None,
    precomputed_del_topk: Sequence[tuple[int, int, int]] | None,
    gold_identity: Sequence[int] | None,
    gold_feature: Sequence[int] | None,
    include_gold_train: bool,
) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    if include_gold_train:
        if gold_identity is None or gold_feature is None:
            raise ValueError("Training candidates require identity and feature gold slots.")
        candidates.append(
            RepairCandidate(
                identity_slots=tuple(int(value) for value in gold_identity),  # type: ignore[arg-type]
                feature_slots=tuple(int(value) for value in gold_feature),  # type: ignore[arg-type]
                source="gold_train",
            )
        )

    candidate_map = heuristics.candidates_for(context)
    for action, patterns in (("add", candidate_map.add), ("delete", candidate_map.delete)):
        triples = _instantiate_patterns(
            patterns,
            placeholder_ids=placeholder_ids,
            none_class=NONE_CLASS_INDEX,
            max_values=cfg.heuristic_max_values,
            max_candidates=cfg.heuristic_max_candidates,
        )
        for triple in triples:
            candidate = _candidate_from_identity(
                candidate_from_triple(triple, action=action),
                identity_to_feature=identity_to_feature,
                source="heuristic",
            )
            if candidate is not None:
                candidates.append(candidate)

    feature_to_identity = _feature_to_identity_map(
        identity_to_feature,
        num_target_ids=num_target_ids,
    )
    add_topk, del_topk = _proposal_triples(
        proposal_logits=proposal_logits,
        cfg=cfg,
        slot_allowed_ids=slot_allowed_ids,
        precomputed_add_topk=precomputed_add_topk,
        precomputed_del_topk=precomputed_del_topk,
    )
    for action, triples in (("add", add_topk), ("delete", del_topk)):
        for triple in triples:
            candidate = _candidate_from_feature(
                candidate_from_triple(triple, action=action),
                feature_to_identity=feature_to_identity,
                source="proposal",
            )
            if candidate is not None:
                candidates.append(candidate)

    deduped: list[RepairCandidate] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for candidate in candidates:
        if any(value < 0 or value >= num_target_ids for value in candidate.feature_slots):
            continue
        if candidate.identity_slots in seen:
            continue
        seen.add(candidate.identity_slots)
        deduped.append(candidate)
        if len(deduped) >= cfg.max_candidates_total:
            break
    return deduped


def build_training_candidates(
    *,
    graph: Data | None = None,
    gold_feature_slots: Sequence[int] | None = None,
    gold_identity_slots: Sequence[int] | None = None,
    context: ViolationContext,
    heuristics: ConstraintRepairHeuristics,
    proposal_logits: torch.Tensor,
    cfg: CandidateConfig,
    placeholder_ids: set[int],
    num_target_ids: int,
    identity_to_feature: Sequence[int] | None = None,
    slot_allowed_ids=None,
    precomputed_add_topk=None,
    precomputed_del_topk=None,
) -> tuple[list[RepairCandidate], int]:
    if graph is not None:
        if gold_feature_slots is None:
            gold_feature_slots = gold_candidate(graph)
        if gold_identity_slots is None:
            target = getattr(graph, "y_identity", None)
            gold_identity_slots = (
                gold_feature_slots
                if target is None
                else tuple(int(value) for value in (target[0] if target.dim() == 2 else target).tolist())
            )
    if gold_feature_slots is None:
        raise ValueError("Training candidates require gold_feature_slots or graph.y.")
    if gold_identity_slots is None:
        gold_identity_slots = gold_feature_slots
    candidates = _build_candidate_pool(
        context=context,
        heuristics=heuristics,
        proposal_logits=proposal_logits,
        cfg=cfg,
        placeholder_ids=placeholder_ids,
        num_target_ids=num_target_ids,
        identity_to_feature=identity_to_feature,
        slot_allowed_ids=slot_allowed_ids,
        precomputed_add_topk=precomputed_add_topk,
        precomputed_del_topk=precomputed_del_topk,
        gold_identity=gold_identity_slots,
        gold_feature=gold_feature_slots,
        include_gold_train=cfg.include_gold_during_training,
    )
    gold_identity_tuple = tuple(int(value) for value in gold_identity_slots)
    for index, candidate in enumerate(candidates):
        if candidate.identity_slots == gold_identity_tuple:
            return candidates, index
    return candidates, -1


def build_inference_candidates(
    *,
    context: ViolationContext,
    heuristics: ConstraintRepairHeuristics,
    proposal_logits: torch.Tensor,
    cfg: CandidateConfig,
    placeholder_ids: set[int],
    num_target_ids: int,
    identity_to_feature: Sequence[int] | None = None,
    slot_allowed_ids=None,
    precomputed_add_topk=None,
    precomputed_del_topk=None,
) -> list[RepairCandidate]:
    return _build_candidate_pool(
        context=context,
        heuristics=heuristics,
        proposal_logits=proposal_logits,
        cfg=cfg,
        placeholder_ids=placeholder_ids,
        num_target_ids=num_target_ids,
        identity_to_feature=identity_to_feature,
        slot_allowed_ids=slot_allowed_ids,
        precomputed_add_topk=precomputed_add_topk,
        precomputed_del_topk=precomputed_del_topk,
        gold_identity=None,
        gold_feature=None,
        include_gold_train=False,
    )


def build_candidates(
    *,
    graph: Data | None = None,
    gold_slots: Sequence[int] | None = None,
    **kwargs,
) -> tuple[list[tuple[int, int, int, int, int, int]], int]:
    """Legacy wrapper retained for external callers; it never force-adds gold when disabled."""
    cfg: CandidateConfig = kwargs["cfg"]
    if cfg.include_gold_during_training:
        candidates, gold_index = build_training_candidates(
            graph=graph,
            gold_feature_slots=gold_slots,
            gold_identity_slots=gold_slots,
            **kwargs,
        )
    else:
        candidates = build_inference_candidates(**kwargs)
        gold_index = -1
        if gold_slots is not None:
            gold_tuple = tuple(int(value) for value in gold_slots)
            for index, candidate in enumerate(candidates):
                if candidate.identity_slots == gold_tuple:
                    gold_index = index
                    break
    return candidate_feature_tuples(candidates), gold_index


def score_candidates_from_logits(
    proposal_logits: torch.Tensor,
    candidate_slots: torch.Tensor,
) -> torch.Tensor:
    """Score candidate edits by summing the corresponding per-slot proposal logits."""
    if proposal_logits.dim() != 2 or proposal_logits.size(0) != NUM_SLOTS:
        raise ValueError(
            f"proposal_logits must be shaped ({NUM_SLOTS}, vocab), got {tuple(proposal_logits.shape)}"
        )
    if candidate_slots.dim() != 2 or candidate_slots.size(-1) != NUM_SLOTS:
        raise ValueError(
            f"candidate_slots must be shaped (num_candidates, {NUM_SLOTS}), got {tuple(candidate_slots.shape)}"
        )
    candidate_slots = candidate_slots.to(device=proposal_logits.device, dtype=torch.long)
    if candidate_slots.numel() == 0:
        return proposal_logits.new_zeros((0,))
    if int(candidate_slots.min().item()) < 0 or int(candidate_slots.max().item()) >= proposal_logits.size(-1):
        raise ValueError("candidate_slots contains out-of-range ids for proposal logits.")
    gathered = proposal_logits.gather(1, candidate_slots.transpose(0, 1)).transpose(0, 1)
    return gathered.sum(dim=-1)
