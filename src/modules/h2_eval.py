"""H2 evaluation helpers for executable constraint-factor semantics.

This module is intentionally read-only with respect to model artifacts.  It
computes additional reports from existing processed graphs and checkpoints and
writes them to a separate H2 evaluation directory.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import IterableDataset

from modules.data_encoders import GraphStreamDataset
from modules.repair_eval import (
    RepairSample,
    evaluate_global_repair_samples,
    evaluate_repair_samples,
)

logger = logging.getLogger(__name__)

NONE_CLASS_INDEX = 0
FACTOR_TO_LOCAL_EDGE_TYPES = {4, 5, 6}
DENSITY_BUCKETS = (
    ("1", 1, 1),
    ("2_4", 2, 4),
    ("5_16", 5, 16),
    ("17_64", 17, 64),
    ("65_plus", 65, None),
)
EXPOSURE_BUCKETS = (
    ("unseen", 0, 0),
    ("low_1_10", 1, 10),
    ("medium_11_100", 11, 100),
    ("high_gt100", 101, None),
)


@dataclass(frozen=True)
class H2RunSupport:
    """Sidecar data needed for H2 repair/global diagnostics."""

    repair_support: Any | None = None
    global_support: Any | None = None


class TransformedGraphDataset(IterableDataset):
    """Apply an in-memory transform while iterating graph datasets."""

    def __init__(self, dataset: Sequence[Data] | GraphStreamDataset, transform: Callable[[Data], Data]):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self) -> Iterator[Data]:
        for graph in self.dataset:
            yield self.transform(graph)


def density_bucket(count: int) -> str:
    for name, low, high in DENSITY_BUCKETS:
        if count >= low and (high is None or count <= high):
            return name
    return "0"


def exposure_bucket(count: int) -> str:
    for name, low, high in EXPOSURE_BUCKETS:
        if count >= low and (high is None or count <= high):
            return name
    return "unknown"


def factor_count(graph: Data) -> int:
    ids = getattr(graph, "factor_constraint_ids", None)
    if ids is None:
        node_index = getattr(graph, "factor_node_index", None)
        return int(torch.as_tensor(node_index).numel()) if node_index is not None else 0
    return int(torch.as_tensor(ids).view(-1).numel())


def factor_pressure_overlap(graph: Data) -> dict[str, int | float]:
    edge_index = getattr(graph, "edge_index", None)
    edge_type = getattr(graph, "edge_type", None)
    if edge_index is None or edge_type is None:
        return {
            "factor_pressure_edges": 0,
            "pressure_target_nodes": 0,
            "max_factors_per_target": 0,
            "mean_factors_per_target": 0.0,
            "shared_pressure_target_nodes": 0,
        }
    edge_index = torch.as_tensor(edge_index)
    edge_type = torch.as_tensor(edge_type).view(-1)
    if edge_index.numel() == 0 or edge_type.numel() == 0:
        return {
            "factor_pressure_edges": 0,
            "pressure_target_nodes": 0,
            "max_factors_per_target": 0,
            "mean_factors_per_target": 0.0,
            "shared_pressure_target_nodes": 0,
        }
    mask = torch.zeros(edge_type.numel(), dtype=torch.bool)
    for etype in FACTOR_TO_LOCAL_EDGE_TYPES:
        mask |= edge_type.cpu() == etype
    if not bool(mask.any()):
        return {
            "factor_pressure_edges": 0,
            "pressure_target_nodes": 0,
            "max_factors_per_target": 0,
            "mean_factors_per_target": 0.0,
            "shared_pressure_target_nodes": 0,
        }
    src = edge_index[0, mask].cpu().tolist()
    dst = edge_index[1, mask].cpu().tolist()
    target_to_factors: dict[int, set[int]] = defaultdict(set)
    for factor_id, target_id in zip(src, dst):
        target_to_factors[int(target_id)].add(int(factor_id))
    counts = [len(values) for values in target_to_factors.values()]
    return {
        "factor_pressure_edges": int(mask.sum().item()),
        "pressure_target_nodes": len(counts),
        "max_factors_per_target": max(counts) if counts else 0,
        "mean_factors_per_target": float(sum(counts) / len(counts)) if counts else 0.0,
        "shared_pressure_target_nodes": sum(1 for value in counts if value > 1),
    }


def clone_with_factor_pressure_mask(graph: Data, mode: str) -> Data:
    """Return a cloned graph with factor-to-local pressure edges masked."""

    if mode == "normal":
        return graph.clone()
    if mode not in {"no_factor_pressure", "primary_only_pressure", "secondary_only_pressure"}:
        raise ValueError(f"Unknown H2 counterfactual mode: {mode}")

    cloned = graph.clone()
    edge_index = getattr(cloned, "edge_index", None)
    edge_type = getattr(cloned, "edge_type", None)
    factor_node_index = getattr(cloned, "factor_node_index", None)
    if edge_index is None or edge_type is None or factor_node_index is None:
        return cloned

    edge_type = torch.as_tensor(edge_type).view(-1)
    if edge_index.numel() == 0 or edge_type.numel() == 0:
        return cloned

    pressure_mask = torch.zeros(edge_type.numel(), dtype=torch.bool, device=edge_type.device)
    for etype in FACTOR_TO_LOCAL_EDGE_TYPES:
        pressure_mask |= edge_type == etype
    if not bool(pressure_mask.any()):
        return cloned

    keep = torch.ones(edge_type.numel(), dtype=torch.bool, device=edge_type.device)
    if mode == "no_factor_pressure":
        keep &= ~pressure_mask
    else:
        factor_nodes = torch.as_tensor(factor_node_index, dtype=torch.long, device=edge_index.device).view(-1)
        primary_idx = int(getattr(cloned, "primary_factor_index", -1))
        primary_node = None
        if 0 <= primary_idx < factor_nodes.numel():
            primary_node = int(factor_nodes[primary_idx].item())
        src = edge_index[0]
        primary_edge_mask = pressure_mask & (src == primary_node) if primary_node is not None else torch.zeros_like(pressure_mask)
        if mode == "primary_only_pressure":
            keep &= (~pressure_mask) | primary_edge_mask
        else:
            keep &= (~pressure_mask) | (pressure_mask & ~primary_edge_mask)

    cloned.edge_index = edge_index[:, keep]
    cloned.edge_type = edge_type[keep]
    return cloned


def count_train_factor_exposure(train_data: Iterable[Data]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for graph in train_data:
        ids = getattr(graph, "factor_constraint_ids", None)
        if ids is None:
            continue
        for value in torch.as_tensor(ids).view(-1).tolist():
            counts[int(value)] += 1
    return counts


def _as_flat_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).detach().cpu().view(-1)


def _safe_div(num: int | float, denom: int | float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float | int | None]:
    total = len(labels)
    if total == 0:
        return {
            "support": 0,
            "positive_rate": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": None,
            "auprc": None,
            "ece": None,
        }
    preds = [1 if score >= 0.5 else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    unique_labels = set(int(v) for v in labels)
    auroc = None
    auprc = None
    if len(unique_labels) > 1:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
    return {
        "support": total,
        "positive_rate": _safe_div(sum(labels), total),
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "ece": _expected_calibration_error(labels, scores),
    }


def _expected_calibration_error(labels: Sequence[int], scores: Sequence[float], *, bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    ece = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        if idx == bins - 1:
            selected = [(y, s) for y, s in zip(labels, scores) if low <= s <= high]
        else:
            selected = [(y, s) for y, s in zip(labels, scores) if low <= s < high]
        if not selected:
            continue
        acc = sum(1 for y, s in selected if (1 if s >= 0.5 else 0) == y) / len(selected)
        conf = sum(s for _, s in selected) / len(selected)
        ece += (len(selected) / total) * abs(acc - conf)
    return float(ece)


def aggregate_semantic_records(records: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not bool(record.get("checkable", False)):
            continue
        grouped[tuple(record.get(key, "unknown") for key in group_keys)].append(record)
    rows: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        labels = [int(item["label"]) for item in items]
        scores = [float(item["score"]) for item in items]
        row = {name: value for name, value in zip(group_keys, key_values)}
        row.update(_classification_metrics(labels, scores))
        rows.append(row)
    return rows


def _triples_from_slots(row: torch.Tensor) -> dict[str, tuple[int, int, int] | None]:
    values = [int(v) for v in row.view(-1).tolist()]
    add = tuple(values[:3])
    delete = tuple(values[3:6])
    return {
        "add": add if all(v != NONE_CLASS_INDEX for v in add) else None,
        "del": delete if all(v != NONE_CLASS_INDEX for v in delete) else None,
    }


def _repair_samples(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    kinds: Sequence[str],
) -> list[RepairSample]:
    samples: list[RepairSample] = []
    for idx, kind in enumerate(kinds):
        samples.append(
            RepairSample(
                constraint_type=str(kind),
                predicted=_triples_from_slots(predictions[idx]),
                gold=_triples_from_slots(targets[idx]),
            )
        )
    return samples


def _fidelity_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    tp = fp = fn = 0
    for pred, gold in zip(predictions, targets):
        pred_triples = {
            (action, triple)
            for action, triple in _triples_from_slots(pred).items()
            if triple is not None
        }
        gold_triples = {
            (action, triple)
            for action, triple in _triples_from_slots(gold).items()
            if triple is not None
        }
        tp += len(pred_triples & gold_triples)
        fp += len(pred_triples - gold_triples)
        fn += len(gold_triples - pred_triples)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _summarize_repair_metrics(repair_metrics: dict[str, Any] | None) -> dict[str, float | int]:
    if not repair_metrics:
        return {"primary_fix_rate": 0.0, "primary_total": 0}
    total = exact = alternative = 0
    for actions in (repair_metrics.get("per_constraint_type") or {}).values():
        if not isinstance(actions, dict):
            continue
        for counts in actions.values():
            if not isinstance(counts, dict):
                continue
            total += int(counts.get("total", 0))
            exact += int(counts.get("exact", 0))
            alternative += int(counts.get("alternative", 0))
    return {
        "primary_fix_rate": _safe_div(exact + alternative, total),
        "primary_total": total,
    }


def _summarize_global_metrics(global_metrics: dict[str, Any] | None) -> dict[str, float | int]:
    if not global_metrics:
        return {}
    overall = global_metrics.get("overall") if isinstance(global_metrics, dict) else None
    if not isinstance(overall, dict):
        return {}
    disruption = overall.get("disruption", {})
    if not isinstance(disruption, dict):
        disruption = {}
    return {
        "gfr": float(overall.get("gfr", 0.0)),
        "srr": float(overall.get("srr", 0.0)),
        "sir": float(overall.get("sir", 0.0)),
        "disruption_add_mean": float(disruption.get("added_triples_mean", 0.0)),
        "disruption_del_mean": float(disruption.get("deleted_triples_mean", 0.0)),
        "disruption_total_ops_mean": float(disruption.get("total_ops_mean", 0.0)),
        "disruption_changed_mean": float(disruption.get("changed_triples_mean", 0.0)),
    }


def _constraint_family_lookup(global_support: Any | None) -> Callable[[int, int | None], str]:
    registry_by_id = getattr(getattr(global_support, "evaluator", None), "_registry_by_id", None)

    def lookup(constraint_id: int, factor_type: int | None) -> str:
        entry = None
        if isinstance(registry_by_id, dict):
            entry = registry_by_id.get(int(constraint_id))
        family = getattr(entry, "constraint_family", None) if entry is not None else None
        if family:
            return str(family)
        return f"type_{factor_type}" if factor_type is not None and factor_type >= 0 else "unknown"

    return lookup


@torch.no_grad()
def collect_h2_variant_outputs(
    *,
    model,
    dataset,
    device: torch.device,
    variant_name: str,
    exposure_counts: Counter[int],
    family_lookup: Callable[[int, int | None], str],
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    kinds: list[str] = []
    graph_records: list[dict[str, Any]] = []
    factor_records: list[dict[str, Any]] = []
    unsupported_reason = None
    graph_offset = 0

    model.eval()
    for batch in loader:
        batch_graphs = batch.to_data_list()
        batch = batch.to(device)
        outputs = model(batch)
        if not isinstance(outputs, dict) or outputs.get("edit_logits") is None:
            raise RuntimeError("H2 evaluation requires model outputs with edit_logits.")
        logits = outputs["edit_logits"]
        batch_predictions = logits.argmax(dim=-1).detach().cpu()
        predictions.append(batch_predictions)
        targets.append(batch.y.detach().cpu())
        kinds.extend(str(getattr(graph, "constraint_type", "UNKNOWN") or "UNKNOWN") for graph in batch_graphs)

        factor_logits_pre = outputs.get("factor_logits_pre")
        factor_logits_post = outputs.get("factor_logits_post_gold")
        factor_graph_index = outputs.get("factor_graph_index")
        if factor_logits_pre is None or factor_graph_index is None:
            unsupported_reason = "model did not emit factor_logits_pre/factor_graph_index"
        else:
            factor_logits_pre_cpu = torch.as_tensor(factor_logits_pre).detach().cpu().view(-1)
            factor_logits_post_cpu = (
                torch.as_tensor(factor_logits_post).detach().cpu().view(-1)
                if factor_logits_post is not None
                else None
            )
            factor_graph_index_cpu = torch.as_tensor(factor_graph_index).detach().cpu().view(-1)
            local_factor_offsets: Counter[int] = Counter()
            for factor_pos in range(factor_graph_index_cpu.numel()):
                local_graph_idx = int(factor_graph_index_cpu[factor_pos].item())
                graph = batch_graphs[local_graph_idx]
                local_factor_idx = local_factor_offsets[local_graph_idx]
                local_factor_offsets[local_graph_idx] += 1
                ids = getattr(graph, "factor_constraint_ids", None)
                types = getattr(graph, "factor_types", None)
                if ids is None:
                    continue
                ids_t = _as_flat_tensor(ids, dtype=torch.long)
                if local_factor_idx >= ids_t.numel():
                    continue
                types_t = _as_flat_tensor(types, dtype=torch.long) if types is not None else torch.empty((0,), dtype=torch.long)
                factor_type = int(types_t[local_factor_idx].item()) if local_factor_idx < types_t.numel() else -1
                constraint_id = int(ids_t[local_factor_idx].item())
                primary_idx = int(getattr(graph, "primary_factor_index", -1))
                exposure = int(exposure_counts.get(constraint_id, 0))
                common = {
                    "variant": variant_name,
                    "graph_index": graph_offset + local_graph_idx,
                    "constraint_type": str(getattr(graph, "constraint_type", "UNKNOWN") or "UNKNOWN"),
                    "factor_index": local_factor_idx,
                    "constraint_id": constraint_id,
                    "factor_type": factor_type,
                    "factor_family": family_lookup(constraint_id, factor_type),
                    "is_primary": local_factor_idx == primary_idx,
                    "primary_or_secondary": "primary" if local_factor_idx == primary_idx else "secondary",
                    "train_exposure": exposure,
                    "exposure_bucket": exposure_bucket(exposure),
                    "density_bucket": density_bucket(factor_count(graph)),
                }

                pre_checkable = getattr(graph, "factor_checkable_pre", None)
                pre_labels = getattr(graph, "factor_satisfied_pre", None)
                if pre_labels is not None:
                    labels_t = _as_flat_tensor(pre_labels, dtype=torch.long)
                    check_t = (
                        _as_flat_tensor(pre_checkable, dtype=torch.bool)
                        if pre_checkable is not None
                        else torch.ones(labels_t.numel(), dtype=torch.bool)
                    )
                    if local_factor_idx < labels_t.numel() and local_factor_idx < check_t.numel():
                        factor_records.append(
                            {
                                **common,
                                "state": "pre",
                                "checkable": bool(check_t[local_factor_idx].item()),
                                "label": int(labels_t[local_factor_idx].item()),
                                "score": float(torch.sigmoid(factor_logits_pre_cpu[factor_pos]).item()),
                            }
                        )

                post_checkable = getattr(graph, "factor_checkable_post_gold", None)
                post_labels = getattr(graph, "factor_satisfied_post_gold", None)
                if factor_logits_post_cpu is not None and post_labels is not None:
                    labels_t = _as_flat_tensor(post_labels, dtype=torch.long)
                    check_t = (
                        _as_flat_tensor(post_checkable, dtype=torch.bool)
                        if post_checkable is not None
                        else torch.ones(labels_t.numel(), dtype=torch.bool)
                    )
                    if local_factor_idx < labels_t.numel() and local_factor_idx < check_t.numel():
                        factor_records.append(
                            {
                                **common,
                                "state": "post_gold",
                                "checkable": bool(check_t[local_factor_idx].item()),
                                "label": int(labels_t[local_factor_idx].item()),
                                "score": float(torch.sigmoid(factor_logits_post_cpu[factor_pos]).item()),
                            }
                        )

        for idx, graph in enumerate(batch_graphs):
            overlap = factor_pressure_overlap(graph)
            graph_records.append(
                {
                    "variant": variant_name,
                    "graph_index": graph_offset + idx,
                    "constraint_type": str(getattr(graph, "constraint_type", "UNKNOWN") or "UNKNOWN"),
                    "factor_count": factor_count(graph),
                    "density_bucket": density_bucket(factor_count(graph)),
                    **overlap,
                }
            )
        graph_offset += len(batch_graphs)

    return {
        "variant": variant_name,
        "predictions": torch.cat(predictions, dim=0) if predictions else torch.empty((0, 6), dtype=torch.long),
        "targets": torch.cat(targets, dim=0) if targets else torch.empty((0, 6), dtype=torch.long),
        "kinds": kinds,
        "graph_records": graph_records,
        "factor_records": factor_records,
        "unsupported_reason": unsupported_reason,
    }


def _subset_tensor(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        return torch.empty((0, tensor.size(-1)), dtype=tensor.dtype)
    return tensor[torch.tensor(indices, dtype=torch.long)]


def _bucket_eval_rows(
    *,
    variant_output: Mapping[str, Any],
    support: H2RunSupport,
    bucket_field: str,
) -> list[dict[str, Any]]:
    graph_records = list(variant_output.get("graph_records", []))
    predictions = variant_output["predictions"]
    targets = variant_output["targets"]
    kinds = list(variant_output.get("kinds", []))
    rows: list[dict[str, Any]] = []
    for bucket in sorted({str(record.get(bucket_field, "unknown")) for record in graph_records}):
        indices = [idx for idx, record in enumerate(graph_records) if str(record.get(bucket_field, "unknown")) == bucket]
        if not indices:
            continue
        pred_subset = _subset_tensor(predictions, indices)
        target_subset = _subset_tensor(targets, indices)
        kind_subset = [kinds[idx] for idx in indices]
        row: dict[str, Any] = {
            "variant": variant_output["variant"],
            bucket_field: bucket,
            "support": len(indices),
        }
        fidelity = _fidelity_metrics(pred_subset, target_subset)
        row.update({f"fidelity_{key}": value for key, value in fidelity.items()})
        samples = _repair_samples(pred_subset, target_subset, kind_subset)
        if support.repair_support is not None:
            contexts = [support.repair_support.contexts[idx] for idx in indices]
            repair_metrics = evaluate_repair_samples(
                samples=samples,
                contexts=contexts,
                heuristics=support.repair_support.heuristics,
                actions=("add", "del"),
            )
            row.update(_summarize_repair_metrics(repair_metrics))
        if support.global_support is not None:
            global_metrics = evaluate_global_repair_samples(
                samples=samples,
                rows=[support.global_support.rows[idx] for idx in indices],
                evaluator=support.global_support.evaluator,
                none_class=NONE_CLASS_INDEX,
                pre_vectors=None,
            )
            row.update(_summarize_global_metrics(global_metrics))
        rows.append(row)
    return rows


def _overall_eval_row(variant_output: Mapping[str, Any], support: H2RunSupport) -> dict[str, Any]:
    predictions = variant_output["predictions"]
    targets = variant_output["targets"]
    kinds = list(variant_output.get("kinds", []))
    row: dict[str, Any] = {
        "variant": variant_output["variant"],
        "support": int(predictions.size(0)),
    }
    fidelity = _fidelity_metrics(predictions, targets)
    row.update({f"fidelity_{key}": value for key, value in fidelity.items()})
    samples = _repair_samples(predictions, targets, kinds)
    if support.repair_support is not None:
        repair_metrics = evaluate_repair_samples(
            samples=samples,
            contexts=support.repair_support.contexts,
            heuristics=support.repair_support.heuristics,
            actions=("add", "del"),
        )
        row.update(_summarize_repair_metrics(repair_metrics))
    if support.global_support is not None:
        global_metrics = evaluate_global_repair_samples(
            samples=samples,
            rows=support.global_support.rows,
            evaluator=support.global_support.evaluator,
            none_class=NONE_CLASS_INDEX,
            pre_vectors=None,
        )
        row.update(_summarize_global_metrics(global_metrics))
    return row


def _delta_rows(rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    normal_by_key = {
        tuple(row.get(key) for key in keys): row
        for row in rows
        if row.get("variant") == "normal"
    }
    delta_fields = (
        "fidelity_f1",
        "primary_fix_rate",
        "gfr",
        "srr",
        "sir",
        "disruption_total_ops_mean",
    )
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row.get("variant") == "normal":
            continue
        key = tuple(row.get(k) for k in keys)
        base = normal_by_key.get(key)
        if base is None:
            continue
        out = {"variant": row.get("variant")}
        for field_name, field_value in zip(keys, key):
            out[field_name] = field_value
        for field in delta_fields:
            if field in row and field in base:
                out[f"delta_{field}"] = float(row[field]) - float(base[field])
        deltas.append(out)
    return deltas


def _prediction_change_summary(
    base_predictions: torch.Tensor,
    variant_predictions: torch.Tensor,
    indices: Sequence[int],
) -> dict[str, float | int]:
    if not indices:
        return {
            "prediction_delta_support": 0,
            "prediction_changed_rate": 0.0,
            "slot_changed_rate": 0.0,
            "changed_slots_mean": 0.0,
        }
    index_tensor = torch.tensor(indices, dtype=torch.long)
    base_subset = base_predictions[index_tensor]
    variant_subset = variant_predictions[index_tensor]
    changed = base_subset != variant_subset
    return {
        "prediction_delta_support": len(indices),
        "prediction_changed_rate": float(changed.any(dim=1).float().mean().item()),
        "slot_changed_rate": float(changed.float().mean().item()),
        "changed_slots_mean": float(changed.sum(dim=1).float().mean().item()),
    }


def _prediction_delta_rows(
    variant_outputs: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    normal_output = next((item for item in variant_outputs if item.get("variant") == "normal"), None)
    if normal_output is None:
        return []
    normal_predictions = normal_output["predictions"]
    rows: list[dict[str, Any]] = []
    for output in variant_outputs:
        variant = output.get("variant")
        if variant == "normal":
            continue
        graph_records = list(output.get("graph_records", []))
        if keys:
            key_values = sorted({tuple(record.get(key) for key in keys) for record in graph_records})
        else:
            key_values = [tuple()]
        for key_value in key_values:
            if keys:
                indices = [
                    idx
                    for idx, record in enumerate(graph_records)
                    if tuple(record.get(key) for key in keys) == key_value
                ]
            else:
                indices = list(range(len(graph_records)))
            row: dict[str, Any] = {"variant": variant}
            for field_name, field_value in zip(keys, key_value):
                row[field_name] = field_value
            row.update(
                _prediction_change_summary(
                    normal_predictions,
                    output["predictions"],
                    indices,
                )
            )
            rows.append(row)
    return rows


def _merge_delta_rows(
    metric_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in metric_rows:
        key = (row.get("variant"), *(row.get(name) for name in keys))
        merged[key] = dict(row)
    for row in prediction_rows:
        key = (row.get("variant"), *(row.get(name) for name in keys))
        merged.setdefault(key, dict(row)).update(row)
    return list(merged.values())


def write_h2_report(
    *,
    model,
    train_data,
    test_data,
    device: torch.device,
    output_dir: Path,
    support: H2RunSupport,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exposure_counts = count_train_factor_exposure(train_data)
    family_lookup = _constraint_family_lookup(support.global_support)
    variants = {
        "normal": test_data,
        "no_factor_pressure": TransformedGraphDataset(
            test_data, lambda graph: clone_with_factor_pressure_mask(graph, "no_factor_pressure")
        ),
        "primary_only_pressure": TransformedGraphDataset(
            test_data, lambda graph: clone_with_factor_pressure_mask(graph, "primary_only_pressure")
        ),
        "secondary_only_pressure": TransformedGraphDataset(
            test_data, lambda graph: clone_with_factor_pressure_mask(graph, "secondary_only_pressure")
        ),
    }

    variant_outputs = []
    for variant_name, dataset in variants.items():
        logger.info("Running H2 variant: %s", variant_name)
        variant_outputs.append(
            collect_h2_variant_outputs(
                model=model,
                dataset=dataset,
                device=device,
                variant_name=variant_name,
                exposure_counts=exposure_counts,
                family_lookup=family_lookup,
                batch_size=batch_size,
            )
        )

    normal_output = next(item for item in variant_outputs if item["variant"] == "normal")
    factor_records = list(normal_output.get("factor_records", []))
    semantic_rows = aggregate_semantic_records(factor_records, ("state", "factor_family", "factor_type"))
    transfer_rows = aggregate_semantic_records(
        factor_records,
        ("state", "exposure_bucket", "primary_or_secondary", "factor_family"),
    )
    density_semantic_rows = aggregate_semantic_records(
        factor_records,
        ("state", "density_bucket", "factor_family", "factor_type"),
    )

    density_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for output in variant_outputs:
        overall_rows.append(_overall_eval_row(output, support))
        density_rows.extend(_bucket_eval_rows(variant_output=output, support=support, bucket_field="density_bucket"))
        counterfactual_rows.extend(
            _bucket_eval_rows(variant_output=output, support=support, bucket_field="density_bucket")
        )

    graph_rows = [record for output in variant_outputs for record in output.get("graph_records", [])]
    graph_density_summary = _graph_density_summary(graph_rows)
    counterfactual_delta_rows = _merge_delta_rows(
        _delta_rows(counterfactual_rows, keys=("density_bucket",)),
        _prediction_delta_rows(variant_outputs, keys=("density_bucket",)),
        keys=("density_bucket",),
    )
    overall_delta_rows = _merge_delta_rows(
        _delta_rows(overall_rows, keys=()),
        _prediction_delta_rows(variant_outputs, keys=()),
        keys=(),
    )

    _write_csv(output_dir / "factor_semantics.csv", semantic_rows)
    _write_csv(output_dir / "transfer_slices.csv", transfer_rows)
    _write_csv(output_dir / "density_slices.csv", density_rows)
    _write_csv(output_dir / "density_factor_semantics.csv", density_semantic_rows)
    _write_csv(output_dir / "counterfactual_masking.csv", counterfactual_rows)
    _write_csv(output_dir / "counterfactual_deltas.csv", counterfactual_delta_rows)
    _write_csv(output_dir / "counterfactual_overall_deltas.csv", overall_delta_rows)
    _write_csv(output_dir / "graph_density.csv", graph_density_summary)

    report = {
        "status": "ok" if not normal_output.get("unsupported_reason") else "partial",
        "unsupported_reason": normal_output.get("unsupported_reason"),
        "unsupported": {
            "factor_semantics": normal_output.get("unsupported_reason"),
        }
        if normal_output.get("unsupported_reason")
        else {},
        "train_factor_exposure": {
            "unique_factor_constraints": len(exposure_counts),
            "total_factor_occurrences": int(sum(exposure_counts.values())),
            "bucket_counts": dict(Counter(exposure_bucket(count) for count in exposure_counts.values())),
        },
        "factor_semantics": semantic_rows,
        "transfer_slices": transfer_rows,
        "overall": overall_rows,
        "overall_deltas": overall_delta_rows,
        "density_slices": density_rows,
        "density_factor_semantics": density_semantic_rows,
        "counterfactual_deltas": counterfactual_delta_rows,
        "graph_density": graph_density_summary,
        "artifacts": {
            "factor_semantics_csv": "factor_semantics.csv",
            "transfer_slices_csv": "transfer_slices.csv",
            "density_slices_csv": "density_slices.csv",
            "density_factor_semantics_csv": "density_factor_semantics.csv",
            "counterfactual_masking_csv": "counterfactual_masking.csv",
            "counterfactual_deltas_csv": "counterfactual_deltas.csv",
            "counterfactual_overall_deltas_csv": "counterfactual_overall_deltas.csv",
            "graph_density_csv": "graph_density.csv",
        },
    }
    with (output_dir / "h2_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def _graph_density_summary(graph_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in graph_rows:
        grouped[(str(row.get("variant", "unknown")), str(row.get("density_bucket", "unknown")))].append(row)
    out: list[dict[str, Any]] = []
    for (variant, bucket), rows in sorted(grouped.items()):
        out.append(
            {
                "variant": variant,
                "density_bucket": bucket,
                "support": len(rows),
                "factor_count_mean": sum(float(r.get("factor_count", 0)) for r in rows) / len(rows),
                "pressure_target_nodes_mean": sum(float(r.get("pressure_target_nodes", 0)) for r in rows) / len(rows),
                "shared_pressure_target_nodes_mean": sum(
                    float(r.get("shared_pressure_target_nodes", 0)) for r in rows
                )
                / len(rows),
                "max_factors_per_target_max": max(int(r.get("max_factors_per_target", 0)) for r in rows),
            }
        )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


__all__ = [
    "H2RunSupport",
    "aggregate_semantic_records",
    "clone_with_factor_pressure_mask",
    "count_train_factor_exposure",
    "density_bucket",
    "exposure_bucket",
    "factor_pressure_overlap",
    "write_h2_report",
]
