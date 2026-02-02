from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm.autonotebook import tqdm

from modules.data_encoders import GlobalIntEncoder, GraphStreamDataset
from modules.repair_eval import ConstraintRepairHeuristics, TriplePattern, ViolationContext

if TYPE_CHECKING:
    from modules.config import DynamicReweightingConfig, FixProbabilityLossConfig

logger = logging.getLogger(__name__)


PREDICTION_SLOT_LABELS = [
    "Slot 0 - add_subject",
    "Slot 1 - add_predicate",
    "Slot 2 - add_object",
    "Slot 3 - del_subject",
    "Slot 4 - del_predicate",
    "Slot 5 - del_object",
]


# General-purpose training helpers shared across scripts.
_DISABLE_TQDM_ENV = os.environ.get("DISABLE_TQDM_PROGRESS", "").lower() in {"1", "true", "yes"}
_STDERR_IS_TTY = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def progress_bar(*args, **kwargs):
    """
    Show tqdm only on interactive terminals (stderr is a TTY) and when not disabled via env.
    tqdm still writes to stderr so it won't pollute logs captured from stdout.
    """
    kwargs.setdefault("disable", _DISABLE_TQDM_ENV or not _STDERR_IS_TTY)
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(*args, **kwargs)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph_dataset(path: Path) -> list[Data] | GraphStreamDataset:
    """Return either an in-memory list or a lazy stream of graphs."""
    with path.open("rb") as f:
        first_object = pickle.load(f)

    logger.debug(f"First object type in dataset: {type(first_object)!r}")

    if isinstance(first_object, list):
        logger.info(f"Loaded dataset into memory with {len(first_object)} graphs")
        return first_object
    if isinstance(first_object, Data):
        logger.info("Streaming dataset detected; graphs will be lazily loaded from disk")
        return GraphStreamDataset(path)

    raise TypeError(
        f"Unsupported object type {type(first_object)!r} in dataset {path}."
        " Expected list[Data] or a pickled Data stream."
    )


# Dynamic constraint weighting utilities.
class DynamicConstraintWeighter:
    """
    Adjusts per-constraint loss weights using validation metrics or on-the-fly batch
    observations. Designed to prioritise underperforming constraint types without
    destabilising those already converged.
    """

    __slots__ = ("_cfg", "_enabled", "_weights")

    def __init__(self, cfg: DynamicReweightingConfig | None) -> None:
        from modules.config import DynamicReweightingConfig  # Local import to avoid cycles

        if cfg is None:
            cfg = DynamicReweightingConfig()
        if not isinstance(cfg, DynamicReweightingConfig):
            cfg = DynamicReweightingConfig.from_mapping(cfg)  # type: ignore[arg-type]
        self._cfg = cfg
        self._enabled = bool(cfg.enabled)
        self._weights: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def update_frequency(self) -> str:
        return self._cfg.update_frequency

    def weights_for(self, constraint_types: Sequence[str]) -> list[float]:
        """Return the current weights for the given constraint types."""
        if not self._enabled:
            return [1.0] * len(constraint_types)

        weights: list[float] = []
        for constraint in constraint_types:
            key = str(constraint) if constraint not in (None, "None") else "UNKNOWN"
            weight = self._weights.setdefault(key, 1.0)
            weights.append(weight)
        return weights

    def observe_batch(self, constraint_types: Sequence[str], loss_matrix: torch.Tensor) -> None:
        """Update weights using the current batch when configured for batch-level updates."""
        if not self._enabled or self._cfg.update_frequency != "batch":
            return
        if loss_matrix.numel() == 0:
            return

        with torch.no_grad():
            losses = loss_matrix.detach().mean(dim=1).cpu().tolist()

        # Aggregate losses per constraint type.
        aggregated: dict[str, list[float]] = {}
        for constraint, loss in zip(constraint_types, losses):
            key = str(constraint) if constraint not in (None, "None") else "UNKNOWN"
            aggregated.setdefault(key, []).append(float(loss))

        # Compute difficulty as average loss per constraint type.
        difficulty: dict[str, float] = {
            key: float(sum(values) / max(len(values), 1)) for key, values in aggregated.items()
        }
        self._apply_difficulty(difficulty)

    def update_from_metrics(self, metrics: Mapping[str, Mapping[str, float]] | None) -> None:
        """Update weights using per-constraint validation metrics."""
        if not self._enabled or not metrics:
            return

        difficulty: dict[str, float] = {}
        for constraint, payload in metrics.items():
            if not isinstance(payload, Mapping):
                continue

            metric_scores: list[float] = []
            for metric_key in self._cfg.target_metrics:
                value = payload.get(metric_key)
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue

                if metric_key.lower().startswith("acc"):
                    metric_scores.append(max(0.0, 1.0 - numeric_value / 100.0))
                else:
                    metric_scores.append(max(numeric_value, 0.0))

            if metric_scores:
                difficulty[constraint] = float(sum(metric_scores) / len(metric_scores))

        self._apply_difficulty(difficulty)

    def _apply_difficulty(self, difficulty: Mapping[str, float]) -> None:
        if not difficulty:
            return

        # Avoid zero or negative difficulty values.
        epsilon = 1e-8
        sanitized: dict[str, float] = {}
        for key, value in difficulty.items():
            sanitized_value = float(value)
            if sanitized_value < 0.0:
                sanitized_value = 0.0
            sanitized[key] = sanitized_value + epsilon

        # Compute average difficulty across all constraint types.
        avg_difficulty = sum(sanitized.values()) / len(sanitized)
        if avg_difficulty <= 0.0:
            avg_difficulty = 1.0

        scale = max(self._cfg.scale, 0.0) # How aggressively to adjust weights.
        smoothing = min(max(self._cfg.smoothing, 0.0), 1.0) # Weight update smoothing factor.
        min_weight = self._cfg.min_weight # Minimum allowed weight.
        max_weight = max(min_weight, self._cfg.max_weight) # Maximum allowed weight.

        # Update weights based on constraint type relative difficulty.
        for key, difficulty_value in sanitized.items():
            relative = difficulty_value / avg_difficulty # Relative difficulty to average.
            target_weight = 1.0 + scale * (relative - 1.0) # Target weight adjustment.
            prev_weight = self._weights.get(key, 1.0) # Previous weight (default 1.0).
            blended_weight = (smoothing * prev_weight) + ((1.0 - smoothing) * target_weight) # Smooth update.
            clamped_weight = min(max(blended_weight, min_weight), max_weight) # Clamp to allowed range.
            self._weights[key] = clamped_weight # Apply updated weight.

        if not self._weights:
            return

        mean_weight = sum(self._weights.values()) / len(self._weights)
        if mean_weight <= 0.0:
            return

        # Renormalise so the average stays close to one while respecting clamps.
        for key in list(self._weights):
            renormalised = self._weights[key] / mean_weight
            self._weights[key] = min(max(renormalised, min_weight), max_weight)



# Device / resource monitoring helpers.
def log_cuda_memory(prefix: str, device: torch.device, level: int = logging.INFO) -> None:
    """Log current and peak CUDA memory stats for the given device."""
    if device.type != "cuda":
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device_index) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device_index) / 1024**3

    logger.log(
        level,
        (
            f"{prefix} | allocated={allocated:.2f} GB reserved={reserved:.2f} GB"
            f" peak_allocated={max_allocated:.2f} GB peak_reserved={max_reserved:.2f} GB"
        ),
    )


# Target vocabulary helpers.
def load_precomputed_target_vocabs(
    base_path: Path,
    splits: Iterable[str] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return precomputed target vocabularies if available."""
    vocab_file = base_path / "target_vocabs.json"
    if not vocab_file.exists():
        return None

    with vocab_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    def _normalize(items: Iterable[Any]) -> set[int]:
        normalized: set[int] = set()
        for item in items:
            try:
                normalized.add(int(item))
            except (TypeError, ValueError):
                continue
        return normalized

    entity_ids: set[int] = set()
    predicate_ids: set[int] = set()

    if splits:
        per_split = payload.get("per_split") or {}
        for split in splits:
            split_payload = per_split.get(split)
            if not isinstance(split_payload, dict):
                continue
            entity_ids.update(_normalize(split_payload.get("entity_class_ids", [])))
            predicate_ids.update(_normalize(split_payload.get("predicate_class_ids", [])))

    if not entity_ids:
        entity_ids.update(_normalize(payload.get("entity_class_ids", [])))
    if not predicate_ids:
        predicate_ids.update(_normalize(payload.get("predicate_class_ids", [])))

    if not entity_ids and not predicate_ids:
        return None

    entity_ids.add(0)
    predicate_ids.add(0)

    return tuple(sorted(entity_ids)), tuple(sorted(predicate_ids))


# Soft constraint violation fix probabilities functions.
def placeholder_ids_from_encoder(encoder: GlobalIntEncoder) -> dict[str, int]:
    """Return placeholder token ids (e.g. 'subject') if they exist in the encoder."""
    placeholders: dict[str, int] = {}
    for token in ("subject", "predicate", "object", "other_subject", "other_predicate", "other_object"):
        idx = encoder.encode(token, add_new=False)
        if idx:
            placeholders[token] = idx
    return placeholders


def _slot_mass(prob_matrix: torch.Tensor, slot_index: int, allowed: frozenset[int] | None) -> torch.Tensor:
    """Sum probability mass assigned to the allowed ids for a specific slot."""
    if allowed is None:
        return prob_matrix.new_tensor(1.0)
    valid = [idx for idx in allowed if 0 <= idx < prob_matrix.size(-1)]
    if not valid:
        return prob_matrix.new_tensor(0.0)
    index = torch.tensor(sorted(set(valid)), dtype=torch.long, device=prob_matrix.device)
    return prob_matrix[slot_index, index].sum()


def _candidate_union_probability(
    prob_matrix: torch.Tensor,
    patterns: Sequence[TriplePattern],
    slot_indices: tuple[int, int, int],
) -> torch.Tensor:
    """Return probability that any candidate pattern is satisfied for the slot trio."""
    if not patterns:
        return prob_matrix.new_tensor(0.0)
    triple_probs: list[torch.Tensor] = []
    for pattern in patterns:
        s_prob = _slot_mass(prob_matrix, slot_indices[0], pattern.subjects)
        p_prob = _slot_mass(prob_matrix, slot_indices[1], pattern.predicates)
        o_prob = _slot_mass(prob_matrix, slot_indices[2], pattern.objects)
        triple_probs.append(s_prob * p_prob * o_prob)
    stacked = torch.stack(triple_probs)
    return torch.clamp(stacked.sum(), max=1.0)


def compute_fix_probabilities(
    logits: torch.Tensor,
    contexts: Sequence[ViolationContext],
    heuristics: ConstraintRepairHeuristics,
) -> torch.Tensor:
    """
    Convert logits into per-graph fix probabilities by reusing the constraint heuristics.

    Parameters
    ----------
    logits: tensor shaped [batch, 6, num_nodes]
    contexts: violation metadata aligned with the batch order
    heuristics: ConstraintRepairHeuristics instance
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3-D (B,6,num_nodes); got shape {tuple(logits.shape)}")
    if logits.size(1) != len(PREDICTION_SLOT_LABELS):
        raise ValueError(f"logits second dimension must be {len(PREDICTION_SLOT_LABELS)}")
    probs = torch.softmax(logits, dim=-1)
    add_indices = (0, 1, 2)
    del_indices = (3, 4, 5)
    batch_scores: list[torch.Tensor] = []
    for row_idx, context in enumerate(contexts):
        candidate_map = heuristics.candidates_for(context)
        add_prob = _candidate_union_probability(probs[row_idx], candidate_map.add, add_indices)
        del_prob = _candidate_union_probability(probs[row_idx], candidate_map.delete, del_indices)
        fix_prob = 1.0 - (1.0 - add_prob) * (1.0 - del_prob)
        batch_scores.append(fix_prob)
    return torch.stack(batch_scores)


# Fix probability scheduling utilities.
class FixProbabilityScheduler:
    """
    Epoch-based decay scheduler for the fix-probability coefficient.

    Supports exponential (default) and linear decay with an optional warmup.
    """

    __slots__ = ("_cfg",)

    def __init__(self, cfg: FixProbabilityLossConfig) -> None:
        from modules.config import FixProbabilityLossConfig  # Local import

        if not isinstance(cfg, FixProbabilityLossConfig):
            cfg = FixProbabilityLossConfig.from_mapping(cfg)  # type: ignore[arg-type]
        self._cfg = cfg

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def weight_for_progress(self, progress_epochs: float) -> float:
        """Return λ for the given (possibly fractional) epoch progress."""
        if not self.enabled:
            return 0.0
        cfg = self._cfg
        start = max(0.0, float(cfg.initial_weight))
        end = max(0.0, float(cfg.final_weight))
        decay_epochs = max(float(cfg.decay_epochs), 1e-6)
        warmup = max(float(cfg.warmup_epochs), 0.0)

        if progress_epochs <= warmup:
            return start

        decay_progress = progress_epochs - warmup
        if cfg.schedule == "linear":
            blend = min(decay_progress / decay_epochs, 1.0)
            weight = start + (end - start) * blend
        else:  # exponential
            decay = math.exp(-decay_progress / decay_epochs)
            weight = end + (start - end) * decay

        return max(weight, 0.0)


# Per-constraint metric tracking helpers.
class ConstraintMetricsAccumulator:
    """Accumulates per-constraint losses and accuracies for a single epoch."""

    __slots__ = ("_stats",)

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "loss_sum": 0.0,
                "slot_count": 0,
                "graph_count": 0,
                "all6_correct": 0,
                "slot_correct": 0,
            }
        )

    def update(
        self,
        constraint_kinds: Sequence[str],
        loss_per_graph: Sequence[float],
        slot_correct: Sequence[int],
        all6_correct: Sequence[int],
        num_slots: int,
    ) -> None:
        if not constraint_kinds:
            return

        for idx, kind in enumerate(constraint_kinds):
            key = str(kind) if kind not in (None, "None") else "UNKNOWN"
            stats = self._stats[key]
            stats["loss_sum"] += float(loss_per_graph[idx])
            stats["slot_count"] += int(num_slots)
            stats["graph_count"] += 1
            stats["all6_correct"] += int(all6_correct[idx])
            stats["slot_correct"] += int(slot_correct[idx])

    def as_epoch_metrics(self) -> dict[str, dict[str, float]]:
        """Return averaged metrics per constraint type."""
        results: dict[str, dict[str, float]] = {}
        for kind, stats in self._stats.items():
            graph_count = stats["graph_count"]
            slot_count = stats["slot_count"]
            if graph_count == 0 or slot_count == 0:
                continue
            results[kind] = {
                "loss": stats["loss_sum"] / slot_count,
                "acc_all6": 100.0 * stats["all6_correct"] / graph_count,
                "acc_slot": 100.0 * stats["slot_correct"] / slot_count,
                "graph_count": graph_count,
                "slot_count": slot_count,
            }
        return results


def extract_constraint_types(batch: Data, expected_len: int) -> list[str]:
    """Extract constraint type labels from the batch, padding with UNKNOWN when absent."""
    raw = getattr(batch, "constraint_type", None)
    if raw is None:
        return ["UNKNOWN"] * expected_len

    if isinstance(raw, (list, tuple)):
        values = list(raw)
    elif torch.is_tensor(raw):
        if raw.numel() == expected_len:
            values = raw.detach().cpu().tolist()
        else:
            logger.warning(
                "Constraint type tensor has unexpected shape %s (expected %s); falling back to UNKNOWN.",
                tuple(raw.shape),
                expected_len,
            )
            return ["UNKNOWN"] * expected_len
    elif isinstance(raw, str):
        values = [raw] * expected_len
    else:
        values = [raw] * expected_len

    if len(values) != expected_len:
        if len(values) < expected_len:
            values.extend(["UNKNOWN"] * (expected_len - len(values)))
        else:
            values = values[:expected_len]
        logger.warning(
            "Constraint type list length mismatch (got %s, expected %s). Adjusting to fit batch.",
            len(values),
            expected_len,
        )

    return [str(v) if v is not None else "UNKNOWN" for v in values]


# Constraint history aggregation utilities.
def _append_metric(
    entry: dict[str, list],
    key: str,
    value: float | int | None,
    fill_value: float | int | None,
    epoch_index: int,
) -> None:
    """Append a metric value for the current epoch, padding earlier epochs if necessary."""
    history_list = entry.setdefault(key, [])
    if len(history_list) < epoch_index:
        history_list.extend([fill_value] * (epoch_index - len(history_list)))
    history_list.append(fill_value if value is None else value)


def update_per_constraint_history(
    store: dict[str, dict[str, list]],
    train_metrics: dict[str, dict[str, float]],
    val_metrics: dict[str, dict[str, float]],
    epoch_index: int,
) -> None:
    """Merge per-constraint metrics from the current epoch into the persistent history store."""
    all_types = set(store) | set(train_metrics) | set(val_metrics)
    metric_mappings = (
        ("train_loss", "loss", None),
        ("val_loss", "loss", None),
        ("train_acc_all6", "acc_all6", None),
        ("val_acc_all6", "acc_all6", None),
        ("train_acc_slot", "acc_slot", None),
        ("val_acc_slot", "acc_slot", None),
        ("train_graph_count", "graph_count", 0),
        ("val_graph_count", "graph_count", 0),
        ("train_slot_count", "slot_count", 0),
        ("val_slot_count", "slot_count", 0),
    )

    for constraint_type in sorted(all_types):
        entry = store.setdefault(constraint_type, {})
        train_entry = train_metrics.get(constraint_type)
        val_entry = val_metrics.get(constraint_type)

        for key, metric_key, fill_value in metric_mappings:
            source = train_entry if key.startswith("train") else val_entry
            metric_value = source.get(metric_key) if source else None
            _append_metric(entry, key, metric_value, fill_value, epoch_index)


# Training history plotting helpers.
def _sanitize_constraint_type(constraint_type: str) -> str:
    """Return a filesystem-safe representation of a constraint type name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", constraint_type.strip() or "UNKNOWN")
    return safe[:80] or "UNKNOWN"


def _to_plot_values(values: Sequence[float | None]) -> list[float]:
    """Convert None entries into NaNs for plotting continuity."""
    return [float("nan") if value is None else float(value) for value in values]


def _slot_label(slot_key: str | int) -> str:
    try:
        idx = int(slot_key)
    except (TypeError, ValueError):
        return f"Slot {slot_key}"
    if 0 <= idx < len(PREDICTION_SLOT_LABELS):
        return PREDICTION_SLOT_LABELS[idx]
    return f"Slot {idx}"


def plot_training_history(history: dict, output_file: Path) -> None:
    max_history_len = max((len(values) for values in history.values() if isinstance(values, list)), default=0)
    epochs = list(range(1, max_history_len + 1))
    plot_path = output_file.with_suffix(".png")
    metrics = [
        ("Loss", ("train_loss", "val_loss")),
        ("All-6 Accuracy", ("train_acc_all6", "val_acc_all6")),
        ("Slot Accuracy", ("train_acc_slot", "val_acc_slot")),
    ]
    active_metrics = [(title, keys) for title, keys in metrics if any(len(history.get(key, [])) for key in keys)]
    if active_metrics:
        fig, axes = plt.subplots(
            len(active_metrics),
            1,
            figsize=(8, 3 * len(active_metrics)),
            sharex=True,
        )
        if len(active_metrics) == 1:
            axes = [axes]

        for ax, (title, keys) in zip(axes, active_metrics):
            plotted = False
            for key, label in zip(keys, ("Train", "Validation")):
                values = history.get(key, [])
                if not values:
                    continue
                ax.plot(epochs[: len(values)], values, label=label)
                plotted = True
            if plotted:
                ax.set_title(title)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                ax.legend()

        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved learning curves plot to %s", plot_path)

    per_constraint = history.get("per_constraint")
    if isinstance(per_constraint, dict) and per_constraint:
        per_constraint_dir = output_file.parent / f"{output_file.stem}_per_constraint"
        per_constraint_dir.mkdir(parents=True, exist_ok=True)

        metric_keys = (
            ("train_loss", "val_loss", "Loss"),
            ("train_acc_all6", "val_acc_all6", "All-6 Accuracy"),
            ("train_acc_slot", "val_acc_slot", "Slot Accuracy"),
        )

        for constraint_type, metrics_dict in sorted(per_constraint.items()):
            lengths = [len(metrics_dict.get(key, [])) for key_pair in metric_keys for key in key_pair[:2]]
            max_len = max(lengths, default=0)
            if max_len == 0:
                continue

            epochs_local = list(range(1, max_len + 1))

            fig, axes = plt.subplots(len(metric_keys), 1, figsize=(8, 3 * len(metric_keys)), sharex=True)
            if len(metric_keys) == 1:
                axes = [axes]

            has_plot = False
            for ax, (train_key, val_key, title) in zip(axes, metric_keys):
                train_values = _to_plot_values(metrics_dict.get(train_key, []))
                val_values = _to_plot_values(metrics_dict.get(val_key, []))

                if len(train_values) < max_len:
                    train_values.extend([float("nan")] * (max_len - len(train_values)))
                if len(val_values) < max_len:
                    val_values.extend([float("nan")] * (max_len - len(val_values)))

                stacked = np.array(train_values + val_values, dtype=float)
                if not np.isfinite(stacked).any():
                    continue

                ax.plot(epochs_local, train_values, label="Train")
                ax.plot(epochs_local, val_values, label="Validation")
                ax.set_title(title)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                ax.legend()
                has_plot = True

            if has_plot:
                fig.suptitle(f"Constraint: {constraint_type}")
                axes[-1].set_xlabel("Epoch")
                fig.tight_layout(rect=(0, 0.03, 1, 0.97))
                file_name = f"{output_file.stem}_{_sanitize_constraint_type(constraint_type)}.png"
                fig.savefig(per_constraint_dir / file_name, dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.close(fig)
        logger.info("Saved per-constraint learning curves to %s", per_constraint_dir)

    per_slot = history.get("per_slot")
    if isinstance(per_slot, dict) and per_slot:
        per_slot_dir = output_file.parent / f"{output_file.stem}_per_slot"
        per_slot_dir.mkdir(parents=True, exist_ok=True)
        slots_rendered = False

        def _sort_key(item: tuple[str, dict[str, list]]) -> tuple[int, str]:
            key, _ = item
            key_str = str(key)
            return (
                int(key_str) if key_str.isdigit() else float("inf"),
                key_str,
            )

        for slot_key, metrics_dict in sorted(per_slot.items(), key=_sort_key):
            max_len = max(
                len(metrics_dict.get("train_loss", [])),
                len(metrics_dict.get("val_loss", [])),
                len(metrics_dict.get("train_acc", [])),
                len(metrics_dict.get("val_acc", [])),
            )
            if max_len == 0:
                continue

            epochs_local = list(range(1, max_len + 1))

            def _normalize(values: Sequence[float | None]) -> list[float]:
                arr = _to_plot_values(values)
                if len(arr) < max_len:
                    arr.extend([float("nan")] * (max_len - len(arr)))
                return arr

            train_loss = _normalize(metrics_dict.get("train_loss", []))
            val_loss = _normalize(metrics_dict.get("val_loss", []))
            train_acc = _normalize(metrics_dict.get("train_acc", []))
            val_acc = _normalize(metrics_dict.get("val_acc", []))

            stacked = np.array(train_loss + val_loss + train_acc + val_acc, dtype=float)
            if not np.isfinite(stacked).any():
                continue

            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            axes[0].plot(epochs_local, train_loss, label="Train")
            axes[0].plot(epochs_local, val_loss, label="Validation")
            axes[0].set_title("Loss")
            axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axes[0].legend()

            axes[1].plot(epochs_local, train_acc, label="Train")
            axes[1].plot(epochs_local, val_acc, label="Validation")
            axes[1].set_title("Accuracy")
            axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axes[1].legend()
            axes[1].set_xlabel("Epoch")

            fig.suptitle(_slot_label(slot_key))
            fig.tight_layout(rect=(0, 0.03, 1, 0.97))
            file_name = f"{output_file.stem}_slot_{_sanitize_constraint_type(str(slot_key))}.png"
            fig.savefig(per_slot_dir / file_name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            slots_rendered = True

        if slots_rendered:
            logger.info("Saved per-slot learning curves to %s", per_slot_dir)


__all__ = [
    "ConstraintMetricsAccumulator",
    "DynamicConstraintWeighter",
    "FixProbabilityScheduler",
    "PREDICTION_SLOT_LABELS",
    "load_graph_dataset",
    "progress_bar",
    "set_seed",
    "compute_fix_probabilities",
    "extract_constraint_types",
    "load_precomputed_target_vocabs",
    "log_cuda_memory",
    "placeholder_ids_from_encoder",
    "plot_training_history",
    "update_per_constraint_history",
]
