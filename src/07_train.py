#!/usr/bin/env python3

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from modules.config import ModelConfig, TrainingConfig
from modules.data_encoders import (
    GlobalIntEncoder,
    GraphStreamDataset,
    dataset_variant_name,
    infer_node_feature_spec,
)
from modules.model_store import (
    config_copy_path,
    config_tag_from_path,
    ensure_run_dir,
    get_checkpoint_path,
    history_path,
)
from modules.models import BaseGraphModel, build_model
from modules.repair_eval import ConstraintRepairHeuristics, load_violation_contexts
from modules.training_utils import (
    ConstraintMetricsAccumulator,
    DynamicConstraintWeighter,
    FixProbabilityScheduler,
    compute_fix_probabilities,
    extract_constraint_types,
    load_graph_dataset,
    load_precomputed_target_vocabs,
    log_cuda_memory,
    placeholder_ids_from_encoder,
    plot_training_history,
    progress_bar,
    set_seed,
    update_per_constraint_history,
)

NUM_SLOTS = 6
NONE_CLASS_INDEX = 0

logger = logging.getLogger(__name__)


ENTITY_SLOT_INDICES: tuple[int, ...] = (0, 2, 3, 5)
PREDICATE_SLOT_INDICES: tuple[int, ...] = (1, 4)


def derive_target_class_ids(
    *datasets: list[Data] | GraphStreamDataset | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Aggregate entity/predicate target vocabularies observed in the provided datasets."""

    entity_ids: set[int] = {0}
    predicate_ids: set[int] = {0}

    def update_from_tensor(targets: torch.Tensor) -> None:
        if targets.dim() == 1:
            tensor = targets.view(1, -1)
        elif targets.dim() == 2:
            tensor = targets
        else:
            tensor = targets.view(-1, targets.size(-1))

        entity_values = tensor[:, ENTITY_SLOT_INDICES].reshape(-1)
        predicate_values = tensor[:, PREDICATE_SLOT_INDICES].reshape(-1)

        entity_ids.update(int(v) for v in entity_values.tolist())
        predicate_ids.update(int(v) for v in predicate_values.tolist())

    # Accumulate from all datasets
    for dataset in datasets:
        if dataset is None:
            continue
        iterator = dataset if isinstance(dataset, list) else iter(dataset)
        for graph in iterator:
            targets = getattr(graph, "y", None)
            if targets is None:
                continue
            if targets.dtype not in (torch.long, torch.int64):
                targets = targets.to(dtype=torch.long)
            update_from_tensor(targets)

    entity_sorted = tuple(sorted(entity_ids))
    predicate_sorted = tuple(sorted(predicate_ids))

    return entity_sorted, predicate_sorted


def train(
    model: BaseGraphModel,
    train_data: list[Data] | GraphStreamDataset,
    val_data: list[Data] | GraphStreamDataset,
    num_graph_nodes: int,
    train_cfg: TrainingConfig,
    device: torch.device | str = torch.device("cpu"),
    fix_loss_state: dict[str, object] | None = None,
):
    # Normalise the device argument because config may pass strings.
    if isinstance(device, str):
        device = torch.device(device)

    # Optional fix-probability regulariser state (scheduler, heuristics, contexts).
    fix_scheduler = fix_heuristics = train_contexts = val_contexts = None

    if fix_loss_state:
        fix_scheduler, fix_heuristics, train_contexts, val_contexts = (
            fix_loss_state.get("scheduler"),
            fix_loss_state.get("heuristics"),
            fix_loss_state.get("train_contexts"),
            fix_loss_state.get("val_contexts"),
        )

    # Unpack training configuration.
    batch_size = train_cfg.batch_size
    num_epochs = train_cfg.num_epochs
    early_stopping_rounds = train_cfg.early_stopping_rounds
    grad_clip = train_cfg.grad_clip if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0 else None
    pin_memory = device.type == "cuda" if train_cfg.pin_memory is None else train_cfg.pin_memory

    # Device introspection (CUDA memory tracking/debugging).
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info(f"GPU: {torch.cuda.get_device_name(device_index)}")
        log_cuda_memory("Initial GPU state", device)

    # Dataset bookkeeping / loader construction.
    train_dataset_info = "streaming" if isinstance(train_data, IterableDataset) else "in-memory"
    val_dataset_info = "streaming" if isinstance(val_data, IterableDataset) else "in-memory"

    try:
        train_size = len(train_data)
        logger.info(f"Train dataset ({train_dataset_info}) contains {train_size} graphs")
    except TypeError:
        logger.info(f"Train dataset is {train_dataset_info}; size will be determined lazily")

    try:
        val_size = len(val_data)
        logger.info(f"Validation dataset ({val_dataset_info}) contains {val_size} graphs")
    except TypeError:
        logger.info(f"Validation dataset is {val_dataset_info}; size will be determined lazily")

    train_is_iterable = isinstance(train_data, IterableDataset)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=(not train_is_iterable),
        pin_memory=pin_memory,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=train_cfg.num_workers,
    )

    # Optimiser / scheduler / loss boilerplate.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.scheduler_factor,
        patience=train_cfg.scheduler_patience,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # Dynamic constraint reweighting / early stopping setup.
    dynamic_weighter = DynamicConstraintWeighter(train_cfg.constraint_loss.dynamic_reweighting)
    try:
        train_total_batches = len(train_loader)
    except TypeError:
        train_total_batches = None

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    # Rolling training history for logging + persistence.
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc_all6": [],
        "val_acc_all6": [],
        "train_acc_slot": [],
        "val_acc_slot": [],
    }

    epoch_iter = progress_bar(range(num_epochs), desc="Training Epochs", leave=True)

    for epoch in epoch_iter:
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started")
        if device.type == "cuda" and logger.isEnabledFor(logging.DEBUG):
            log_cuda_memory(f"Epoch {epoch + 1} pre-forward", device, level=logging.DEBUG)

        # ------- Training -------
        model.train()
        train_graph_loss_sum = 0.0  # combined loss (per graph)
        train_graph_count = 0  # graphs processed (for averaging)
        train_slots = 0  # total number of slots processed
        train_correct = 0  # all-6 correct predictions
        train_total = 0  # total number of graphs
        train_slot_correct = 0  # per-slot correct predictions
        train_slot_total = 0  # total number of slots processed
        train_slot_loss_sums = [0.0] * NUM_SLOTS  # per-slot loss sums
        train_slot_counts = [0] * NUM_SLOTS  # per-slot sample counts
        train_slot_correct_slots = [0] * NUM_SLOTS  # per-slot correct predictions

        train_constraint_metrics = ConstraintMetricsAccumulator()
        data_iter = progress_bar(train_loader, desc="Training Batches", leave=False)
        for batch_idx, data in enumerate(data_iter, start=1):
            data = data.to(device, non_blocking=True)
            targets = data.y.long()

            # Validation checks
            assert targets.dim() == 2 and targets.size(1) == NUM_SLOTS, (
                f"targets must be (B,{NUM_SLOTS}), got {tuple(targets.shape)}"
            )
            t_min, t_max = targets.min().item(), targets.max().item()
            assert 0 <= t_min and t_max < model.num_target_ids, (
                f"Expected targets in [0,{model.num_target_ids}), got [{t_min},{t_max}]"
            )

            optimizer.zero_grad(set_to_none=True)

            out = model(data)
            assert out.dim() == 3 and out.size(1) == NUM_SLOTS and out.size(2) == model.num_target_ids, (
                f"Expected out (batch_size,{NUM_SLOTS},{model.num_target_ids}), got {tuple(out.shape)}"
            )

            out_flat = out.reshape(-1, out.size(-1))
            targets_flat = targets.reshape(-1)

            # Constraint type extraction and loss computation
            per_slot_loss = criterion(out_flat, targets_flat)
            loss_matrix = per_slot_loss.view(-1, NUM_SLOTS)
            graph_loss = loss_matrix.mean(dim=1)

            # Optional - Constraint-is-fixed loss term
            fix_weight = 0.0
            if fix_scheduler and fix_scheduler.enabled and fix_heuristics is not None and train_contexts is not None:
                context_tensor = getattr(data, "context_index", None)
                if context_tensor is not None:
                    context_indices = context_tensor.detach().cpu().tolist()
                    if len(context_indices) == graph_loss.size(0):
                        batch_contexts = [train_contexts[int(idx)] for idx in context_indices]
                        if train_total_batches:
                            batch_progress = (batch_idx - 1) / max(train_total_batches, 1)
                        else:
                            batch_progress = 0.0
                        progress = epoch + max(batch_progress, 0.0)
                        fix_weight = fix_scheduler.weight_for_progress(progress)
                        if fix_weight > 0.0:
                            fix_probs = compute_fix_probabilities(out, batch_contexts, fix_heuristics)
                            fix_penalty = 1.0 - fix_probs
                            graph_loss = graph_loss + fix_weight * fix_penalty

            # Optional - Rebalance weights based on constraint types
            constraint_types = extract_constraint_types(data, loss_matrix.size(0))
            sample_weights = dynamic_weighter.weights_for(constraint_types)

            if dynamic_weighter.enabled:
                weight_tensor = torch.tensor(
                    sample_weights,
                    device=graph_loss.device,
                    dtype=graph_loss.dtype,
                )
                weighted_loss = (graph_loss * weight_tensor).mean()
            else:
                weighted_loss = graph_loss.mean()

            weighted_loss.backward()

            train_graph_loss_sum += graph_loss.detach().sum().item()
            train_graph_count += graph_loss.numel()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if logger.isEnabledFor(logging.DEBUG):
                if batch_idx == 1:
                    logger.debug(
                        "First training batch stats: num_graphs=%s num_nodes=%s num_edges=%s",
                        getattr(data, "num_graphs", "?"),
                        getattr(data, "num_nodes", "?"),
                        getattr(data, "num_edges", "?"),
                    )
                if device.type == "cuda" and batch_idx % 10 == 0:
                    log_cuda_memory(
                        f"Epoch {epoch + 1} batch {batch_idx}",
                        device,
                        level=logging.DEBUG,
                    )

            # Accuracy (all-6 and per-slot)
            _, predicted = torch.max(out_flat, 1)
            predicted_reshaped = predicted.reshape(-1, NUM_SLOTS)
            targets_reshaped = targets_flat.reshape(-1, NUM_SLOTS)

            all_correct = (predicted_reshaped == targets_reshaped).all(dim=1)
            train_correct += all_correct.sum().item()
            train_total += all_correct.size(0)

            train_slot_correct += (predicted == targets_flat).sum().item()
            train_slot_total += targets_flat.numel()

            # Loss accumulation (per-slot averaging) uses unweighted losses.
            train_slots += targets_flat.numel()

            slot_loss_sums = loss_matrix.sum(dim=0).detach().cpu().tolist()
            batch_graphs = loss_matrix.size(0)
            slot_correct_matrix = (predicted_reshaped == targets_reshaped).to(dtype=torch.long)
            slot_correct_counts = slot_correct_matrix.sum(dim=0).detach().cpu().tolist()
            for idx in range(NUM_SLOTS):
                train_slot_loss_sums[idx] += slot_loss_sums[idx]
                train_slot_counts[idx] += batch_graphs
                train_slot_correct_slots[idx] += int(slot_correct_counts[idx])

            # Per-constraint aggregation
            loss_per_graph = loss_matrix.detach().cpu().sum(dim=1).tolist()
            slot_correct_per_graph = slot_correct_matrix.sum(dim=1).detach().cpu().tolist()
            all_correct_per_graph = all_correct.to(dtype=torch.long).detach().cpu().tolist()
            train_constraint_metrics.update(
                constraint_types,
                loss_per_graph,
                slot_correct_per_graph,
                all_correct_per_graph,
                NUM_SLOTS,
            )
            dynamic_weighter.observe_batch(constraint_types, loss_matrix)

            data_iter.set_postfix(
                {
                    "loss": f"{weighted_loss.item():.4f}",
                    "all6_acc": f"{100 * train_correct / max(train_total, 1):.2f}%",
                    "slot_acc": f"{100 * train_slot_correct / max(train_slot_total, 1):.2f}%",
                }
            )

        # Epoch training metrics
        avg_train_loss = train_graph_loss_sum / max(train_graph_count, 1)
        train_acc_all6 = 100 * train_correct / max(train_total, 1)
        train_acc_slot = 100 * train_slot_correct / max(train_slot_total, 1)
        train_slot_avg_loss = [train_slot_loss_sums[idx] / max(train_slot_counts[idx], 1) for idx in range(NUM_SLOTS)]
        train_slot_acc_per_slot = [
            100 * train_slot_correct_slots[idx] / max(train_slot_counts[idx], 1) for idx in range(NUM_SLOTS)
        ]
        train_per_constraint = train_constraint_metrics.as_epoch_metrics()

        # ------- Validation -------
        model.eval()
        val_constraint_metrics = ConstraintMetricsAccumulator()
        val_graph_loss_sum = 0.0  # combined validation loss
        val_graph_count = 0  # graphs processed
        val_slots = 0  # total number of slots processed
        val_correct = 0  # all-6 correct predictions
        val_total = 0  # total number of graphs
        val_slot_correct = 0  # per-slot correct predictions
        val_slot_total = 0  # total number of slots processed
        val_slot_loss_sums = [0.0] * NUM_SLOTS
        val_slot_counts = [0] * NUM_SLOTS
        val_slot_correct_slots = [0] * NUM_SLOTS

        with torch.no_grad():
            for batch_idx, data in enumerate(
                progress_bar(val_loader, desc="Validation Batches", leave=False),
                start=1,
            ):
                data = data.to(device, non_blocking=True)
                targets = data.y.long()

                out = model(data)
                assert out.dim() == 3 and out.size(1) == NUM_SLOTS and out.size(2) == model.num_target_ids, (
                    f"Expected out (B,{NUM_SLOTS},{model.num_target_ids}), got {tuple(out.shape)}"
                )

                out_flat = out.reshape(-1, out.size(-1))
                targets_flat = targets.reshape(-1)

                per_slot_loss = criterion(out_flat, targets_flat)
                loss_matrix = per_slot_loss.view(-1, NUM_SLOTS)
                graph_loss = loss_matrix.mean(dim=1)

                # Optional - Constraint-is-fixed loss term
                if (
                    fix_scheduler
                    and fix_scheduler.enabled
                    and fix_heuristics is not None
                    and val_contexts is not None
                ):
                    context_tensor = getattr(data, "context_index", None)
                    if context_tensor is not None:
                        context_indices = context_tensor.detach().cpu().tolist()
                        if len(context_indices) == graph_loss.size(0):
                            batch_contexts = [val_contexts[int(idx)] for idx in context_indices]
                            progress = epoch + 1.0
                            fix_weight = fix_scheduler.weight_for_progress(progress)
                            if fix_weight > 0.0:
                                fix_probs = compute_fix_probabilities(out, batch_contexts, fix_heuristics)
                                graph_loss = graph_loss + fix_weight * (1.0 - fix_probs)

                # Accumulate validation metrics
                val_graph_loss_sum += graph_loss.sum().item()
                val_graph_count += graph_loss.numel()
                val_slots += targets_flat.numel()

                _, predicted = torch.max(out_flat, 1)
                predicted_reshaped = predicted.reshape(-1, NUM_SLOTS)
                targets_reshaped = targets_flat.reshape(-1, NUM_SLOTS)

                all_correct = (predicted_reshaped == targets_reshaped).all(dim=1)
                val_correct += all_correct.sum().item()
                val_total += all_correct.size(0)

                val_slot_correct += (predicted == targets_flat).sum().item()
                val_slot_total += targets_flat.numel()

                slot_loss_sums = loss_matrix.sum(dim=0).detach().cpu().tolist()
                batch_graphs = loss_matrix.size(0)
                slot_correct_matrix = (predicted_reshaped == targets_reshaped).to(dtype=torch.long)
                slot_correct_counts = slot_correct_matrix.sum(dim=0).detach().cpu().tolist()
                for idx in range(NUM_SLOTS):
                    val_slot_loss_sums[idx] += slot_loss_sums[idx]
                    val_slot_counts[idx] += batch_graphs
                    val_slot_correct_slots[idx] += int(slot_correct_counts[idx])

                loss_per_graph = loss_matrix.detach().cpu().sum(dim=1).tolist()
                slot_correct_per_graph = slot_correct_matrix.sum(dim=1).detach().cpu().tolist()
                all_correct_per_graph = all_correct.to(dtype=torch.long).detach().cpu().tolist()
                constraint_types = extract_constraint_types(data, len(loss_per_graph))
                val_constraint_metrics.update(
                    constraint_types,
                    loss_per_graph,
                    slot_correct_per_graph,
                    all_correct_per_graph,
                    NUM_SLOTS,
                )

                if logger.isEnabledFor(logging.DEBUG):
                    if batch_idx == 1:
                        logger.debug(
                            "First validation batch stats: num_graphs=%s num_nodes=%s num_edges=%s",
                            getattr(data, "num_graphs", "?"),
                            getattr(data, "num_nodes", "?"),
                            getattr(data, "num_edges", "?"),
                        )
                    if device.type == "cuda" and batch_idx % 5 == 0:
                        log_cuda_memory(
                            f"Epoch {epoch + 1} validation batch {batch_idx}",
                            device,
                            level=logging.DEBUG,
                        )

        # Epoch validation metrics
        avg_val_loss = val_graph_loss_sum / max(val_graph_count, 1)
        val_acc_all6 = 100 * val_correct / max(val_total, 1)
        val_acc_slot = 100 * val_slot_correct / max(val_slot_total, 1)
        val_slot_avg_loss = [val_slot_loss_sums[idx] / max(val_slot_counts[idx], 1) for idx in range(NUM_SLOTS)]
        val_slot_acc_per_slot = [
            100 * val_slot_correct_slots[idx] / max(val_slot_counts[idx], 1) for idx in range(NUM_SLOTS)
        ]
        val_per_constraint = val_constraint_metrics.as_epoch_metrics()

        logger.info(
            "Epoch %s samples | train=%s validation=%s",
            epoch + 1,
            train_total,
            val_total,
        )

        scheduler.step(avg_val_loss)

        # Snapshot aggregate metrics for this epoch.
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc_all6"].append(train_acc_all6)
        history["val_acc_all6"].append(val_acc_all6)
        history["train_acc_slot"].append(train_acc_slot)
        history["val_acc_slot"].append(val_acc_slot)

        # Record per-slot diagnostics (helps inspect slot-specific drift).
        per_slot_history = history.setdefault("per_slot", {})
        for slot_idx in range(NUM_SLOTS):
            slot_key = str(slot_idx)
            slot_entry = per_slot_history.setdefault(
                slot_key,
                {
                    "train_loss": [],
                    "val_loss": [],
                    "train_acc": [],
                    "val_acc": [],
                },
            )
            slot_entry["train_loss"].append(train_slot_avg_loss[slot_idx])
            slot_entry["val_loss"].append(val_slot_avg_loss[slot_idx])
            slot_entry["train_acc"].append(train_slot_acc_per_slot[slot_idx])
            slot_entry["val_acc"].append(val_slot_acc_per_slot[slot_idx])

        # Merge per-constraint metrics for later plotting / reweighting.
        per_constraint_history = history.setdefault("per_constraint", {})
        epoch_index = len(history["train_loss"]) - 1
        update_per_constraint_history(
            per_constraint_history,
            train_per_constraint,
            val_per_constraint,
            epoch_index,
        )
        dynamic_weighter.update_from_metrics(val_per_constraint)

        if train_per_constraint:
            top_train = sorted(train_per_constraint.items(), key=lambda item: item[1]["graph_count"], reverse=True)[
                :3
            ]
            logger.debug(
                "Top train constraint metrics: %s",
                "; ".join(
                    f"{name}: loss={metrics['loss']:.4f} acc_all6={metrics['acc_all6']:.2f}%"
                    f" acc_slot={metrics['acc_slot']:.2f}% (graphs={metrics['graph_count']})"
                    for name, metrics in top_train
                ),
            )
        if val_per_constraint:
            top_val = sorted(val_per_constraint.items(), key=lambda item: item[1]["graph_count"], reverse=True)[:3]
            logger.info(
                "Val constraint metrics: %s",
                "; ".join(
                    f"{name}: loss={metrics['loss']:.4f} acc_all6={metrics['acc_all6']:.2f}%"
                    f" acc_slot={metrics['acc_slot']:.2f}% (graphs={metrics['graph_count']})"
                    for name, metrics in top_val
                ),
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_rounds:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        epoch_iter.set_postfix(
            {
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "train_all6": f"{train_acc_all6:.2f}%",
                "val_all6": f"{val_acc_all6:.2f}%",
                "train_slot": f"{train_acc_slot:.2f}%",
                "val_slot": f"{val_acc_slot:.2f}%",
                "best_val": f"{best_val_loss:.4f}",
            }
        )

        logger.info(
            "Epoch %s summary | train_loss=%.4f val_loss=%.4f train_all6=%.2f%% val_all6=%.2f%%"
            " train_slot=%.2f%% val_slot=%.2f%%",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            train_acc_all6,
            val_acc_all6,
            train_acc_slot,
            val_acc_slot,
        )

        if device.type == "cuda":
            log_cuda_memory(f"Epoch {epoch + 1} post-epoch", device)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    if device.type == "cuda":
        log_cuda_memory("Training complete", device)
        torch.cuda.empty_cache()

    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the graphs dataset.")

    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        required=True,
        help="JSON file with model hyperparameters.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Verbosity level for logging output.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    return args


def main():
    set_seed(42)
    args = parse_args()

    # Load experiment configuration (model + training sections).
    config_path = Path(args.experiment_config)
    with config_path.open("r", encoding="utf-8") as f:
        experiment_config = json.load(f)
    model_cfg = ModelConfig.from_mapping(experiment_config["model_config"])
    training_cfg = TrainingConfig.from_mapping(experiment_config["training_config"])

    # Prepare run directory
    config_tag = config_tag_from_path(config_path)
    run_directory = ensure_run_dir(model_cfg.dataset_variant, model_cfg.encoding, model_cfg.model, config_tag)
    logger.info("Artifacts will be stored in %s", run_directory)

    logger.info(
        "Starting experiment | dataset=%s min_occurrance=%s encoding=%s model=%s log_level=%s",
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        model_cfg.encoding,
        model_cfg.model,
        args.log_level,
    )

    # Dataset paths
    dataset_variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    path = Path("data/processed") / dataset_variant
    logger.debug("Resolved dataset base path to %s", path)

    train_data_path = path / f"train_graph-{model_cfg.encoding}.pkl"
    val_data_path = path / f"val_graph-{model_cfg.encoding}.pkl"
    train_data = load_graph_dataset(train_data_path)
    val_data = load_graph_dataset(val_data_path)

    # Infer node feature spec and target vocabularies
    use_node_embeddings, feature_dim, feature_dtype, role_spec = infer_node_feature_spec(train_data, val_data)
    logger.info(
        "Detected node features | use_node_embeddings=%s feature_dim=%s dtype=%s role_flags=%s (types=%s)",
        use_node_embeddings,
        feature_dim,
        feature_dtype,
        "enabled" if role_spec.enabled else "absent",
        role_spec.num_types if role_spec.enabled else 0,
    )

    # Target vocabularies for distinct entity and predicate predictions
    precomputed_targets = load_precomputed_target_vocabs(path, splits=("train", "val"))
    if precomputed_targets is not None:
        entity_class_ids, predicate_class_ids = precomputed_targets
    else:
        entity_class_ids, predicate_class_ids = derive_target_class_ids(train_data, val_data)

    model_cfg = model_cfg.updated(
        use_node_embeddings=use_node_embeddings,
        entity_class_ids=entity_class_ids,
        predicate_class_ids=predicate_class_ids,
        use_role_embeddings=role_spec.enabled,
        num_role_types=role_spec.num_types if role_spec.enabled else 0,
    )

    # Load and freeze int encoder
    encoder = GlobalIntEncoder()
    interim_path = Path("data/interim/") / dataset_variant
    encoder.load(interim_path / "globalintencoder.txt")
    encoder.freeze()

    # Optional - Constraint-is-fixed loss components (heuristics + contexts).
    fix_loss_cfg = training_cfg.fix_probability_loss
    fix_loss_state: dict[str, object] | None = None
    fix_scheduler: FixProbabilityScheduler | None = None

    if fix_loss_cfg.enabled:
        fix_scheduler = FixProbabilityScheduler(fix_loss_cfg)
        placeholder_ids = placeholder_ids_from_encoder(encoder)
        heuristics = ConstraintRepairHeuristics(
            encoder=encoder,
            placeholder_ids=placeholder_ids,
            none_class=NONE_CLASS_INDEX,
        )

        def _prepare_contexts(split: str, dataset_obj: list[Data] | GraphStreamDataset) -> list | None:
            if not isinstance(dataset_obj, list):
                logger.warning(
                    "Fix probability loss disabled for %s split: dataset is streamed and cannot carry contexts.",
                    split,
                )
                return None
            contexts = load_violation_contexts(interim_path, split, none_class=NONE_CLASS_INDEX)
            if len(contexts) != len(dataset_obj):
                logger.warning(
                    "Fix probability loss disabled: split=%s has %s graphs but %s contexts.",
                    split,
                    len(dataset_obj),
                    len(contexts),
                )
                return None
            for idx, graph in enumerate(dataset_obj):
                setattr(graph, "context_index", idx)
            return contexts

        train_contexts = _prepare_contexts("train", train_data)
        val_contexts = _prepare_contexts("val", val_data)

        if train_contexts and val_contexts:
            fix_loss_state = {
                "scheduler": fix_scheduler,
                "heuristics": heuristics,
                "train_contexts": train_contexts,
                "val_contexts": val_contexts,
            }
            logger.info(
                "Fix probability loss enabled | initial_weight=%.3f final_weight=%.3f decay=%s schedule=%s",
                fix_loss_cfg.initial_weight,
                fix_loss_cfg.final_weight,
                fix_loss_cfg.decay_epochs,
                fix_loss_cfg.schedule,
            )
        else:
            logger.warning("Fix probability loss disabled due to missing contexts.")
            fix_loss_state = None
            fix_scheduler = None
    else:
        fix_scheduler = None

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Selected compute device: %s", device)

    vocab_from_filtered = len(encoder._global_id_to_unfiltered_global_id)
    if vocab_from_filtered > 0:
        num_input_graph_nodes = vocab_from_filtered + 1
    else:
        num_input_graph_nodes = len(encoder._encoding) + 1
    logger.info("Encoder vocabulary size (graph nodes): %s", num_input_graph_nodes)

    # Build model
    model = build_model(
        model_cfg.model,
        num_input_graph_nodes,
        model_cfg,
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters | total=%s trainable=%s", total_params, trainable_params)
    logger.debug("Model architecture: %s", model)
    if hasattr(model, "head_hidden"):
        logger.info(
            "Model dimensions | mp_hidden=%s head_hidden=%s branch_hidden=%s",
            getattr(model, "hidden_channels", "?"),
            model.head_hidden,
            getattr(model, "branch_hidden", "?"),
        )
        logger.info(
            "Model regularisation | dropout=%.3f num_layers=%s",
            getattr(model, "dropout", float("nan")),
            model.num_layers,
        )
    logger.info(
        "Role features | enabled=%s num_role_types=%s embedding_dim=%s",
        model_cfg.use_role_embeddings,
        model_cfg.num_role_types,
        getattr(model, "role_embedding_dim", 0),
    )
    if device.type == "cuda":
        log_cuda_memory("Model moved to device", device, level=logging.DEBUG)

    # Train model
    model, history = train(
        model,
        train_data,
        val_data,
        num_input_graph_nodes,
        train_cfg=training_cfg,
        device=device,
        fix_loss_state=fix_loss_state,
    )

    if device.type == "cuda":
        log_cuda_memory("Post-training state", device)

    model_path = get_checkpoint_path(run_directory)
    # Save model state and minimal config
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_graph_nodes": num_input_graph_nodes,
            "model_name": model_cfg.model,
            "model_cfg": model_cfg.to_dict(),
            "training_cfg": training_cfg.to_dict(),
        },
        model_path,
    )
    logger.info("Saved model checkpoint to %s", model_path)

    # Persist a copy of the configuration alongside the run
    config_destination = config_copy_path(run_directory)
    if config_destination.resolve(strict=False) != config_path.resolve(strict=False):
        shutil.copyfile(config_path, config_destination)

    # Save training history
    history_file = history_path(run_directory)
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f)
    logger.info("Persisted training history to %s", history_file)
    plot_training_history(history, history_file)


if __name__ == "__main__":
    main()
