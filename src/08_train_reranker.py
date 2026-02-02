#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from modules.config import ModelConfig
from modules.data_encoders import (
    GlobalIntEncoder,
    dataset_variant_name,
    infer_node_feature_spec,
)
from modules.model_store import (
    config_copy_path,
    config_tag_from_path,
    ensure_run_dir,
    get_checkpoint_path,
    history_path,
    resolve_run_dir,
)
from modules.models import build_model
from modules.repair_eval import ConstraintRepairHeuristics, ViolationContext, load_violation_contexts
from modules.reranker import CandidateReranker, RerankerConfig, build_reranker
from modules.reranker_eval import CandidateConstraintEvaluator
from modules.training_utils import (
    load_graph_dataset,
    placeholder_ids_from_encoder,
    progress_bar,
    set_seed,
)

NUM_SLOTS = 6
NONE_CLASS_INDEX = 0

logger = logging.getLogger(__name__)


@dataclass
class RerankerTrainingConfig:
    batch_size: int = 32
    num_epochs: int = 5
    early_stopping_rounds: int = 3
    grad_clip: float | None = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    objective: str = "main"  # main | global_fix
    regression_weight: float = 0.5
    topk_candidates: int = 20
    topk_per_slot: int = 5
    heuristic_max_candidates: int = 30
    heuristic_max_values: int = 3
    include_gold: bool = True
    max_candidates_total: int = 80
    assume_complete_entity_facts: bool = True
    constraint_scope: str = "local"  # local | focus

    @classmethod
    def from_mapping(cls, data: dict | None) -> "RerankerTrainingConfig":
        payload = dict(data or {})
        objective = str(payload.get("objective", cls.objective)).lower()
        if objective not in {"main", "global_fix"}:
            raise ValueError("training_config.objective must be 'main' or 'global_fix'")
        constraint_scope = str(payload.get("constraint_scope", cls.constraint_scope)).lower()
        if constraint_scope not in {"local", "focus"}:
            raise ValueError("training_config.constraint_scope must be 'local' or 'focus'")
        return cls(
            batch_size=int(payload.get("batch_size", cls.batch_size)),
            num_epochs=int(payload.get("num_epochs", cls.num_epochs)),
            early_stopping_rounds=int(payload.get("early_stopping_rounds", cls.early_stopping_rounds)),
            grad_clip=payload.get("grad_clip", cls.grad_clip),
            learning_rate=float(payload.get("learning_rate", cls.learning_rate)),
            weight_decay=float(payload.get("weight_decay", cls.weight_decay)),
            scheduler_factor=float(payload.get("scheduler_factor", cls.scheduler_factor)),
            scheduler_patience=int(payload.get("scheduler_patience", cls.scheduler_patience)),
            num_workers=int(payload.get("num_workers", cls.num_workers)),
            pin_memory=bool(payload.get("pin_memory", cls.pin_memory)),
            objective=objective,
            regression_weight=float(payload.get("regression_weight", cls.regression_weight)),
            topk_candidates=int(payload.get("topk_candidates", cls.topk_candidates)),
            topk_per_slot=int(payload.get("topk_per_slot", cls.topk_per_slot)),
            heuristic_max_candidates=int(payload.get("heuristic_max_candidates", cls.heuristic_max_candidates)),
            heuristic_max_values=int(payload.get("heuristic_max_values", cls.heuristic_max_values)),
            include_gold=bool(payload.get("include_gold", cls.include_gold)),
            max_candidates_total=int(payload.get("max_candidates_total", cls.max_candidates_total)),
            assume_complete_entity_facts=bool(
                payload.get("assume_complete_entity_facts", cls.assume_complete_entity_facts)
            ),
            constraint_scope=constraint_scope,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "early_stopping_rounds": self.early_stopping_rounds,
            "grad_clip": self.grad_clip,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_patience": self.scheduler_patience,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "objective": self.objective,
            "regression_weight": self.regression_weight,
            "topk_candidates": self.topk_candidates,
            "topk_per_slot": self.topk_per_slot,
            "heuristic_max_candidates": self.heuristic_max_candidates,
            "heuristic_max_values": self.heuristic_max_values,
            "include_gold": self.include_gold,
            "max_candidates_total": self.max_candidates_total,
            "assume_complete_entity_facts": self.assume_complete_entity_facts,
            "constraint_scope": self.constraint_scope,
        }


def _load_experiment_config(path: Path) -> tuple[ModelConfig, RerankerConfig, RerankerTrainingConfig, dict]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a JSON object.")

    model_cfg = ModelConfig.from_mapping(payload.get("model_config", {}))
    reranker_cfg = RerankerConfig.from_mapping(payload.get("reranker_config", {}))
    training_cfg = RerankerTrainingConfig.from_mapping(payload.get("training_config", {}))
    proposal_cfg = dict(payload.get("proposal_config", {}))
    return model_cfg, reranker_cfg, training_cfg, proposal_cfg


def _load_encoder(interim_path: Path) -> GlobalIntEncoder:
    encoder = GlobalIntEncoder()
    encoder.load(interim_path / "globalintencoder.txt")
    encoder.freeze()
    return encoder


def _load_parquet_rows(interim_path: Path, split: str) -> list:
    import pandas as pd

    path = interim_path / f"df_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet split not found at {path}")
    columns = [
        "constraint_id",
        "constraint_type",
        "subject",
        "predicate",
        "object",
        "other_subject",
        "other_predicate",
        "other_object",
        "constraint_predicates",
        "constraint_objects",
        "subject_predicates",
        "subject_objects",
        "object_predicates",
        "object_objects",
        "other_entity_predicates",
        "other_entity_objects",
        "local_constraint_ids",
        "local_constraint_ids_focus",
    ]
    df = pd.read_parquet(path)
    existing = [col for col in columns if col in df.columns]
    if existing:
        df = df[existing]
    return list(df.itertuples(index=False))


def _resolve_proposal_checkpoint(
    proposal_cfg: dict,
    *,
    model_cfg: ModelConfig,
) -> Path:
    if "checkpoint_path" in proposal_cfg:
        return Path(proposal_cfg["checkpoint_path"])
    if "run_dir" in proposal_cfg:
        return Path(proposal_cfg["run_dir"]) / "checkpoint.pth"
    if "config_tag" in proposal_cfg or "model" in proposal_cfg:
        model_name = proposal_cfg.get("model", model_cfg.model)
        config_tag = proposal_cfg.get("config_tag")
        run_dir = resolve_run_dir(
            model_cfg.dataset_variant,
            model_cfg.encoding,
            model_name,
            config_tag,
        )
        return run_dir / "checkpoint.pth"
    raise ValueError("proposal_config requires checkpoint_path, run_dir, or model/config_tag.")


def _load_proposal_model(
    proposal_cfg: dict,
    *,
    num_input_graph_nodes: int,
    device: torch.device,
    fallback_model_cfg: ModelConfig,
) -> nn.Module:
    checkpoint_path = _resolve_proposal_checkpoint(proposal_cfg, model_cfg=fallback_model_cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Proposal checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", fallback_model_cfg.model)
    model_cfg_payload = checkpoint.get("model_cfg", None)
    model_cfg = ModelConfig.from_mapping(model_cfg_payload) if model_cfg_payload else fallback_model_cfg
    model = build_model(model_name, num_input_graph_nodes, model_cfg)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def _gold_candidate(graph: Data) -> tuple[int, int, int, int, int, int]:
    y = getattr(graph, "y", None)
    if y is None:
        raise ValueError("Graph missing y target tensor.")
    if y.dim() == 2:
        y = y[0]
    return tuple(int(v) for v in y.tolist())


def _candidate_from_triple(triple: tuple[int, int, int], *, action: str) -> tuple[int, int, int, int, int, int]:
    if action == "add":
        return (triple[0], triple[1], triple[2], 0, 0, 0)
    return (0, 0, 0, triple[0], triple[1], triple[2])


def _select_values(values: Iterable[int] | None, *, placeholder_ids: set[int], none_class: int, max_values: int) -> list[int]:
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
            return unique
    if unique:
        return unique
    for value in values:
        if value in (none_class, None):
            continue
        if value in seen:
            continue
        seen.add(int(value))
        unique.append(int(value))
        if len(unique) >= max_values:
            break
    return unique


def _instantiate_patterns(
    patterns: Sequence,
    *,
    placeholder_ids: set[int],
    none_class: int,
    max_values: int,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for pattern in patterns:
        subjects = _select_values(pattern.subjects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        predicates = _select_values(pattern.predicates, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        objects = _select_values(pattern.objects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        if not subjects or not predicates or not objects:
            continue
        for s in subjects:
            for p in predicates:
                for o in objects:
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
) -> list[tuple[int, int, int]]:
    topk_per_slot = max(1, min(topk_per_slot, logits.size(-1)))
    slot_vals = []
    slot_ids = []
    for slot in slots:
        vals, ids = torch.topk(logits[slot], k=topk_per_slot)
        slot_vals.append(vals.cpu())
        slot_ids.append(ids.cpu())
    combos: list[tuple[float, int, int, int]] = []
    for i in range(topk_per_slot):
        for j in range(topk_per_slot):
            for k in range(topk_per_slot):
                score = float(slot_vals[0][i] + slot_vals[1][j] + slot_vals[2][k])
                combos.append((score, int(slot_ids[0][i]), int(slot_ids[1][j]), int(slot_ids[2][k])))
    combos.sort(key=lambda x: x[0], reverse=True)
    return [(s, p, o) for _, s, p, o in combos[:topk_triples]]


def _build_candidates(
    *,
    graph: Data,
    context: ViolationContext,
    heuristics: ConstraintRepairHeuristics,
    proposal_logits: torch.Tensor,
    cfg: RerankerTrainingConfig,
    placeholder_ids: set[int],
    num_target_ids: int,
) -> tuple[list[tuple[int, int, int, int, int, int]], int]:
    candidates: list[tuple[int, int, int, int, int, int]] = []

    if cfg.include_gold:
        candidates.append(_gold_candidate(graph))

    candidate_map = heuristics.candidates_for(context)
    add_triples = _instantiate_patterns(
        candidate_map.add,
        placeholder_ids=placeholder_ids,
        none_class=NONE_CLASS_INDEX,
        max_values=cfg.heuristic_max_values,
        max_candidates=cfg.heuristic_max_candidates,
    )
    del_triples = _instantiate_patterns(
        candidate_map.delete,
        placeholder_ids=placeholder_ids,
        none_class=NONE_CLASS_INDEX,
        max_values=cfg.heuristic_max_values,
        max_candidates=cfg.heuristic_max_candidates,
    )
    candidates.extend(_candidate_from_triple(triple, action="add") for triple in add_triples)
    candidates.extend(_candidate_from_triple(triple, action="delete") for triple in del_triples)

    add_slots = (0, 1, 2)
    del_slots = (3, 4, 5)
    add_topk = _topk_triples_from_logits(
        proposal_logits,
        slots=add_slots,
        topk_triples=cfg.topk_candidates,
        topk_per_slot=cfg.topk_per_slot,
    )
    del_topk = _topk_triples_from_logits(
        proposal_logits,
        slots=del_slots,
        topk_triples=cfg.topk_candidates,
        topk_per_slot=cfg.topk_per_slot,
    )
    candidates.extend(_candidate_from_triple(triple, action="add") for triple in add_topk)
    candidates.extend(_candidate_from_triple(triple, action="delete") for triple in del_topk)

    deduped: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for cand in candidates:
        if any(v < 0 or v >= num_target_ids for v in cand):
            continue
        if cand in seen:
            continue
        seen.add(cand)
        deduped.append(cand)
        if len(deduped) >= cfg.max_candidates_total:
            break

    gold = _gold_candidate(graph)
    if any(v < 0 or v >= num_target_ids for v in gold):
        raise ValueError("Gold candidate contains out-of-range ids for target vocabulary.")
    if gold not in seen:
        deduped.insert(0, gold)
    gold_index = deduped.index(gold)

    return deduped, gold_index


def _evaluate_candidate_set(
    evaluator: CandidateConstraintEvaluator,
    row: Any,
    *,
    candidates: Sequence[Sequence[int]],
    primary_index: int,
) -> list:
    metrics: list = []
    for cand in candidates:
        metrics.append(
            evaluator.evaluate(row, candidate_slots=cand, primary_factor_index=primary_index)
        )
    return metrics


def _aggregate_epoch_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    totals: dict[str, float] = {}
    for record in records:
        for key, value in record.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: total / len(records) for key, total in totals.items()}


def _run_epoch(
    *,
    model: CandidateReranker,
    proposal_model: nn.Module,
    loader: DataLoader,
    contexts: Sequence[ViolationContext],
    rows: Sequence[Any],
    heuristics: ConstraintRepairHeuristics,
    evaluator: CandidateConstraintEvaluator,
    device: torch.device,
    cfg: RerankerTrainingConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    proposal_model.eval()

    epoch_loss = 0.0
    epoch_records: list[dict[str, float]] = []

    for batch in progress_bar(loader, desc="train" if is_train else "val"):
        batch = batch.to(device)
        with torch.no_grad():
            proposal_outputs = proposal_model(batch)
            proposal_logits = proposal_outputs["edit_logits"].detach()

        graph_emb = model.encode_graphs(batch)
        graphs = batch.to_data_list()

        total_loss = 0.0
        batch_count = 0
        for idx, graph in enumerate(graphs):
            context_index = int(getattr(graph, "context_index"))
            context = contexts[context_index]
            row = rows[context_index]
            candidates, gold_index = _build_candidates(
                graph=graph,
                context=context,
                heuristics=heuristics,
                proposal_logits=proposal_logits[idx],
                cfg=cfg,
                placeholder_ids=set(heuristics.placeholder_ids.values()),
                num_target_ids=model.num_target_ids,
            )
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(graph_emb[idx], candidate_tensor)
            log_probs = F.log_softmax(scores, dim=0)
            probs = log_probs.exp()

            metrics = _evaluate_candidate_set(
                evaluator,
                row,
                candidates=candidates,
                primary_index=int(getattr(graph, "primary_factor_index", 0)),
            )

            primary_oracle = max(m.primary_satisfied for m in metrics)
            global_oracle = max(m.global_satisfied_fraction for m in metrics)

            best_idx = int(torch.argmax(scores).item())
            primary_best = metrics[best_idx].primary_satisfied
            global_best = metrics[best_idx].global_satisfied_fraction
            regress_best = metrics[best_idx].secondary_regressions

            record = {
                "primary_oracle": primary_oracle,
                "primary_chosen": primary_best,
                "global_oracle": global_oracle,
                "global_chosen": global_best,
                "regressions_chosen": regress_best,
                "candidate_count": float(len(candidates)),
            }
            epoch_records.append(record)

            if cfg.objective == "global_fix":
                satisfaction = torch.tensor(
                    [m.global_satisfied_fraction for m in metrics],
                    dtype=torch.float32,
                    device=device,
                )
                expected_satisfaction = torch.sum(probs * satisfaction)
                loss = -expected_satisfaction
            else:
                ce_loss = -log_probs[gold_index]
                regressions = torch.tensor(
                    [m.secondary_regressions for m in metrics],
                    dtype=torch.float32,
                    device=device,
                )
                gold_regression = regressions[gold_index]
                reg_penalty = torch.sum(probs * torch.clamp(regressions - gold_regression, min=0.0))
                loss = ce_loss + cfg.regression_weight * reg_penalty

            total_loss += loss
            batch_count += 1

        if batch_count == 0:
            continue

        batch_loss = total_loss / batch_count
        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

        epoch_loss += float(batch_loss.item())

    avg_loss = epoch_loss / max(len(loader), 1)
    metrics = _aggregate_epoch_metrics(epoch_records)
    return avg_loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate-based reranker.")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="Path to reranker experiment config JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)

    model_cfg, reranker_cfg, training_cfg, proposal_cfg = _load_experiment_config(args.experiment_config)

    dataset_variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    processed_root = Path("data") / "processed" / dataset_variant
    interim_path = Path("data") / "interim" / dataset_variant

    train_path = processed_root / f"train_graph-{model_cfg.encoding}.pkl"
    val_path = processed_root / f"val_graph-{model_cfg.encoding}.pkl"

    train_data = load_graph_dataset(train_path)
    val_data = load_graph_dataset(val_path)
    if not isinstance(train_data, list) or not isinstance(val_data, list):
        raise RuntimeError("Reranker training requires in-memory datasets (list[Data]).")

    encoder = _load_encoder(interim_path)
    placeholder_ids = placeholder_ids_from_encoder(encoder)
    heuristics = ConstraintRepairHeuristics(
        encoder=encoder,
        placeholder_ids=placeholder_ids,
        none_class=NONE_CLASS_INDEX,
    )

    train_contexts = load_violation_contexts(interim_path, "train", none_class=NONE_CLASS_INDEX)
    val_contexts = load_violation_contexts(interim_path, "val", none_class=NONE_CLASS_INDEX)
    if len(train_contexts) != len(train_data) or len(val_contexts) != len(val_data):
        raise RuntimeError("Mismatch between graph dataset size and violation contexts.")

    for idx, graph in enumerate(train_data):
        setattr(graph, "context_index", idx)
    for idx, graph in enumerate(val_data):
        setattr(graph, "context_index", idx)

    train_rows = _load_parquet_rows(interim_path, "train")
    val_rows = _load_parquet_rows(interim_path, "val")
    if len(train_rows) != len(train_data) or len(val_rows) != len(val_data):
        raise RuntimeError("Mismatch between parquet rows and graph dataset size.")

    registry_path = Path("data") / "interim" / f"constraint_registry_{model_cfg.dataset_variant}.parquet"
    if not registry_path.exists():
        raise FileNotFoundError(f"Constraint registry not found at {registry_path}")

    use_encoded_ids = True
    try:
        sample_id = getattr(train_rows[0], "constraint_id", None)
        if sample_id is None or isinstance(sample_id, str):
            use_encoded_ids = False
    except Exception:
        use_encoded_ids = True

    evaluator = CandidateConstraintEvaluator(
        str(registry_path),
        encoder=encoder if use_encoded_ids else None,
        assume_complete=training_cfg.assume_complete_entity_facts,
        constraint_scope=training_cfg.constraint_scope,
        use_encoded_ids=use_encoded_ids,
    )

    infer_node_feature_spec(train_data)

    vocab_from_filtered = len(encoder._global_id_to_unfiltered_global_id)
    if vocab_from_filtered > 0:
        num_input_graph_nodes = vocab_from_filtered + 1
    else:
        num_input_graph_nodes = len(encoder._encoding) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proposal_model = _load_proposal_model(
        proposal_cfg,
        num_input_graph_nodes=num_input_graph_nodes,
        device=device,
        fallback_model_cfg=model_cfg,
    )

    model = build_reranker(
        num_input_graph_nodes=num_input_graph_nodes,
        model_cfg=model_cfg,
        reranker_cfg=reranker_cfg,
    )
    model.to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=training_cfg.scheduler_factor,
        patience=training_cfg.scheduler_patience,
    )

    run_dir = ensure_run_dir(model_cfg.dataset_variant, model_cfg.encoding, "RERANKER", config_tag_from_path(args.experiment_config))

    best_val = float("inf")
    best_epoch = -1
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
    }

    for epoch in range(training_cfg.num_epochs):
        train_loss, train_metrics = _run_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=train_loader,
            contexts=train_contexts,
            rows=train_rows,
            heuristics=heuristics,
            evaluator=evaluator,
            device=device,
            cfg=training_cfg,
            optimizer=optimizer,
        )
        val_loss, val_metrics = _run_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=val_loader,
            contexts=val_contexts,
            rows=val_rows,
            heuristics=heuristics,
            evaluator=evaluator,
            device=device,
            cfg=training_cfg,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

        logger.info(
            "Epoch %s | train_loss=%.4f val_loss=%.4f primary=%.3f/%.3f global=%.3f/%.3f",
            epoch + 1,
            train_loss,
            val_loss,
            train_metrics.get("primary_chosen", 0.0),
            train_metrics.get("primary_oracle", 0.0),
            val_metrics.get("global_chosen", 0.0),
            val_metrics.get("global_oracle", 0.0),
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            model_path = get_checkpoint_path(run_dir)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_graph_nodes": num_input_graph_nodes,
                    "model_name": "RERANKER",
                    "model_cfg": model_cfg.to_dict(),
                    "training_cfg": training_cfg.to_dict(),
                    "reranker_cfg": reranker_cfg.to_dict(),
                },
                model_path,
            )

        if epoch - best_epoch >= training_cfg.early_stopping_rounds:
            logger.info("Early stopping at epoch %s", epoch + 1)
            break

    config_destination = config_copy_path(run_dir)
    if config_destination.resolve(strict=False) != args.experiment_config.resolve(strict=False):
        shutil.copyfile(args.experiment_config, config_destination)

    history_file = history_path(run_dir)
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    logger.info("Training complete. Best val loss=%.4f", best_val)


if __name__ == "__main__":
    main()
