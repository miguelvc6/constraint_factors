#!/usr/bin/env python3
"""
Generate a focused 5-config hyperparameter sweep for the appendix-only chooser model:
- ``M1C`` proposal + chooser(Fix-1) + typed pressure
- Full dataset defaults: dataset_variant="full", min_occurrence=100
  (resolved to "full_minocc100")
- Default encoding: "node_id"

It writes configs under:
  models/hp_m1c_<tag>__full__<encoding>/config.json

Run:
  python scripts/make_hparam_search_configs_m1.py \
    --processed-root data/processed \
    --models-root models \
    --encoding node_id \
    --num-configs 5
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from modules.data_encoders import (
    base_dataset_name,
    dataset_variant_name,
    discover_graph_artifacts,
    graph_dataset_filename,
)

SAFE_STREAMING_NUM_WORKERS = 2
SAFE_STREAMING_PIN_MEMORY = False
VALIDATION_SUBSET_SIZE = 25_000
CHEAPER_NUM_EPOCHS = 15
CHEAPER_EARLY_STOPPING_ROUNDS = 5
CHEAPER_SCHEDULER_FACTOR = 0.5
CHEAPER_SCHEDULER_PATIENCE = 1


def _torch_load_trusted(path: Path) -> Any:
    """Load trusted local torch payloads (PyTorch 2.6+ compatible)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_first_data_obj(path: Path) -> Any | None:
    """Load the first graph-like object from a graph artifact path."""
    try:
        if path.suffix == ".pt":
            payload = _torch_load_trusted(path)
        else:
            with path.open("rb") as fh:
                payload = pickle.Unpickler(fh).load()
        if isinstance(payload, list):
            return payload[0] if payload else None
        return payload
    except Exception:
        return None


def _infer_num_factor_types(sample_data: Any) -> int:
    """
    Infer num_factor_types from the sample Data object.
    Prefers factor_types (your current pipeline), falls back to factor_type_id(s) if present.
    """
    if sample_data is None:
        return 0

    for attr in ("factor_types", "factor_type_id", "factor_type_ids"):
        if hasattr(sample_data, attr):
            v = getattr(sample_data, attr)
            try:
                return int(v.max().item()) + 1
            except Exception:
                pass

    return 0


def _infer_num_factor_types_from_registry(dataset_variant: str, interim_root: Path = Path("data/interim")) -> int:
    candidate_names = [dataset_variant, base_dataset_name(dataset_variant)]
    base_name = base_dataset_name(dataset_variant)
    if "_strat" in base_name:
        candidate_names.append(base_name.split("_strat", 1)[0])
    candidates = [
        interim_root / f"constraint_registry_{candidate}.parquet"
        for candidate in dict.fromkeys(candidate_names)
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "registry_json" in df.columns and len(df) > 0:
            payload = json.loads(df.iloc[0]["registry_json"])
            indices = [
                int(item["constraint_type_index"])
                for item in payload.values()
                if isinstance(item, dict) and item.get("constraint_type_index") is not None
            ]
            if indices:
                return max(indices) + 1
        for column in ("constraint_type_index", "constraint_type_id"):
            if column in df.columns and len(df[column].dropna()) > 0:
                return int(df[column].max()) + 1
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


@dataclass(frozen=True)
class HP:
    tag: str
    # training
    batch_size: int
    lr: float
    wd: float
    dropout: float
    num_layers: int
    hidden: int
    head_hidden: int
    # chooser
    beta: float
    gamma: float
    chooser_w: float
    topk: int
    max_cands: int
    # factor loss
    factor_w: float
    # pressure
    pressure_mode: str  # "concat" | "gate"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    ap.add_argument("--encoding", type=str, default="node_id")
    ap.add_argument("--num-configs", type=int, default=5)
    ap.add_argument("--min-occurrence", type=int, default=100)
    ap.add_argument("--dataset-variant", type=str, default="full")
    ap.add_argument(
        "--seed", type=int, default=0, help="Optional: stored in config as metadata if you use it downstream."
    )
    args = ap.parse_args()

    variant = dataset_variant_name(args.dataset_variant, args.min_occurrence)
    encoding = args.encoding
    min_occ = int(args.min_occurrence)

    train_graph = args.processed_root / variant / graph_dataset_filename("train", encoding)
    artifacts = discover_graph_artifacts(train_graph)
    if not artifacts:
        raise FileNotFoundError(
            "Missing graph artifacts.\n"
            f"Checked base path: {train_graph}\n"
            f"Expected monolithic file or shards matching: {train_graph.parent}/{train_graph.stem}-shard*"
        )

    num_factor_types = _infer_num_factor_types_from_registry(variant)
    if num_factor_types <= 0:
        sample = _load_first_data_obj(artifacts[0].path)
        num_factor_types = _infer_num_factor_types(sample)

    # ---- Focused shortlist (5 configs) ----
    # These are the selected candidates for constrained-budget model selection.
    grid = [
        # Beta sweep on concat pressure
        HP("c1", 256, 3.0e-4, 1.1e-4, 0.17, 4, 400, 400, 0.5, 0.0, 0.5, 20, 80, 0.10, "concat"),
        HP("c2", 256, 3.0e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 0.5, 20, 80, 0.10, "concat"),
        HP("c3", 256, 3.0e-4, 1.1e-4, 0.17, 4, 400, 400, 2.0, 0.0, 0.5, 20, 80, 0.10, "concat"),
        # Pressure-mode ablation
        HP("g0", 256, 3.0e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 0.5, 20, 80, 0.10, "gate"),
        # Gamma stress test
        HP("c10", 256, 3.0e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.2, 0.35, 20, 80, 0.10, "concat"),
    ]

    if args.num_configs > len(grid):
        raise ValueError(
            f"Requested --num-configs={args.num_configs}, but focused sweep only defines {len(grid)} configs."
        )

    # truncate to requested count (useful for quick smoke runs)
    grid = grid[: args.num_configs]

    created = 0
    for hp in grid:
        exp_name = f"hp_m1c_{hp.tag}__{variant}__{encoding}"
        exp_dir = args.models_root / exp_name
        cfg_path = exp_dir / "config.json"

        payload = {
            "model_config": {
                "dataset_variant": variant,
                "encoding": encoding,
                "min_occurrence": min_occ,
                "model": "GIN_PRESSURE",
                "factor_executor_impl": "per_type_v1",
                # backbone (inspired by ESWC best, adapted)
                "num_layers": hp.num_layers,
                "hidden_channels": hp.hidden,
                "head_hidden": hp.head_hidden,
                "dropout": hp.dropout,
                "use_node_embeddings": False,
                "use_role_embeddings": True,
                "role_embedding_dim": 16,
                "use_edge_attributes": True,
                "use_edge_subtraction": False,
                "num_role_types": 6,
                # factors / pressure
                "num_factor_types": int(num_factor_types),
                "pressure_enabled": True,
                "pressure_type_conditioning": hp.pressure_mode,  # concat|gate
                "pressure_residual_scale": 0.1,
            },
            "training_config": {
                "batch_size": hp.batch_size,
                "num_epochs": CHEAPER_NUM_EPOCHS,
                "early_stopping_rounds": CHEAPER_EARLY_STOPPING_ROUNDS,
                "learning_rate": hp.lr,
                "weight_decay": hp.wd,
                "scheduler_factor": CHEAPER_SCHEDULER_FACTOR,
                "scheduler_patience": CHEAPER_SCHEDULER_PATIENCE,
                "num_workers": SAFE_STREAMING_NUM_WORKERS,
                "pin_memory": SAFE_STREAMING_PIN_MEMORY,
                "validate_factor_labels": True,
                "validation_subset_size": VALIDATION_SUBSET_SIZE,
                "fix_probability_loss": {"enabled": False},
                "factor_loss": {
                    "enabled": True,
                    "weight_pre": hp.factor_w,
                    "weight_post_gold": hp.factor_w,
                    "only_checkable": True,
                    "per_graph_reduction": "mean",
                },
                "chooser": {
                    "enabled": True,
                    "loss_mode": "fix1",
                    "topk_candidates": hp.topk,
                    "max_candidates_total": hp.max_cands,
                    "beta_no_regression": hp.beta,
                    "gamma_primary": hp.gamma,
                    "loss_weight": hp.chooser_w,
                },
                # keep policy choice off for main optimization sweep
                "policy_filter_strict": True,
            },
            # optional metadata (ignored by strict loaders if unknown — remove if loader is strict about unknown keys)
            # If your loader is strict, delete the "meta" block.
            "meta": {
                "seed": args.seed,
                "family": "hp_search_m1c_appendix",
                "note": "Appendix sweep for chooser-based safe-factor model (M1C).",
            },
        }

        # If config loaders are STRICT and do not allow unknown keys, comment out/remove "meta".
        # (Your reference says loaders are strict on unknown keys.)
        payload.pop("meta", None)

        _write_json(cfg_path, payload)
        created += 1

    print(f"[ok] wrote {created} configs under {args.models_root}")
    print(f"Target graph artifacts checked: {train_graph} (found {len(artifacts)})")
    print(f"Resolved dataset_variant: {variant}")
    print(f"Next: run scheduler over models/hp_m1_*__{variant}__{encoding}/config.json")


if __name__ == "__main__":
    main()
