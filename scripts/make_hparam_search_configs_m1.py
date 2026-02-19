#!/usr/bin/env python3
"""
Generate ~20 hyperparameter-search configs for the MAIN model (Fix-1):
- Proposal + Chooser(Fix-1) + Typed Pressure + Factor Loss
- Full dataset defaults: dataset_variant="full", min_occurrence=100
  (resolved to "full_minocc100")
- Default encoding: "node_id"

It writes configs under:
  models/hp_m1_<tag>__full__<encoding>/config.json

Run:
  python scripts/make_hparam_search_configs_m1.py \
    --processed-root data/processed \
    --models-root models \
    --encoding node_id \
    --num-configs 20
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from modules.data_encoders import dataset_variant_name, discover_graph_artifacts


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
    ap.add_argument("--num-configs", type=int, default=20)
    ap.add_argument("--min-occurrence", type=int, default=100)
    ap.add_argument("--dataset-variant", type=str, default="full")
    ap.add_argument(
        "--seed", type=int, default=0, help="Optional: stored in config as metadata if you use it downstream."
    )
    args = ap.parse_args()

    variant = dataset_variant_name(args.dataset_variant, args.min_occurrence)
    encoding = args.encoding
    min_occ = int(args.min_occurrence)

    train_graph = args.processed_root / variant / f"train_graph-{encoding}.pkl"
    artifacts = discover_graph_artifacts(train_graph)
    if not artifacts:
        raise FileNotFoundError(
            "Missing graph artifacts.\n"
            f"Checked base path: {train_graph}\n"
            f"Expected monolithic file or shards matching: {train_graph.parent}/{train_graph.stem}-shard*"
        )

    sample = _load_first_data_obj(artifacts[0].path)
    num_factor_types = _infer_num_factor_types(sample)

    # ---- Search design (20 configs) ----
    # Center around your ESWC best-ish backbone (4 layers, hidden ~400, dropout ~0.17, lr ~7.5e-4, wd ~1.1e-4),
    # but adapt to new objective by scanning:
    # - chooser beta (no-regression weight) as main trade-off dial
    # - candidate cap/topk to test candidate search pressure
    # - pressure type conditioning concat vs gate
    # - modest LR/WD/dropout and hidden variations
    #
    # Keep epochs/patience suitable for full runs (you can adjust later).
    grid = [
        # Base family (concat) – beta sweep, mild LR/WD variations
        HP("c0", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 0.3, 0.0, 20, 80, 0.10, "concat"),
        HP("c1", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 0.5, 0.0, 20, 80, 0.10, "concat"),
        HP("c2", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c3", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 2.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c4", 256, 5.0e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c5", 256, 1.0e-3, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c6", 256, 7.5e-4, 5.0e-5, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c7", 256, 7.5e-4, 2.0e-4, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "concat"),
        # Candidate-set sensitivity (concat)
        HP("c8", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 10, 40, 0.10, "concat"),
        HP("c9", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 30, 120, 0.10, "concat"),
        # Primary satisfaction emphasis (gamma)
        HP("c10", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.2, 20, 80, 0.10, "concat"),
        HP("c11", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.5, 20, 80, 0.10, "concat"),
        # Backbone capacity + regularization (concat)
        HP("c12", 256, 7.5e-4, 1.1e-4, 0.12, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c13", 256, 7.5e-4, 1.1e-4, 0.25, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c14", 256, 7.5e-4, 1.1e-4, 0.17, 3, 320, 320, 1.0, 0.0, 20, 80, 0.10, "concat"),
        HP("c15", 256, 7.5e-4, 1.1e-4, 0.17, 5, 512, 512, 1.0, 0.0, 20, 80, 0.10, "concat"),
        # Pressure mode ablation: gate vs concat (keep same as strong configs)
        HP("g0", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.10, "gate"),
        HP("g1", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 0.5, 0.0, 20, 80, 0.10, "gate"),
        HP("g2", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 2.0, 0.0, 20, 80, 0.10, "gate"),
        # Factor loss weight scan (concat)
        HP("f0", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.05, "concat"),
        HP("f1", 256, 7.5e-4, 1.1e-4, 0.17, 4, 400, 400, 1.0, 0.0, 20, 80, 0.20, "concat"),
    ]

    # truncate/extend to requested count
    grid = grid[: args.num_configs]

    created = 0
    for hp in grid:
        exp_name = f"hp_m1_{hp.tag}__{variant}__{encoding}"
        exp_dir = args.models_root / exp_name
        cfg_path = exp_dir / "config.json"

        payload = {
            "model_config": {
                "dataset_variant": variant,
                "encoding": encoding,
                "min_occurrence": min_occ,
                "model": "GIN_PRESSURE",
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
            },
            "training_config": {
                "batch_size": hp.batch_size,
                "num_epochs": 30,
                "early_stopping_rounds": 6,
                "learning_rate": hp.lr,
                "weight_decay": hp.wd,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "num_workers": 4,
                "pin_memory": True,
                "validate_factor_labels": True,
                "fix_probability_loss": {"enabled": False},
                "factor_loss": {
                    "enabled": True,
                    "weight_pre": hp.factor_w,
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
                },
                # keep policy choice off for main optimization sweep
                "policy_filter_strict": True,
            },
            # optional metadata (ignored by strict loaders if unknown — remove if loader is strict about unknown keys)
            # If your loader is strict, delete the "meta" block.
            "meta": {
                "seed": args.seed,
                "family": "hp_search_m1_fix1",
                "note": "Main model sweep on full dataset (proposal+chooser+typed pressure).",
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
