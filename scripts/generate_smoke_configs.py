#!/usr/bin/env python3

"""Generate a small suite of configs that exercise 07_train.py code paths.

This does NOT attempt a full grid search. It targets coverage for:
- fix_probability_loss on/off with different schedules
- dynamic reweighting on/off and batch/epoch updates
- factor loss on/off with different reductions
- factor pressure model variant

Adjust the base template below if your data uses different encoding or dimensions.
"""

import argparse
import json
from pathlib import Path


VALIDATION_SUBSET_SIZE = 25_000


def _base_config() -> dict:
    return {
        "training_config": {
            "batch_size": 64,
            "num_epochs": 2,
            "early_stopping_rounds": 1,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "num_workers": 0,
            "pin_memory": False,
            "validation_subset_size": VALIDATION_SUBSET_SIZE,
            "constraint_loss": {
                "dynamic_reweighting": {
                    "enabled": False,
                    "update_frequency": "epoch",
                    "scale": 1.0,
                    "min_weight": 0.5,
                    "max_weight": 3.0,
                    "smoothing": 0.2,
                }
            },
            "fix_probability_loss": {
                "enabled": False,
            },
            "factor_loss": {
                "enabled": False,
                "weight_pre": 0.1,
                "pos_weight": None,
                "only_checkable": True,
                "per_graph_reduction": "mean",
            },
            "validate_factor_labels": False,
        },
        "model_config": {
            "dataset_variant": "sample",
            "encoding": "text_embedding",
            "min_occurrence": 100,
            "num_embedding_size": 768,
            "model": "GIN",
            "num_layers": 2,
            "hidden_channels": 128,
            "head_hidden": 128,
            "dropout": 0.1,
            "use_node_embeddings": False,
            "use_role_embeddings": True,
            "role_embedding_dim": 8,
            "num_role_types": 8,
            "use_edge_attributes": False,
            "use_edge_subtraction": False,
            "num_factor_types": 0,
            "factor_type_embedding_dim": 8,
            "pressure_enabled": False,
        },
    }


def _write_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate smoke configs for 07_train.py")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("model-configs/generated_smoke"),
        help="Output directory for generated configs.",
    )
    args = parser.parse_args()

    configs: list[tuple[str, dict]] = []

    base = _base_config()
    configs.append(("00_baseline.json", base))

    # Dynamic reweighting (batch update)
    cfg = json.loads(json.dumps(base))
    cfg["training_config"]["constraint_loss"]["dynamic_reweighting"].update(
        {"enabled": True, "update_frequency": "batch"}
    )
    configs.append(("01_dyn_reweight_batch.json", cfg))

    # Fix probability loss (linear schedule)
    cfg = json.loads(json.dumps(base))
    cfg["training_config"]["fix_probability_loss"] = {
        "enabled": True,
        "initial_weight": 0.5,
        "final_weight": 0.05,
        "decay_epochs": 2,
        "warmup_epochs": 0,
        "schedule": "linear",
    }
    configs.append(("02_fixprob_linear.json", cfg))

    # Factor loss enabled (sum reduction + pos_weight)
    cfg = json.loads(json.dumps(base))
    cfg["training_config"]["factor_loss"] = {
        "enabled": True,
        "weight_pre": 0.2,
        "pos_weight": 2.0,
        "only_checkable": True,
        "per_graph_reduction": "sum",
    }
    cfg["training_config"]["validate_factor_labels"] = True
    configs.append(("03_factor_loss_sum.json", cfg))

    # Pressure model variant
    cfg = json.loads(json.dumps(base))
    cfg["model_config"]["model"] = "GIN_PRESSURE"
    cfg["model_config"]["pressure_enabled"] = True
    cfg["model_config"]["num_factor_types"] = 8
    configs.append(("04_pressure_model.json", cfg))

    # Role embeddings off
    cfg = json.loads(json.dumps(base))
    cfg["model_config"]["use_role_embeddings"] = False
    configs.append(("05_no_role_embeddings.json", cfg))

    for name, payload in configs:
        _write_config(args.out_dir / name, payload)

    print(f"Wrote {len(configs)} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
