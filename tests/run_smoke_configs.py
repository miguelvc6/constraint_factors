#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import shutil
import pickle


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _resolve_runner(preferred: str | None) -> list[str]:
    if preferred:
        return preferred.split()
    for candidate in ("uv",):
        if shutil.which(candidate):
            return [candidate, "run"]
    return [sys.executable]


def _has_factor_labels(dataset_path: Path) -> bool:
    if not dataset_path.exists():
        return False
    try:
        with dataset_path.open("rb") as fh:
            first = pickle.load(fh)
    except Exception:
        return False
    graph = first[0] if isinstance(first, list) and first else first
    if graph is None:
        return False
    return getattr(graph, "factor_checkable_pre", None) is not None and getattr(graph, "factor_satisfied_pre", None) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 07_train.py over generated smoke configs.")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("model-configs/generated_smoke"),
        help="Directory of generated configs.",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path("scripts/generate_smoke_configs.py"),
        help="Path to the config generator script.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("src/07_train.py"),
        help="Training script to run.",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default=None,
        help="Optional command prefix to run Python (e.g., 'uv run').",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("DISABLE_TQDM_PROGRESS", "1")

    runner = _resolve_runner(args.runner)
    _run([*runner, str(args.generator), "--out-dir", str(args.configs_dir)], repo_root)

    configs = sorted(args.configs_dir.glob("*.json"))
    if not configs:
        raise SystemExit(f"No configs found in {args.configs_dir}")

    for cfg in configs:
        with cfg.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        model_cfg = payload.get("model_config", {})
        training_cfg = payload.get("training_config", {})
        factor_cfg = training_cfg.get("factor_loss", {})
        validate_factor = bool(training_cfg.get("validate_factor_labels"))
        factor_enabled = bool(factor_cfg.get("enabled"))

        dataset_variant = model_cfg.get("dataset_variant", "full")
        encoding = model_cfg.get("encoding", "text_embedding")
        data_path = repo_root / "data/processed" / dataset_variant / f"train_graph-{encoding}.pkl"

        if (factor_enabled or validate_factor) and not _has_factor_labels(data_path):
            print(f"Skipping {cfg.name}: factor labels not present in {data_path}")
            continue

        _run([*runner, str(args.train_script), "--experiment-config", str(cfg)], repo_root)


if __name__ == "__main__":
    main()
