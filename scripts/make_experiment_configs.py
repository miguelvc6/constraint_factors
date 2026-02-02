#!/usr/bin/env python3
"""
Generate experiment config directories under models/<exp_name>/config.json.

Assumptions (matches your current pipeline + scheduler style):
- Graphs live in: data/processed/<variant>/<split>_graph-<encoding>.pkl
- Scheduler enumerates: models/**/config.json
- Proposal training consumes: { "model_config": ..., "training_config": ... }
- Reranker training consumes: { "model_config": ..., "reranker_config": ..., "training_config": ..., "proposal_config": ... }

Run:
  python scripts/11_make_experiment_configs.py \
    --processed-root data/processed \
    --models-root models \
    --include-ablations

"""

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

VARIANT_MINOCC_RE = re.compile(r"minocc(\d+)", re.IGNORECASE)


def _parse_min_occurrence(variant: str) -> int:
    m = VARIANT_MINOCC_RE.search(variant)
    if not m:
        return 1
    return int(m.group(1))


def _iter_graph_files(processed_root: Path) -> Iterable[tuple[str, str, Path]]:
    """
    Yield (variant, encoding, train_graph_path) pairs.

    Looks for:
      data/processed/<variant>/train_graph-<encoding>.pkl
    Ignores shards for now (you can extend later if needed).
    """
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    for variant_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        variant = variant_dir.name
        for path in sorted(variant_dir.glob("train_graph-*.pkl")):
            # train_graph-<encoding>.pkl
            enc = path.name[len("train_graph-") : -len(".pkl")]
            if "-shard" in enc.lower():
                continue
            yield variant, enc, path


def _load_first_data_obj(pkl_path: Path) -> Any | None:
    """
    Graph files are a pickle stream of torch_geometric.data.Data objects.
    Load the first one without reading the full stream.
    """
    try:
        with pkl_path.open("rb") as fh:
            unpickler = pickle.Unpickler(fh)
            return unpickler.load()
    except Exception:
        return None


def _infer_num_factor_types(sample_data: Any) -> int | None:
    """
    Try to infer embedding size for factor-type IDs from a sample Data object.

    We look for:
      - factor_type_id: Tensor-like (N_factors,)
      - factor_type_ids: same idea
      - factor_type (rare): might be strings -> cannot infer
    """
    if sample_data is None:
        return None

    for attr in ("factor_type_id", "factor_type_ids"):
        if hasattr(sample_data, attr):
            value = getattr(sample_data, attr)
            try:
                # torch tensor supports .max().item()
                max_id = int(value.max().item())
                return max_id + 1
            except Exception:
                pass

    return None


@dataclass(frozen=True)
class ProposalExperiment:
    name: str
    model_name: str
    factor_loss_enabled: bool
    pressure_enabled: bool
    pressure_type_conditioning: str
    fix_prob_enabled: bool
    validate_factor_labels: bool = False
    factor_weight_pre: float = 0.1
    enable_policy_choice: bool = False
    policy_num_classes: int = 6


@dataclass(frozen=True)
class RerankerExperiment:
    name: str
    objective: str  # "main" | "global_fix"
    proposal_ref_config_tag: str  # which proposal to load (config_tag)
    constraint_scope: str = "local"  # local | focus
    disabled: bool = False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    ap.add_argument("--limit", type=int, default=0, help="Limit variant/encoding pairs (0 = no limit).")
    ap.add_argument("--include-ablations", action="store_true")
    args = ap.parse_args()

    pairs = list(_iter_graph_files(args.processed_root))
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        raise SystemExit(
            f"No train_graph-*.pkl found under {args.processed_root}. "
            "Expected data/processed/<variant>/train_graph-<encoding>.pkl"
        )

    # Define the core experiment set (minimum to start).
    proposal_exps: list[ProposalExperiment] = [
        # M0: ESWC-like (edit imitation). Keep fix-prob ON if you want closer match to ESWC pipeline.
        ProposalExperiment(
            name="m0_eswc_like",
            model_name="GIN",
            factor_loss_enabled=False,
            pressure_enabled=False,
            pressure_type_conditioning="none",
            fix_prob_enabled=True,
            validate_factor_labels=False,
            enable_policy_choice=False,
        ),
        # M1: Main model (Fix 1): factor loss + pressure
        ProposalExperiment(
            name="m1_main_fix1",
            model_name="GIN_PRESSURE",
            factor_loss_enabled=True,
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            fix_prob_enabled=False,
            validate_factor_labels=True,  # required for factor loss + global metrics
            factor_weight_pre=0.1,
            enable_policy_choice=False,
        ),
        ProposalExperiment(
            name="m3_policy_choice",
            model_name="GIN_PRESSURE",
            factor_loss_enabled=True,
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            fix_prob_enabled=False,
            validate_factor_labels=True,
            factor_weight_pre=0.1,
            enable_policy_choice=True,
            policy_num_classes=6,
        ),
    ]

    if args.include_ablations:
        # Ablation: factor loss ON, pressure OFF (tests whether pressure mechanism matters)
        proposal_exps.append(
            ProposalExperiment(
                name="a1_factorloss_no_pressure",
                model_name="GIN",
                factor_loss_enabled=True,
                pressure_enabled=False,
                pressure_type_conditioning="none",
                fix_prob_enabled=False,
                validate_factor_labels=True,
                factor_weight_pre=0.1,
                enable_policy_choice=False,
            )
        )
        # Ablation: pressure ON, factor loss OFF (tests whether auxiliary labels matter)
        proposal_exps.append(
            ProposalExperiment(
                name="a2_pressure_no_factorloss",
                model_name="GIN_PRESSURE",
                factor_loss_enabled=False,
                pressure_enabled=True,
                pressure_type_conditioning="none",
                fix_prob_enabled=False,
                validate_factor_labels=False,
                enable_policy_choice=False,
            )
        )
        # Ablation: pressure typed vs untyped (keep loss same as main model)
        proposal_exps.append(
            ProposalExperiment(
                name="a3_pressure_untyped",
                model_name="GIN_PRESSURE",
                factor_loss_enabled=True,
                pressure_enabled=True,
                pressure_type_conditioning="none",
                fix_prob_enabled=False,
                validate_factor_labels=True,
                factor_weight_pre=0.1,
                enable_policy_choice=False,
            )
        )
        proposal_exps.append(
            ProposalExperiment(
                name="a4_pressure_typed",
                model_name="GIN_PRESSURE",
                factor_loss_enabled=True,
                pressure_enabled=True,
                pressure_type_conditioning="concat",
                fix_prob_enabled=False,
                validate_factor_labels=True,
                factor_weight_pre=0.1,
                enable_policy_choice=False,
            )
        )

    reranker_exps: list[RerankerExperiment] = [
        # M1 reranker: Fix 1 (imitation + no-regression vs gold)
        RerankerExperiment(
            name="m1_fix1_reranker",
            objective="main",
            proposal_ref_config_tag="m1_main_fix1",
            constraint_scope="local",
            disabled=False,
        ),
        # M2: Global Fix reranker (objective global_fix)
        RerankerExperiment(
            name="m2_global_fix_reranker",
            objective="global_fix",
            proposal_ref_config_tag="m1_main_fix1",
            constraint_scope="local",
            disabled=False,
        ),
        # M3: Policy Choice placeholder (disabled until implemented)
        RerankerExperiment(
            name="m3_policy_choice_reranker",
            objective="main",
            proposal_ref_config_tag="m1_main_fix1",
            constraint_scope="local",
            disabled=True,
        ),
    ]

    created = 0
    for variant, encoding, train_graph_path in pairs:
        min_occ = _parse_min_occurrence(variant)
        sample = _load_first_data_obj(train_graph_path)
        num_factor_types = _infer_num_factor_types(sample)

        # Fall back safely if factors aren't yet in Data; you can override manually later.
        # NOTE: if your model code requires this, you *must* make sure it's correct.
        if num_factor_types is None:
            # Conservative default; better to fail loudly than train with wrong size.
            num_factor_types = 0

        # --- proposal configs ---
        for exp in proposal_exps:
            exp_dir = args.models_root / f"{exp.name}__{variant}__{encoding}"
            cfg_path = exp_dir / "config.json"

            payload: dict[str, Any] = {
                "model_config": {
                    "dataset_variant": variant,
                    "encoding": encoding,
                    "model": exp.model_name,
                    "min_occurrence": min_occ,
                    # The following keys are only used if present in your current ModelConfig:
                    "num_factor_types": int(num_factor_types),
                    "pressure_enabled": bool(exp.pressure_enabled),
                    "pressure_type_conditioning": exp.pressure_type_conditioning,
                    "enable_policy_choice": bool(exp.enable_policy_choice),
                    "policy_num_classes": int(exp.policy_num_classes),
                    # optional: "factor_type_embedding_dim": 16,
                },
                "training_config": {
                    "validate_factor_labels": bool(exp.validate_factor_labels),
                    "fix_probability_loss": {
                        "enabled": bool(exp.fix_prob_enabled),
                        # leave schedule defaults unless you want ESWC exact reproduction
                    },
                    "factor_loss": {
                        "enabled": bool(exp.factor_loss_enabled),
                        "weight_pre": float(exp.factor_weight_pre),
                        # keep defaults: only_checkable=True, per_graph_reduction="mean"
                    },
                    "policy_filter_strict": True,
                },
            }
            _write_json(cfg_path, payload)
            created += 1

        # --- reranker configs ---
        for exp in reranker_exps:
            exp_dir = args.models_root / f"{exp.name}__{variant}__{encoding}"
            cfg_path = exp_dir / "config.json"

            payload = {
                "disabled": bool(exp.disabled),
                "model_config": {
                    "dataset_variant": variant,
                    "encoding": encoding,
                    # used for run_dir naming in reranker script; actual reranker is separate
                    "model": "RERANKER",
                    "min_occurrence": min_occ,
                    "num_factor_types": int(num_factor_types),
                },
                "reranker_config": {
                    # rely on defaults in RerankerConfig unless you want to override
                },
                "training_config": {
                    "objective": exp.objective,  # "main" or "global_fix"
                    "constraint_scope": exp.constraint_scope,
                    # you can bump these later; start conservative
                    "topk_candidates": 20,
                    "topk_per_slot": 5,
                    "include_gold": True,
                    "max_candidates_total": 80,
                    "regression_weight": 0.5,  # beta for no-regression vs gold (Fix 1)
                },
                "proposal_config": {
                    # 08_train_reranker.py can resolve this using resolve_run_dir(...)
                    # It uses dataset_variant + encoding from *this* model_config,
                    # model defaults to proposal_config.model or model_cfg.model.
                    "model": "GIN_PRESSURE",
                    "config_tag": exp.proposal_ref_config_tag,
                },
            }
            _write_json(cfg_path, payload)
            created += 1

    print(f"[ok] wrote {created} configs under {args.models_root}")
    print("Next: update scheduler to skip disabled configs and to route to 07_train / 08_train_reranker.")


if __name__ == "__main__":
    main()
