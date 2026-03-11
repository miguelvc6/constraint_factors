#!/usr/bin/env python3
"""
Generate the paper-facing experiment bundle under ``models/<exp_name>/config.json``.

Default output:
- ``b0_eswc_reproduction``
- ``a1_factorized_imitation``
- ``m1c_safe_factor_chooser``
- ``m1d_safe_factor_direct``
- ``g0_globalfix_reference``

Optional appendix / ablation configs are only emitted with ``--include-experimental``.
"""

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from modules.data_encoders import graph_dataset_filename

VARIANT_MINOCC_RE = re.compile(r"minocc(\d+)", re.IGNORECASE)
FACTORIZED_RE = re.compile(r"^train_graph-(?P<encoding>.+)\.pkl$")
FACTORIZED_SHARD_RE = re.compile(r"^train_graph-(?P<encoding>.+)-shard\d+\.(?:pkl|pt)$")
PASSIVE_RE = re.compile(r"^train_graph_repr-eswc_passive-(?P<encoding>.+)\.pkl$")
PASSIVE_SHARD_RE = re.compile(r"^train_graph_repr-eswc_passive-(?P<encoding>.+)-shard\d+\.(?:pkl|pt)$")


def _parse_min_occurrence(variant: str) -> int:
    match = VARIANT_MINOCC_RE.search(variant)
    return int(match.group(1)) if match else 1


def _torch_load_trusted(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _discover_artifacts(path: Path) -> list[Path]:
    if path.exists():
        return [path]
    artifacts: list[Path] = []
    artifacts.extend(sorted(path.parent.glob(f"{path.stem}-shard*.pkl")))
    artifacts.extend(sorted(path.parent.glob(f"{path.stem}-shard*.pt")))
    return artifacts


def _load_first_data_obj(path: Path) -> Any | None:
    try:
        artifacts = _discover_artifacts(path)
        if not artifacts:
            return None
        first_path = artifacts[0]
        if first_path.suffix == ".pt":
            payload = _torch_load_trusted(first_path)
        else:
            with first_path.open("rb") as fh:
                payload = pickle.Unpickler(fh).load()
        if isinstance(payload, list):
            return payload[0] if payload else None
        return payload
    except Exception:
        return None


def _infer_num_factor_types(sample_data: Any) -> int:
    if sample_data is None:
        return 0
    for attr in ("factor_types", "factor_type_id", "factor_type_ids"):
        if hasattr(sample_data, attr):
            value = getattr(sample_data, attr)
            try:
                return int(value.max().item()) + 1
            except Exception:
                pass
    return 0


def _iter_variant_encodings(processed_root: Path) -> Iterable[tuple[str, str]]:
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    for variant_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        encodings: set[str] = set()
        for candidate in sorted(variant_dir.iterdir()):
            if not candidate.is_file():
                continue
            for pattern in (FACTORIZED_RE, FACTORIZED_SHARD_RE, PASSIVE_RE, PASSIVE_SHARD_RE):
                match = pattern.match(candidate.name)
                if match:
                    encodings.add(match.group("encoding"))
                    break
        for encoding in sorted(encodings):
            yield variant_dir.name, encoding


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


@dataclass(frozen=True)
class ProposalExperiment:
    name: str
    model_name: str
    constraint_representation: str
    pressure_enabled: bool
    pressure_type_conditioning: str
    chooser_enabled: bool = False
    chooser_loss_mode: str = "fix1"
    chooser_loss_weight: float = 0.5
    chooser_beta_no_regression: float = 0.5
    chooser_gamma_primary: float = 1.0
    direct_safety_enabled: bool = False
    direct_safety_alpha_primary: float = 1.0
    direct_safety_beta_secondary: float = 0.5
    validate_factor_labels: bool = False
    include_gold_candidates: bool = True
    enable_policy_choice: bool = False


@dataclass(frozen=True)
class RerankerExperiment:
    name: str
    objective: str
    proposal_name: str
    constraint_scope: str = "local"


def _proposal_config_payload(
    *,
    exp: ProposalExperiment,
    variant: str,
    encoding: str,
    min_occurrence: int,
    num_factor_types: int,
) -> dict[str, Any]:
    dynamic_reweighting = exp.name != "b0_eswc_reproduction"
    return {
        "model_config": {
            "dataset_variant": variant,
            "encoding": encoding,
            "min_occurrence": min_occurrence,
            "model": exp.model_name,
            "constraint_representation": exp.constraint_representation,
            "factor_executor_impl": "per_type_v1",
            "use_edge_attributes": True,
            "use_edge_subtraction": False,
            "use_role_embeddings": True,
            "role_embedding_dim": 16,
            "pressure_enabled": exp.pressure_enabled,
            "pressure_type_conditioning": exp.pressure_type_conditioning,
            "pressure_residual_scale": 0.1,
            "num_factor_types": int(num_factor_types),
            "enable_policy_choice": exp.enable_policy_choice,
            "policy_num_classes": 6,
        },
        "training_config": {
            "batch_size": 256,
            "num_epochs": 30,
            "early_stopping_rounds": 6,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 2,
            "num_workers": 4,
            "pin_memory": True,
            "validate_factor_labels": exp.validate_factor_labels,
            "constraint_loss": {
                "dynamic_reweighting": {
                    "enabled": dynamic_reweighting,
                }
            },
            "fix_probability_loss": {
                "enabled": False,
            },
            "factor_loss": {
                "enabled": exp.constraint_representation == "factorized",
                "weight_pre": 0.1,
                "weight_post_gold": 0.1,
            },
            "chooser": {
                "enabled": exp.chooser_enabled,
                "loss_mode": exp.chooser_loss_mode,
                "loss_weight": exp.chooser_loss_weight,
                "beta_no_regression": exp.chooser_beta_no_regression,
                "gamma_primary": exp.chooser_gamma_primary,
                "topk_candidates": 20,
                "max_candidates_total": 80,
            },
            "direct_safety": {
                "enabled": exp.direct_safety_enabled,
                "alpha_primary": exp.direct_safety_alpha_primary,
                "beta_secondary": exp.direct_safety_beta_secondary,
                "topk_candidates": 20,
                "max_candidates_total": 80,
            },
            "policy_filter_strict": True,
        },
    }


def _reranker_config_payload(
    *,
    exp: RerankerExperiment,
    variant: str,
    encoding: str,
    min_occurrence: int,
    num_factor_types: int,
    proposal_config_tag: str,
) -> dict[str, Any]:
    return {
        "model_config": {
            "dataset_variant": variant,
            "encoding": encoding,
            "model": "RERANKER",
            "min_occurrence": min_occurrence,
            "constraint_representation": "factorized",
            "num_factor_types": int(num_factor_types),
        },
        "reranker_config": {},
        "training_config": {
            "batch_size": 64,
            "num_epochs": 20,
            "early_stopping_rounds": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 2,
            "objective": exp.objective,
            "regression_weight": 0.5,
            "topk_candidates": 20,
            "topk_per_slot": 5,
            "heuristic_max_candidates": 30,
            "heuristic_max_values": 3,
            "include_gold": True,
            "max_candidates_total": 80,
            "assume_complete_entity_facts": True,
            "constraint_scope": exp.constraint_scope,
        },
        "proposal_config": {
            "model": "GIN_PRESSURE",
            "config_tag": proposal_config_tag,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--limit", type=int, default=0, help="Limit variant/encoding pairs (0 = no limit).")
    parser.add_argument("--include-experimental", action="store_true")
    args = parser.parse_args()

    pairs = list(_iter_variant_encodings(args.processed_root))
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(
            "No graph artifacts found under "
            f"{args.processed_root}.\n"
            "Build the paper graph artifacts first, for example:\n"
            "  PYTHONPATH=src .venv/bin/python src/05_constraint_labeler.py --dataset full --min-occurrence 100 --constraint-scope local\n"
            "  PYTHONPATH=src .venv/bin/python src/06_graph.py --dataset full --min-occurrence 100 --encoding node_id --constraint-scope local --constraint-representation factorized\n"
            "  PYTHONPATH=src .venv/bin/python src/06_graph.py --dataset full --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive"
        )

    canonical_proposals: list[ProposalExperiment] = [
        ProposalExperiment(
            name="b0_eswc_reproduction",
            model_name="GIN",
            constraint_representation="eswc_passive",
            pressure_enabled=False,
            pressure_type_conditioning="none",
            validate_factor_labels=False,
        ),
        ProposalExperiment(
            name="a1_factorized_imitation",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            validate_factor_labels=True,
        ),
        ProposalExperiment(
            name="m1c_safe_factor_chooser",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            chooser_enabled=True,
            chooser_loss_mode="fix1",
            chooser_loss_weight=0.5,
            chooser_beta_no_regression=0.5,
            chooser_gamma_primary=1.0,
            validate_factor_labels=True,
        ),
        ProposalExperiment(
            name="m1d_safe_factor_direct",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            direct_safety_enabled=True,
            direct_safety_alpha_primary=1.0,
            direct_safety_beta_secondary=0.5,
            validate_factor_labels=True,
        ),
    ]
    canonical_rerankers: list[RerankerExperiment] = [
        RerankerExperiment(
            name="g0_globalfix_reference",
            objective="global_fix",
            proposal_name="a1_factorized_imitation",
            constraint_scope="local",
        )
    ]

    experimental_proposals: list[ProposalExperiment] = [
        ProposalExperiment(
            name="x1_policy_choice_appendix",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            validate_factor_labels=True,
            enable_policy_choice=True,
        ),
        ProposalExperiment(
            name="x2_factor_loss_only_appendix",
            model_name="GIN",
            constraint_representation="factorized",
            pressure_enabled=False,
            pressure_type_conditioning="none",
            validate_factor_labels=True,
        ),
    ]
    experimental_rerankers: list[RerankerExperiment] = [
        RerankerExperiment(
            name="x3_fix1_reranker_appendix",
            objective="main",
            proposal_name="a1_factorized_imitation",
            constraint_scope="local",
        )
    ]

    created = 0
    for variant, encoding in pairs:
        min_occurrence = _parse_min_occurrence(variant)
        factorized_path = args.processed_root / variant / graph_dataset_filename("train", encoding)
        passive_path = args.processed_root / variant / graph_dataset_filename(
            "train",
            encoding,
            constraint_representation="eswc_passive",
        )
        sample = _load_first_data_obj(factorized_path) or _load_first_data_obj(passive_path)
        num_factor_types = _infer_num_factor_types(sample)

        proposal_experiments = list(canonical_proposals)
        reranker_experiments = list(canonical_rerankers)
        if args.include_experimental:
            proposal_experiments.extend(experimental_proposals)
            reranker_experiments.extend(experimental_rerankers)

        for exp in proposal_experiments:
            exp_dir_name = f"{exp.name}__{variant}__{encoding}"
            cfg_path = args.models_root / exp_dir_name / "config.json"
            payload = _proposal_config_payload(
                exp=exp,
                variant=variant,
                encoding=encoding,
                min_occurrence=min_occurrence,
                num_factor_types=num_factor_types,
            )
            if exp.name == "x2_factor_loss_only_appendix":
                payload["training_config"]["factor_loss"]["enabled"] = True
            _write_json(cfg_path, payload)
            created += 1

        for exp in reranker_experiments:
            exp_dir_name = f"{exp.name}__{variant}__{encoding}"
            proposal_config_tag = f"{exp.proposal_name}__{variant}__{encoding}"
            cfg_path = args.models_root / exp_dir_name / "config.json"
            payload = _reranker_config_payload(
                exp=exp,
                variant=variant,
                encoding=encoding,
                min_occurrence=min_occurrence,
                num_factor_types=num_factor_types,
                proposal_config_tag=proposal_config_tag,
            )
            _write_json(cfg_path, payload)
            created += 1

    print(f"[ok] wrote {created} configs under {args.models_root}")


if __name__ == "__main__":
    main()
