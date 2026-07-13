from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.candidates import CandidateConfig, build_inference_candidates
from modules.constraint_semantics import RegistryEntry, build_constraint_instance
from modules.data_encoders import ConstraintGraphData, GlobalIntEncoder
from modules.repair_eval import (
    CandidateRepairs,
    RepairSample,
    TriplePattern,
    ViolationContext,
    evaluate_global_repair_samples,
)
from modules.provenance import build_run_provenance, file_sha256, write_run_manifest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_module(ROOT / "src" / "09_eval.py", "eval_09_integrity_test")
CONFIG_GENERATOR = _load_module(
    ROOT / "scripts" / "make_experiment_configs.py",
    "make_experiment_configs_integrity_test",
)


class _Heuristics:
    placeholder_ids: dict[str, int] = {}

    def candidates_for(self, context: ViolationContext) -> CandidateRepairs:
        del context
        return CandidateRepairs(
            add=[TriplePattern(frozenset({2}), frozenset({3}), frozenset({4}))]
        )


def _context() -> ViolationContext:
    return ViolationContext(
        constraint_type="single",
        constraint_id=10,
        subject=2,
        predicate=3,
        object=4,
        other_subject=0,
        other_predicate=0,
        other_object=0,
        constraint_predicates=(),
        constraint_objects=(),
    )


def test_inference_candidates_have_no_gold_input_and_preserve_identity() -> None:
    logits = torch.zeros((6, 5), dtype=torch.float32)
    mapping = [0, 1, 2, 3, 4, 1]
    candidates = build_inference_candidates(
        context=_context(),
        heuristics=_Heuristics(),
        proposal_logits=logits,
        cfg=CandidateConfig(force_include_gold_train=True),
        placeholder_ids=set(),
        num_target_ids=5,
        identity_to_feature=mapping,
    )

    assert candidates
    assert candidates[0].identity_slots[:3] == (2, 3, 4)
    assert candidates[0].feature_slots[:3] == (2, 3, 4)
    assert all(candidate.source != "gold_train" for candidate in candidates)
    with pytest.raises(TypeError):
        build_inference_candidates(  # type: ignore[call-arg]
            graph=object(),
            context=_context(),
            heuristics=_Heuristics(),
            proposal_logits=logits,
            cfg=CandidateConfig(),
            placeholder_ids=set(),
            num_target_ids=5,
        )


def test_feature_predictions_only_resolve_unique_identities() -> None:
    mapping = [0, 1, 1, 2]
    features = torch.tensor([[0, 1, 2]], dtype=torch.long)
    identities = EVAL._feature_to_unique_identity(features, mapping)
    assert identities.tolist() == [[0, -1, 3]]


def _details(*, secondary_count: int, regressions: int, primary_checkable_post: bool = True):
    pre_checkable = [True] * (secondary_count + 1)
    pre_satisfied = [0] + [1] * secondary_count
    post_checkable = [primary_checkable_post] + [True] * secondary_count
    post_satisfied = [1 if primary_checkable_post else 0] + [1] * secondary_count
    for index in range(regressions):
        post_satisfied[index + 1] = 0
    return {
        "local_constraint_ids": list(range(secondary_count + 1)),
        "primary_factor_index": 0,
        "pre_checkable": pre_checkable,
        "pre_satisfied": pre_satisfied,
        "post_checkable": post_checkable,
        "post_satisfied": post_satisfied,
        "focus_preserved": 1,
        "focus_deleted": 0,
        "candidate_deletes_focus": 0,
        "pre_global_satisfied_fraction": 0.0,
        "post_global_satisfied_fraction": 1.0,
    }


class _Evaluator:
    def __init__(self, details):
        self.details = iter(details)

    def evaluate_full(self, *args, **kwargs):
        del args, kwargs
        return next(self.details)


def test_global_metrics_use_pooled_srr_and_primary_transitions() -> None:
    details = [
        _details(secondary_count=1, regressions=1),
        _details(secondary_count=9, regressions=0),
        _details(secondary_count=0, regressions=0, primary_checkable_post=False),
    ]
    empty = {"add": None, "del": None}
    samples = [RepairSample("single", predicted=empty, gold=empty) for _ in details]
    metrics = evaluate_global_repair_samples(
        samples=samples,
        rows=[object() for _ in details],
        evaluator=_Evaluator(details),
        none_class=0,
    )["overall"]

    assert metrics["srr"] == pytest.approx(0.1)
    assert metrics["srr_macro_defined"] == pytest.approx(0.5)
    assert metrics["srr_defined_support"] == 2
    assert metrics["primary_fix_rate"] == pytest.approx(2 / 3)
    assert metrics["primary_post_uncheckable_total"] == 1


def _encoder_with_tokens(*tokens: str) -> GlobalIntEncoder:
    encoder = GlobalIntEncoder()
    for token in tokens:
        encoder.encode(f"http://www.wikidata.org/entity/{token}", add_new=True)
    encoder.freeze()
    return encoder


def test_registry_parameter_semantics_are_family_specific() -> None:
    encoder = _encoder_with_tokens("P10", "P20", "Q30", "Q40")
    base = dict(
        constraint_type_raw="",
        constraint_type_item="",
        constraint_type_index=1,
        constraint_family="",
        constraint_label="",
        constraint_family_supported=True,
        constrained_property_raw="P10",
        param_predicates_raw=("P2306", "P2305", "P2303", "P4680"),
        param_objects_raw=("P20", "Q30", "Q40", "Q46466787"),
    )

    inverse = build_constraint_instance(
        1,
        RegistryEntry(**base),
        encoder=encoder,
        constraint_type_name="inverse",
        constraint_type_id=1,
        default_relation_predicates=[],
    )
    required = build_constraint_instance(
        2,
        RegistryEntry(**base),
        encoder=encoder,
        constraint_type_name="itemRequiresStatement",
        constraint_type_id=2,
        default_relation_predicates=[],
    )

    p20 = encoder.encode("http://www.wikidata.org/entity/P20", add_new=False)
    q30 = encoder.encode("http://www.wikidata.org/entity/Q30", add_new=False)
    q40 = encoder.encode("http://www.wikidata.org/entity/Q40", add_new=False)
    assert inverse.inverse_properties == [p20]
    assert inverse.required_properties == set()
    assert required.required_properties == {p20}
    assert required.allowed_items == {q30}
    assert required.exceptions == {q40}
    assert required.applies_to_main_value is True


def test_constraint_graph_data_offsets_local_node_references_only() -> None:
    def graph() -> ConstraintGraphData:
        return ConstraintGraphData(
            x=torch.tensor([1, 2, 3]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr_non_flattened=torch.tensor([1]),
            factor_node_index=torch.tensor([2]),
            primary_factor_index=0,
        )

    batch = Batch.from_data_list([graph(), graph()])
    assert batch.edge_attr_non_flattened.tolist() == [1, 4]
    assert batch.factor_node_index.tolist() == [2, 5]
    assert batch.primary_factor_index.tolist() == [0, 0]


def test_canonical_a1_configuration_uses_shared_pressure() -> None:
    experiment = CONFIG_GENERATOR.ProposalExperiment(
        name="a1_factorized_imitation",
        model_name="GIN_PRESSURE",
        constraint_representation="factorized",
        pressure_enabled=True,
        pressure_type_conditioning="concat",
    )
    payload = CONFIG_GENERATOR._proposal_config_payload(
        exp=experiment,
        variant="toy_minocc2",
        encoding="node_id",
        min_occurrence=2,
        num_factor_types=3,
    )
    assert payload["model_config"]["pressure_module_sharing"] == "shared"


def test_run_provenance_hashes_effective_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pth"
    dataset_manifest = tmp_path / "dataset_manifest.json"
    graph_manifest = tmp_path / "train_graph.pkl.manifest.json"
    config_path.write_text(
        json.dumps({"model_config": {"model": "toy"}, "training_config": {"epochs": 1}}),
        encoding="utf-8",
    )
    torch.save({"model_state": {}}, checkpoint_path)
    dataset_manifest.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    graph_manifest.write_text(json.dumps({"graph_schema_version": 2}), encoding="utf-8")
    model = torch.nn.Linear(2, 3)

    payload = build_run_provenance(
        repository_root=ROOT,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model=model,
        seed=42,
        dataset_manifest_path=dataset_manifest,
        graph_manifest_paths=[graph_manifest],
    )
    manifest_path = write_run_manifest(tmp_path, payload)

    assert payload["checkpoint"]["sha256"] == file_sha256(checkpoint_path)
    assert payload["parameters"]["total"] == 9
    assert payload["source"]["source_tree_sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["seed"] == 42
