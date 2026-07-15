from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.candidates import CandidateConfig, build_inference_candidates
from modules.class_hierarchy import (
    CLASS_HIERARCHY_FILENAME,
    ClassHierarchy,
    build_training_class_hierarchy,
    write_class_hierarchy,
)
from modules.constraint_semantics import RegistryEntry, build_constraint_instance
from modules.data_encoders import ConstraintGraphData, GlobalIntEncoder
from modules.repair_eval import (
    CandidateRepairs,
    ConstraintRepairHeuristics,
    RepairSample,
    TriplePattern,
    ViolationContext,
    evaluate_global_repair_samples,
)
from modules.provenance import build_run_provenance, file_sha256, write_run_manifest
from modules.reranker_eval import CandidateConstraintEvaluator


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_module(ROOT / "src" / "09_eval.py", "eval_09_integrity_test")
LABELER = _load_module(ROOT / "src" / "05_constraint_labeler.py", "labeler_05_integrity_test")
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


def test_evaluation_builds_feature_identity_lookup_once(monkeypatch) -> None:
    mapping = [0, 1, 1, 2]
    graphs = []
    for _ in range(2):
        graph = ConstraintGraphData(
            x=torch.tensor([1], dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.full((1, 6), 3, dtype=torch.long),
        )
        graph.y_identity = graph.y.clone()
        graph.y_feature = torch.full((1, 6), 2, dtype=torch.long)
        graph.constraint_type = "single"
        graphs.append(graph)

    class DenseFeatureModel(torch.nn.Module):
        def forward(self, data):
            logits = torch.zeros((data.num_graphs, 6, 3), dtype=torch.float32)
            logits[:, :, 2] = 1.0
            return logits

    original_builder = EVAL._build_feature_to_unique_identity
    build_count = 0

    def tracked_builder(identity_to_feature):
        nonlocal build_count
        build_count += 1
        return original_builder(identity_to_feature)

    monkeypatch.setattr(EVAL, "_build_feature_to_unique_identity", tracked_builder)
    metrics = EVAL.eval(
        DenseFeatureModel(),
        graphs,
        batch_size=1,
        device="cpu",
        identity_to_feature=mapping,
    )

    assert build_count == 1
    assert metrics["micro_f1"] == 1.0


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
        uri = f"http://www.wikidata.org/entity/{token}"
        encoder.encode(uri, add_new=True)
        encoder.encode(f"<{uri}>", add_new=True)
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
    )
    required = build_constraint_instance(
        2,
        RegistryEntry(**base),
        encoder=encoder,
        constraint_type_name="itemRequiresStatement",
        constraint_type_id=2,
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


@pytest.mark.parametrize(
    ("relation_mode", "expected_predicates"),
    [
        ("Q21503252", {"P31"}),
        ("Q21514624", {"P279"}),
        ("Q30208840", {"P31", "P279"}),
    ],
)
def test_p2309_relation_modes_resolve_to_statement_predicates(
    relation_mode: str,
    expected_predicates: set[str],
) -> None:
    encoder = _encoder_with_tokens(
        "P10",
        "P31",
        "P279",
        "P2308",
        "P2309",
        "Q30",
        relation_mode,
    )
    entry = RegistryEntry(
        constraint_type_raw="",
        constraint_type_item="Q21503250",
        constraint_type_index=1,
        constraint_family="type",
        constraint_label="type",
        constraint_family_supported=True,
        constrained_property_raw="P10",
        param_predicates_raw=("P2308", "P2309"),
        param_objects_raw=("Q30", relation_mode),
    )
    instance = build_constraint_instance(
        1,
        entry,
        encoder=encoder,
        constraint_type_name="type",
        constraint_type_id=1,
    )

    decoded = {
        str(encoder.decode(predicate)).rsplit("/", 1)[-1]
        for predicate in instance.relation_predicates
    }
    assert decoded == expected_predicates


def test_type_constraint_without_mandatory_p2309_is_unresolved() -> None:
    encoder = _encoder_with_tokens("P10", "P31", "P279", "P2308", "Q30")
    entry = RegistryEntry(
        constraint_type_raw="",
        constraint_type_item="Q21503250",
        constraint_type_index=1,
        constraint_family="type",
        constraint_label="type",
        constraint_family_supported=True,
        constrained_property_raw="P10",
        param_predicates_raw=("P2308",),
        param_objects_raw=("Q30",),
    )
    instance = build_constraint_instance(
        1,
        entry,
        encoder=encoder,
        constraint_type_name="type",
        constraint_type_id=1,
    )
    assert instance.relation_predicates == []


def test_type_primary_reconstructs_pre_state_before_gold_addition() -> None:
    encoder = _encoder_with_tokens(
        "P10",
        "P31",
        "P279",
        "P2308",
        "P2309",
        "Q1",
        "Q2",
        "Q30",
        "Q21503252",
    )
    encoder._frozen = False
    subject_placeholder = encoder.encode("subject", add_new=True)
    constraint_id = encoder.encode("constraint::type-test", add_new=True)
    encoder.freeze()

    def encoded(token: str) -> int:
        return encoder.encode(
            f"http://www.wikidata.org/entity/{token}",
            add_new=False,
        )

    subject = encoded("Q1")
    obj = encoded("Q2")
    constrained_property = encoded("P10")
    instance_of = encoded("P31")
    allowed_class = encoded("Q30")
    row = {
        "constraint_id": constraint_id,
        "constraint_type": "type",
        "subject": subject,
        "predicate": constrained_property,
        "object": obj,
        "other_subject": 0,
        "other_predicate": 0,
        "other_object": 0,
        "add_subject": subject_placeholder,
        "add_predicate": instance_of,
        "add_object": allowed_class,
        "del_subject": 0,
        "del_predicate": 0,
        "del_object": 0,
        # The serialized snapshot already contains the later addition.
        "subject_predicates": [constrained_property, instance_of],
        "subject_objects": [obj, allowed_class],
        "object_predicates": [],
        "object_objects": [],
        "other_entity_predicates": [],
        "other_entity_objects": [],
        "local_constraint_ids": [constraint_id],
        "local_constraint_ids_focus": [constraint_id],
    }
    registry = {
        constraint_id: RegistryEntry(
            constraint_type_raw="",
            constraint_type_item="Q21503250",
            constraint_type_index=1,
            constraint_family="type",
            constraint_label="type",
            constraint_family_supported=True,
            constrained_property_raw="P10",
            param_predicates_raw=("P2308", "P2309"),
            param_objects_raw=("Q30", "Q21503252"),
        )
    }

    labeled, _, _ = LABELER._process_dataframe(
        pd.DataFrame([row]),
        registry,
        encoder=encoder,
        assume_complete=True,
        use_encoded_ids=True,
        constraint_scope="local",
        factor_family_policy="supported_only",
    )
    result = labeled.iloc[0]
    assert bool(result.primary_checkable_pre) is True
    assert int(result.primary_satisfied_pre) == 0
    assert bool(result.primary_checkable_post_gold) is True
    assert int(result.primary_satisfied_post_gold) == 1
    assert result.primary_validation_reason == "valid"

    context = ViolationContext(
        constraint_type="type",
        constraint_id=constraint_id,
        subject=subject,
        predicate=constrained_property,
        object=obj,
        other_subject=0,
        other_predicate=0,
        other_object=0,
        constraint_predicates=(encoded("P2308"), encoded("P2309")),
        constraint_objects=(allowed_class, encoded("Q21503252")),
    )
    heuristics = ConstraintRepairHeuristics(
        encoder=encoder,
        placeholder_ids={},
        none_class=0,
    )
    additions = heuristics.candidates_for(context).add
    assert additions
    assert {predicate for pattern in additions for predicate in (pattern.predicates or ())} == {
        instance_of
    }


def test_type_primary_uses_transitive_training_hierarchy() -> None:
    encoder = _encoder_with_tokens(
        "P10",
        "P31",
        "P279",
        "P2308",
        "P2309",
        "Q1",
        "Q2",
        "Q5",
        "Q30",
        "Q21503252",
    )
    encoder._frozen = False
    subject_placeholder = encoder.encode("subject", add_new=True)
    constraint_id = encoder.encode("constraint::transitive-type-test", add_new=True)
    encoder.freeze()

    def encoded(token: str) -> int:
        return encoder.encode(
            f"http://www.wikidata.org/entity/{token}",
            add_new=False,
        )

    subject = encoded("Q1")
    obj = encoded("Q2")
    constrained_property = encoded("P10")
    instance_of = encoded("P31")
    human = encoded("Q5")
    person = encoded("Q30")
    row = {
        "constraint_id": constraint_id,
        "constraint_type": "type",
        "subject": subject,
        "predicate": constrained_property,
        "object": obj,
        "other_subject": 0,
        "other_predicate": 0,
        "other_object": 0,
        "add_subject": subject_placeholder,
        "add_predicate": instance_of,
        "add_object": human,
        "del_subject": 0,
        "del_predicate": 0,
        "del_object": 0,
        "subject_predicates": [constrained_property],
        "subject_objects": [obj],
        "object_predicates": [],
        "object_objects": [],
        "other_entity_predicates": [],
        "other_entity_objects": [],
        "local_constraint_ids": [constraint_id],
        "local_constraint_ids_focus": [constraint_id],
    }
    registry = {
        constraint_id: RegistryEntry(
            constraint_type_raw="",
            constraint_type_item="Q21503250",
            constraint_type_index=1,
            constraint_family="type",
            constraint_label="type",
            constraint_family_supported=True,
            constrained_property_raw="P10",
            param_predicates_raw=("P2308", "P2309"),
            param_objects_raw=("Q30", "Q21503252"),
        )
    }
    hierarchy = ClassHierarchy.from_edges([(human, person), (person, human)])

    labeled, _, _ = LABELER._process_dataframe(
        pd.DataFrame([row]),
        registry,
        encoder=encoder,
        assume_complete=True,
        use_encoded_ids=True,
        constraint_scope="local",
        factor_family_policy="supported_only",
        class_hierarchy=hierarchy,
    )
    result = labeled.iloc[0]
    assert int(result.primary_satisfied_pre) == 0
    assert int(result.primary_satisfied_post_gold) == 1
    assert result.primary_gold_repair_status == "verified"
    assert bool(result.primary_gold_repair_verified) is True
    assert hierarchy.ancestors_including_self(human) == frozenset({human, person})


def test_training_hierarchy_artifact_is_deterministic(tmp_path: Path) -> None:
    train_path = tmp_path / "df_train.parquet"
    pd.DataFrame(
        [
            {
                "subject": 10,
                "subject_predicates": [9, 9],
                "subject_objects": [20, 20],
                "object": 20,
                "object_predicates": [9],
                "object_objects": [30],
                "other_subject": 10,
                "other_object": 40,
                "other_entity_predicates": [9],
                "other_entity_objects": [50],
            }
        ]
    ).to_parquet(train_path, index=False)

    hierarchy = build_training_class_hierarchy(
        train_path,
        p279_predicate_id=9,
        batch_size=1,
    )
    assert list(hierarchy.direct_edges()) == [(10, 20), (20, 30), (40, 50)]
    assert hierarchy.reaches_any([10], {30}) is True

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = write_class_hierarchy(
        hierarchy,
        first,
        p279_predicate_id=9,
        source_dataset_variant="fixture",
        source_manifest_path=None,
    )
    second_manifest = write_class_hierarchy(
        hierarchy,
        second,
        p279_predicate_id=9,
        source_dataset_variant="fixture",
        source_manifest_path=None,
    )
    assert first_manifest["outputs"] == second_manifest["outputs"]
    assert list(ClassHierarchy.load(first / CLASS_HIERARCHY_FILENAME).direct_edges()) == [
        (10, 20),
        (20, 30),
        (40, 50),
    ]


def test_shared_subject_comparison_facts_belong_to_other_object() -> None:
    row = next(
        pd.DataFrame(
            [
                {
                    "subject": 10,
                    "predicate": 2,
                    "object": 20,
                    "other_subject": 10,
                    "other_predicate": 3,
                    "other_object": 30,
                    "subject_predicates": [],
                    "subject_objects": [],
                    "object_predicates": [],
                    "object_objects": [],
                    "other_entity_predicates": [9],
                    "other_entity_objects": [40],
                }
            ]
        ).itertuples(index=False)
    )
    facts, _ = LABELER._build_facts_state(
        row,
        p_local={2, 3, 9},
        assume_complete=True,
        cast_int=True,
    )
    assert facts[30][9] == {40}
    assert 9 not in facts[10]


def test_candidate_evaluator_loads_persisted_class_hierarchy(tmp_path: Path) -> None:
    encoder = _encoder_with_tokens(
        "P10",
        "P31",
        "P2308",
        "P2309",
        "Q1",
        "Q2",
        "Q5",
        "Q30",
        "Q999",
        "Q21503250",
        "Q21503252",
    )

    def encoded(token: str) -> int:
        return encoder.encode(
            f"http://www.wikidata.org/entity/{token}",
            add_new=False,
        )

    registry_payload = {
        "Q999": {
            "constraint_type": "Q21503250",
            "constraint_type_item": "Q21503250",
            "constraint_type_index": 1,
            "constraint_family": "type",
            "constraint_label": "subject type constraint",
            "constraint_family_supported": True,
            "constrained_property": "P10",
            "param_predicates": ["P2308", "P2309"],
            "param_objects": ["Q30", "Q21503252"],
        }
    }
    registry_path = tmp_path / "registry.parquet"
    pd.DataFrame(
        {"registry_json": [json.dumps(registry_payload)]}
    ).to_parquet(registry_path, index=False)

    hierarchy_root = tmp_path / "hierarchy"
    write_class_hierarchy(
        ClassHierarchy.from_edges([(encoded("Q5"), encoded("Q30"))]),
        hierarchy_root,
        p279_predicate_id=1,
        source_dataset_variant="fixture",
        source_manifest_path=None,
    )
    row = next(
        pd.DataFrame(
            [
                {
                    "constraint_id": encoded("Q999"),
                    "constraint_type": "type",
                    "subject": encoded("Q1"),
                    "predicate": encoded("P10"),
                    "object": encoded("Q2"),
                    "other_subject": 0,
                    "other_predicate": 0,
                    "other_object": 0,
                    "add_subject": encoded("Q1"),
                    "add_predicate": encoded("P31"),
                    "add_object": encoded("Q5"),
                    "del_subject": 0,
                    "del_predicate": 0,
                    "del_object": 0,
                    "subject_predicates": [encoded("P10")],
                    "subject_objects": [encoded("Q2")],
                    "object_predicates": [],
                    "object_objects": [],
                    "other_entity_predicates": [],
                    "other_entity_objects": [],
                    "factor_constraint_ids": [encoded("Q999")],
                    "local_constraint_ids": [encoded("Q999")],
                }
            ]
        ).itertuples(index=False)
    )
    candidate = [
        encoded("Q1"),
        encoded("P31"),
        encoded("Q5"),
        0,
        0,
        0,
    ]
    direct_evaluator = CandidateConstraintEvaluator(
        str(registry_path),
        encoder=encoder,
        assume_complete=True,
        constraint_scope="local",
        use_encoded_ids=True,
    )
    hierarchy_evaluator = CandidateConstraintEvaluator(
        str(registry_path),
        encoder=encoder,
        assume_complete=True,
        constraint_scope="local",
        use_encoded_ids=True,
        class_hierarchy_path=hierarchy_root / CLASS_HIERARCHY_FILENAME,
    )

    assert direct_evaluator.evaluate_full(
        row,
        candidate_slots=candidate,
        primary_factor_index=0,
    )["primary_satisfied"] == 0
    assert hierarchy_evaluator.evaluate_full(
        row,
        candidate_slots=candidate,
        primary_factor_index=0,
    )["primary_satisfied"] == 1


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
    assert payload["training_config"]["num_epochs"] == 15
    assert payload["training_config"]["validation_subset_size"] is None

    reranker_payload = CONFIG_GENERATOR._reranker_config_payload(
        exp=CONFIG_GENERATOR.RerankerExperiment(
            name="g0_globalfix_reference",
            objective="global_fix",
            proposal_name="a1_factorized_imitation",
        ),
        variant="toy_minocc2",
        encoding="node_id",
        min_occurrence=2,
        num_factor_types=3,
        proposal_config_tag="a1_factorized_imitation",
    )
    assert reranker_payload["training_config"]["num_epochs"] == 15
    assert reranker_payload["training_config"]["validation_subset_size"] is None


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
