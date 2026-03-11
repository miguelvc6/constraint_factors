from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig, TrainingConfig
from modules.data_encoders import graph_dataset_filename
from modules.model_store import config_tag_from_path


def test_graph_dataset_filename_by_representation() -> None:
    assert graph_dataset_filename("train", "node_id") == "train_graph-node_id.pkl"
    assert (
        graph_dataset_filename("train", "node_id", constraint_representation="eswc_passive")
        == "train_graph_repr-eswc_passive-node_id.pkl"
    )


def test_config_tag_uses_parent_directory_for_config_json() -> None:
    config_path = Path("models/m1d_safe_factor_direct__full_minocc100__node_id/config.json")
    assert config_tag_from_path(config_path) == "m1d_safe_factor_direct__full_minocc100__node_id"


def test_paper_surface_configs_accept_new_fields() -> None:
    model_cfg = ModelConfig.from_mapping({"constraint_representation": "eswc_passive"})
    training_cfg = TrainingConfig.from_mapping(
        {
            "direct_safety": {
                "enabled": True,
                "alpha_primary": 2.0,
                "beta_secondary": 0.25,
                "topk_candidates": 10,
                "max_candidates_total": 40,
            }
        }
    )

    assert model_cfg.constraint_representation == "eswc_passive"
    assert training_cfg.direct_safety.enabled is True
    assert training_cfg.direct_safety.alpha_primary == 2.0
    assert training_cfg.direct_safety.beta_secondary == 0.25


if __name__ == "__main__":
    test_graph_dataset_filename_by_representation()
    test_config_tag_uses_parent_directory_for_config_json()
    test_paper_surface_configs_accept_new_fields()
    print("paper surface tests passed")
