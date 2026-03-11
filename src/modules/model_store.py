"""Utilities for organizing model artifacts under the unified ``models/`` directory."""

import re
from pathlib import Path
from typing import Iterable

MODELS_ROOT = Path("models")
TEMPLATES_DIR = MODELS_ROOT / "templates"
DEFAULT_CONFIG_TAG = "default"
DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"
TRAINING_HISTORY_NAME = "training_history.json"
CONFIG_FILE_NAME = "config.json"
EVAL_DIR_NAME = "evaluations"

_SANITIZE_REGEX = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_fragment(name: str | Path | None, fallback: str = DEFAULT_CONFIG_TAG) -> str:
    """Return a filesystem-friendly fragment derived from ``name``."""
    if name is None:
        return fallback
    if isinstance(name, Path):
        name = name.stem
    if not isinstance(name, str):
        name = str(name)
    cleaned = _SANITIZE_REGEX.sub("_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def config_tag_from_path(path: Path | None) -> str:
    """Derive a config tag from a given path."""
    if path is None:
        return DEFAULT_CONFIG_TAG
    candidate = Path(path)
    if candidate.name == CONFIG_FILE_NAME and candidate.parent.name:
        return sanitize_fragment(candidate.parent.name, fallback=DEFAULT_CONFIG_TAG)
    return sanitize_fragment(candidate, fallback=DEFAULT_CONFIG_TAG)


def run_slug(dataset_variant: str, encoding: str, model_name: str, config_tag: str | None = None) -> str:
    """Build a normalized slug identifying a model run."""
    parts = [
        sanitize_fragment(dataset_variant, fallback="dataset"),
        sanitize_fragment(encoding, fallback="encoding"),
        sanitize_fragment(model_name, fallback="model").upper(),
        sanitize_fragment(config_tag, fallback=DEFAULT_CONFIG_TAG),
    ]
    return "-".join(parts[:2]) + f"_{parts[2]}_{parts[3]}"


def run_dir(dataset_variant: str, encoding: str, model_name: str, config_tag: str | None = None) -> Path:
    """Return the directory path for a specific model run."""
    slug = run_slug(dataset_variant, encoding, model_name, config_tag)
    return MODELS_ROOT / slug


def ensure_run_dir(dataset_variant: str, encoding: str, model_name: str, config_tag: str | None = None) -> Path:
    """Create the run directory if needed and return it."""
    directory = run_dir(dataset_variant, encoding, model_name, config_tag)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_checkpoint_path(run_directory: Path) -> Path:
    """Return the checkpoint file location within ``run_directory``."""
    return run_directory / DEFAULT_CHECKPOINT_NAME


def history_path(run_directory: Path) -> Path:
    """Return the training history file path for a run."""
    return run_directory / TRAINING_HISTORY_NAME


def config_copy_path(run_directory: Path) -> Path:
    """Return the stored config copy path for a run."""
    return run_directory / CONFIG_FILE_NAME


def evaluations_dir(run_directory: Path, create: bool = False) -> Path:
    """Return the evaluations directory, optionally creating it."""
    path = run_directory / EVAL_DIR_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def baseline_dir(dataset_variant: str, encoding: str, create: bool = False) -> Path:
    """Return the baseline directory for a dataset/encoding pair."""
    directory = MODELS_ROOT / "baselines" / sanitize_fragment(dataset_variant) / sanitize_fragment(encoding)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_run_dirs(dataset_variant: str, encoding: str, model_name: str) -> list[Path]:
    """List all stored run directories matching the given identifiers."""
    slug_with_placeholder = run_slug(dataset_variant, encoding, model_name, config_tag="placeholder")
    prefix = slug_with_placeholder.rsplit("_", 1)[0] + "_"
    candidates: list[Path] = []
    if not MODELS_ROOT.exists():
        return candidates
    for entry in MODELS_ROOT.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(prefix):
            candidates.append(entry)
    return sorted(candidates)


def resolve_run_dir(
    dataset_variant: str,
    encoding: str,
    model_name: str,
    config_tag: str | None = None,
) -> Path:
    """Resolve a unique run directory, optionally constrained by ``config_tag``."""
    if config_tag:
        candidate = run_dir(dataset_variant, encoding, model_name, config_tag)
        if not candidate.exists():
            raise FileNotFoundError(
                f"Model run directory not found for dataset={dataset_variant}, encoding={encoding}, "
                f"model={model_name}, config_tag={config_tag}. Expected at {candidate}"
            )
        return candidate

    candidates = list_run_dirs(dataset_variant, encoding, model_name)
    if not candidates:
        raise FileNotFoundError(
            f"No model runs stored for dataset={dataset_variant}, encoding={encoding}, model={model_name}."
        )
    if len(candidates) == 1:
        return candidates[0]
    joined = ", ".join(entry.name for entry in candidates)
    raise FileNotFoundError("Multiple model runs found; please specify --config-tag. Candidates: " + joined)


def available_config_tags(dataset_variant: str, encoding: str, model_name: str) -> list[str]:
    """Return the discovered config tags for the given model runs."""
    prefix = run_slug(dataset_variant, encoding, model_name, config_tag="placeholder")
    prefix = prefix.rsplit("_", 1)[0] + "_"
    tags: list[str] = []
    for directory in list_run_dirs(dataset_variant, encoding, model_name):
        if directory.name.startswith(prefix):
            tags.append(directory.name[len(prefix) :])
    return tags


def iter_eval_files(dataset_variant: str, encoding: str, model_name: str) -> Iterable[Path]:
    """Yield evaluation result files for each stored run."""
    for run in list_run_dirs(dataset_variant, encoding, model_name):
        eval_dir = run / EVAL_DIR_NAME
        if not eval_dir.exists():
            continue
        yield from sorted(eval_dir.glob("*.json"))
