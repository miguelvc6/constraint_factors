import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping


def _filter_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:
    valid_fields = {f.name for f in fields(cls)}
    unknown = set(data) - valid_fields
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration keys for {cls.__name__}: {unknown_list}")
    return {key: data[key] for key in valid_fields if key in data}


def _load_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Configuration file at {path} must contain an object at the top level.")
    return loaded


def _normalize_class_ids(value: Any) -> tuple[int, ...] | None:
    """Return ``value`` as a tuple[int, ...] if provided, otherwise ``None``."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, (list, set, frozenset)):
        return tuple(int(v) for v in value)
    if isinstance(value, (int,)):
        return (int(value),)
    if hasattr(value, "tolist"):
        return tuple(int(v) for v in value.tolist())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(int(v) for v in value)
    raise TypeError(f"Expected iterable of ints for class ids, got {type(value)!r}")


@dataclass
class DynamicReweightingConfig:
    enabled: bool = False  # Toggle dynamic per-constraint loss weighting.
    target_metrics: tuple[str, ...] = ("loss",)  # Validation metrics used to derive difficulty.
    update_frequency: str = "epoch"  # Either "epoch" (default) or "batch".
    scale: float = 1.0  # Strength of the reweighting relative to uniform weights.
    min_weight: float = 0.5  # Lower clamp for generated weights.
    max_weight: float = 3.0  # Upper clamp for generated weights.
    smoothing: float = 0.2  # Interpolation factor toward previous weights (0 = overwrite).

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DynamicReweightingConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "DynamicReweightingConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        if "target_metrics" in filtered:
            value = filtered["target_metrics"]
            if isinstance(value, str):
                filtered["target_metrics"] = (value,)
            else:
                filtered["target_metrics"] = tuple(str(v) for v in value)

        if "update_frequency" in filtered:
            freq = str(filtered["update_frequency"]).lower()
            if freq not in {"epoch", "batch"}:
                raise ValueError("DynamicReweightingConfig.update_frequency must be 'epoch' or 'batch'")
            filtered["update_frequency"] = freq

        for float_field in ("scale", "min_weight", "max_weight", "smoothing"):
            if float_field in filtered and filtered[float_field] is not None:
                filtered[float_field] = float(filtered[float_field])

        if "smoothing" in filtered:
            smoothing_value = filtered["smoothing"]
            if not 0.0 <= smoothing_value <= 1.0:
                raise ValueError("DynamicReweightingConfig.smoothing must be between 0 and 1 inclusive")

        if "min_weight" in filtered or "max_weight" in filtered:
            min_weight = filtered.get("min_weight", self.min_weight)
            max_weight = filtered.get("max_weight", self.max_weight)
            if max_weight < min_weight:
                raise ValueError("DynamicReweightingConfig.max_weight must be >= min_weight")

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target_metrics": list(self.target_metrics),
            "update_frequency": self.update_frequency,
            "scale": self.scale,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "smoothing": self.smoothing,
        }


@dataclass
class ConstraintLossConfig:
    dynamic_reweighting: DynamicReweightingConfig = field(default_factory=DynamicReweightingConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ConstraintLossConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "ConstraintLossConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        dynamic_payload = filtered.pop("dynamic_reweighting", None)
        current.update(filtered)

        if dynamic_payload is not None:
            if isinstance(dynamic_payload, DynamicReweightingConfig):
                current["dynamic_reweighting"] = dynamic_payload
            else:
                current["dynamic_reweighting"] = self.dynamic_reweighting.updated(dynamic_payload)

        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dynamic_reweighting": self.dynamic_reweighting.to_dict(),
        }


@dataclass
class FixProbabilityLossConfig:
    enabled: bool = False  # Toggle the fix-aware loss term.
    initial_weight: float = 0.5  # Weight at the start (after warmup).
    final_weight: float = 0.05  # Asymptotic weight once decay finishes.
    decay_epochs: float = 40.0  # Time constant (exponential) or span (linear).
    warmup_epochs: float = 0.0  # Epochs to hold the initial weight before decay.
    schedule: str = "exponential"  # 'exponential' (default) or 'linear'.

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "FixProbabilityLossConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "FixProbabilityLossConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        for key in ("initial_weight", "final_weight", "decay_epochs", "warmup_epochs"):
            if key in filtered and filtered[key] is not None:
                filtered[key] = float(filtered[key])

        if "schedule" in filtered and filtered["schedule"] is not None:
            value = str(filtered["schedule"]).lower()
            if value not in {"exponential", "linear"}:
                raise ValueError("FixProbabilityLossConfig.schedule must be 'exponential' or 'linear'")
            filtered["schedule"] = value

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "initial_weight": self.initial_weight,
            "final_weight": self.final_weight,
            "decay_epochs": self.decay_epochs,
            "warmup_epochs": self.warmup_epochs,
            "schedule": self.schedule,
        }


@dataclass
class FactorLossConfig:
    enabled: bool = False
    weight_pre: float = 0.1
    pos_weight: float | None = None
    only_checkable: bool = True
    per_graph_reduction: str = "mean"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "FactorLossConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "FactorLossConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        if "weight_pre" in filtered and filtered["weight_pre"] is not None:
            filtered["weight_pre"] = float(filtered["weight_pre"])
        if "pos_weight" in filtered and filtered["pos_weight"] is not None:
            filtered["pos_weight"] = float(filtered["pos_weight"])
        if "only_checkable" in filtered and filtered["only_checkable"] is not None:
            filtered["only_checkable"] = bool(filtered["only_checkable"])
        if "per_graph_reduction" in filtered and filtered["per_graph_reduction"] is not None:
            value = str(filtered["per_graph_reduction"]).lower()
            if value not in {"mean", "sum"}:
                raise ValueError("FactorLossConfig.per_graph_reduction must be 'mean' or 'sum'")
            filtered["per_graph_reduction"] = value

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight_pre": self.weight_pre,
            "pos_weight": self.pos_weight,
            "only_checkable": self.only_checkable,
            "per_graph_reduction": self.per_graph_reduction,
        }


@dataclass
class ModelConfig:
    dataset_variant: str = "full" 
    """Which intermediate dataset variant to consume."""
    encoding: str = "text_embedding"
    """Node feature encoding that selects graph files."""
    model: str = "GIN"
    """Identifier for the GNN architecture to instantiate."""
    min_occurrence: int = 100
    """Frequency threshold used when building the dataset."""
    num_embedding_size: int = 128  
    """Width of learned embeddings for integer node ids."""
    num_layers: int = 2  
    """Number of message-passing layers in the backbone."""
    hidden_channels: int = 128  
    """Channel size inside the message-passing stack."""
    head_hidden: int = 128
    """Hidden width shared by the prediction heads."""
    dropout: float = 0.5  
    """Dropout probability applied to head activations."""
    use_node_embeddings: bool = True  
    """Toggle between embedding integer ids or passing features through."""
    use_role_embeddings: bool = False  
    """Whether to append learned focus-role embeddings to node features."""
    role_embedding_dim: int = 8  
    """Dimensionality of each learned role embedding vector."""
    num_role_types: int = 4  
    """Number of distinct role ids expected in role_flags tensors."""
    use_edge_attributes: bool = False 
    """Whether to use edge attributes, instead of treating edges as nodes."""
    use_edge_subtraction: bool = False 
    """Whether to use edge subtraction, which requires use_edge_attributes to be True."""
    entity_class_ids: tuple[int, ...] | None = None  
    """Optional vocabulary subset for entity targets."""
    predicate_class_ids: tuple[int, ...] | None = None  
    """Optional vocabulary subset for predicate targets."""
    num_factor_types: int = 0
    """Number of distinct factor type ids (0 disables type conditioning)."""
    factor_type_embedding_dim: int = 8
    """Embedding dim for factor type conditioning."""
    pressure_enabled: bool = False
    """Toggle factor pressure injection during message passing."""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        return cls().updated(data)

    @classmethod
    def from_path(cls, path: Path | None) -> "ModelConfig":
        if path is None:
            return cls()
        return cls.from_mapping(_load_mapping(path))

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "ModelConfig":
        """Update the configuration with values from ``data`` and ``overrides``."""
        data_dict = dict(data or {})
        if "hidden" in data_dict and "hidden_channels" not in data_dict:
            data_dict["hidden_channels"] = data_dict.pop("hidden")
        filtered = _filter_fields(type(self), data_dict)
        filtered.update({k: v for k, v in overrides.items() if v is not None})

        for field_name in ("entity_class_ids", "predicate_class_ids"):
            if field_name in filtered:
                filtered[field_name] = _normalize_class_ids(filtered[field_name])

        if "role_embedding_dim" in filtered and filtered["role_embedding_dim"] is not None:
            filtered["role_embedding_dim"] = int(filtered["role_embedding_dim"])
        if "num_role_types" in filtered and filtered["num_role_types"] is not None:
            filtered["num_role_types"] = int(filtered["num_role_types"])
        if "use_role_embeddings" in filtered and filtered["use_role_embeddings"] is not None:
            filtered["use_role_embeddings"] = bool(filtered["use_role_embeddings"])
        if "num_factor_types" in filtered and filtered["num_factor_types"] is not None:
            filtered["num_factor_types"] = int(filtered["num_factor_types"])
        if "factor_type_embedding_dim" in filtered and filtered["factor_type_embedding_dim"] is not None:
            filtered["factor_type_embedding_dim"] = int(filtered["factor_type_embedding_dim"])
        if "pressure_enabled" in filtered and filtered["pressure_enabled"] is not None:
            filtered["pressure_enabled"] = bool(filtered["pressure_enabled"])

        current = asdict(self)
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    batch_size: int = 124  # Number of graphs per optimization step.
    num_epochs: int = 5  # Maximum number of training epochs.
    early_stopping_rounds: int = 5  # Patience before early stopping triggers.
    grad_clip: float | None = 1.0  # Gradient norm cap; set None to disable clipping.
    learning_rate: float = 1e-3  # Base learning rate for Adam.
    weight_decay: float = 5e-4  # L2 penalty applied through Adam weight decay.
    scheduler_factor: float = 0.5  # Multiplicative drop factor for the LR scheduler.
    scheduler_patience: int = 3  # Epochs with no improvement before lowering LR.
    num_workers: int = 0  # Worker processes used by DataLoader.
    pin_memory: bool | None = None  # Override DataLoader pin_memory behaviour (None keeps the default).
    validate_factor_labels: bool = False  # Enable strict factor label assertions per batch.
    constraint_loss: ConstraintLossConfig = field(default_factory=ConstraintLossConfig)
    fix_probability_loss: FixProbabilityLossConfig = field(default_factory=FixProbabilityLossConfig)
    factor_loss: FactorLossConfig = field(default_factory=FactorLossConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        return cls().updated(data)

    @classmethod
    def from_path(cls, path: Path | None) -> "TrainingConfig":
        if path is None:
            return cls()
        return cls.from_mapping(_load_mapping(path))

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "TrainingConfig":
        payload = dict(data or {})
        dynamic_fallback = payload.pop("dynamic_reweighting", None)

        filtered = _filter_fields(type(self), payload)
        filtered.update({k: v for k, v in overrides.items() if v is not None})

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        constraint_update = filtered.pop("constraint_loss", None)
        fix_loss_update = filtered.pop("fix_probability_loss", None)
        factor_loss_update = filtered.pop("factor_loss", None)

        if dynamic_fallback is not None:
            if constraint_update is None:
                constraint_update = {"dynamic_reweighting": dynamic_fallback}
            else:
                if isinstance(constraint_update, ConstraintLossConfig):
                    constraint_update = constraint_update.to_dict()
                if isinstance(constraint_update, Mapping):
                    constraint_update = dict(constraint_update)
                    if "dynamic_reweighting" not in constraint_update:
                        constraint_update["dynamic_reweighting"] = dynamic_fallback
                else:
                    raise TypeError("constraint_loss must be mapping-compatible when combining configuration sources")

        current.update(filtered)

        if constraint_update is not None:
            if isinstance(constraint_update, ConstraintLossConfig):
                current["constraint_loss"] = constraint_update
            else:
                current["constraint_loss"] = self.constraint_loss.updated(constraint_update)

        if fix_loss_update is not None:
            if isinstance(fix_loss_update, FixProbabilityLossConfig):
                current["fix_probability_loss"] = fix_loss_update
            else:
                current["fix_probability_loss"] = self.fix_probability_loss.updated(fix_loss_update)

        if factor_loss_update is not None:
            if isinstance(factor_loss_update, FactorLossConfig):
                current["factor_loss"] = factor_loss_update
            else:
                current["factor_loss"] = self.factor_loss.updated(factor_loss_update)

        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constraint_loss"] = self.constraint_loss.to_dict()
        payload["fix_probability_loss"] = self.fix_probability_loss.to_dict()
        payload["factor_loss"] = self.factor_loss.to_dict()
        return payload


__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "ConstraintLossConfig",
    "DynamicReweightingConfig",
    "FixProbabilityLossConfig",
    "FactorLossConfig",
]
