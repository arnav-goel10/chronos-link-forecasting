from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from chronos_link_forecasting.data.splits import TemporalSplit

type FlashAttentionMode = Literal["disabled", "auto"]
type Device = Literal["cpu", "cuda"]

_EXPERIMENT_KEYS = {
    "data_path",
    "timestamp_column",
    "id_column",
    "target_column",
    "past_covariates",
    "prediction_length",
    "quantile_levels",
    "split",
    "model_id",
    "device",
    "seed",
}
_TRAINING_KEYS = {
    "bf16",
    "flash_attention_2",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "learning_rate",
}


@dataclass(frozen=True)
class TrainingConfig:
    bf16: bool
    flash_attention_2: FlashAttentionMode
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float

    def __post_init__(self) -> None:
        positive_integer_fields = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
        }
        for name, value in positive_integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path
    timestamp_column: str
    id_column: str
    target_column: str
    past_covariates: tuple[str, ...]
    prediction_length: int
    quantile_levels: tuple[float, ...]
    split: TemporalSplit
    model_id: str
    device: Device
    seed: int
    training: TrainingConfig | None = None

    def __post_init__(self) -> None:
        if self.prediction_length < 1:
            raise ValueError("prediction_length must be at least 1")
        if any(level <= 0 or level >= 1 for level in self.quantile_levels):
            raise ValueError("quantile_levels must be between 0 and 1")
        if not {0.1, 0.5, 0.9}.issubset(self.quantile_levels):
            raise ValueError("quantile_levels must include 0.1, 0.5, and 0.9")
        if self.device == "cuda" and self.model_id != "amazon/chronos-2":
            raise ValueError("CUDA experiments require model_id amazon/chronos-2")
        if self.device == "cuda" and self.training is None:
            raise ValueError("CUDA experiments require training settings")
        if self.device == "cuda" and self.training is not None:
            if not self.training.bf16:
                raise ValueError("CUDA Chronos training requires bf16")
            if self.training.flash_attention_2 != "auto":
                raise ValueError(
                    "CUDA Chronos training requires runtime FlashAttention-2 detection"
                )

    @classmethod
    def from_toml(cls, path: Path) -> ExperimentConfig:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)
        _require_exact_keys(raw, _EXPERIMENT_KEYS, "experiment", optional={"training"})

        split_raw = _require_mapping(raw, "split")
        _require_exact_keys(
            split_raw,
            {"train_end", "validation_end", "test_end"},
            "split",
        )

        training: TrainingConfig | None = None
        if "training" in raw:
            training_raw = _require_mapping(raw, "training")
            _require_exact_keys(training_raw, _TRAINING_KEYS, "training")
            training = TrainingConfig(
                bf16=_require_bool(training_raw, "bf16"),
                flash_attention_2=_require_flash_attention_mode(
                    training_raw, "flash_attention_2"
                ),
                per_device_train_batch_size=_require_int(
                    training_raw, "per_device_train_batch_size"
                ),
                gradient_accumulation_steps=_require_int(
                    training_raw, "gradient_accumulation_steps"
                ),
                max_steps=_require_int(training_raw, "max_steps"),
                learning_rate=_require_float(training_raw, "learning_rate"),
            )

        return cls(
            data_path=Path(_require_str(raw, "data_path")),
            timestamp_column=_require_str(raw, "timestamp_column"),
            id_column=_require_str(raw, "id_column"),
            target_column=_require_str(raw, "target_column"),
            past_covariates=_require_str_tuple(raw, "past_covariates"),
            prediction_length=_require_int(raw, "prediction_length"),
            quantile_levels=_require_float_tuple(raw, "quantile_levels"),
            split=TemporalSplit(
                _require_str(split_raw, "train_end"),
                _require_str(split_raw, "validation_end"),
                _require_str(split_raw, "test_end"),
            ),
            model_id=_require_str(raw, "model_id"),
            device=_require_device(raw, "device"),
            seed=_require_int(raw, "seed"),
            training=training,
        )


def _require_exact_keys(
    values: Mapping[str, Any],
    required: set[str],
    section: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(values)
    missing = required - actual
    unexpected = actual - required - optional
    if missing:
        raise ValueError(f"{section} is missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(
            f"{section} has unexpected fields: {', '.join(sorted(unexpected))}"
        )


def _require_mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a TOML table")
    return cast(dict[str, Any], value)


def _require_str(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_float(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _require_bool(values: Mapping[str, Any], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _require_str_tuple(values: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(value)


def _require_float_tuple(values: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = values.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{key} must be an array of numbers")
    return tuple(float(item) for item in value)


def _require_device(values: Mapping[str, Any], key: str) -> Device:
    value = _require_str(values, key)
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return cast(Device, value)


def _require_flash_attention_mode(
    values: Mapping[str, Any], key: str
) -> FlashAttentionMode:
    value = _require_str(values, key)
    if value not in {"disabled", "auto"}:
        raise ValueError("flash_attention_2 must be disabled or auto")
    return cast(FlashAttentionMode, value)
