from pathlib import Path

import pytest

from chronos_link_forecasting.config import ExperimentConfig

REPOSITORY_ROOT = Path(__file__).parents[1]


def _write_config(
    tmp_path: Path,
    *,
    prediction_length: int = 12,
    quantile_levels: str = "[0.1, 0.5, 0.9]",
    model_id: str = "seasonal-naive",
    device: str = "cpu",
    include_training: bool = False,
) -> Path:
    training = ""
    if include_training:
        training = """
[training]
bf16 = true
flash_attention_2 = "auto"
per_device_train_batch_size = 32
gradient_accumulation_steps = 8
max_steps = 1000
learning_rate = 1e-5
"""

    path = tmp_path / "experiment.toml"
    path.write_text(
        f"""
data_path = "data/sample/link-hourly.csv"
timestamp_column = "timestamp"
id_column = "item_id"
target_column = "target"
past_covariates = ["volume", "market_cap"]
prediction_length = {prediction_length}
quantile_levels = {quantile_levels}
model_id = "{model_id}"
device = "{device}"
seed = 42

[split]
train_end = "2025-01-01T00:00:00Z"
validation_end = "2025-02-01T00:00:00Z"
test_end = "2025-03-01T00:00:00Z"
{training}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_cpu_smoke_config_loads_typed_forecasting_contract() -> None:
    config = ExperimentConfig.from_toml(REPOSITORY_ROOT / "configs" / "cpu-smoke.toml")

    assert config.data_path == Path("data/sample/link-hourly.csv")
    assert config.timestamp_column == "timestamp"
    assert config.id_column == "item_id"
    assert config.target_column == "target"
    assert config.past_covariates == ("volume", "market_cap")
    assert config.prediction_length == 12
    assert config.quantile_levels == (0.1, 0.5, 0.9)
    assert config.model_id == "seasonal-naive"
    assert config.device == "cpu"
    assert config.seed == 42
    assert config.training is None


def test_h100_config_requires_runtime_detection_for_flash_attention() -> None:
    config = ExperimentConfig.from_toml(REPOSITORY_ROOT / "configs" / "h100-full.toml")

    assert config.model_id == "amazon/chronos-2"
    assert config.device == "cuda"
    assert config.training is not None
    assert config.training.bf16 is True
    assert config.training.flash_attention_2 == "auto"
    assert config.training.per_device_train_batch_size == 32
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.max_steps == 1000
    assert config.training.learning_rate == pytest.approx(1e-5)


@pytest.mark.parametrize("prediction_length", [0, -1])
def test_prediction_length_must_be_positive(
    tmp_path: Path, prediction_length: int
) -> None:
    path = _write_config(tmp_path, prediction_length=prediction_length)

    with pytest.raises(ValueError, match="prediction_length must be at least 1"):
        ExperimentConfig.from_toml(path)


@pytest.mark.parametrize(
    "quantile_levels",
    ["[0.0, 0.5, 0.9]", "[0.1, 0.5, 1.0]", "[-0.1, 0.5, 0.9]"],
)
def test_quantiles_must_be_strictly_between_zero_and_one(
    tmp_path: Path, quantile_levels: str
) -> None:
    path = _write_config(tmp_path, quantile_levels=quantile_levels)

    with pytest.raises(ValueError, match="quantile_levels must be between 0 and 1"):
        ExperimentConfig.from_toml(path)


@pytest.mark.parametrize("quantile_levels", ["[0.5, 0.9]", "[0.1, 0.9]", "[0.1, 0.5]"])
def test_required_quantiles_cannot_be_omitted(
    tmp_path: Path, quantile_levels: str
) -> None:
    path = _write_config(tmp_path, quantile_levels=quantile_levels)

    with pytest.raises(
        ValueError, match=r"quantile_levels must include 0\.1, 0\.5, and 0\.9"
    ):
        ExperimentConfig.from_toml(path)


def test_cuda_config_rejects_non_chronos_model(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        model_id="seasonal-naive",
        device="cuda",
        include_training=True,
    )

    with pytest.raises(
        ValueError, match="CUDA experiments require model_id amazon/chronos-2"
    ):
        ExperimentConfig.from_toml(path)


def test_cpu_config_allows_baseline_model(tmp_path: Path) -> None:
    path = _write_config(tmp_path, model_id="seasonal-naive", device="cpu")

    config = ExperimentConfig.from_toml(path)

    assert config.model_id == "seasonal-naive"


def test_cuda_config_requires_training_settings(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        model_id="amazon/chronos-2",
        device="cuda",
        include_training=False,
    )

    with pytest.raises(ValueError, match="CUDA experiments require training settings"):
        ExperimentConfig.from_toml(path)
