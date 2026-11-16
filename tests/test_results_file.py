"""Guards the committed benchmark figures and the README table built from them."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from chronos_link_forecasting.evaluate import (
    METRIC_KEYS,
    MODEL_NAMES,
    SCHEMA_VERSION,
    render_markdown_table,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "link_usd_1h_backtest.json"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(RESULTS.read_text())
    return data


def test_results_schema(payload: dict[str, Any]) -> None:
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["provenance"]["github_run_id"].isdigit()
    assert payload["data"]["venue"] in {"coinbase", "kraken"}
    assert payload["data"]["rows"] > 0
    assert len(payload["data"]["sha256_timestamp_target_csv"]) == 64

    backtest = payload["backtest"]
    assert backtest["quantile_levels"] == [0.1, 0.5, 0.9]
    assert backtest["windows"] > 0
    assert (
        backtest["scored_points"]
        == backtest["windows"] * backtest["prediction_length_hours"]
    )
    assert backtest["context_length_hours"] <= payload["data"]["rows"]

    assert set(payload["models"]) == set(MODEL_NAMES)
    for row in payload["models"].values():
        for key in METRIC_KEYS:
            assert math.isfinite(row[key])
        assert row["weighted_quantile_loss"] > 0
        assert row["p50_mae"] > 0
        assert 0.0 <= row["coverage_80"] <= 1.0
        assert row["interval_width_80"] > 0
    assert payload["environment"]["chronos_forecasting"]


def test_ranking_matches_the_scores(payload: dict[str, Any]) -> None:
    ranking = payload["ranking_by_weighted_quantile_loss"]
    losses = [payload["models"][name]["weighted_quantile_loss"] for name in ranking]
    assert sorted(ranking) == sorted(payload["models"])
    assert losses == sorted(losses)


def test_readme_table_is_rendered_from_the_committed_file(
    payload: dict[str, Any],
) -> None:
    readme = (ROOT / "README.md").read_text()
    assert render_markdown_table(payload) in readme
    assert payload["provenance"]["github_run_id"] in readme
