from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from chronos_link_forecasting.data.candles import (
    fetch_coinbase,
    fetch_hourly_link,
    parse_coinbase_candles,
    parse_kraken_ohlc,
    to_hourly_series,
)

START = pd.Timestamp("2026-01-01T00:00:00Z")


def _epoch(hours: int) -> int:
    return int((START + pd.Timedelta(hours=hours)).timestamp())


def test_coinbase_rows_are_sorted_and_deduplicated() -> None:
    rows = [
        [_epoch(1), 1.0, 2.0, 1.5, 11.0, 5.0],
        [_epoch(0), 1.0, 2.0, 1.5, 10.0, 4.0],
        [_epoch(1), 1.0, 2.0, 1.5, 12.0, 6.0],
    ]

    frame = parse_coinbase_candles(rows)

    assert frame["timestamp"].tolist() == [START, START + pd.Timedelta(hours=1)]
    assert frame["close"].tolist() == [10.0, 12.0]


def test_coinbase_rejects_error_payload() -> None:
    with pytest.raises(ValueError, match="unexpected Coinbase"):
        parse_coinbase_candles({"message": "rate limited"})


def test_kraken_parses_close_and_volume_and_raises_on_errors() -> None:
    payload = {
        "error": [],
        "result": {
            "LINKUSD": [[_epoch(0), "1", "2", "0.5", "10.5", "1.2", "7.0", 3]],
            "last": _epoch(0),
        },
    }
    frame = parse_kraken_ohlc(payload)
    assert frame["close"].tolist() == [10.5]
    assert frame["volume"].tolist() == [7.0]

    with pytest.raises(ValueError, match="Kraken error"):
        parse_kraken_ohlc({"error": ["EQuery:Unknown asset pair"]})


def test_coinbase_pages_cover_the_window_without_overlap() -> None:
    end = START + pd.Timedelta(hours=650)
    urls: list[str] = []

    def fake_fetch(url: str) -> Any:
        urls.append(url)
        query = parse_qs(urlsplit(url).query)
        first = pd.Timestamp(query["start"][0])
        last = pd.Timestamp(query["end"][0])
        hours = int((last - first) / pd.Timedelta(hours=1)) + 1
        assert hours <= 300
        return [
            [int((first + pd.Timedelta(hours=h)).timestamp()), 0, 0, 0, 1.0 + h, 1]
            for h in reversed(range(hours))
        ]

    frame = fetch_coinbase("LINK-USD", START, end, fetch=fake_fetch, pause_seconds=0)

    assert len(urls) == 3
    assert len(frame) == 650
    assert frame["timestamp"].is_unique
    assert frame["timestamp"].iloc[-1] == end - pd.Timedelta(hours=1)


def test_hourly_series_carries_gaps_forward_and_drops_leading_hours() -> None:
    candles = pd.DataFrame(
        {
            "timestamp": [
                START + pd.Timedelta(hours=1),
                START + pd.Timedelta(hours=3),
            ],
            "close": [5.0, 7.0],
            "volume": [1.0, 1.0],
        }
    )

    frame, filled = to_hourly_series(
        candles, "LINK-USD", START, START + pd.Timedelta(hours=5)
    )

    assert frame["timestamp"].iloc[0] == START + pd.Timedelta(hours=1)
    assert frame["target"].tolist() == [5.0, 5.0, 7.0, 7.0]
    assert frame["filled"].tolist() == [False, True, False, True]
    assert filled == 2


def test_auto_venue_falls_back_to_kraken_and_records_why() -> None:
    end = START + pd.Timedelta(hours=4)

    def fake_fetch(url: str) -> Any:
        if "coinbase" in url:
            raise OSError("blocked")
        return {
            "error": [],
            "result": {
                "LINKUSD": [
                    [_epoch(h), "0", "0", "0", str(10 + h), "0", "1", 1]
                    for h in range(-20, 5)
                ],
                "last": 0,
            },
        }

    frame, meta = fetch_hourly_link(1, end=end, fetch=fake_fetch)

    assert meta["venue"] == "kraken"
    assert "coinbase failed" in meta["notes"][0]
    assert frame["timestamp"].iloc[-1] == end - pd.Timedelta(hours=1)
    assert len(frame) == 24
