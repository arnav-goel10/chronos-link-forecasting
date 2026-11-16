"""Public hourly LINK candles from exchanges that need no API key.

Coinbase Exchange is the primary venue because it serves deep hourly history in
300-candle pages. Kraken is the fallback; its public OHLC endpoint only returns
the most recent 720 candles, so a Kraken run covers about 30 days.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

import pandas as pd

COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
HOUR = pd.Timedelta(hours=1)
_COINBASE_PAGE_CANDLES = 300
_USER_AGENT = "chronos-link-forecasting/0.1"

JsonFetcher = Callable[[str], Any]


def http_json(url: str, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_coinbase_candles(rows: Any) -> pd.DataFrame:
    """Parse ``[time, low, high, open, close, volume]`` rows, newest first."""
    if not isinstance(rows, list):
        raise ValueError(f"unexpected Coinbase candle payload: {rows!r:.200}")
    records = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError(f"unexpected Coinbase candle row: {row!r}")
        records.append(
            {
                "timestamp": pd.Timestamp(int(row[0]), unit="s", tz="UTC"),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    return _candle_frame(records)


def parse_kraken_ohlc(payload: Any) -> pd.DataFrame:
    """Parse Kraken ``[time, open, high, low, close, vwap, volume, count]`` rows."""
    if not isinstance(payload, dict):
        raise ValueError("unexpected Kraken payload")
    errors = payload.get("error") or []
    if errors:
        raise ValueError(f"Kraken error: {errors}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Kraken payload has no result")
    series = [value for key, value in result.items() if key != "last"]
    if len(series) != 1 or not isinstance(series[0], list):
        raise ValueError("Kraken payload must contain exactly one OHLC series")
    records = []
    for row in series[0]:
        if not isinstance(row, list) or len(row) < 7:
            raise ValueError(f"unexpected Kraken OHLC row: {row!r}")
        records.append(
            {
                "timestamp": pd.Timestamp(int(row[0]), unit="s", tz="UTC"),
                "close": float(row[4]),
                "volume": float(row[6]),
            }
        )
    return _candle_frame(records)


def _candle_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=["timestamp", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates(subset="timestamp", keep="last")
    return frame.sort_values("timestamp").reset_index(drop=True)


def fetch_coinbase(
    product: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch: JsonFetcher = http_json,
    pause_seconds: float = 0.35,
) -> pd.DataFrame:
    """Fetch hourly candles whose open time lies in ``[start, end)``."""
    frames = []
    cursor = start
    url = COINBASE_URL.format(product=urllib.parse.quote(product))
    while cursor < end:
        page_end = min(cursor + _COINBASE_PAGE_CANDLES * HOUR, end)
        # Coinbase treats ``end`` as inclusive, so ask for the last open time.
        query = urllib.parse.urlencode(
            {
                "granularity": 3600,
                "start": cursor.isoformat(),
                "end": (page_end - HOUR).isoformat(),
            }
        )
        frames.append(parse_coinbase_candles(fetch(f"{url}?{query}")))
        cursor = page_end
        if pause_seconds and cursor < end:
            time.sleep(pause_seconds)
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
    )
    in_range = (combined["timestamp"] >= start) & (combined["timestamp"] < end)
    return combined.loc[in_range].reset_index(drop=True)


def fetch_kraken(
    pair: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch: JsonFetcher = http_json,
) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {"pair": pair, "interval": 60, "since": int(start.timestamp()) - 1}
    )
    candles = parse_kraken_ohlc(fetch(f"{KRAKEN_URL}?{query}"))
    # Kraken's newest candle is still open; the ``< end`` filter drops it.
    in_range = (candles["timestamp"] >= start) & (candles["timestamp"] < end)
    return candles.loc[in_range].reset_index(drop=True)


def to_hourly_series(
    candles: pd.DataFrame,
    item_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Reindex closes onto a regular hourly grid in ``[start, end)``.

    Hours with no trades carry the previous close forward, which uses only past
    information. Leading hours before the first candle are dropped rather than
    back-filled. Returns the frame and the number of carried-forward hours.
    """
    if candles.empty:
        raise ValueError("no candles were returned for the requested window")
    grid = pd.date_range(start, end, freq="h", inclusive="left")
    closes = candles.set_index("timestamp")["close"].reindex(grid)
    closes = closes.iloc[int(closes.notna().to_numpy().argmax()) :]
    filled = closes.isna()
    closes = closes.ffill()
    frame = pd.DataFrame(
        {
            "item_id": item_id,
            "timestamp": closes.index,
            "target": closes.to_numpy(dtype="float64"),
            "filled": filled.to_numpy(dtype=bool),
        }
    )
    return frame, int(filled.sum())


def fetch_hourly_link(
    days: int,
    end: pd.Timestamp | None = None,
    venue: str = "auto",
    fetch: JsonFetcher = http_json,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch ``days`` of completed hourly LINK/USD closes ending at ``end``."""
    if days < 1:
        raise ValueError("days must be positive")
    if venue not in {"auto", "coinbase", "kraken"}:
        raise ValueError("venue must be auto, coinbase or kraken")
    now = pd.Timestamp.now(tz="UTC")
    end = end if end is not None else now
    end = (end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")).floor(
        "h"
    )
    start = end - pd.Timedelta(days=days)
    notes: list[str] = []

    candles: pd.DataFrame | None = None
    used = venue
    if venue in {"auto", "coinbase"}:
        try:
            candles = fetch_coinbase("LINK-USD", start, end, fetch=fetch)
            used = "coinbase"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            if venue == "coinbase":
                raise
            notes.append(f"coinbase failed, fell back to kraken: {error}")
    if candles is None:
        candles = fetch_kraken("LINKUSD", start, end, fetch=fetch)
        used = "kraken"

    frame, filled_hours = to_hourly_series(candles, "LINK-USD", start, end)
    meta: dict[str, Any] = {
        "venue": used,
        "symbol": "LINK-USD" if used == "coinbase" else "LINKUSD",
        "granularity": "1h",
        "target": "close",
        "requested_start": start.isoformat(),
        "requested_end_exclusive": end.isoformat(),
        "fetched_at": now.isoformat(),
        "candles_returned": len(candles),
        "rows": len(frame),
        "filled_hours": filled_hours,
        "notes": notes,
    }
    return frame, meta
