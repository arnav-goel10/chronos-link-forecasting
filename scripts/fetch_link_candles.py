"""Download public hourly LINK/USD closes (no API key) to Parquet.

    python scripts/fetch_link_candles.py --days 180 --out artifacts/link_usd_1h.parquet

Writes the Parquet file and a ``.meta.json`` sidecar recording the venue, the
requested window and how many hours were carried forward. Data is pulled at
run time and is not committed to the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chronos_link_forecasting.data.candles import fetch_hourly_link


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch hourly LINK/USD candles")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument(
        "--end", help="exclusive UTC end, floored to the hour; defaults to now"
    )
    parser.add_argument(
        "--venue", choices=("auto", "coinbase", "kraken"), default="auto"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else None
    frame, meta = fetch_hourly_link(args.days, end=end, venue=args.venue)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
