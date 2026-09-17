"""Fetch real US Treasury constant-maturity yields and write data/treasury_yields.csv.

Two sources, pick with --source:

  treasury  (default) : direct from the US Treasury XML feed via the `ust` package
                        (the primary publisher of the daily par yield curve).
  fred                : Federal Reserve H.15 series via pandas-datareader (needs a
                        free FRED API key in the FRED_API_KEY environment variable).
  github              : a public GitHub mirror of the Treasury feed, no key needed.

Examples:
    pip install ust                 # for --source treasury
    python scripts/fetch_yields.py --source treasury --start 2018 --end 2019

    pip install pandas-datareader   # for --source fred
    export FRED_API_KEY=your_key
    python scripts/fetch_yields.py --source fred --start 2018-01-01 --end 2019-12-31

    python scripts/fetch_yields.py --source github --start 2018 --end 2019
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request

import pandas as pd

OUT_COLS = [1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
GITHUB_URL = "https://raw.githubusercontent.com/epogrebnyak/data-ust/master/ust.csv"
GITHUB_MAP = {
    "BC_1MONTH": 1 / 12, "BC_3MONTH": 0.25, "BC_6MONTH": 0.5, "BC_1YEAR": 1,
    "BC_2YEAR": 2, "BC_3YEAR": 3, "BC_5YEAR": 5, "BC_7YEAR": 7,
    "BC_10YEAR": 10, "BC_20YEAR": 20, "BC_30YEAR": 30,
}
FRED_MAP = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2,
    "DGS3": 3, "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30,
}


def _from_github(start, end):
    raw = urllib.request.urlopen(GITHUB_URL, timeout=60).read().decode()
    df = pd.read_csv(io.StringIO(raw), parse_dates=["date"])
    df = df[["date"] + list(GITHUB_MAP)].rename(columns={k: round(v, 4) for k, v in GITHUB_MAP.items()})
    return df


def _from_fred(start, end):
    from pandas_datareader import data as web
    frames = {}
    for code, tau in FRED_MAP.items():
        frames[round(tau, 4)] = web.DataReader(code, "fred", start, end)[code]
    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df.reset_index()


def _from_treasury(start, end):
    from ust import read_rates  # pip install ust
    y0 = int(str(start)[:4])
    y1 = int(str(end)[:4])
    df = read_rates(start_year=y0, end_year=y1)
    ren = {"BC_1MONTH": 1 / 12, "BC_3MONTH": 0.25, "BC_6MONTH": 0.5, "BC_1YEAR": 1,
           "BC_2YEAR": 2, "BC_3YEAR": 3, "BC_5YEAR": 5, "BC_7YEAR": 7,
           "BC_10YEAR": 10, "BC_20YEAR": 20, "BC_30YEAR": 30}
    df = df.reset_index().rename(columns={k: round(v, 4) for k, v in ren.items()})
    keep = ["date"] + [round(v, 4) for v in ren.values()]
    return df[[c for c in keep if c in df.columns]]


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["treasury", "fred", "github"], default="treasury")
    ap.add_argument("--start", default="2018")
    ap.add_argument("--end", default="2019")
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "treasury_yields.csv"))
    args = ap.parse_args()

    fn = {"treasury": _from_treasury, "fred": _from_fred, "github": _from_github}[args.source]
    df = fn(args.start, args.end)

    df["date"] = pd.to_datetime(df["date"])
    y0, y1 = int(str(args.start)[:4]), int(str(args.end)[:4])
    df = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)]

    mat = [c for c in df.columns if c != "date"]
    df[mat] = df[mat].replace(0.0, pd.NA)
    df = df.dropna(subset=mat)

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} daily curves from '{args.source}' to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
