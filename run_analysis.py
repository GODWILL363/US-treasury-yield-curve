"""Fit Nelson-Siegel and Svensson curves to a panel of Treasury yields.

Usage:
    python run_analysis.py                       # uses data/treasury_yields.csv
    python run_analysis.py --data path/to.csv    # your own wide-format file

Input format: first column 'date', remaining columns are maturities in
years (headers like 0.25, 1, 2, 10, 30), values are yields in percent.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from yield_curve.fit import fit_nelson_siegel, fit_svensson
from yield_curve.plotting import plot_fit, plot_factors


def load_panel(path: str):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    tau = df.columns.astype(float).to_numpy()
    return df, tau


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data", "treasury_yields.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "outputs"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, tau = load_panel(args.data)

    levels, slopes, curvs, ns_rmse, sv_rmse = [], [], [], [], []
    last_ns = last_sv = last_y = None

    for date, row in df.iterrows():
        y = row.to_numpy(dtype=float)
        ns = fit_nelson_siegel(tau, y)
        sv = fit_svensson(tau, y)
        f = ns.factors
        levels.append(f["level"])
        slopes.append(f["slope"])
        curvs.append(f["curvature"])
        ns_rmse.append(ns.rmse)
        sv_rmse.append(sv.rmse)
        last_ns, last_sv, last_y = ns, sv, y
        print(f"{date.date()}  level={f['level']:6.3f}  slope={f['slope']:6.3f}  "
              f"curv={f['curvature']:6.3f}  NS_rmse={ns.rmse:.3f}  NSS_rmse={sv.rmse:.3f}")

    factors = pd.DataFrame(
        {"level": levels, "slope": slopes, "curvature": curvs,
         "ns_rmse": ns_rmse, "svensson_rmse": sv_rmse},
        index=df.index,
    )
    factors.to_csv(os.path.join(args.outdir, "ns_factors.csv"))

    # Curve plot for the most recent date in the panel.
    plot_fit(tau, last_y, last_ns, last_sv,
             title=f"US Treasury yield curve  {df.index[-1].date()}",
             path=os.path.join(args.outdir, "curve_fit.png"))
    plot_factors(df.index, levels, slopes, curvs,
                 path=os.path.join(args.outdir, "factors.png"))

    print(f"\nMean NS RMSE:       {np.mean(ns_rmse):.4f}")
    print(f"Mean Svensson RMSE: {np.mean(sv_rmse):.4f}")
    print(f"Wrote factors and plots to {args.outdir}")


if __name__ == "__main__":
    main()
