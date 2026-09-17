"""Build a self-contained HTML report for the yield-curve analysis.

Re-fits the curves, embeds the plots, explains the Nelson-Siegel and Svensson
models and how they are fitted, and interprets the results with the numbers
computed from the data.

Usage:
    python report.py                 # bundled real data -> outputs/report.html
Then open outputs/report.html in a browser.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from yield_curve.fit import fit_nelson_siegel, fit_svensson
from yield_curve.plotting import plot_fit, plot_factors


def _img(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html(df: pd.DataFrame, tau) -> str:
    dates, levels, slopes, curvs, ns_rmse, sv_rmse = [], [], [], [], [], []
    last = None
    for date, row in df.iterrows():
        y = row.to_numpy(dtype=float)
        ns = fit_nelson_siegel(tau, y)
        sv = fit_svensson(tau, y)
        f = ns.factors
        dates.append(date); levels.append(f["level"]); slopes.append(f["slope"])
        curvs.append(f["curvature"]); ns_rmse.append(ns.rmse); sv_rmse.append(sv.rmse)
        last = (date, y, ns, sv)

    tmp = tempfile.mkdtemp()
    fp = os.path.join(tmp, "fit.png"); xp = os.path.join(tmp, "factors.png")
    plot_fit(tau, last[1], last[2], last[3],
             title=f"US Treasury yield curve  {last[0].date()}", path=fp)
    plot_factors(dates, levels, slopes, curvs, path=xp)
    fit_img, fac_img = _img(fp), _img(xp)

    # dynamic numbers
    d0, d1, n = df.index.min().date(), df.index.max().date(), len(df)
    lvl0, lvl1 = levels[0], levels[-1]
    slope_min = min(slopes); slope_min_date = dates[int(np.argmin(slopes))].date()
    mean_ns, mean_sv = float(np.mean(ns_rmse)), float(np.mean(sv_rmse))
    ns = last[2]

    inversion = ("The slope even turned negative (an inverted curve) around "
                 f"{slope_min_date}, when short yields rose above long yields."
                 if slope_min < 0 else
                 "The slope stayed positive over the sample, so the curve never fully inverted.")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>US Treasury Yield Curve — Method and Results</title>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['$$','$$']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
 :root{{--ink:#1a1a1a;--muted:#555;--line:#e2e2e2;--accent:#1f3864;--panel:#f7f8fa;}}
 *{{box-sizing:border-box;}}
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
   max-width:860px;margin:0 auto;padding:2.2rem 1.2rem 4rem;line-height:1.55;}}
 h1{{font-size:1.7rem;color:var(--accent);margin:0 0 .2rem;}}
 h2{{font-size:1.25rem;color:var(--accent);margin:2.2rem 0 .6rem;border-bottom:2px solid var(--accent);padding-bottom:.25rem;}}
 h3{{font-size:1.05rem;margin:1.2rem 0 .4rem;}}
 .sub{{color:var(--muted);margin:0 0 1.4rem;}}
 p{{margin:.6rem 0;}}
 table{{border-collapse:collapse;margin:1rem 0;font-size:.94rem;}}
 th,td{{border:1px solid var(--line);padding:.4rem .7rem;text-align:left;}}
 th{{background:var(--panel);}} td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
 img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;margin:.6rem 0;}}
 .panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem 1.1rem;margin:1rem 0;}}
 .note{{border-left:4px solid var(--accent);padding:.3rem 0 .3rem 1rem;color:var(--muted);}}
 code{{background:var(--panel);padding:.05rem .3rem;border-radius:4px;font-size:.9em;}}
 footer{{margin-top:2.5rem;color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:1rem;}}
</style></head><body>

<h1>US Treasury Yield Curve: Nelson-Siegel and Svensson</h1>
<p class="sub">Fitted to real Treasury data, {d0} to {d1} ({n} daily curves).</p>

<div class="panel"><strong>What this shows.</strong> How two standard term-structure
models describe the whole yield curve with a few parameters, how those parameters
are estimated, and what the fitted level, slope, and curvature did over the sample.</div>

<h2>1. The Nelson-Siegel model</h2>
<p>The idea is to describe the yield at every maturity \\(\\tau\\) with three factors
instead of one number per maturity:</p>
$$ y(\\tau) = \\beta_0
 + \\beta_1\\underbrace{{\\frac{{1-e^{{-\\tau/\\lambda}}}}{{\\tau/\\lambda}}}}_{{\\text{{slope loading}}}}
 + \\beta_2\\underbrace{{\\left(\\frac{{1-e^{{-\\tau/\\lambda}}}}{{\\tau/\\lambda}} - e^{{-\\tau/\\lambda}}\\right)}}_{{\\text{{curvature loading}}}} $$
<p>Each factor has an economic reading:</p>
<ul>
<li><strong>Level</strong> \\(\\beta_0\\): a constant loading of 1 at every maturity, so it shifts the whole curve up or down. It is the long-run yield the curve approaches as \\(\\tau\\to\\infty\\).</li>
<li><strong>Slope</strong> \\(\\beta_1\\): its loading is 1 at the short end and decays to 0 at the long end, so it moves short rates relative to long. We report slope as long minus short, \\(-\\beta_1\\).</li>
<li><strong>Curvature</strong> \\(\\beta_2\\): its loading is a hump that is near 0 at both ends and peaks in the middle, so it bends the belly of the curve.</li>
<li>\\(\\lambda\\) sets where that hump sits (how fast the short-rate influence decays).</li>
</ul>
<p>The <strong>Svensson</strong> extension adds a second curvature term with its own
\\(\\beta_3\\) and decay \\(\\lambda_2\\), so it can fit curves with two bends (a second
hump at the long end) that plain Nelson-Siegel cannot.</p>

<h2>2. How the curve is fitted</h2>
<p>For a given day with observed yields \\(y_i\\) at maturities \\(\\tau_i\\), we choose the
parameters that minimize the sum of squared pricing errors:</p>
$$ \\min_{{\\beta_0,\\beta_1,\\beta_2,\\lambda}}\\; \\sum_i \\big(y_i - y(\\tau_i)\\big)^2 $$
<p>Because \\(\\lambda\\) enters non-linearly, this is a non-linear least squares problem,
solved with a trust-region method (<code>scipy.optimize.least_squares</code>) with
\\(\\lambda\\) kept positive. Fit quality is the root-mean-square error,
\\(\\;\\text{{RMSE}}=\\sqrt{{\\frac1N\\sum_i(y_i-y(\\tau_i))^2}}\\), in the same percentage units
as the yields.</p>

<h2>3. Results</h2>
<h3>Fitted curve, {last[0].date()}</h3>
<img alt="Fitted curve" src="data:image/png;base64,{fit_img}">
<p>Fitted Nelson-Siegel factors on this day: level {ns.factors['level']:.2f},
slope {ns.factors['slope']:.2f}, curvature {ns.factors['curvature']:.2f}
(RMSE {ns.rmse:.3f}).</p>

<h3>Factors over time</h3>
<img alt="Factors over time" src="data:image/png;base64,{fac_img}">

<table>
<tr><th>Model</th><th class="num">Mean RMSE (%)</th></tr>
<tr><td>Nelson-Siegel</td><td class="num">{mean_ns:.4f}</td></tr>
<tr><td>Svensson</td><td class="num">{mean_sv:.4f}</td></tr>
</table>

<h2>4. Interpretation</h2>
<p><strong>Three numbers capture the whole curve.</strong> Across {n} days the
Nelson-Siegel fit has an average error of only {mean_ns:.3f} percentage points, so
the eleven observed maturities each day are summarized well by just level, slope,
and curvature. Svensson's extra hump lowers the average error further to
{mean_sv:.3f}, which is expected since it has more parameters.</p>
<p><strong>What the factors did.</strong> The level moved from about {lvl0:.2f} to
{lvl1:.2f} over the sample, so the general level of yields
{"fell" if lvl1 < lvl0 else "rose"}. The slope (long minus short) reached its low of
{slope_min:.2f} around {slope_min_date}. {inversion}</p>
<p class="note"><strong>Why this matters.</strong> Reducing a curve to level, slope,
and curvature is the standard first step for rates work: it makes curves comparable
across days, feeds duration and carry calculations, and a flattening or inverting
slope is one of the most watched recession signals. The same factors drive the bond
returns used in the portfolio project.</p>

<footer>Generated by <code>report.py</code>. Data source in <code>data/SOURCES.md</code>.
Regenerate with <code>python report.py</code>.</footer>
</body></html>
"""


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data", "treasury_yields.csv"))
    ap.add_argument("--out", default=os.path.join(here, "outputs", "report.html"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=["date"]).set_index("date")
    tau = df.columns.astype(float).to_numpy()
    html = build_html(df, tau)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
