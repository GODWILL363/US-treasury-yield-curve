"""Plots for fitted yield curves and factor time series."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_fit(tau, observed, ns_fit, sv_fit=None, title="US Treasury yield curve", path=None):
    grid = np.linspace(min(tau), max(tau), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, observed, color="black", zorder=3, label="Observed par yields")
    ax.plot(grid, ns_fit.predict(grid), color="#c0392b", lw=2, label=f"Nelson-Siegel (RMSE {ns_fit.rmse:.3f})")
    if sv_fit is not None:
        ax.plot(grid, sv_fit.predict(grid), color="#2471a3", lw=2, ls="--",
                label=f"Svensson (RMSE {sv_fit.rmse:.3f})")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (percent)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_factors(dates, level, slope, curvature, path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dates, level, label="Level (beta0)", color="#1f3864")
    ax.plot(dates, slope, label="Slope (long - short)", color="#c0392b")
    ax.plot(dates, curvature, label="Curvature (beta2)", color="#27ae60")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.set_title("Nelson-Siegel factors over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig
