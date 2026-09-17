"""Nelson-Siegel and Nelson-Siegel-Svensson term-structure models.

Yields and maturities are in the same units the data uses. Maturities
(tau) are in years. Yields are in percent (e.g. 4.25 for 4.25%).
"""

from __future__ import annotations

import numpy as np


def _loading(tau: np.ndarray, lam: float) -> np.ndarray:
    """Slope loading: (1 - exp(-tau/lam)) / (tau/lam)."""
    x = tau / lam
    # limit of the loading as tau -> 0 is 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x == 0, 1.0, (1.0 - np.exp(-x)) / x)
    return out


def nelson_siegel(tau: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    """Nelson-Siegel fitted yield for each maturity tau.

    beta0 is the long-run level, beta1 the short-term (slope) factor,
    beta2 the medium-term (curvature) factor, lam the decay parameter.
    """
    tau = np.asarray(tau, dtype=float)
    slope = _loading(tau, lam)
    curve = slope - np.exp(-tau / lam)
    return beta0 + beta1 * slope + beta2 * curve


def svensson(
    tau: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lam1: float,
    lam2: float,
) -> np.ndarray:
    """Nelson-Siegel-Svensson fitted yield. Adds a second curvature hump."""
    tau = np.asarray(tau, dtype=float)
    slope = _loading(tau, lam1)
    curve1 = slope - np.exp(-tau / lam1)
    curve2 = _loading(tau, lam2) - np.exp(-tau / lam2)
    return beta0 + beta1 * slope + beta2 * curve1 + beta3 * curve2
