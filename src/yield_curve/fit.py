"""Fit Nelson-Siegel and Svensson curves to observed yields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .models import nelson_siegel, svensson


@dataclass
class NSFit:
    beta0: float
    beta1: float
    beta2: float
    lam: float
    rmse: float

    def predict(self, tau):
        return nelson_siegel(tau, self.beta0, self.beta1, self.beta2, self.lam)

    @property
    def factors(self) -> dict:
        # Standard readings of the three factors.
        return {
            "level": self.beta0,
            "slope": -self.beta1,          # long minus short
            "curvature": self.beta2,
        }


@dataclass
class SvenssonFit:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lam1: float
    lam2: float
    rmse: float

    def predict(self, tau):
        return svensson(tau, self.beta0, self.beta1, self.beta2, self.beta3, self.lam1, self.lam2)


def _rmse(resid: np.ndarray) -> float:
    return float(np.sqrt(np.mean(resid ** 2)))


def fit_nelson_siegel(tau, yields) -> NSFit:
    """Fit NS by nonlinear least squares.

    Starting values: level at the longest yield, slope from short minus long,
    curvature at zero, lambda near the 2-3 year area where the hump usually sits.
    """
    tau = np.asarray(tau, dtype=float)
    yields = np.asarray(yields, dtype=float)

    b0 = yields[-1]
    b1 = yields[0] - yields[-1]
    x0 = [b0, b1, 0.0, 2.0]

    def resid(p):
        return nelson_siegel(tau, p[0], p[1], p[2], p[3]) - yields

    res = least_squares(
        resid,
        x0=x0,
        bounds=([-30, -30, -30, 0.05], [30, 30, 30, 30]),
        method="trf",
    )
    p = res.x
    return NSFit(p[0], p[1], p[2], p[3], _rmse(res.fun))


def fit_svensson(tau, yields) -> SvenssonFit:
    """Fit NSS by nonlinear least squares."""
    tau = np.asarray(tau, dtype=float)
    yields = np.asarray(yields, dtype=float)

    b0 = yields[-1]
    b1 = yields[0] - yields[-1]
    x0 = [b0, b1, 0.0, 0.0, 2.0, 5.0]

    def resid(p):
        return svensson(tau, p[0], p[1], p[2], p[3], p[4], p[5]) - yields

    res = least_squares(
        resid,
        x0=x0,
        bounds=([-30, -30, -30, -30, 0.05, 0.05], [30, 30, 30, 30, 30, 30]),
        method="trf",
    )
    p = res.x
    return SvenssonFit(p[0], p[1], p[2], p[3], p[4], p[5], _rmse(res.fun))
