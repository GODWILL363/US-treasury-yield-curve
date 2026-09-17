import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yield_curve.models import nelson_siegel, svensson
from yield_curve.fit import fit_nelson_siegel, fit_svensson


TAU = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


def test_ns_short_end_equals_beta0_plus_beta1():
    # As tau -> 0 the fitted yield approaches beta0 + beta1.
    y = nelson_siegel(np.array([1e-6]), 3.0, -1.0, 0.5, 2.0)[0]
    assert abs(y - (3.0 - 1.0)) < 1e-3


def test_ns_long_end_approaches_beta0():
    y = nelson_siegel(np.array([100.0]), 3.0, -1.0, 0.5, 2.0)[0]
    assert abs(y - 3.0) < 0.1


def test_fit_recovers_ns_params():
    true = nelson_siegel(TAU, 3.0, -1.5, 2.0, 1.8)
    fit = fit_nelson_siegel(TAU, true)
    assert fit.rmse < 1e-6
    assert np.allclose(fit.predict(TAU), true, atol=1e-4)


def test_svensson_fits_at_least_as_well_as_ns():
    y = np.array([2.4, 2.5, 2.6, 2.5, 2.47, 2.49, 2.56, 2.66, 2.83, 2.97])
    ns = fit_nelson_siegel(TAU, y)
    sv = fit_svensson(TAU, y)
    # more parameters should not fit worse
    assert sv.rmse <= ns.rmse + 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
