import pandas as pd
import numpy as np
from typing import Callable
import scipy

# from scipy.optimize import curve_fit
# from scipy.interpolate import UnivariateSpline

ArrayLike = np.ndarray | list[float] | pd.Series
CurveForm = Callable[[ArrayLike, float, float], ArrayLike]


def marginal_effects_curve_form(x: float, alpha: float, beta: float) -> float:
    """Return the marginal effects from the curve at x, given parameters alpha and beta."""
    return alpha - beta * np.log(x)


class MarginalEffectsCurve:
    params: list[float] | None

    def __init__(self, curve_form: CurveForm):
        self.curve_form = curve_form
        self.params = None

    def curve(self, x: ArrayLike) -> np.ndarray:
        if self.params is None:
            raise ValueError("Curve parameters have not been fit.")
        return self.curve_form(x, *self.params)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.params, _ = scipy.optimize.curve_fit(self.curve_form, x, y)
