# EVOLVE-BLOCK-START
"""
Refined 4-parameter scaling law for LLM finetuning:

    L(N) = d + a * (N + c)^(-b)

Parameters:
    a > 0   : scale of the power‐law decay
    b > 0   : exponent
    c ≥ 0   : horizontal offset to handle small N
    d ≥ 0   : irreducible loss floor

Fitting strategy:
    1. Estimate floor d via 0.9 * min(observed loss).
    2. Subtract floor and perform a log‐linear least squares to initialize (a, b).
    3. Use a single robust nonlinear least‐squares solve (Huber loss) with bounds.
    4. Fallback to analytic init if optimization fails.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predict loss given training data sizes via a 4‐parameter scaling law:
        L(N) = d + a * (N + c)^(-b)

    Args:
        data_points: array‐like of N (training sizes)
        params: [a, b, c, d]

    Returns:
        numpy array of predicted L(N)
    """
    a, b, c, d = params
    x = np.asarray(data_points, dtype=np.float64)
    # ensure c ≥ 0, shift
    x_shift = x + max(c, 0.0)
    # guard against non-positive
    x_shift = np.maximum(x_shift, 1e-12)
    return d + a * np.power(x_shift, -b)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4‐parameter scaling law to data (N, L) by robust nonlinear least squares.

    Args:
        data_points: array‐like of training sizes
        loss_values: array‐like of observed losses

    Returns:
        best_params: array [a, b, c, d]
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # 1) Floor estimate
    y_min, y_max = np.min(y), np.max(y)
    d0 = max(0.0, 0.9 * y_min)

    # 2) Log-linear initialization for a, b (assuming c=0, d=d0)
    y_adj = y - d0
    # avoid non-positive
    y_adj = np.where(y_adj <= 0, y_min * 1e-3 + 1e-6, y_adj)
    logx = np.log(x + 1e-6)
    logy = np.log(y_adj)
    # solve logy ≈ log(a) - b * logx  => [ -logx, 1 ] * [b, log(a)] = logy
    A = np.vstack([-logx, np.ones_like(logx)]).T
    sol, *_ = np.linalg.lstsq(A, logy, rcond=None)
    b0, loga0 = sol
    a0 = np.exp(loga0)
    b0 = max(b0, 1e-3)

    # c0 small offset
    c0 = 1e-6

    init = np.array([a0, b0, c0, d0], dtype=np.float64)

    # Bounds: a>0, b>0, c≥0, d≥0
    lower = [1e-8, 1e-8, 0.0, 0.0]
    upper = [
        (y_max - y_min) * 20 + 1e-6,  # a
        10.0,                        # b
        np.max(x) * 2 + 1.0,         # c
        y_max * 2 + 1e-6             # d
    ]

    # Residuals for least_squares
    def resid(p):
        return scaling_law_func(x, p) - y

    # Robust solve with Huber loss
    try:
        res = least_squares(
            resid,
            init,
            bounds=(lower, upper),
            loss='huber',
            f_scale=0.1,
            xtol=1e-12,
            ftol=1e-12,
            max_nfev=2000
        )
        if res.success:
            return res.x
    except Exception:
        pass

    # Fallback to analytic init if optimize fails
    return init

# record number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END