# EVOLVE-BLOCK-START
"""
Improved scaling law discovery for LLM finetuning scenarios.

We fit a 4‐parameter form:
    L(N) = a * (N + c)^(-b) + d

where:
    a > 0      : scale of the power‐law decay
    b > 0      : exponent
    c >= 0     : horizontal shift to handle small N
    d >= 0      : irreducible loss (baseline)

Key enhancements:
  - Analytic log–log initialization for (a, b) via linear regression
  - Adaptive initial guess for c and d
  - Curve‐fit with bounds for fast, robust fitting
  - Fallback to multi‐start L-BFGS-B local optimization if curve‐fit fails
  - Simple, stable function form with 4 parameters
"""
import numpy as np
from scipy.optimize import curve_fit, minimize

def scaling_law_func(data_points, params):
    """
    Four-parameter scaling law:
        y = a * (x + c)^(-b) + d

    Args:
        data_points: 1D array of training data sizes
        params: [a, b, c, d]
    Returns:
        1D array of predicted losses
    """
    a, b, c, d = params
    x = np.asarray(data_points, dtype=float)
    # Enforce non-negative shift
    c0 = float(c) if c >= 0 else 0.0
    x_shift = x + c0 + 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        y = a * np.power(x_shift, -b) + d
    # Clamp to finite
    return np.nan_to_num(y, nan=1e6, posinf=1e6, neginf=-1e6)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to given data.

    Args:
        data_points: 1D array of training data sizes
        loss_values: 1D array of observed losses
    Returns:
        params: optimized [a, b, c, d]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Bounds for parameters: a>0, b>0, c>=0, d>=0
    bounds = (
        [1e-8, 1e-6, 0.0, 0.0],   # lower
        [1e6,  10.0, 1e7, np.max(y)]  # upper
    )

    # 1) Analytic initialization
    # Baseline d0 ~ min observed loss * 0.9 (to avoid zero shift)
    d0 = max(0.0, np.min(y) * 0.9)
    # Horizontal shift c0 ~ small fraction of min(x)
    c0 = max(1e-8, np.min(x) * 0.01)
    # Linearize: log(y - d0) = log(a) - b * log(x + c0)
    y_shift = np.clip(y - d0, a_min=1e-12, a_max=None)
    X = np.log(x + c0)
    Y = np.log(y_shift)
    # Fit slope/intercept
    slope, intercept = np.polyfit(X, Y, 1)
    b0 = max(1e-6, -slope)
    a0 = max(1e-8, np.exp(intercept))
    p0 = [a0, b0, c0, d0]

    # 2) Try bounded curve_fit for speed & robustness
    def _curve(x_arr, a, b, c, d):
        return a * np.power(x_arr + np.clip(c, 0.0, None) + 1e-12, -b) + d

    try:
        popt, _ = curve_fit(
            _curve, x, y, p0, bounds=bounds, maxfev=5000
        )
        return list(popt)
    except Exception:
        pass  # fall through to multi-start local search

    # 3) Multi-start L-BFGS-B fallback
    def obj(params):
        y_pred = scaling_law_func(x, params)
        return np.mean((y_pred - y)**2)

    best_params = None
    best_obj = np.inf
    rng = np.random.default_rng(seed=42)
    # include analytic p0 plus 5 random inits
    inits = [p0] + [
        [
            rng.uniform(bounds[0][i], bounds[1][i])
            for i in range(4)
        ]
        for _ in range(5)
    ]
    for init in inits:
        try:
            res = minimize(
                obj, init,
                method='L-BFGS-B',
                bounds=list(zip(*bounds)),
                options={'ftol':1e-9, 'gtol':1e-6, 'maxiter':500}
            )
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            continue

    # If all fails, return analytic estimate
    return best_params.tolist() if best_params is not None else p0

# Informational attribute
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END