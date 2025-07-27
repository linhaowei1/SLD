# EVOLVE-BLOCK-START
"""
Enhanced scaling law discovery for LLM finetuning scenarios.
Function form: L(N) = A * (N + N0)^(-α) + B
Parameters: A, α, B, N0 (4 total)
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict loss given training size and scaling-law parameters.

    Args:
        data_points: 1D array of training set sizes (N).
        params: array_like of 4 parameters [A, alpha, B, N0].

    Returns:
        loss_pred: array of predicted losses.
    """
    x = np.asarray(data_points, dtype=float)
    A, alpha, B, N0 = params
    # enforce non-negativity
    alpha = max(alpha, 1e-12)
    N0 = max(N0, 0.0)
    # avoid zero or negative base
    x_eff = np.maximum(x + N0, 1e-8)
    return A * x_eff ** (-alpha) + B


def fit_scaling_law(data_points, loss_values):
    """
    Fit the scaling law L(N) = A*(N+N0)^(-α) + B to observed losses.

    Uses a log–linear initialization plus multi-start L-BFGS-B.

    Args:
        data_points: array_like of training sizes.
        loss_values: array_like of observed losses.

    Returns:
        best_params: ndarray of shape (4,) with [A, alpha, B, N0].
    """
    # Prepare data
    x = np.asarray(data_points, dtype=float).ravel()
    y = np.asarray(loss_values, dtype=float).ravel()
    mask = (x >= 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        # Not enough data: return defaults
        return np.array([1e-3, 0.5, np.mean(y), 1.0])

    # -------- Initial guess via log–linear fit --------
    # Shift floor B0 to ensure positivity
    B0 = max(np.min(y) * 0.9, 1e-8)
    y_shift = y - B0
    # Clip for log-domain
    y_shift = np.maximum(y_shift, 1e-8)
    x_clip = np.maximum(x, 1.0)

    # Perform linear regression on log-log
    Ly = np.log(y_shift)
    Lx = np.log(x_clip)
    M = np.vstack([Lx, np.ones_like(Lx)]).T
    try:
        sol, *_ = np.linalg.lstsq(M, Ly, rcond=None)
        slope, intercept = sol
        alpha0 = max(-slope, 1e-3)
        A0 = max(np.exp(intercept), 1e-8)
    except Exception:
        # Fallback defaults
        alpha0, A0 = 0.5, (np.max(y) - np.min(y)) * (np.min(x) + 1) ** 0.5

    # Initial N0
    N0_0 = max(np.min(x), 1.0)
    init = np.array([A0, alpha0, B0, N0_0])

    # -------- Optimization setup --------
    bounds = [
        (1e-12, None),           # A > 0
        (1e-12, 10.0),           # alpha in (0, 10]
        (0.0, np.max(y) * 2),    # B >= 0
        (0.0, np.max(x) * 10)    # N0 >= 0
    ]

    def objective(p):
        pred = scaling_law_func(x, p)
        return np.mean((pred - y) ** 2)

    best_params = None
    best_val = np.inf

    # Multi-start candidates: analytic + randomized
    inits = [init]
    for scale in (0.2, 0.5, 1.0):
        perturb = init * (1 + scale * (np.random.rand(4) - 0.5))
        inits.append(perturb)

    for p0 in inits:
        try:
            res = minimize(
                objective,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue

    # Fallback to initial if no successful run
    if best_params is None:
        best_params = init.copy()

    return best_params

# Declare expected number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END