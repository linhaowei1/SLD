# EVOLVE-BLOCK-START
"""
Refined 4-parameter scaling law for LLM fine-tuning:

    L(N) = A * (N + N0)^(-alpha) + B

Parameters:
    A     > 0       (scale of power-law decay)
    alpha > 0       (exponent)
    B     >= 0      (irreducible loss floor)
    N0    >= 0      (horizontal shift for small-N behavior)

We fit this using:
  1) Median-based B0 initialization from large-N region
  2) Grid-search over N0 candidates to initialize A0, alpha0 via
     log–log linear regression
  3) Multi-start bounded robust least squares with relative weighting
  4) Final unweighted refinement to minimize MSE
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predict loss given training sizes and scaling-law parameters.

    Args:
        data_points: array_like of shape (M,)
        params: array_like [A, alpha, B, N0]

    Returns:
        ndarray of shape (M,) with predicted losses
    """
    A, alpha, B, N0 = params
    # enforce positivity/feasible ranges
    A = max(A, 1e-12)
    alpha = max(alpha, 1e-12)
    B = max(B, 0.0)
    N0 = max(N0, 0.0)

    x = np.asarray(data_points, dtype=float)
    x_eff = np.maximum(x + N0, 1e-8)
    return A * x_eff ** (-alpha) + B

def fit_scaling_law(data_points, loss_values):
    """
    Fit L(N)=A*(N+N0)^(-alpha)+B to observed losses.

    Args:
        data_points: array_like of shape (M,)
        loss_values: array_like of shape (M,)

    Returns:
        ndarray [A, alpha, B, N0]
    """
    # Clean and validate inputs
    x = np.asarray(data_points, float).ravel()
    y = np.asarray(loss_values, float).ravel()
    mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        # Fallback trivial fit
        return np.array([1e-3, 0.5, np.mean(y) if y.size else 0.0, 1.0])

    # INITIALIZATION STEP

    # 1) Estimate B0 as median of losses at largest N
    idx_desc = np.argsort(-x)
    k = max(3, len(x)//10)
    B0 = float(np.median(y[idx_desc[:k]]))
    B0 = max(B0 * 0.99, 1e-8)  # slight under-estimate

    # 2) Grid-search over N0 candidates to init A0, alpha0
    x_min = np.min(x)
    candidates = np.unique(np.array([0.0, 0.5*x_min, x_min, 2.0*x_min]))
    best_ssr = np.inf
    A0, alpha0, N00 = 1.0, 0.5, 0.0

    y_shift = y - B0
    # Avoid non-positive
    for N0_c in candidates:
        x_eff = x + N0_c
        mask_pos = y_shift > 0
        if np.sum(mask_pos) < 3:
            continue
        lx = np.log(x_eff[mask_pos])
        ly = np.log(y_shift[mask_pos])
        # linear fit in log-log
        slope, intercept = np.polyfit(lx, ly, 1)
        # SSR in log domain
        ssr = np.sum((ly - (intercept + slope*lx))**2)
        if ssr < best_ssr:
            best_ssr = ssr
            N00 = max(N0_c, 0.0)
            alpha0 = max(-slope, 1e-6)
            A0 = max(np.exp(intercept), 1e-12)

    # 3) Initial parameter vector
    init = np.array([A0, alpha0, B0, N00], dtype=float)

    # Bounds
    lb = np.array([1e-12, 1e-12, 0.0, 0.0], dtype=float)
    ub = np.array([np.inf, 10.0, np.max(y)*1.5, np.max(x)*5.0], dtype=float)

    # Precompute denominator for relative weighting (to lower NMSE)
    eps = 1e-8
    scale = (np.max(y) - np.min(y)) * 0.01 + eps
    denom = y + scale

    # Residual functions
    def resid_rel(p):
        return (scaling_law_func(x, p) - y) / denom

    def resid_abs(p):
        return scaling_law_func(x, p) - y

    # Multi-start initialization set
    rng = np.random.RandomState(123)
    inits = [init]
    for _ in range(4):
        noise = 0.2 * rng.randn(4)
        p_try = init * (1.0 + noise)
        p_try = np.clip(p_try, lb + 1e-8, ub - 1e-8)
        inits.append(p_try)

    best_p = init.copy()
    best_cost = np.inf

    # Robust weighted least-squares (relative errors)
    for p0 in inits:
        try:
            res = least_squares(
                resid_rel,
                p0,
                bounds=(lb, ub),
                loss='huber',
                f_scale=0.1,
                xtol=1e-9,
                ftol=1e-9,
                max_nfev=2000
            )
            # Evaluate true MSE cost
            pred = scaling_law_func(x, res.x)
            cost = np.mean((pred - y)**2)
            if cost < best_cost and res.success:
                best_cost = cost
                best_p = res.x.copy()
        except:
            pass

    # Final unweighted refinement to directly minimize MSE
    try:
        res2 = least_squares(
            resid_abs,
            best_p,
            bounds=(lb, ub),
            loss='linear',
            xtol=1e-9,
            ftol=1e-9,
            max_nfev=1000
        )
        if res2.success:
            best_p = res2.x
    except:
        pass

    # Enforce constraints
    best_p[0] = max(best_p[0], 1e-12)
    best_p[1] = max(best_p[1], 1e-12)
    best_p[2] = max(best_p[2], 0.0)
    best_p[3] = max(best_p[3], 0.0)
    return best_p

# Expose number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END