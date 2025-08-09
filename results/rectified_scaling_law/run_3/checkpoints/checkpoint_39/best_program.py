# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict loss using 4-parameter scaling law:
        L(N) = B + A * (N + C)^(-alpha)
    where params = [lnA, lnalpha, lnC, lnB].
    """
    X = np.asarray(data_points).reshape(-1).astype(float)
    p = np.asarray(params).reshape(-1)
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    A, alpha, C, B = np.exp(p)
    return B + A * (X + C) ** (-alpha)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to (data_points, loss_values).
    Returns optimized log-domain params [lnA, lnalpha, lnC, lnB].
    Uses a small hybrid objective (natural + relative MSE), analytic gradients,
    and a simple size-based weighting to favor large-data generalization.
    """
    X = np.asarray(data_points).reshape(-1).astype(float)
    y = np.asarray(loss_values).reshape(-1).astype(float)
    eps = 1e-8

    # 1) Natural-domain initial guesses
    B0 = max(np.min(y) * 0.9, eps)
    C0 = max(np.median(X) * 0.1, eps)

    # 2) Quick log-linear fit for A and alpha
    y_shift = y - B0
    if np.any(y_shift <= 0):
        min_pos = np.min(y_shift[y_shift > 0]) if np.any(y_shift > 0) else eps
        y_shift = np.clip(y_shift, min_pos, None)
    logX = np.log(X + C0)
    logY = np.log(y_shift)
    slope, intercept = np.polyfit(logX, logY, 1)
    alpha0 = max(-slope, eps)
    A0 = max(np.exp(intercept), eps)

    p0 = np.log([A0, alpha0, C0, B0])

    # 3) Weighting scheme: emphasize larger N
    w = X / np.sum(X)      # sum(w) == 1
    lam = 0.2              # relative‐error weight

    def obj_and_grad(p):
        # unpack
        A, alpha, C, B = np.exp(p)
        Xc = X + C
        pred = B + A * Xc ** (-alpha) + eps
        resid = pred - y
        log_resid = np.log(pred) - np.log(y + eps)

        # objective = ∑ w * resid^2  +  lam * ∑ w * log_resid^2
        obj = np.sum(w * resid * resid) + lam * np.sum(w * log_resid * log_resid)

        # analytic gradient w.r.t. log-parameters
        dA     = A * Xc ** (-alpha)
        dalpha = -A * alpha * Xc ** (-alpha) * np.log(Xc)
        dC     = -A * alpha * C * Xc ** (-alpha - 1)
        dB     = B * np.ones_like(X)

        # gradient from natural MSE term
        grad_n = 2.0 * np.array([
            np.sum(w * resid * dA),
            np.sum(w * resid * dalpha),
            np.sum(w * resid * dC),
            np.sum(w * resid * dB),
        ])

        # gradient from relative (log) MSE term
        grad_r = 2.0 * lam * np.array([
            np.sum(w * log_resid * (dA / pred)),
            np.sum(w * log_resid * (dalpha / pred)),
            np.sum(w * log_resid * (dC / pred)),
            np.sum(w * log_resid * (dB / pred)),
        ])

        grad = grad_n + grad_r
        return obj, grad

    result = minimize(
        fun=lambda p: obj_and_grad(p)[0],
        x0=p0,
        jac=lambda p: obj_and_grad(p)[1],
        method='L-BFGS-B'
    )

    return result.x if result.success else p0
# EVOLVE-BLOCK-END