import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Shifted power law with log‐domain parameters:
      L(D) = c + A * (D + d0)^(-alpha)

    params = [pA, pα, pd0, pc], each log‐transformed:
      A     = exp(pA)
      α     = exp(pα)
      d0    = exp(pd0)
      c     = exp(pc)
    This enforces positivity and numerical stability.
    """
    D = np.asarray(data_points).ravel().astype(float)
    p = np.asarray(params).ravel()
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    pA, p_alpha, p_d0, p_c = p
    A     = np.exp(pA)
    alpha = np.exp(p_alpha)
    d0    = np.exp(p_d0)
    c     = np.exp(p_c)
    # shifted data size
    D_shift = D + d0
    return c + A * D_shift**(-alpha)


def fit_scaling_law(data_points, loss_values):
    """
    Fits the 4‐parameter shifted‐power‐law to (data_points, loss_values)
    by weighted nonlinear least squares in the log‐domain. Minimizes
    relative error to balance absolute and relative fit quality.
    Returns optimized params = [pA, pα, pd0, pc].
    """
    D = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)

    # 1) Initial guess for c slightly below the minimum observed loss
    c0 = max(np.min(y) * 0.9, 1e-4)
    # 2) Small horizontal shift to avoid singularity
    d0_0 = max(np.min(D) * 0.1, 1.0)

    # 3) Linearize log‐log (for D + d0_0) to estimate A0, alpha0
    mask = y > c0
    if mask.sum() >= 2:
        X_lin = np.log(D[mask] + d0_0)
        Y_lin = np.log(y[mask] - c0)
        slope, intercept = np.polyfit(X_lin, Y_lin, 1)
        alpha0 = max(-slope, 1e-3)
        A0     = max(np.exp(intercept), 1e-3)
    else:
        alpha0 = 0.5
        A0     = max(np.max(y) - c0, 1e-3)

    # 4) Pack into log‐domain initial vector
    p0 = np.log([A0, alpha0, d0_0, c0])

    # Residual: relative error (predicted - actual) / actual
    def resid(p):
        pred = scaling_law_func(D, p)
        return (pred - y) / np.maximum(y, 1e-8)

    # Analytical Jacobian of residuals w.r.t. p = [pA, pα, pd0, pc]
    def jac(p):
        pA, p_alpha, p_d0, p_c = p
        A     = np.exp(pA)
        alpha = np.exp(p_alpha)
        d0    = np.exp(p_d0)
        c     = np.exp(p_c)

        D_shift = D + d0
        pow_term = D_shift**(-alpha)

        # ∂L/∂pA = A * D_shift^(−alpha)
        dL_dA = pow_term * A
        # ∂L/∂pα = −A * D_shift^(−alpha) * ln(D_shift) * alpha
        dL_dalpha = -A * pow_term * np.log(D_shift) * alpha
        # ∂L/∂pd0 = −A * alpha * D_shift^(−alpha−1) * d0
        dL_dd0 = -A * alpha * D_shift**(-alpha - 1) * d0
        # ∂L/∂pc = c
        dL_dc = c

        denom = np.maximum(y, 1e-8)
        # Stack into Jacobian matrix of shape (N, 4)
        return np.vstack([
            dL_dA    / denom,
            dL_dalpha/ denom,
            dL_dd0   / denom,
            dL_dc    / denom
        ]).T

    # Bounds in log‐domain to keep d0 and c positive but constrained
    lower = [-np.inf, -np.inf, np.log(1e-8),    np.log(1e-8)]
    upper = [ np.inf,  np.inf, np.log(D.max()*10), np.log(np.min(y))]

    result = least_squares(
        resid,
        p0,
        jac=jac,
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
        method='trf',
        verbose=0
    )

    return result.x if result.success else p0

# EVOLVE-BLOCK-END