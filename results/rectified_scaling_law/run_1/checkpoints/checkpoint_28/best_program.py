# EVOLVE-BLOCK-START
"""
Refined 4-parameter shifted power‐law:
    L(N) = B + A * (N + C)^(-α)
Parameters are kept positive via log-reparameterization.
Fitting minimizes relative squared residuals with analytic Jacobian
and bound constraints for numerical stability.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    data_points: array of shape (N,) or (N,1) of data sizes
    params:      array of 4 log-domain parameters [pA, pα, pC, pB]
    Returns:     Predicted loss array of shape (N,)
    """
    X = np.asarray(data_points).ravel().astype(float)
    p = np.asarray(params).ravel()
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    # Exponentiate to enforce positivity
    A, α, C, B = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), np.exp(p[3])
    # Compute shifted power‐law prediction
    return B + A * (X + C) ** (-α)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter shifted power‐law to (data_points, loss_values).
    Returns optimized log-domain params = [pA, pα, pC, pB].
    """
    X = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)
    # Heuristic initial guess in natural domain
    B0 = max(np.min(y) * 0.95, 1e-6)                     # near lower loss bound
    C0 = max(np.min(X) * 0.1, 1e-6)                      # small horizontal shift
    α0 = 0.7                                            # moderate decay
    A0 = max((np.max(y) - B0) * ((np.min(X) + C0) ** α0), 1e-6)
    # Pack into log-domain
    p0 = np.log([A0, α0, C0, B0])

    # Weighted residuals: relative error to handle wide loss range
    def resid(p):
        pred = scaling_law_func(X, p)
        # avoid division by zero
        w = np.maximum(y, 1e-8)
        return (pred - y) / w

    # Analytic Jacobian of the weighted residuals
    def jac(p):
        A, α, C, B = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), np.exp(p[3])
        Xc = X + C
        pow_term = Xc ** (-α)
        log_Xc = np.log(Xc)
        w = np.maximum(y, 1e-8)
        J = np.empty((X.size, 4), dtype=float)
        # ∂r/∂pA =   A * Xc^(-α) / w
        J[:, 0] = A * pow_term / w
        # ∂r/∂pα =  -A * Xc^(-α) * ln(Xc) * α / w
        J[:, 1] = -A * pow_term * log_Xc * α / w
        # ∂r/∂pC =  -A * α * Xc^(-α-1) * C / w
        J[:, 2] = -A * α * (Xc ** (-α - 1)) * C / w
        # ∂r/∂pB =   B / w
        J[:, 3] = B / w
        return J

    # Bound log-parameters to avoid degenerate fits
    lower = np.log([1e-6, 1e-3, 1e-6, 1e-6])
    upper = np.log([1e3, 5.0, 1e6, np.max(y) * 1.1])

    result = least_squares(
        resid, p0, jac=jac,
        bounds=(lower, upper),
        xtol=1e-9, ftol=1e-9, max_nfev=2000
    )
    # return optimized log-params (fallback to p0 if optimization fails)
    return result.x if result.success else p0
# EVOLVE-BLOCK-END