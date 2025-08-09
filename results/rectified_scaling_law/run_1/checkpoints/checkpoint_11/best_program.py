# EVOLVE-BLOCK-START
"""
Enhanced 4-parameter shifted power-law scaling function for LLM finetuning loss:
    L(N) = B + A * (N + C)^(−α)
Parameters are represented in log-domain for positivity and numerical stability:
    params = [log_A, log_α, log_C, log_B]
Fitting is done via bounded Levenberg–Marquardt least‐squares with analytic Jacobian.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predict loss given data sizes and log-domain parameters.
    Inputs:
      data_points: array-like of shape (N,) or (N,1) with data sizes
      params:      array-like of 4 log-domain parameters [pA, pα, pC, pB]
    Returns:
      preds: array of length N with predicted losses
    """
    X = np.asarray(data_points).ravel().astype(float)
    p = np.asarray(params).ravel()
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    # Unpack and exponentiate for positivity
    A     = np.exp(p[0])
    α     = np.exp(p[1])
    C     = np.exp(p[2])
    B     = np.exp(p[3])
    # Compute shifted power‐law
    return B + A * (X + C) ** (-α)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter shifted power-law to empirical (data_points, loss_values).
    Returns optimized log-domain params = [pA, pα, pC, pB].
    """
    X = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)

    # Heuristic initial guess in natural domain
    B0 = max(np.min(y) * 0.9, 1e-8)                      # near minimum observed loss
    C0 = max(np.min(X) * 0.5, 1e-8)                     # small horizontal shift
    α0 = 0.5                                            # moderate decay
    # Choose A0 so that at smallest X: A*(X+C)^(-α) ≈ (max(y)-B0)
    A0 = max((np.max(y) - B0) * ( (np.min(X) + C0) ** α0 ), 1e-8)

    # Pack initial log-domain parameters
    p0 = np.log([A0, α0, C0, B0])

    # Residuals function
    def resid(p):
        return scaling_law_func(X, p) - y

    # Analytic Jacobian of residuals w.r.t. log-domain params
    def jac(p):
        A     = np.exp(p[0])
        α     = np.exp(p[1])
        C     = np.exp(p[2])
        B     = np.exp(p[3])
        Xc    = X + C
        # Precompute common terms
        pow_term = Xc ** (-α)
        log_term = np.log(Xc)
        # Allocate Jacobian: shape (N,4)
        J = np.empty((X.size, 4), dtype=float)
        # ∂r/∂pA = ∂(B + A·Xc^(−α) − y)/∂pA = A·Xc^(−α)
        J[:, 0] = A * pow_term
        # ∂r/∂pα = ∂(...)/∂α · ∂α/∂pα = [−A·Xc^(−α)·ln(Xc)]·α
        J[:, 1] = -A * pow_term * log_term * α
        # ∂r/∂pC = ∂(...)/∂C · ∂C/∂pC = [−A·α·Xc^(−α−1)]·C
        J[:, 2] = -A * α * (Xc ** (-α - 1)) * C
        # ∂r/∂pB = ∂(...)/∂B · ∂B/∂pB = 1·B
        J[:, 3] = B
        return J

    # Perform bounded least-squares optimization
    result = least_squares(
        resid,
        p0,
        jac=jac,
        bounds=([-np.inf, -np.inf, -np.inf, -np.inf],
                [ np.inf,  np.inf,  np.inf,  np.inf]),
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=1000,
        verbose=0
    )

    # Return optimized log-params (fallback to p0 if failure)
    return result.x if result.success else p0
# EVOLVE-BLOCK-END