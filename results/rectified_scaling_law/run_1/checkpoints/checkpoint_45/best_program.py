import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predicts loss as a 4-parameter shifted power law:
      L(N) = B + A * (N + C)^(-alpha)
    where B, A, C, alpha >= 0.

    Inputs:
      data_points: array-like of shape (N,) or (N,1) with data sizes N
      params:      array-like of 4 parameters [B, A, C, alpha]
    Returns:
      preds: numpy.ndarray of shape (N,) with predicted losses
    """
    N = np.asarray(data_points).ravel().astype(float)
    B, A, C, alpha = params
    # Ensure positivity inside the power
    return B + A * np.power(N + C, -alpha)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law L(N) = B + A*(N + C)^(-alpha)
    to the provided (data_points, loss_values) via robust, weighted,
    bounded least-squares in the log-parameter space.

    Returns optimized natural-domain params = [B, A, C, alpha].
    """
    # Prepare data
    X = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)
    # Small constants
    eps = 1e-8

    # === INITIAL GUESS HEURISTICS ===
    # Asymptotic floor just below minimal observed loss
    B0 = max(np.min(y) * 0.9, eps)
    # Horizontal shift as small fraction of minimal data size
    C0 = max(np.min(X) * 0.1, eps)

    # Shifted target for log–log fitting
    y_shift = np.clip(y - B0, eps, None)
    X_shift = X + C0

    # Estimate slope/intercept: log(y_shift) ≈ intercept + slope * log(X_shift)
    try:
        slope, intercept = np.polyfit(np.log(X_shift), np.log(y_shift), 1)
        alpha0 = max(-slope, 1e-4)
        A0     = max(np.exp(intercept), eps)
    except Exception:
        # Fallback defaults
        alpha0 = 0.5
        A0     = max(np.max(y) - B0, eps)

    # Pack natural-domain guess and convert to log-domain
    p0 = np.log([B0, A0, C0, alpha0])

    # === RESIDUAL AND JACOBIAN IN LOG-PARAM SPACE ===
    def _residuals(p_log):
        # Recover positive parameters
        B, A, C, alpha = np.exp(p_log)
        Xc = X + C
        pred = B + A * Xc**(-alpha)
        # Relative residual: balances scale across losses
        return (pred - y) / (y + eps)

    def _jacobian(p_log):
        B, A, C, alpha = np.exp(p_log)
        Xc = X + C
        # Derivatives of L wrt natural params
        dL_dB     = np.ones_like(X)                # ∂L/∂B
        dL_dA     = Xc ** (-alpha)                 # ∂L/∂A
        dL_dC     = -A * alpha * Xc**(-alpha - 1)  # ∂L/∂C
        dL_dalpha = -A * Xc**(-alpha) * np.log(Xc) # ∂L/∂alpha
        # Chain-rule: dL/d(log-param) = dL/dparam * param
        dB = dL_dB     * B
        dA = dL_dA     * A
        dC = dL_dC     * C
        da = dL_dalpha * alpha
        # Stack and apply relative weighting
        W = 1.0 / (y + eps)
        J = np.vstack((dB * W, dA * W, dC * W, da * W)).T
        return J

    # === OPTIMIZATION ===
    result = least_squares(
        fun=_residuals,
        x0=p0,
        jac=_jacobian,
        loss='huber',
        f_scale=1.0,
        xtol=1e-9,
        ftol=1e-9,
        max_nfev=5000
    )

    # Recover and return natural-domain parameters
    p_opt = result.x if result.success else p0
    return np.exp(p_opt)