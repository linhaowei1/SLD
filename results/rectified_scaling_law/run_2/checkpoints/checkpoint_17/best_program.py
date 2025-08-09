import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    4-parameter shifted power law:
      L(D) = c + A * (D + d0)^(-alpha)

    params = [c, A, alpha, d0]
    """
    D = np.asarray(data_points).ravel().astype(float)
    c, A, alpha, d0 = params
    # ensure positivity inside exponent
    D_shift = np.maximum(D + d0, 1e-8)
    return c + A * D_shift**(-alpha)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law by bounded least squares.
    Returns params = [c, A, alpha, d0].
    """
    D = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)

    # Heuristic for c0: asymptotic floor near min(y)
    c0 = max(0.0, y.min() * 0.9)

    # Quick log-log linearization to get A0, alpha0
    y_shift = y - c0
    # Avoid non-positive before taking log
    eps = np.finfo(float).tiny
    y_shift = np.where(y_shift > eps, y_shift, eps)
    logslope, logintercept = np.polyfit(np.log(D + eps), np.log(y_shift), 1)
    alpha0 = max(1e-4, -logslope)
    A0     = max(1e-4, np.exp(logintercept))

    # Small shift
    d0_0 = max(1.0, 0.1 * D.min())

    x0 = np.array([c0, A0, alpha0, d0_0], dtype=float)

    # Bounds: 
    #   0 ≤ c ≤ min(y)
    #   0 ≤ A
    #   0 ≤ alpha ≤ 5
    #   0 ≤ d0 ≤ max(D)
    lower = [0.0,        0.0,      0.0,    0.0]
    upper = [y.min(),   10 * A0,   5.0,    D.max()]

    # Residuals
    def resid(p):
        return scaling_law_func(D, p) - y

    # Jacobian of residuals
    def jac(p):
        c, A, alpha, d0 = p
        D_shift = np.maximum(D + d0, 1e-8)
        denom    = D_shift**(-alpha)
        # dL/dc = 1
        # dL/dA = denom
        # dL/dalpha = -A * log(D_shift) * denom
        # dL/dd0 = -A * alpha * D_shift^(-alpha-1)
        return np.vstack((
            np.ones_like(D),
            denom,
            -A * np.log(D_shift) * denom,
            -A * alpha * D_shift**(-alpha - 1),
        )).T

    result = least_squares(
        resid,
        x0,
        jac=jac,
        bounds=(lower, upper),
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=2000,
        verbose=0
    )

    return result.x

# EVOLVE-BLOCK-END