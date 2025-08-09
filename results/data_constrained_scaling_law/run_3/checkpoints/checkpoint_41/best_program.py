import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict loss from [tokens, params, unique_tokens] using a 7-parameter
    additive power‐law with a floor:
      loss ≈ C0
             + amp_tkn * tokens^(−dec_tkn)
             + amp_prm * params^(−dec_prm)
             + amp_uniq * unique_tokens^(−dec_uniq)

    Inputs:
      data_points: array‐like of shape (N,3) [tokens, params, unique_tokens]
      params:      array‐like of length 7:
                   [C0,
                    amp_tkn, dec_tkn,
                    amp_prm, dec_prm,
                    amp_uniq, dec_uniq]
    Returns:
      preds: np.ndarray of shape (N,) with predicted losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 7:
        raise ValueError(f"Expected 7 parameters [C0, amp_tkn, dec_tkn, amp_prm, dec_prm, amp_uniq, dec_uniq], got {p.size}")
    C0, amp_t, dec_t, amp_p, dec_p, amp_u, dec_u = p

    # Avoid zeros/negatives in power computations
    eps = 1e-12
    tkn = np.maximum(X[:, 0], eps)
    prm = np.maximum(X[:, 1], eps)
    uniq = np.maximum(X[:, 2], eps)

    # Compute additive power-law contributions plus floor
    pred = (
        C0
        + amp_t * np.power(tkn, -dec_t)
        + amp_p * np.power(prm, -dec_p)
        + amp_u * np.power(uniq, -dec_u)
    )
    return pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7 parameters of the additive power‐law scaling law by minimizing
    relative error residuals via least squares.

    Returns:
      params_opt: array of length 7
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")
    y = np.asarray(loss_values, dtype=float).ravel()

    # Ensure strictly positive for relative residuals
    y = np.maximum(y, 1e-12)

    # Initial floor guess: 90% of min loss (to allow some headroom)
    y_min, y_max = np.min(y), np.max(y)
    C0_0 = max(0.0, 0.9 * y_min)

    # Distribute remaining mean residual equally across three amplitudes
    mean_resid = np.mean(y - C0_0)
    amp0 = max(mean_resid / 3.0, 1e-12)

    # Initial decay exponents
    dec0 = 0.5

    # Initial parameter vector: [C0, amp_tkn, dec_tkn, amp_prm, dec_prm, amp_uniq, dec_uniq]
    p0 = np.array([C0_0,
                   amp0, dec0,
                   amp0, dec0,
                   amp0, dec0], dtype=float)

    # Bounds: floor ≥ 0, amps ≥ tiny positive, decays ≥ 0
    lower = np.array([0.0,
                      1e-12, 0.0,
                      1e-12, 0.0,
                      1e-12, 0.0], dtype=float)
    upper = np.full(7, np.inf, dtype=float)

    # Residual function: relative error to balance across scales
    def resid_fn(p):
        pred = scaling_law_func(X, p)
        return (pred - y) / (y + 1e-12)

    # Solve with Trust-Region-Reflective least squares
    sol = least_squares(
        resid_fn,
        p0,
        bounds=(lower, upper),
        method='trf',
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=10000
    )

    # Return optimized params or fallback to initial
    return sol.x if sol.success else p0
# EVOLVE-BLOCK-END