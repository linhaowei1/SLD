# EVOLVE-BLOCK-START
"""
Refined Domain Mixture Scaling Law for LLM Training Scenarios

We adopt a 7-parameter per-domain form (35 total parameters), blending
inverse/self-power and competition terms with flexible exponents:

  L_i = A_i
        + B_i * p_i^{ -C_i }
        + D_i * (1 - p_i)^{ E_i }
        + F_i * [ p_i * (1 - p_i)^{ G_i } ]

Transforms ensure positivity and numerical stability:
  B, D, F = exp(raw)
  C, E, G = softplus(raw) + eps > 0
"""
import numpy as np
from scipy.optimize import least_squares

def _softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses from 5-domain mixture proportions.

    Args:
      proportions: array_like (n_samples, 5)
      params:       array_like (35,) raw parameters

    Returns:
      losses: ndarray (n_samples, 5)
    """
    P = np.atleast_2d(proportions).astype(np.float64)
    n, d = P.shape
    assert d == 5, "Expect 5 domain proportions"
    # Clip for stability
    P_safe = np.clip(P, 1e-12, 1.0)

    # Prepare parameters
    theta = np.array(params, dtype=np.float64).flatten()
    if theta.size < 35:
        theta = np.concatenate([theta, np.zeros(35 - theta.size)])
    else:
        theta = theta[:35]
    raw = theta.reshape(5, 7)

    # Compute losses
    L = np.zeros((n, 5), dtype=np.float64)
    for i in range(5):
        A_raw, B_raw, C_raw, D_raw, E_raw, F_raw, G_raw = raw[i]
        A = A_raw
        B = np.exp(B_raw)
        C = _softplus(C_raw) + 1e-6
        D = np.exp(D_raw)
        E = _softplus(E_raw) + 1e-6
        F = np.exp(F_raw)
        G = _softplus(G_raw) + 1e-6

        p_i = P_safe[:, i]
        one_minus = 1.0 - p_i

        term_inv   = B * (p_i ** (-C))
        term_comp  = D * (one_minus ** E)
        term_cross = F * (p_i * (one_minus ** G))

        L[:, i] = A + term_inv + term_comp + term_cross

    return L

def fit_scaling_law(proportions, loss_values):
    """
    Fit the refined scaling law to observed losses.

    Args:
      proportions:  array_like (n_samples, 5)
      loss_values:  array_like (n_samples, 5)

    Returns:
      best_params: ndarray (35,) optimized raw params
    """
    X = np.atleast_2d(proportions).astype(np.float64)
    Y = np.atleast_2d(loss_values).astype(np.float64)
    n, d = X.shape
    assert d == 5 and Y.shape == (n, 5)

    # Inverse softplus to initialize exponents ≈ 1.0
    inv_sp = lambda y: np.log(np.exp(y) - 1.0 + 1e-8)
    c0 = inv_sp(1.0)

    # Build initial parameter vector
    init = np.zeros(35, dtype=np.float64)
    for i in range(5):
        init[i*7 + 0] = np.mean(Y[:, i])  # A_i
        init[i*7 + 1] = 0.0               # B_i_raw -> exp(0)=1
        init[i*7 + 2] = c0                # C_i_raw -> softplus≈1
        init[i*7 + 3] = 0.0               # D_i_raw -> exp(0)=1
        init[i*7 + 4] = c0                # E_i_raw -> softplus≈1
        init[i*7 + 5] = 0.0               # F_i_raw -> exp(0)=1
        init[i*7 + 6] = c0                # G_i_raw -> softplus≈1

    # Residual vector for least_squares
    def residuals(theta):
        return (scaling_law_func(X, theta) - Y).ravel()

    best_params = init.copy()
    best_cost = np.inf

    # Multi-start optimization
    for attempt in range(3):
        start = init if attempt == 0 else init + 0.05 * np.random.randn(35)
        res = least_squares(residuals,
                            x0=start,
                            method='trf',
                            ftol=1e-9,
                            xtol=1e-9,
                            max_nfev=2000)
        if res.success and res.cost < best_cost:
            best_cost = res.cost
            best_params = res.x

    return best_params

# Attach expected parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END