import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(proportions, params):
    """
    Scaling law: For each domain i:
      α_i : bias
      γ_i : exponent
      β_i1..β_i5 : weights on negative-log proportions
    Li = α_i + (sum_j β_ij * ( -log(p_j + ε) ))^γ_i

    Args:
      proportions: array [n_samples, 5] of domain proportions (rows sum to 1)
      params     : flat array of length 35 (5 domains × 7 params each)
                   ordering per domain i: [α_i, γ_i, β_i1, β_i2, β_i3, β_i4, β_i5]

    Returns:
      losses: array [n_samples, 5]
    """
    proportions = np.atleast_2d(proportions).astype(float)
    n_samples, n_dom = proportions.shape
    assert n_dom == 5, "Expect 5 domain proportions"

    # Reshape params into [5,7]
    p = np.array(params, dtype=float).reshape(5, 7)
    alphas = p[:, 0]            # [5]
    gammas = p[:, 1]            # [5]
    betas  = p[:, 2:]           # [5,5]

    # Avoid log(0)
    eps = 1e-8
    logp = -np.log(proportions + eps)   # [n_samples,5]

    # Weighted sum: for each sample and domain: sum_j β_ij * (-log p_j)
    # logp @ betas.T => [n_samples,5]
    weighted = logp.dot(betas.T)         # [n_samples,5]

    # Enforce non-negative weighted sums
    weighted = np.maximum(weighted, eps)

    # raise to gamma: each column raises to its γ
    losses = alphas + np.power(weighted, gammas)

    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the 35 parameters of the scaling law via non-linear least squares.

    Args:
      proportions: [n_samples,5]
      loss_values: [n_samples,5]

    Returns:
      params: flat array length 35
    """
    proportions = np.atleast_2d(proportions).astype(float)
    loss_values = np.atleast_2d(loss_values).astype(float)
    n_samples, n_dom = proportions.shape
    assert n_dom == 5 and loss_values.shape == (n_samples, 5)

    # Build initial guess: for each domain i
    # α_i = 0.5 * min_observed_loss_i, γ_i = 1.0, β_ij = 1.0
    init = np.zeros(35, dtype=float)
    for i in range(5):
        base = 7*i
        init[base + 0] = 0.5 * np.min(loss_values[:, i])
        init[base + 1] = 1.0
        init[base + 2: base + 7] = 1.0

    # Bounds: α_i >= 0; γ_i in [0.01, 10]; β_ij >= 0
    lower = np.zeros(35, dtype=float)
    upper = np.full(35, np.inf, dtype=float)
    for i in range(5):
        base = 7*i
        lower[base + 0] = 0.0       # α_i
        lower[base + 1] = 0.01      # γ_i
        upper[base + 1] = 10.0
        lower[base + 2: base + 7] = 0.0  # β_ij

    # Residual function: flatten the difference
    def residuals(params):
        pred = scaling_law_func(proportions, params)
        return (pred - loss_values).ravel()

    # Solve with Levenberg-Marquardt or Trust Region Reflective
    res = least_squares(
        residuals,
        init,
        bounds=(lower, upper),
        method='trf',
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=2000
    )

    return res.x

# Attach metadata
scaling_law_func.num_params = 35