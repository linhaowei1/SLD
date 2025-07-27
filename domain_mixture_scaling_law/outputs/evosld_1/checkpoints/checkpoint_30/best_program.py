# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(proportions, params):
    """
    Vectorized 7-parameter per-domain scaling law:
      Li = A_i
           + B_i * p_i^( -C_i )
           + D_i * (1 - p_i)^( E_i )
           + F_i * (p_i * (1 - p_i))
           + G_i * sqrt( sum_{j≠i} p_j^2 )

    Args:
        proportions: array [n_samples, 5], each row sums to 1
        params:      flat array length 35 (5 domains × 7 params)

    Returns:
        losses: array [n_samples, 5]
    """
    P = np.atleast_2d(proportions)
    n_samples, n_domains = P.shape
    assert n_domains == 5, "Expect 5 domain proportions"
    # reshape params into [5 domains × 7 params]
    p = params.reshape(5, 7)
    # safe clipping
    eps = 1e-9
    pi   = np.clip(P,      eps, 1.0)
    comp = np.clip(1.0 - P, eps, 1.0)
    # cross-domain interaction: sum of squares of other proportions
    sq_sum = np.sum(P**2, axis=1, keepdims=True)
    cross  = np.sqrt(np.clip(sq_sum - P**2, 0.0, None))
    # unpack per-domain parameters
    A = p[:, 0]  # baseline
    B = p[:, 1]  # self-power scale
    C = p[:, 2]  # self-power exponent
    D = p[:, 3]  # complement-power scale
    E = p[:, 4]  # complement-power exponent
    F = p[:, 5]  # interaction linear
    G = p[:, 6]  # cross sqrt scale
    # compute Li for all samples and domains via broadcasting
    # shapes: (n_samples,5) each term
    term_self = B[np.newaxis, :] * pi**(-C[np.newaxis, :])
    term_comp = D[np.newaxis, :] * comp**(E[np.newaxis, :])
    term_mix  = F[np.newaxis, :] * (pi * comp)
    term_cross= G[np.newaxis, :] * cross
    losses = A[np.newaxis, :] + term_self + term_comp + term_mix + term_cross
    return losses

# specify expected number of parameters
scaling_law_func.num_params = 35

def fit_scaling_law(proportions, loss_values):
    """
    Fit the 35 parameters via nonlinear least squares.

    Args:
        proportions:  [n_samples,5] domain mixtures
        loss_values:  [n_samples,5] observed losses

    Returns:
        best_params: flat array length 35
    """
    P = np.atleast_2d(proportions)
    L = np.atleast_2d(loss_values)
    n_samples, n_domains = P.shape
    assert n_domains == 5 and L.shape[1] == 5

    # initialize params [5 domains × 7 params]
    init = np.zeros((5, 7))
    for i in range(5):
        yi = L[:, i]
        # baseline A_i ~ median loss
        init[i, 0] = np.median(yi)
        # scales B_i, D_i ~ half the loss range
        scale_est = (np.max(yi) - np.min(yi)) * 0.5
        init[i, 1] = max(scale_est, 1e-3)
        init[i, 3] = init[i, 1]
        # exponents start at 1
        init[i, 2] = 1.0
        init[i, 4] = 1.0
        # small interaction weights
        init[i, 5] = 0.0
        init[i, 6] = 0.0

    x0 = init.ravel()
    # bounds: B,C,D,E ≥ 0 ; A,F,G unbounded
    lower = np.tile([ -np.inf,  0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf ], 5)
    upper = np.tile([  np.inf,  np.inf, np.inf, np.inf, np.inf,  np.inf,  np.inf ], 5)

    def residuals(x):
        pred = scaling_law_func(P, x)
        return (pred - L).ravel()

    # solve with robust least-squares
    res = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=3000,
        verbose=0
    )

    return res.x
# EVOLVE-BLOCK-END