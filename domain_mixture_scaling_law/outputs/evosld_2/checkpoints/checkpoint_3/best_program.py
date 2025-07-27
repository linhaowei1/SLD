# EVOLVE-BLOCK-START
"""
Refined domain-mixture scaling law for LLM training:
Each domain's loss is modeled as
   L_i = a_i + sum_{j=1..5} b_{i,j} * (p_j)^{c_i}
where:
  - a_i    : bias term for domain i
  - b_{i,j}: weight of proportion p_j on domain i
  - c_i    : shared exponent for all p_j in domain i

Total parameters per domain = 1 (a_i) + 5 (b_{i,j}) + 1 (c_i) = 7
Total parameters overall = 5 domains × 7 = 35
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses given mixture proportions and parameters.
    Args:
      proportions: array-like, shape [n_samples, 5], each row sums to 1
      params     : array-like, shape [35,] (5 domains × 7 params)
    Returns:
      losses_pred: np.ndarray, shape [n_samples, 5]
    """
    P = np.atleast_2d(proportions).astype(float)
    n, d = P.shape
    assert d == 5, "Expect 5 domain proportions"
    # ensure no zeros for power
    P = np.clip(P, 1e-8, 1.0)

    # reshape params → (5 domains, 7 params)
    M = np.array(params, dtype=float).reshape(5, 7)
    a   = M[:, 0]        # shape [5,]
    b   = M[:, 1:6]      # shape [5,5]
    c   = M[:, 6]        # shape [5,]

    # compute p_j^{c_i} for each domain i and sample k
    # P_pow shape: [n_samples, 5 domains, 5 proportions]
    # but we can vectorize: for each domain i: P**c[i]
    losses = np.zeros((n, 5))
    for i in range(5):
        Pci = P ** c[i]             # [n,5]
        losses[:, i] = a[i] + Pci.dot(b[i])
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the 35 scaling-law parameters to data via nonlinear least squares.
    Args:
      proportions: [n_samples,5] array of domain mix proportions
      loss_values: [n_samples,5] array of observed losses
    Returns:
      params_opt : array of length 35
    """
    P = np.asarray(proportions, dtype=float)
    Y = np.asarray(loss_values, dtype=float)
    n, d = P.shape
    assert d == 5 and Y.shape == (n, 5)

    # avoid zeros
    P = np.clip(P, 1e-8, 1.0)

    # Initial guess: perform linear regression per domain for y_i ~ [1, P]
    # to get a_i and b_{i,j} (assuming c_i=1)
    init = np.zeros((5, 7))
    X_lin = np.concatenate([np.ones((n,1)), P], axis=1)  # [n,6]
    for i in range(5):
        # solve [n × 6] β ≈ Y[:,i]
        β, *_ = np.linalg.lstsq(X_lin, Y[:, i], rcond=None)
        init[i, 0] = β[0]        # a_i
        init[i, 1:6] = β[1:]     # b_{i,j}
        init[i, 6] = 1.0         # c_i
    x0 = init.ravel()

    # bounds: keep exponents in [0.1, 5], others free
    lb = -np.inf * np.ones_like(x0)
    ub =  np.inf * np.ones_like(x0)
    # indices of c_i are 6, 13, 20, 27, 34
    for idx in (6, 13, 20, 27, 34):
        lb[idx] = 0.1
        ub[idx] = 5.0

    def residuals(x):
        pred = scaling_law_func(P, x)
        return (pred - Y).ravel()

    # solve nonlinear least squares
    sol = least_squares(
        residuals, x0,
        bounds=(lb, ub),
        method='trf',
        ftol=1e-9,
        xtol=1e-9,
        verbose=0
    )
    return sol.x

# annotate expected parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END