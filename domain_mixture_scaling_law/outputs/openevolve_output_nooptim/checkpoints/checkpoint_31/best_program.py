import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    Domain‐mixture exponential scaling law with domain‐specific sensitivities:
        L_i(r) = c_i + k_i * exp( sum_j t_{i,j} * r_j )
    where for each of the 5 domains i:
      - c_i      : domain bias (5 params)
      - k_i      : domain scale (5 params, enforced positive via exp)
      - t_{i,j}  : sensitivity of domain i to mixture proportion j (5×5 = 25 params)
    Total parameters = 5 (c) + 5 (k_raw) + 25 (t) = 35.

    Args:
      proportions: array, shape [n_samples, 5], rows sum to 1.0
      params:      array-like, length 35

    Returns:
      loss_predictions: array, shape [n_samples, 5]
    """
    proportions = np.atleast_2d(proportions)
    n_samples, n_domains = proportions.shape
    assert n_domains == 5, "Expect 5 domain proportions per sample"

    # Prepare parameter vector of length 35
    p = np.array(params, dtype=float).ravel()
    if p.size < 35:
        p = np.concatenate([p, np.ones(35 - p.size)])
    else:
        p = p[:35]

    # Extract c_i (5), raw scale k_raw_i (5), and sensitivity matrix t_{i,j} (25)
    c     = p[0:5]                    # biases
    k_raw = p[5:10]                   # raw scales
    t_mat = p[10:35].reshape(5, 5)    # sensitivities

    # Enforce positivity of scales
    k = np.exp(k_raw)                 # k_i > 0

    # Compute exponent arguments: for each sample n and domain i:
    #   exponent_arg[n,i] = sum_j proportions[n,j] * t_mat[i,j]
    # Vectorized: proportions [n,5] dot t_mat.T [5,5] -> [n,5]
    exponent_arg = proportions.dot(t_mat.T)

    # Compute the exponential term
    exp_term = np.exp(exponent_arg)   # shape [n_samples, 5]

    # Compute domain losses
    # L[n,i] = c_i + k_i * exp_term[n,i]
    loss_predictions = exp_term * k[None, :] + c[None, :]

    return loss_predictions

def fit_scaling_law(proportions, loss_values):
    """
    Fit the 35‐parameter scaling law via BFGS, initializing all params to 1.0.

    Args:
      proportions: array, shape [n_samples, 5]
      loss_values: array, shape [n_samples, 5]

    Returns:
      optimized_params: array, length 35
    """
    def objective(params):
        pred = scaling_law_func(proportions, params)
        return np.mean((pred - loss_values) ** 2)

    # Initialize all 35 parameters to 1.0
    initial_params = np.ones(35)

    result = minimize(objective, initial_params, method='BFGS')
    return result.x

# Inform downstream code the number of expected parameters
scaling_law_func.num_params = 35