# EVOLVE-BLOCK-START
"""
Improved scaling law: mixture-based power law with shared domain mixture weights.

We model each output loss as:
    y_j = a_j + b_j * (sum_i w_i * x_i)^{p_j}

Parameters (20 total):
    u (5):    unnormalized logits for domain-mixture weights w = softmax(u)
    a (5):    output-specific biases
    r (5):    log-scales so that b = exp(r) >= 0
    v (5):    log-exponents so that p = exp(v) >= 0

This parameterization uses 20 parameters (≤ 35), enforces positivity
for mixture weights, scales, and exponents, and shares the same
mixture across all outputs for efficiency and cross-domain generalization.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    X = np.atleast_2d(np.asarray(data_points))  # shape (N, 5)
    prm = np.asarray(params).reshape(-1)
    if prm.size != 20:
        raise ValueError(f"Expected 20 parameters, got {prm.size}")

    # 1) Mixture logits -> softmax weights
    u = prm[0:5]                          # (5,)
    u_max = np.max(u)
    exp_u = np.exp(u - u_max)
    w = exp_u / np.sum(exp_u)             # (5,)

    # 2) Output-specific parameters
    a = prm[5:10]                         # bias terms, (5,)
    r = prm[10:15]                        # log-scale terms
    v = prm[15:20]                        # log-exponent terms

    b = np.exp(r)                         # scale >= 0, (5,)
    p = np.exp(v)                         # exponent >= 0, (5,)

    # 3) Compute the mixture score for each sample
    #    s_n = sum_i w_i * x_{n,i}
    s = X.dot(w)                          # (N,)

    # 4) Predict each output
    #    y_{n,j} = a_j + b_j * s_n^{p_j}
    # Broadcasting to (N, 5)
    Y = a[None, :] + b[None, :] * np.power(s[:, None], p[None, :])
    return Y

def fit_scaling_law(data_points, loss_values):
    X = np.atleast_2d(np.asarray(data_points))  # (N,5)
    y = np.asarray(loss_values)
    if y.ndim == 1:
        y2d = y[:, None]
    else:
        y2d = y
    N, F = X.shape
    if F != 5 or y2d.shape[1] != 5:
        raise ValueError("Expected 5 domain proportions and 5 loss outputs")

    # Parameter vector length = 20
    # Initialize:
    #   u = 0           => uniform mixture weights
    #   a = mean(y)     => bias approx
    #   r = 0           => scale b = 1
    #   v = 0           => exponent p = 1
    mean_y = np.mean(y2d, axis=0)               # (5,)
    init_u = np.zeros(5)
    init_a = mean_y.copy()
    init_r = np.zeros(5)
    init_v = np.zeros(5)
    init_params = np.concatenate([init_u, init_a, init_r, init_v])  # (20,)

    # Objective: mean squared error
    def objective(params):
        pred = scaling_law_func(X, params)
        return np.mean((pred - y2d) ** 2)

    # Optimize with L-BFGS-B
    result = minimize(objective, init_params, method='L-BFGS-B')
    if result.success:
        return result.x
    else:
        # fallback to initial if optimization fails
        return init_params
# EVOLVE-BLOCK-END