# EVOLVE-BLOCK-START
"""
Improved Domain Mixture Scaling Law for LLM Training Scenarios

We model each domain's loss Li as a function of its own proportion pi and the
complement (1 - pi), capturing both direct and mixture effects:

    Li = A_i + B_i * (pi + ε)^{C_i} + D_i * (1 - pi + ε)^{E_i}

Each domain has 5 parameters: [A_i, B_i, C_i, D_i, E_i], for a total of 25.
This form is flexible yet compact, numerically stable, and easy to fit.
"""

import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses given mixture proportions and parameters.

    Args:
        proportions: array-like, shape (n_samples, 5)
            Domain mixture proportions, each row sums to 1.
        params: array-like, shape (25,)
            Parameters stacked as [A_1..A_5, B_1..B_5, C_1..C_5, D_1..D_5, E_1..E_5].

    Returns:
        losses: ndarray, shape (n_samples, 5)
            Predicted loss for each domain and sample.
    """
    proportions = np.atleast_2d(proportions).astype(float)
    n, d = proportions.shape
    assert d == 5, "Expected 5 domain proportions"

    # Unpack parameters: each is length-5
    p = proportions
    eps = 1e-8
    # Clip to avoid zero/negatives in power
    p_safe = np.clip(p, eps, 1.0)
    one_minus_p = np.clip(1.0 - p, eps, 1.0)
    
    # Reshape params: (5 domains, 5 params each)
    params = np.asarray(params).flatten()
    assert params.size == 25, "Expected 25 parameters"
    P = params.reshape(5, 5)
    A, B, C, D, E = P.T  # each is length-5

    # Broadcast to (n_samples, 5)
    A_row = A[None, :]
    B_row = B[None, :]
    C_row = C[None, :]
    D_row = D[None, :]
    E_row = E[None, :]

    # Compute Li = A_i + B_i * p_i^{C_i} + D_i * (1 - p_i)^{E_i}
    term1 = B_row * np.power(p_safe, C_row)
    term2 = D_row * np.power(one_minus_p, E_row)
    losses = A_row + term1 + term2
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law parameters to observed losses.

    Args:
        proportions: ndarray, shape (n_samples, 5)
        loss_values: ndarray, shape (n_samples, 5)

    Returns:
        best_params: ndarray, shape (25,)
            Optimized parameters for the scaling law.
    """
    proportions = np.atleast_2d(proportions).astype(float)
    loss_values = np.atleast_2d(loss_values).astype(float)
    n, d = proportions.shape
    assert d == 5 and loss_values.shape == (n, 5)

    # Initial guess: A = mean(L), B=D=1.0, C=E=0.5
    A0 = np.mean(loss_values, axis=0)
    B0 = np.ones(5)
    C0 = np.full(5, 0.5)
    D0 = np.ones(5)
    E0 = np.full(5, 0.5)
    x0 = np.concatenate([A0, B0, C0, D0, E0])

    # Bounds to keep parameters in a reasonable range
    # A_i in [0, 10*mean_loss_i], B,D in [0, 10], C,E in [0, 5]
    bounds = []
    for i in range(5):
        bounds.append((0.0, 10.0 * A0[i]))   # A_i
    for _ in range(5):
        bounds.append((0.0, 10.0))           # B_i
    for _ in range(5):
        bounds.append((0.0, 5.0))            # C_i
    for _ in range(5):
        bounds.append((0.0, 10.0))           # D_i
    for _ in range(5):
        bounds.append((0.0, 5.0))            # E_i

    # Objective: mean squared error
    def objective(x):
        pred = scaling_law_func(proportions, x)
        return np.mean((pred - loss_values) ** 2)

    # Multi-start to avoid local minima
    best_x = x0.copy()
    best_loss = objective(x0)
    for seed in [0, 1, 2]:
        rng = np.random.RandomState(seed)
        x_init = x0 + rng.randn(25) * 0.1
        res = minimize(
            objective,
            x_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_x = res.x

    return best_x

# Number of parameters expected by the scaling law
scaling_law_func.num_params = 25
# EVOLVE-BLOCK-END