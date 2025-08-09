"""
Scaling law discovery for LLM multi-domain loss prediction
Using a parameter‐efficient nonlinear feature‐exponent model:
each input proportion is raised to a learned exponent, then
linearly combined into each output loss.

Total parameters = 5 exponents + 25 linear weights + 5 biases = 35 ≤ 35.
"""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict multi-domain losses given domain mixture proportions.

    Args:
        data_points: array of shape (N,5) with domain proportions.
        params:      flat array of length 35:
                     - entries 0–4   → exponent‐log vector u (for positive exponents p=exp(u))
                     - entries 5–29  → weight matrix W of shape (5,5), flattened row‐major
                     - entries 30–34 → bias vector b of shape (5,)

    Returns:
        preds: array of shape (N,5), the predicted losses for each domain.
    """
    X = np.atleast_2d(np.asarray(data_points))   # (N,5)
    N, F = X.shape
    assert F == 5, f"Expected 5 input features, got {F}"

    p = np.asarray(params).ravel()
    assert p.size == 35, f"Expected 35 parameters, got {p.size}"

    # unpack parameters
    u = p[0:5]                         # (5,) logarithms of exponents
    W = p[5:30].reshape(5, 5)          # (5,5) weight matrix W[j,i]
    b = p[30:35].reshape(1, 5)         # (1,5) bias vector

    # positive exponents
    exponents = np.exp(u)              # p_i ≥ 0, (5,)

    # nonlinear transform of inputs
    # x_{n,i}^{p_i}
    X_p = np.power(X, exponents)       # (N,5)

    # affine combination
    preds = X_p.dot(W.T) + b           # (N,5)
    return preds


def fit_scaling_law(data_points, loss_values):
    """
    Fit the nonlinear exponent + linear weights model by minimizing MSE.

    Args:
        data_points: array of shape (N,5) with domain proportions.
        loss_values: array of shape (N,5) of observed multi-domain losses.

    Returns:
        params: flat array of length 35 (5 exponents + 25 weights + 5 biases).
    """
    X = np.atleast_2d(np.asarray(data_points))   # (N,5)
    Y = np.atleast_2d(np.asarray(loss_values))   # (N,5)
    N, F = X.shape
    assert F == 5, f"Expected 5 input features, got {F}"
    assert Y.shape == (N, 5), f"Expected loss_values shape {(N,5)}, got {Y.shape}"

    # Initialization:
    # - log‐exponents u_i = 0 → exponents p_i = 1 (identity)
    # - weights W = 0
    # - biases b = mean of each output
    mean_Y = np.mean(Y, axis=0)                  # (5,)
    init_u = np.zeros(5)                         # (5,)
    init_W = np.zeros(25)                        # flattened (5×5)
    init_b = mean_Y.copy()                       # (5,)
    init_params = np.concatenate([init_u, init_W, init_b])  # (35,)

    # Bound the log‐exponents to keep p_i in a reasonable range
    # here u_i ∈ [−3, 3] ⇒ p_i ∈ [exp(−3), exp(3)] ≈ [0.05, 20]
    bounds = [(-3.0, 3.0)] * 5 + [(None, None)] * 25 + [(None, None)] * 5

    def objective(p_flat):
        Y_pred = scaling_law_func(X, p_flat)
        # mean squared error
        return np.mean((Y_pred - Y) ** 2)

    result = minimize(
        objective,
        init_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9}
    )

    if result.success:
        return result.x
    else:
        # fallback to initial params if optimization fails
        return init_params

# EVOLVE-BLOCK-END