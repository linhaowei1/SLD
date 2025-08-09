"""
Scaling law discovery for LLM multi-domain loss prediction
Using a parameter-efficient linear model with self-squared domain features:
  y_j = sum_i W_{i,j} * x_i  +  a_j * (x_j)^2  +  b_j
Total parameters = 5×5 (W) + 5 (a) + 5 (b) = 35 ≤ 35.
"""
import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict multi-domain losses given domain mixture proportions.

    Args:
        data_points: array of shape (N,5) with domain proportions.
        params:      flat array of length 35:
                       - first 25 entries → weight matrix W of shape (5,5)
                       - next 5 entries  → self-squared coefficients a of shape (5,)
                       - last 5 entries  → bias vector b of shape (5,)

    Returns:
        preds: array of shape (N,5), predicted losses for each domain.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N, F = X.shape
    assert F == 5, f"Expected 5 input features, got {F}"
    p = np.asarray(params, dtype=float).ravel()
    assert p.size == 35, f"Expected 35 parameters, got {p.size}"

    # Unpack parameters
    W = p[:25].reshape(F, 5)      # (5,5)
    a = p[25:30].reshape(1, 5)     # (1,5)
    b = p[30:35].reshape(1, 5)     # (1,5)

    # Linear term
    lin = X.dot(W)                 # (N,5)
    # Self-squared nonlinearity
    sq = (X**2) * a                # (N,5)
    # Final prediction
    preds = lin + sq + b           # (N,5)
    return preds

def fit_scaling_law(data_points, loss_values):
    """
    Fit the nonlinear mixture-of-experts scaling law via independent
    ridge-regularized least squares per output domain.

    Args:
        data_points: array of shape (N,5) with domain proportions.
        loss_values: array of shape (N,5) of observed multi-domain losses.

    Returns:
        params: flat array of length 35 (25 weights + 5 self-squared + 5 biases).
    """
    X = np.asarray(data_points, dtype=float)
    Y = np.asarray(loss_values, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    N, F = X.shape
    assert F == 5, f"Expected 5 input features, got {F}"
    assert Y.shape == (N, 5), f"Expected loss_values shape ({N},5), got {Y.shape}"

    lam = 1e-6
    W = np.zeros((F, 5), dtype=float)
    a = np.zeros(5, dtype=float)
    b = np.zeros(5, dtype=float)

    # Fit one output domain at a time to enforce that each squared feature
    # only affects its corresponding output
    for j in range(5):
        yj = Y[:, j]                          # (N,)
        xj_sq = (X[:, j]**2).reshape(N, 1)    # (N,1)
        # Design matrix: [X, x_j^2, bias]
        X_aug = np.hstack([X, xj_sq, np.ones((N, 1))])  # (N,7)

        # Normal equations with ridge regularization
        A = X_aug.T.dot(X_aug) + lam * np.eye(F + 2)     # (7,7)
        B = X_aug.T.dot(yj)                              # (7,)

        theta = np.linalg.solve(A, B)                    # (7,)
        W[:, j] = theta[:F]
        a[j]    = theta[F]
        b[j]    = theta[F + 1]

    # Flatten parameters
    params = np.concatenate([W.ravel(), a, b])
    return params

# EVOLVE-BLOCK-END