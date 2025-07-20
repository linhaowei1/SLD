# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(proportions, params):
    """
    Domain‐mixture scaling law via linear regression on log/raw features:
      For each domain i:
        p = p_i, q = 1 - p_i
        features = [
          1,
          log(p+eps),
          log(q+eps),
          (log(p+eps))^2,
          (log(q+eps))^2,
          p,
          p * q
        ]
        Li = features ⋅ w_i
    params is flattened array of shape (35,) = 5 domains × 7 features.
    """
    proportions = np.atleast_2d(proportions).astype(float)
    n_samples, n_dom = proportions.shape
    assert n_dom == 5, "Expect proportions with 5 domains"
    eps = 1e-8

    # Prepare params: reshape to (5 domains, 7 features)
    params = np.array(params, dtype=float).flatten()
    if params.size != 35:
        raise ValueError(f"Expected 35 params, got {params.size}")
    W = params.reshape(5, 7)

    # Prepare output
    L = np.zeros((n_samples, 5), dtype=float)

    # Compute per-domain predictions
    for i in range(5):
        p = proportions[:, i]
        q = 1.0 - p

        lp = np.log(p + eps)
        lq = np.log(q + eps)

        # Build design matrix for domain i: shape (n_samples, 7)
        X = np.stack([
            np.ones(n_samples),   # intercept
            lp,
            lq,
            lp * lp,
            lq * lq,
            p,
            p * q
        ], axis=1)

        # Predict Li = X @ w_i
        L[:, i] = X.dot(W[i])

    return L


def fit_scaling_law(proportions, loss_values):
    """
    Fit the linearized scaling law via ridge regression per domain.
    Returns a flattened parameter array of length 35.
    """
    proportions = np.atleast_2d(proportions).astype(float)
    loss_values = np.atleast_2d(loss_values).astype(float)
    n_samples, n_dom = proportions.shape
    assert n_dom == 5 and loss_values.shape == (n_samples, 5), \
        "Expect shapes (N,5) for proportions and loss_values"

    eps = 1e-8
    ridge_lambda = 1e-6 * n_samples  # mild regularization

    # Storage for fitted weights
    W = np.zeros((5, 7), dtype=float)

    # Fit each domain separately
    for i in range(5):
        p = proportions[:, i]
        q = 1.0 - p

        lp = np.log(p + eps)
        lq = np.log(q + eps)

        # Design matrix (N x 7)
        X = np.stack([
            np.ones(n_samples),
            lp,
            lq,
            lp * lp,
            lq * lq,
            p,
            p * q
        ], axis=1)

        y = loss_values[:, i]

        # Closed-form ridge solution: w = (X^T X + λI)^(-1) X^T y
        # Do not regularize intercept (first weight)
        Gram = X.T.dot(X)
        # add ridge to diagonal entries 1..6
        Gram[1:, 1:] += ridge_lambda
        rhs = X.T.dot(y)

        w = np.linalg.solve(Gram, rhs)
        W[i] = w

    return W.flatten()


# Expose expected parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END