"""
Scaling law discovery for LLM multi-domain loss prediction
Using a simple, parameter‐efficient linear mixture‐of‐experts model:
each output loss is an affine combination of the five domain proportions.
Total parameters = 5×5 (weights) + 5 (biases) = 30 ≤ 35.
"""
import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict multi-domain losses given domain mixture proportions.

    Args:
        data_points: array of shape (N,5) with domain proportions.
        params:   flat array of length 30:
                  - first 25 entries → weight matrix W of shape (5,5)
                  - last 5 entries    → bias vector b of shape (5,)

    Returns:
        preds: array of shape (N,5), the predicted losses for each domain.
    """
    X = np.atleast_2d(np.asarray(data_points))    # (N,5)
    N, F = X.shape
    assert F == 5, f"Expected 5 input features, got {F}"

    p = np.asarray(params).ravel()
    assert p.size == 30, f"Expected 30 parameters, got {p.size}"

    # unpack parameters
    W = p[:25].reshape(5, 5)    # weight matrix
    b = p[25:].reshape(1, 5)    # bias vector

    # affine prediction
    preds = X.dot(W) + b        # (N,5)
    return preds


def fit_scaling_law(data_points, loss_values):
    """
    Fit the linear mixture-of-experts scaling law via ridge‐regularized least squares.

    Args:
        data_points:  array of shape (N,5) with domain proportions.
        loss_values:  array of shape (N,5) of observed multi-domain losses.

    Returns:
        params: flat array of length 30 (25 weights + 5 biases).
    """
    X = np.atleast_2d(np.asarray(data_points))    # (N,5)
    Y = np.atleast_2d(np.asarray(loss_values))    # (N,5)
    N, F = X.shape
    assert F == 5, f"Expected 5 input features, got {F}"
    assert Y.shape[0] == N and Y.shape[1] == 5, "Expected loss_values shape (N,5)"

    # augment inputs with constant bias term
    X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N,6)

    # ridge regularization for numerical stability
    lam = 1e-6
    A = X_aug.T.dot(X_aug) + lam * np.eye(F + 1)           # (6,6)
    B = X_aug.T.dot(Y)                                     # (6,5)

    # solve for Theta in A @ Theta = B
    Theta = np.linalg.solve(A, B)                         # (6,5)

    # unpack into W and b
    W = Theta[:F, :]    # (5,5)
    b = Theta[F, :]     # (5,)

    # flatten parameters
    params = np.concatenate([W.ravel(), b])
    return params

# EVOLVE-BLOCK-END