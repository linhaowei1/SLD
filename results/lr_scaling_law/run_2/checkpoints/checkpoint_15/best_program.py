# EVOLVE-BLOCK-START
"""
Improved scaling‐law discovery for LLM training:
We model log(loss) as a second‐order polynomial in the log of each feature
(including squares and pairwise interactions) and fit via closed‐form ridge
regression for numerical stability and parameter efficiency.
"""
import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict language‐model loss from hyperparameters via a log‐domain polynomial.

    Args:
      data_points: array of shape (N,4) with columns
                   [lr, bsz, data_size, non_embedding_param_size]
      params:      1D array of length P = 1 + F + F*(F+1)/2 where F=4
                   = 1 (intercept)
                     + 4 (main effects)
                     + 10 (4 squares + 6 pairwise products)
    Returns:
      preds: array of shape (N,) of predicted loss values.
    """
    X = np.asarray(data_points, dtype=float)
    # avoid log(0)
    X = np.maximum(X, 1e-12)
    logX = np.log(X)                # shape (N,4)
    N, F = logX.shape

    p = np.asarray(params, dtype=float).ravel()
    # expected number of parameters
    P_expected = 1 + F + (F * (F + 1)) // 2
    if p.size != P_expected:
        raise ValueError(f"Expected {P_expected} parameters, got {p.size}")

    # build design matrix Phi
    Phi = np.ones((N, P_expected), dtype=float)
    # main effects
    Phi[:, 1:1+F] = logX
    # second-order terms: squares and pairwise interactions
    idx = 1 + F
    for i in range(F):
        for j in range(i, F):
            Phi[:, idx] = logX[:, i] * logX[:, j]
            idx += 1

    # linear model in log-domain
    log_pred = Phi.dot(p)
    # back to original scale
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the log‐domain polynomial scaling law via ridge regression.

    Args:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)
    Returns:
      params: 1D array of learned parameters of length P = 15
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # floor inputs and outputs to avoid log(0)
    X = np.maximum(X, 1e-12)
    y = np.maximum(y, 1e-12)

    logX = np.log(X)   # shape (N,4)
    logy = np.log(y)   # shape (N,)

    N, F = logX.shape
    # total parameters: intercept + F main + F*(F+1)/2 second-order
    P = 1 + F + (F * (F + 1)) // 2

    # build design matrix
    Phi = np.ones((N, P), dtype=float)
    Phi[:, 1:1+F] = logX
    idx = 1 + F
    for i in range(F):
        for j in range(i, F):
            Phi[:, idx] = logX[:, i] * logX[:, j]
            idx += 1

    # ridge regularization for stability (no penalty on intercept)
    ridge = 1e-8
    A = Phi.T.dot(Phi)
    # add ridge to diagonal
    A += ridge * np.eye(P)
    # remove ridge penalty from intercept term
    A[0, 0] -= ridge

    b = Phi.T.dot(logy)
    # solve normal equations
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END