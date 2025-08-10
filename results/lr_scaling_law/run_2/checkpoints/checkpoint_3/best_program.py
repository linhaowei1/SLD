# EVOLVE-BLOCK-START
"""
Improved scaling‐law discovery for LLM training:
We model log(loss) as a low‐order polynomial in the log of each feature,
including pairwise interactions, and fit via closed‐form ridge regression
for stability and efficiency.
"""
import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict language‐model loss from hyperparameters via a log‐domain polynomial.

    data_points: array of shape (N,4) with columns
                 [lr, bsz, data_size, non_embedding_param_size]
    params:      1D array of length P = 1 + 4 + 4*3/2 = 11
                 weights for [intercept,
                              log(lr), log(bsz), log(data_size), log(non_embedding_param_size),
                              all pairwise products of those 4 logs]
    Returns:     array of shape (N,) of predicted loss values.
    """
    X = np.asarray(data_points, dtype=float)
    # floor inputs to avoid log(0)
    X = np.maximum(X, 1e-12)
    # log-transform each feature
    logX = np.log(X)                          # shape (N,4)
    N, F = logX.shape
    # flatten params
    p = np.asarray(params, dtype=float).ravel()
    # expected number of parameters: intercept + F main effects + F*(F-1)/2 interactions
    P_expected = 1 + F + (F*(F-1))//2
    if p.shape[0] != P_expected:
        raise ValueError(f"Expected {P_expected} params but got {p.shape[0]}")
    # build design matrix Phi
    Phi = np.ones((N, P_expected), dtype=float)
    # main effects
    Phi[:, 1:1+F] = logX
    # pairwise interactions
    idx = 1 + F
    for i in range(F):
        for j in range(i+1, F):
            Phi[:, idx] = logX[:, i] * logX[:, j]
            idx += 1
    # linear model in log-domain
    log_pred = Phi.dot(p)
    # back to original domain
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the log-domain polynomial scaling law via ridge regression.

    data_points: array of shape (N,4)
    loss_values: array of shape (N,)
    Returns:      1D array of learned parameters of length 11.
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    # floor inputs and outputs to avoid log(0)
    X = np.maximum(X, 1e-12)
    y = np.maximum(y, 1e-12)
    # log-transform
    logX = np.log(X)          # shape (N,4)
    logy = np.log(y)          # shape (N,)
    N, F = logX.shape
    # number of parameters: intercept + F main + F*(F-1)/2 interactions
    P = 1 + F + (F*(F-1))//2
    # build design matrix Phi
    Phi = np.ones((N, P), dtype=float)
    Phi[:, 1:1+F] = logX
    idx = 1 + F
    for i in range(F):
        for j in range(i+1, F):
            Phi[:, idx] = logX[:, i] * logX[:, j]
            idx += 1
    # ridge regularization (no penalty on intercept)
    ridge = 1e-8
    A = Phi.T.dot(Phi)
    # add ridge to diagonal except intercept term
    A[np.diag_indices(P)] += ridge
    A[0, 0] -= ridge  # remove penalty on bias
    b = Phi.T.dot(logy)
    # solve normal equations
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END