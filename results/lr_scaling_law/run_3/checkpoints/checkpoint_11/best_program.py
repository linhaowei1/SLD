# EVOLVE-BLOCK-START
"""
Enhanced scaling law model for LLM training hyperparameters.
We fit a second‐order polynomial in the log‐domain of each feature,
including quadratic terms and pairwise interactions, via ridge‐regularized
closed‐form regression for stability, efficiency, and improved accuracy.

Model form:
   log(y_pred) = c0
               + sum_i c1_i * log(x_i)
               + sum_i c2_i * (log(x_i))^2
               + sum_{i<j} c3_{ij} * log(x_i)*log(x_j)

where x = [lr, bsz, data_size, non_embedding_param_size],
and y_pred is the predicted LM loss.
"""
import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict language‐model loss from hyperparameters via a 2nd‐order log‐polynomial.

    Args:
      data_points: array‐like of shape (N,4) with columns
                   [lr, bsz, data_size, non_embedding_param_size]
      params:      1D array of length 15:
                   [c0,
                    c1_lr, c1_bsz, c1_data, c1_param,
                    c2_lr, c2_bsz, c2_data, c2_param,
                    c3_lr_bsz, c3_lr_data, c3_lr_param,
                    c3_bsz_data, c3_bsz_param, c3_data_param]

    Returns:
      y_pred:      1D array of length N of predicted losses.
    """
    X = np.asarray(data_points, float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected input with 4 features, got {F}")

    p = np.asarray(params, float).ravel()
    # total parameters = 1 intercept + 4 linear + 4 quadratic + 6 interactions = 15
    P_expected = 1 + F + F + (F*(F-1))//2
    if p.shape[0] != P_expected:
        raise ValueError(f"Expected {P_expected} parameters, got {p.shape[0]}")

    # floor inputs to avoid log(0)
    X_clipped = np.maximum(X, 1e-12)
    logX = np.log(X_clipped)        # shape (N,4)

    # build design matrix Phi
    Phi = np.ones((N, P_expected), float)
    # linear terms
    idx = 1
    Phi[:, idx:idx+F] = logX
    idx += F
    # quadratic terms
    Phi[:, idx:idx+F] = logX**2
    idx += F
    # pairwise interactions
    for i in range(F):
        for j in range(i+1, F):
            Phi[:, idx] = logX[:, i] * logX[:, j]
            idx += 1

    # predict in log‐domain and exponentiate
    log_pred = Phi.dot(p)           # shape (N,)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 2nd‐order log‐polynomial scaling law via ridge‐regularized regression.

    Args:
      data_points: array‐like of shape (N,4)
      loss_values: array‐like of shape (N,)

    Returns:
      params:      1D array of length 15 of fitted coefficients.
    """
    X = np.asarray(data_points, float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    y = np.asarray(loss_values, float).ravel()
    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected data_points with 4 features, got {F}")
    if y.shape[0] != N:
        raise ValueError("Number of data points and loss values must match")

    # floor to avoid log(0)
    X_clipped = np.maximum(X, 1e-12)
    y_clipped = np.maximum(y, 1e-12)

    # log‐transform
    logX = np.log(X_clipped)        # shape (N,4)
    logy = np.log(y_clipped)        # shape (N,)

    # build design matrix Phi
    P = 1 + F + F + (F*(F-1))//2     # 15
    Phi = np.ones((N, P), float)
    idx = 1
    # linear
    Phi[:, idx:idx+F] = logX
    idx += F
    # quadratic
    Phi[:, idx:idx+F] = logX**2
    idx += F
    # pairwise interactions
    for i in range(F):
        for j in range(i+1, F):
            Phi[:, idx] = logX[:, i] * logX[:, j]
            idx += 1

    # ridge‐regularized normal equations
    ridge = 1e-6
    A = Phi.T.dot(Phi)
    # apply ridge only to non‐intercept terms
    A[np.arange(1, P), np.arange(1, P)] += ridge
    b = Phi.T.dot(logy)

    # solve for parameters
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END