import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict Lossu from a 7-parameter quadratic–interaction model in log-space:
      features: 1,
                logP,
                logD,
                logV,
                (logD)^2,
                (logV)^2,
                (logD)*(logV)
      Lossu ≈ params · features

    data_points: array-like, shape (N,3)
                 columns = [P_non_vocab, vocab_size, num_characters]
    params:      array-like, shape (7,)
    Returns:
      preds: ndarray, shape (N,)
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")

    P = X[:, 0]
    V = X[:, 1]
    D = X[:, 2]

    # safe log transform
    eps = 1e-12
    logP = np.log(P + eps)
    logD = np.log(D + eps)
    logV = np.log(V + eps)

    # build design matrix Phi of shape (N,7)
    N = X.shape[0]
    Phi = np.empty((N, 7), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = logP
    Phi[:, 2] = logD
    Phi[:, 3] = logV
    Phi[:, 4] = logD * logD
    Phi[:, 5] = logV * logV
    Phi[:, 6] = logD * logV

    # compute predictions
    p = np.asarray(params, dtype=float).ravel()
    if p.shape[0] != 7:
        raise ValueError("params must have length 7")
    return Phi.dot(p)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7-parameter scaling law by (ridge-regularized) least squares:
      1. Build log-space features [1, logP, logD, logV, (logD)^2, (logV)^2, logD·logV].
      2. Solve (Phi^T Phi + λI) params = Phi^T y for numerical stability.
    Returns:
      params: ndarray, shape (7,)
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")

    y = np.asarray(loss_values, dtype=float).ravel()
    if y.size != X.shape[0]:
        raise ValueError("loss_values length must match data_points")

    P = X[:, 0]
    V = X[:, 1]
    D = X[:, 2]

    # safe log transform
    eps = 1e-12
    logP = np.log(P + eps)
    logD = np.log(D + eps)
    logV = np.log(V + eps)

    # build design matrix Phi
    N = X.shape[0]
    Phi = np.empty((N, 7), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = logP
    Phi[:, 2] = logD
    Phi[:, 3] = logV
    Phi[:, 4] = logD * logD
    Phi[:, 5] = logV * logV
    Phi[:, 6] = logD * logV

    # ridge regularization parameter
    lam = 1e-8
    A = Phi.T.dot(Phi)
    # add small ridge to diagonal for stability
    A[np.diag_indices_from(A)] += lam
    b = Phi.T.dot(y)

    # solve linear system
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END