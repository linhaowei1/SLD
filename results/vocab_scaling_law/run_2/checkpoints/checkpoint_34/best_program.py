import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    7-parameter bilinear–in–log scaling law with small regularization:
      Lossu ≈ c0
            + c1·log(P_non_vocab)
            + c2·log(vocab_size)
            + c3·log(num_characters)
            + c4·[log(vocab_size)]^2
            + c5·[log(num_characters)]^2
            + c6·log(vocab_size)·log(num_characters)

    params: array-like of length 7 [c0, c1, c2, c3, c4, c5, c6]
    data_points: shape (N,3) with columns [P_non_vocab, vocab_size, num_characters]
    Returns predicted Lossu of shape (N,).
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    # unpack
    P = X[:, 0]
    V = X[:, 1]
    D = X[:, 2]
    # logs
    # all inputs strictly positive by design
    lP = np.log(P)
    lV = np.log(V)
    lD = np.log(D)
    # build design matrix Φ with 7 columns:
    # [1, lP, lV, lD, lV^2, lD^2, lV*lD]
    N = X.shape[0]
    Phi = np.empty((N, 7), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = lP
    Phi[:, 2] = lV
    Phi[:, 3] = lD
    Phi[:, 4] = lV * lV
    Phi[:, 5] = lD * lD
    Phi[:, 6] = lV * lD
    # linear predictor
    return Phi.dot(np.asarray(params, dtype=float).ravel())


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7-parameter bilinear–in–log model by ridge-regularized least squares.
    Returns params = [c0, c1, c2, c3, c4, c5, c6].
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    y = np.asarray(loss_values, dtype=float).ravel()
    # logs
    P = X[:, 0]
    V = X[:, 1]
    D = X[:, 2]
    lP = np.log(P)
    lV = np.log(V)
    lD = np.log(D)
    # design matrix Φ
    N = X.shape[0]
    Phi = np.empty((N, 7), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = lP
    Phi[:, 2] = lV
    Phi[:, 3] = lD
    Phi[:, 4] = lV * lV
    Phi[:, 5] = lD * lD
    Phi[:, 6] = lV * lD
    # ridge regularization parameter
    # scale λ with number of samples for stability
    lam = 1e-8 * N
    # normal equations: (ΦᵀΦ + λ I) θ = Φᵀ y
    A = Phi.T.dot(Phi)
    # add regularization to diagonal (do not regularize intercept too heavily)
    A.flat[::8] += lam  # adds lam to each diagonal entry
    b = Phi.T.dot(y)
    # solve for params
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END