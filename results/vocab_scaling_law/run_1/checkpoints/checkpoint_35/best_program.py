import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict Lossu from:
       Lossu ≈ a0 
             + a1·log(P_non_vocab) 
             + a2·log(vocab_size) 
             + a3·log(num_characters)
             + a4·[log(P_non_vocab)]^2
             + a5·[log(vocab_size)]^2
             + a6·[log(num_characters)]^2

    params: array_like of length 7 [a0, a1, a2, a3, a4, a5, a6]
    data_points: (N,3) array: [P_non_vocab, vocab_size, num_characters]
    Returns:
      preds: (N,) array of predicted Lossu
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    # take logs of each positive feature
    # X[:,0]=P_non_vocab, X[:,1]=vocab_size, X[:,2]=num_characters
    L0 = np.log(X[:, 0])
    L1 = np.log(X[:, 1])
    L2 = np.log(X[:, 2])

    # build design matrix of shape (N,7)
    # columns: [1, L0, L1, L2, L0^2, L1^2, L2^2]
    N = X.shape[0]
    Phi = np.empty((N, 7), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = L0
    Phi[:, 2] = L1
    Phi[:, 3] = L2
    Phi[:, 4] = L0 * L0
    Phi[:, 5] = L1 * L1
    Phi[:, 6] = L2 * L2

    # linear combination
    return Phi.dot(np.asarray(params, dtype=float).ravel())


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7-parameter quadratic–in–log model above by ordinary
    least squares. Returns the fitted params array of length 7.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    y = np.asarray(loss_values, dtype=float).ravel()

    # build the same design matrix as in scaling_law_func
    L0 = np.log(X[:, 0])
    L1 = np.log(X[:, 1])
    L2 = np.log(X[:, 2])
    N = X.shape[0]
    Phi = np.empty((N, 7), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = L0
    Phi[:, 2] = L1
    Phi[:, 3] = L2
    Phi[:, 4] = L0 * L0
    Phi[:, 5] = L1 * L1
    Phi[:, 6] = L2 * L2

    # solve least-squares; rcond=None uses default numpy cutoff
    # for numerical stability one could add tiny ridge, e.g. λ=1e-8
    params, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return params
# EVOLVE-BLOCK-END