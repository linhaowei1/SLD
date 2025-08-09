import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict Lossu using a 7-parameter quadratic model in log-space:
      let xp = log(P_non_vocab),
          xv = log(vocab_size),
          xd = log(num_characters)
      then Lossu = b0 + b1*xp + b2*xv + b3*xd + b4*xp^2 + b5*xv^2 + b6*xd^2
    This compact form captures curvature while remaining linear in parameters.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    # Compute log‐features
    xp = np.log(X[:, 0])
    xv = np.log(X[:, 1])
    xd = np.log(X[:, 2])
    p = np.asarray(params, dtype=float).ravel()
    # Evaluate the quadratic model
    return (
        p[0]
        + p[1] * xp
        + p[2] * xv
        + p[3] * xd
        + p[4] * xp**2
        + p[5] * xv**2
        + p[6] * xd**2
    )

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7-parameter quadratic-in-log scaling law by ridge‐regularized
    least squares. Returns a length‐7 parameter vector.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    y = np.asarray(loss_values, dtype=float).ravel()
    # Build log‐features
    xp = np.log(X[:, 0])
    xv = np.log(X[:, 1])
    xd = np.log(X[:, 2])
    # Design matrix: [1, xp, xv, xd, xp^2, xv^2, xd^2]
    Phi = np.column_stack((np.ones_like(xp), xp, xv, xd, xp**2, xv**2, xd**2))
    # Solve (Phi^T Phi + λI) p = Phi^T y for stability
    lam = 1e-6
    P = Phi.shape[1]
    A = Phi.T.dot(Phi) + lam * np.eye(P)
    b = Phi.T.dot(y)
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END