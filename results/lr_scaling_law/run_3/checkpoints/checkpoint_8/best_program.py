# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict language modeling loss from hyperparameters using a power-law
    with an asymptotic loss floor.

    data_points: (N,4) array columns [lr, bsz, data_size, non_embedding_param_size]
    params:      array of shape (6,) or (T,6):
                 [logA, α_P, α_D, α_bsz, α_lr, log_y_inf]

    Returns:
      y_pred: shape (N,) if single param set or (N,T) for multiple sets.
    """
    X = np.asarray(data_points, dtype=float)
    X = np.atleast_2d(X)
    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected 4 features per point, got {F}")
    p = np.asarray(params, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    T, Pcount = p.shape
    if Pcount != 6:
        raise ValueError(f"Expected 6 parameters, got {Pcount}")

    lr    = X[:, 0]
    bsz   = X[:, 1]
    D     = X[:, 2]
    Psize = X[:, 3]

    y_all = np.zeros((N, T), dtype=float)
    for t in range(T):
        logA, aP, aD, aB, aL, log_yinf = p[t]
        A     = np.exp(logA)
        y_inf = np.exp(log_yinf)
        y_all[:, t] = y_inf + A * (Psize**aP) * (D**aD) * (bsz**aB) * (lr**aL)

    return y_all[:, 0] if T == 1 else y_all


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter scaling law by minimizing MSE on observed losses.
    Returns optimized parameter vector [logA, α_P, α_D, α_bsz, α_lr, log_y_inf].
    """
    X = np.asarray(data_points, dtype=float)
    X = np.atleast_2d(X)
    y = np.asarray(loss_values, dtype=float).ravel()

    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected 4 features per point, got {F}")
    if N != y.shape[0]:
        raise ValueError("Number of data points and loss values must match.")

    # Initial guess for asymptotic loss floor y_inf
    min_y = np.min(y)
    y_inf0 = max(min_y * 0.9, 1e-8)

    # Prepare for linearized power-law fit:
    # log(y - y_inf0) ≈ logA + α_P log(P) + α_D log(D) + α_bsz log(bsz) + α_lr log(lr)
    z = y - y_inf0
    if np.any(z <= 0):
        # Shift to ensure positivity
        z = z + (abs(np.min(z)) + 1e-8)
    log_z = np.log(z)

    log_lr   = np.log(X[:, 0])
    log_bsz  = np.log(X[:, 1])
    log_D    = np.log(X[:, 2])
    log_P    = np.log(X[:, 3])

    # Design matrix for linear regression
    M = np.stack([np.ones(N), log_P, log_D, log_bsz, log_lr], axis=1)
    sol, *_ = np.linalg.lstsq(M, log_z, rcond=None)
    logA0, aP0, aD0, aB0, aL0 = sol

    # Pack initial parameter vector
    p0 = np.array([logA0, aP0, aD0, aB0, aL0, np.log(y_inf0)], dtype=float)

    # Objective: MSE in original loss space
    def objective(p):
        y_pred = scaling_law_func(X, p)
        return np.mean((y_pred - y) ** 2)

    # Bounds for stable optimization
    bounds = [
        (None, None),          # logA
        (-5.0, 5.0),           # α_P
        (-5.0, 5.0),           # α_D
        (-5.0, 5.0),           # α_bsz
        (-5.0, 5.0),           # α_lr
        (None, np.log(min_y))  # log_y_inf ≤ log(min observed loss)
    ]

    result = minimize(
        objective,
        p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if result.success:
        return result.x
    else:
        # Fallback to the linearized solution
        return p0
# EVOLVE-BLOCK-END