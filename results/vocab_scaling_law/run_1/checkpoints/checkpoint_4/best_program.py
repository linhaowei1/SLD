import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict Lossu based on a 7-parameter scaling law:
      Lossu = L0 + A*P^{-alpha} + B*D^{-beta} + C*V^{-gamma}
    where P = non-vocabulary parameters,
          V = vocabulary size,
          D = number of characters processed.
    The params vector is:
      [L0, logA, logB, logC, log_alpha, log_beta, log_gamma]
    ensuring A, B, C, alpha, beta, gamma > 0 via exponentiation.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    P_non_vocab = X[:, 0]
    V = X[:, 1]
    D = X[:, 2]

    p = np.asarray(params, dtype=float).ravel()
    # Unpack raw parameters
    L0 = p[0]
    A  = np.exp(p[1])
    B  = np.exp(p[2])
    C  = np.exp(p[3])
    alpha = np.exp(p[4])
    beta  = np.exp(p[5])
    gamma = np.exp(p[6])

    # Compute predicted Lossu
    pred = (L0
            + A * P_non_vocab**(-alpha)
            + B * D**(-beta)
            + C * V**(-gamma))
    return pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7-parameter scaling law to data via L-BFGS-B:
      params = [L0, logA, logB, logC, log_alpha, log_beta, log_gamma]
    Returns the optimized params vector.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    y = np.asarray(loss_values, dtype=float).ravel()

    # Initial guess: median baseline, unit offsets for logs, moderate negative exponents
    L0_init = np.median(y)
    init_params = np.array([L0_init,    # L0
                            0.0,        # logA
                            0.0,        # logB
                            0.0,        # logC
                           -1.0,       # log_alpha
                           -1.0,       # log_beta
                           -1.0])      # log_gamma

    # Bounds: allow any L0, free logs for A,B,C, restrict exponents to reasonable range
    bounds = [
        (None, None),     # L0
        (None, None),     # logA
        (None, None),     # logB
        (None, None),     # logC
        (-10.0, 10.0),    # log_alpha
        (-10.0, 10.0),    # log_beta
        (-10.0, 10.0)     # log_gamma
    ]

    def objective(p):
        pred = scaling_law_func(X, p)
        return np.mean((pred - y) ** 2)

    result = minimize(
        objective,
        init_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-8}
    )

    if result.success:
        return result.x
    else:
        # fallback to initial guess if optimization fails
        return init_params