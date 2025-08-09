import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict Lossu using a 7-parameter multi-resource law:
      Lossu = L0 + A / (P^alpha + B·V^beta + C·D^gamma + ε)
    where
      P = non-vocabulary parameters,
      V = vocabulary size,
      D = number of characters processed.
    params = [L0,
              logA, log_alpha,
              logB, log_beta,
              logC, log_gamma]
    Enforce A, alpha, B, beta, C, gamma > 0 by exponentiating their logs.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    P = X[:, 0]
    V = X[:, 1]
    D = X[:, 2]

    p = np.asarray(params, dtype=float).ravel()
    L0         = p[0]
    A          = np.exp(p[1])
    alpha      = np.exp(p[2])
    B          = np.exp(p[3])
    beta       = np.exp(p[4])
    C          = np.exp(p[5])
    gamma      = np.exp(p[6])

    # denominator with small ε for numerical safety
    denom = (P**alpha) + B * (V**beta) + C * (D**gamma) + 1e-12
    pred  = L0 + A / denom
    return pred

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7-parameter scaling law via L-BFGS-B:
      params = [L0,
                logA, log_alpha,
                logB, log_beta,
                logC, log_gamma]
    Returns the fitted params array of length 7.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    y = np.asarray(loss_values, dtype=float).ravel()

    # Initial guesses
    L0_init        = np.median(y)
    logA_init      = 0.0
    log_alpha_init = np.log(1e-6)
    logB_init      = 0.0
    log_beta_init  = np.log(1.0)
    logC_init      = 0.0
    log_gamma_init = np.log(0.5)

    init_params = np.array([
        L0_init,
        logA_init,      log_alpha_init,
        logB_init,      log_beta_init,
        logC_init,      log_gamma_init
    ], dtype=float)

    # Bounds to keep exponents in a reasonable range
    bounds = [
        (None, None),  # L0 unconstrained
        (-20, 20),     # logA
        (-10, 10),     # log_alpha
        (-20, 20),     # logB
        (-10, 10),     # log_beta
        (-20, 20),     # logC
        (-10, 10)      # log_gamma
    ]

    def objective(p):
        pred = scaling_law_func(X, p)
        # mean squared error
        return np.mean((pred - y)**2)

    result = minimize(
        objective,
        init_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-8}
    )

    if result.success:
        return result.x
    # fallback to initial guess on failure
    return init_params
# EVOLVE-BLOCK-END