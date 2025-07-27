import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Data‐constrained scaling law using separate power‐law terms for
    model size and effective data, with a smooth saturation of data coverage.
    
    Loss = L_inf
         + C_M * M^{-alpha}
         + C_D * Teff^{-beta}

    where Teff = U * (1 - exp(- (T/U/gamma)^{theta} )).

    params (length 7):
      params[0] = L_inf      (irreducible loss floor)
      params[1] = log_Cm     (log amplitude for model‐size term)
      params[2] = alpha      (>0) exponent on model size
      params[3] = log_Cd     (log amplitude for data term)
      params[4] = beta       (>0) exponent on Teff
      params[5] = gamma      (>0) saturation scale in units of T/U
      params[6] = theta      (>0) sharpness of saturation
    """
    L_inf, log_Cm, alpha, log_Cd, beta, gamma, theta = params
    # Recover amplitudes
    C_M = np.exp(log_Cm)
    C_D = np.exp(log_Cd)

    # Ensure arrays and avoid division by zero
    T = np.maximum(tokens, 0.0)
    P = np.maximum(model_size, 1.0)
    U = np.maximum(unique_tokens, 1.0)
    eps = 1e-12
    gamma = np.maximum(gamma, eps)
    theta = np.maximum(theta, eps)

    # ratio of tokens to unique tokens
    r = T / (U + eps)
    # smooth saturation
    Teff = U * (1.0 - np.exp(- (r / gamma)**theta))

    # compute loss
    return L_inf + C_M * P**(-alpha) + C_D * (Teff + eps)**(-beta)


def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7‐parameter scaling law by minimizing mean squared error.
    """
    # Convert inputs to numpy arrays
    T = np.asarray(tokens, dtype=float)
    P = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    Y = np.asarray(loss_values, dtype=float)

    # Basic stats for init
    y_min, y_max = Y.min(), Y.max()
    span = max(1e-3, y_max - y_min)

    # Initialize parameters
    L0      = max(0.0, y_min * 0.9)
    C_M0    = span * 0.5
    C_D0    = span * 0.5
    alpha0  = 0.5
    beta0   = 0.5
    gamma0  = 1.0
    theta0  = 1.0

    init = np.array([
        L0,
        np.log(C_M0 + 1e-8),
        alpha0,
        np.log(C_D0 + 1e-8),
        beta0,
        gamma0,
        theta0
    ], dtype=float)

    # Bounds to enforce positivity and reasonable ranges
    bounds = [
        (0.0,         y_max),    # L_inf
        (-20.0,       20.0),     # log_Cm
        (1e-6,        5.0),      # alpha
        (-20.0,       20.0),     # log_Cd
        (1e-6,        5.0),      # beta
        (1e-6,        100.0),    # gamma
        (1e-6,        10.0)      # theta
    ]

    def mse_obj(p):
        pred = scaling_law_func(T, P, U, p)
        return np.mean((pred - Y)**2)

    res = minimize(
        mse_obj,
        init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-12, 'maxiter':2000}
    )

    # Return best parameters or init if fitting failed
    return res.x if res.success else init

# annotate number of parameters
scaling_law_func.num_params = 7