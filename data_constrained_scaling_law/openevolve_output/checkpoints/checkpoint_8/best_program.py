# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Data-constrained scaling law with token offset:
      L = E + A * (N/N_ref)^(-beta) * ((T+T0)/T_ref)^(-alpha)
               * (1 + C * ((T+T0)/(U/U_ref))^gamma)
    Params (7):
      A      : scale coefficient (>0)
      alpha  : token scaling exponent (>0)
      beta   : model scaling exponent (>0)
      C      : constraint penalty coefficient (>=0)
      gamma  : constraint penalty exponent (>=0)
      E      : irreducible loss floor (>=0)
      T0     : token offset to capture low-T regime (>=0)
    """
    # Unpack and enforce non-negativity/positivity
    A, alpha, beta, C, gamma, E, T0 = params
    A     = max(A,     1e-12)
    alpha = max(alpha, 1e-12)
    beta  = max(beta,  1e-12)
    C     = max(C,     0.0)
    gamma = max(gamma, 0.0)
    E     = max(E,     0.0)
    T0    = max(T0,    0.0)

    # Convert inputs
    T = np.asarray(tokens, dtype=float)
    N = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)

    # Reference scales for normalization (median, clipped)
    T_ref = max(np.median(T), 1.0)
    N_ref = max(np.median(N), 1.0)
    U_ref = max(np.median(U), 1.0)

    # Normalized variables with offset
    Tn = (T + T0) / T_ref
    Nn = N / N_ref
    Un = U / U_ref

    # Core scaling term and data-constraint penalty
    base    = Nn ** (-beta) * Tn ** (-alpha)
    penalty = 1.0 + C * (Tn / Un) ** gamma

    # Predicted loss
    return E + A * base * penalty

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values, initial_params=None):
    """
    Fit the 7-parameter scaling law by minimizing MSE.
    """
    # Vectorize inputs
    T = np.asarray(tokens, dtype=float)
    N = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    L = np.asarray(loss_values, dtype=float)

    # Precompute reference token scale for T0 initialization
    T_ref = max(np.median(T), 1.0)

    # Default initial guess if none provided
    if initial_params is None:
        A0     = 1.0
        alpha0 = 0.3
        beta0  = 0.07
        C0     = 1.0
        gamma0 = 0.5
        E0     = max(np.min(L) * 0.5, 1e-3)
        T00    = 0.1 * T_ref
        initial_params = [A0, alpha0, beta0, C0, gamma0, E0, T00]
    else:
        # pad or truncate to 7 params
        p = list(initial_params)
        if len(p) < 7:
            p += [0.0] * (7 - len(p))
        initial_params = p[:7]

    # Parameter bounds for stability
    bounds = [
        (1e-6, 1e2),         # A
        (1e-6, 2.0),         # alpha
        (1e-6, 2.0),         # beta
        (0.0, 10.0),         # C
        (0.0, 3.0),          # gamma
        (0.0, 10.0),         # E
        (0.0, 10.0 * T_ref)  # T0
    ]

    # Objective: Mean Squared Error
    def objective(p):
        pred = scaling_law_func(T, N, U, p)
        return np.mean((pred - L) ** 2)

    # Run optimization
    res = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

    if res.success:
        return res.x
    else:
        # fallback to initial guess if optimization fails
        return np.array(initial_params, dtype=float)

# Expose number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END