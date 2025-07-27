# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    6-parameter MoE scaling law:
      loss = C
           + A * P^{-alpha}
                 * (1 + B * E^{beta})^{-gamma}

    where:
      E = num_experts (array),
      P = total_parameter_count (array),
      params = [C, A, alpha, B, beta, gamma]
    """
    C, A, alpha, B, beta, gamma = params
    # enforce minimal positive inputs
    E = np.maximum(num_experts, 1.0)
    P = np.maximum(total_parameter_count, 1.0)
    # compute the two decay factors
    term_P = P ** (-alpha)
    term_E = (1.0 + B * (E ** beta)) ** (-gamma)
    return C + A * term_P * term_E

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter scaling law by minimizing MSE via least_squares.
    Returns optimized params = [C, A, alpha, B, beta, gamma].
    """
    # prepare data
    E = np.array(num_experts, dtype=float)
    P = np.array(total_parameter_count, dtype=float)
    L = np.array(loss_values, dtype=float)

    # initial guesses
    C0 = max(np.min(L) * 0.9, 1e-6)
    A0 = max(np.max(L) - np.min(L), 1e-6)
    alpha0 = 0.5
    B0 = 1.0
    beta0 = 0.5
    gamma0 = 1.0
    x0 = np.array([C0, A0, alpha0, B0, beta0, gamma0])

    # bounds to enforce positivity
    lower = [0.0,    1e-12, 1e-6, 1e-12, 1e-6,  1e-6]
    upper = [np.min(L), np.inf, 10.0, np.inf, 10.0, 10.0]

    # residual function
    def residuals(params):
        pred = scaling_law_func(E, P, params)
        return pred - L

    # run least_squares
    res = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        xtol=1e-14,
        ftol=1e-14,
        max_nfev=5000,
        verbose=0
    )

    if res.success:
        return res.x
    else:
        return x0

# expose number of parameters
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END