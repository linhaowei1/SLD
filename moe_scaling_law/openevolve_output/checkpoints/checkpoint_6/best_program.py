# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A six-parameter MoE scaling law:
      loss ≈ C + A / [ (num_experts + E)**alpha * ((total_parameter_count/1e9) + F)**beta ]
    params = [A, alpha, beta, E, F, C]
    """
    A, alpha, beta, E, F, C = params
    ne = np.asarray(num_experts, dtype=float)
    Pg = np.asarray(total_parameter_count, dtype=float) / 1e9
    # small epsilon to prevent divide-by-zero
    denom = np.power(ne + E, alpha) * np.power(Pg + F, beta) + 1e-12
    return C + A / denom

# Inform how many params the model expects
scaling_law_func.num_params = 6

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to observed data.
    Uses bounded nonlinear least-squares for stability.
    """
    ne = np.asarray(num_experts, dtype=float)
    P  = np.asarray(total_parameter_count, dtype=float)
    y  = np.asarray(loss_values, dtype=float)

    # Basic statistics for initial guesses
    y_min, y_max = y.min(), y.max()
    med_ne, med_Pg = np.median(ne), np.median(P) / 1e9

    # Initial parameter guesses
    C0     = max(0.0, 0.9 * y_min)
    alpha0 = 0.5
    beta0  = 0.5
    E0     = max(1.0, med_ne * 0.1)
    F0     = max(1.0, med_Pg * 0.1)
    A0     = max(1e-3, (y_max - C0) * (med_ne + E0)**alpha0 * (med_Pg + F0)**beta0)

    x0 = np.array([A0, alpha0, beta0, E0, F0, C0], dtype=float)

    # Bounds for physical plausibility
    lower = [0, 0, 0, 0, 0, 0]
    upper = [np.inf, 5.0, 5.0, np.inf, np.inf, np.inf]

    # Residuals for least-squares
    def residuals(p):
        return scaling_law_func(ne, P, p) - y

    res = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=5000,
        verbose=0
    )

    return res.x if res.success else x0
# EVOLVE-BLOCK-END