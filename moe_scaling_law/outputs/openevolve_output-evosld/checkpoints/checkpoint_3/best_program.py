# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    MoE scaling law: 
      loss = p0 
           + c_e * E^{-a} 
           + c_p * P^{-b} 
           + c_i * (E^{-a} * P^{-b})
    where:
      E = num_experts (clamped ≥1)
      P = total_parameter_count (clamped ≥1)
    params = [p0, a, b, c_e, c_p, c_i]
    """
    # Ensure numpy arrays and avoid zeros
    E = np.maximum(np.asarray(num_experts, dtype=float), 1.0)
    P = np.maximum(np.asarray(total_parameter_count, dtype=float), 1.0)
    p0, a, b, c_e, c_p, c_i = params

    term_e = c_e * np.power(E, -a)
    term_p = c_p * np.power(P, -b)
    term_i = c_i * term_e * term_p

    return p0 + term_e + term_p + term_i

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the MoE scaling law to data using bounded L-BFGS-B.
    Returns optimized parameters [p0, a, b, c_e, c_p, c_i].
    """
    E = np.asarray(num_experts, dtype=float)
    P = np.asarray(total_parameter_count, dtype=float)
    L = np.asarray(loss_values, dtype=float)

    # Initial guess: bias at median loss, small exponents, unit scales
    p0_init = np.median(L)
    initial_params = np.array([p0_init, 0.3, 0.3, 1.0, 1.0, 0.5])

    # Bounds: p0 free, a,b ≥ 0, c_e,c_p,c_i ≥ 0
    bounds = [
        (None, None),  # p0
        (0.0, None),   # a
        (0.0, None),   # b
        (0.0, None),   # c_e
        (0.0, None),   # c_p
        (0.0, None)    # c_i
    ]

    def objective(params):
        pred = scaling_law_func(E, P, params)
        return np.mean((pred - L) ** 2)

    result = minimize(objective, initial_params,
                      method='L-BFGS-B', bounds=bounds)

    return result.x if result.success else initial_params

# Expose the expected number of parameters
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END