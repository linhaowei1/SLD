# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A six‐parameter scaling law for MoE models capturing:
      - an asymptotic loss floor (p0),
      - pure expert scaling (p1, p2),
      - pure parameter scaling (p3, p4),
      - and their interaction (p5).
    Args:
      num_experts: array‐like of expert counts (E)
      total_parameter_count: array‐like of total dense parameters (N)
      params: [p0, p1, p2, p3, p4, p5]
    Returns:
      Predicted loss array.
    """
    p0, p1, p2, p3, p4, p5 = params
    # avoid zeros / negative
    E = np.maximum(num_experts, 1e-6)
    N = np.maximum(total_parameter_count, 1e-6)
    # power‐law terms
    term_e  = p1 * E ** (-p2)
    term_n  = p3 * N ** (-p4)
    term_en = p5 * (E ** (-p2) * N ** (-p4))
    return p0 + term_e + term_n + term_en

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the six‐parameter MoE scaling law to (E, N) → loss data by minimizing MSE.
    Returns optimized params [p0, p1, p2, p3, p4, p5].
    """
    num_experts = np.asarray(num_experts, dtype=float)
    total_parameter_count = np.asarray(total_parameter_count, dtype=float)
    loss_values = np.asarray(loss_values, dtype=float)

    # Initial guess: floor ~ 0.9 * min loss, other coefs/exponents ~ O(1)
    p0_init = max(np.min(loss_values) * 0.9, 1e-3)
    x0 = np.array([p0_init, 1.0, 0.5, 1.0, 0.5, 0.1])

    # Constraints: all params ≥ 0 (to keep exponents and coefs non-negative)
    bounds = [(0, None)] * 6

    def objective(x):
        pred = scaling_law_func(num_experts, total_parameter_count, x)
        return np.mean((pred - loss_values) ** 2)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 10000}
    )

    if result.success:
        return result.x
    else:
        # fallback to initial guess if optimization fails
        return x0

# metadata: number of parameters used by scaling_law_func
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END