# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    6-parameter multiplicative MoE scaling law:
      L = a
        + b * (P_norm)^(-c)
              * (1 + g * E^d)^(-e)

    where:
      E       = num_experts (array)
      P_norm  = total_parameter_count / 1e9 (array, clipped >=1e-6)
      params  = [a, b, c, d, e, g]
        a: base loss floor
        b: amplitude
        c: P exponent
        d: E exponent inside interaction
        e: overall interaction exponent
        g: scale for expert term
    """
    a, b, c, d, e, g = params
    E = np.maximum(num_experts.astype(np.float64), 0.0)
    P = np.maximum(total_parameter_count.astype(np.float64) / 1e9, 1e-6)
    return a + b * (P ** (-c)) * (1.0 + g * (E ** d)) ** (-e)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter scaling law via robust least-squares with multiple restarts.
    Returns optimized [a, b, c, d, e, g].
    """
    E = np.array(num_experts, dtype=np.float64)
    P = np.array(total_parameter_count, dtype=np.float64)
    L = np.array(loss_values, dtype=np.float64)

    best_cost = np.inf
    best_params = None

    # Prepare bounds for each parameter
    lower = [0.0,    0.0,    1e-6,   1e-6,   1e-6,   0.0  ]
    upper = [np.inf, np.inf, 10.0,   5.0,    10.0,   1e3  ]

    # Grid of initial exponents to avoid local minima
    c_guesses = [0.2, 0.5, 0.8]
    d_guesses = [0.2, 0.5, 1.0]
    e_guesses = [0.2, 0.5, 1.0]

    # Basic scale estimates
    Lmin, Lmax = L.min(), L.max()
    a0_base = max(Lmin * 0.9, 1e-3)
    b0_base = max((Lmax - Lmin) * 0.5, 1e-3)
    g0      = 0.1

    for c0 in c_guesses:
        for d0 in d_guesses:
            for e0 in e_guesses:
                init = np.array([a0_base, b0_base, c0, d0, e0, g0], dtype=np.float64)
                res = least_squares(
                    lambda p: scaling_law_func(E, P, p) - L,
                    x0=init,
                    bounds=(lower, upper),
                    loss='soft_l1',
                    max_nfev=2000
                )
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_params = res.x

    # If optimization failed for all inits, fallback to a simple heuristic
    if best_params is None:
        best_params = np.array([a0_base, b0_base, 0.3, 0.5, 0.3, g0], dtype=np.float64)

    return best_params

# annotate number of learnable parameters
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END