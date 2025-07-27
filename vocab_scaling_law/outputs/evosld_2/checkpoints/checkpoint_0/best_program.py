# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Parametric scaling law for Lossu:
      Lossu ≈ a 
             - b * (Non_vocab_parameters)^(-c) 
             - d * (num_characters)^(-e) 
             - f * (vocab_size)^(-g)
    where params = [a, b, c, d, e, f, g] (7 parameters).
    """
    # Avoid zeros / negative arguments in log
    eps = 1e-8
    a, b, c, d, e, f, g = params
    N = Non_vocab_parameters + eps
    M = num_characters      + eps
    V = vocab_size          + eps

    # Use exp(−exponent * log(x)) for numerical stability
    term_N = b * np.exp(-c * np.log(N))
    term_M = d * np.exp(-e * np.log(M))
    term_V = f * np.exp(-g * np.log(V))

    return a - term_N - term_M - term_V

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter scaling law above by minimizing MSE between predicted
    and observed Lossu. Uses L-BFGS-B with bounds to ensure numerical stability.
    Returns the optimized parameter vector [a, b, c, d, e, f, g].
    """
    # Initial parameter guesses
    a0 = np.median(lossu_values)
    loss_span = np.max(lossu_values) - np.min(lossu_values)
    b0 = d0 = f0 = max(loss_span / 3.0, 1e-2)
    c0 = e0 = g0 = 0.5
    initial = np.array([a0, b0, c0, d0, e0, f0, g0], dtype=float)

    # Bounds: a in [-10,10]; b,d,f ≥ 0; c,e,g in [1e-3, 10]
    bounds = [
        (-10, 10),    # a
        (0.0, 10),    # b
        (1e-3, 10),   # c
        (0.0, 10),    # d
        (1e-3, 10),   # e
        (0.0, 10),    # f
        (1e-3, 10),   # g
    ]

    def objective(params):
        pred = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
        return np.mean((pred - lossu_values) ** 2)

    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-9}
    )

    if result.success:
        return result.x
    else:
        # Fallback to initial guess on failure
        return initial

# Declare number of parameters expected by the scaling law
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END