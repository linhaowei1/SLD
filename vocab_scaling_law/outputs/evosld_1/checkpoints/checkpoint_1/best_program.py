# EVOLVE-BLOCK-START
"""
Vocab scaling law discovery for LLM training scenarios
Evolved: simplified form, feature‐scaled inputs, robust bounded optimization
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict unigram-normalized loss improvement (Lossu) via a 7-parameter power law:
      Lossu = p0
            + p1 * (N_norm)^(-a1)
            + p2 * (C_norm)^(-a2)
            + p3 * (V_norm)^(-a3)

    where
      N_norm = Non_vocab_parameters / 1e7
      C_norm = num_characters       / 1e9
      V_norm = vocab_size           / 1e4

    params = [p0, p1, a1, p2, a2, p3, a3]
    """
    # unpack parameters
    p0, p1, a1, p2, a2, p3, a3 = params
    
    # normalize inputs for numerical stability
    N_norm = Non_vocab_parameters / 1e7
    C_norm = num_characters     / 1e9
    V_norm = vocab_size         / 1e4

    # compute each term
    term_N = p1 * np.power(N_norm, -a1)
    term_C = p2 * np.power(C_norm, -a2)
    term_V = p3 * np.power(V_norm, -a3)

    return p0 + term_N + term_C + term_V

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter scaling law to data by minimizing MSE.
    Returns optimized [p0, p1, a1, p2, a2, p3, a3].
    """
    # initial guess: p0 ~ mean(loss), p1/p2/p3 ~ 1, exponents ~ 0.5
    p0_init = np.mean(lossu_values)
    initial_params = np.array([p0_init, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5])

    # bounds: p0 free, p1/p2/p3 >= 0; exponents in (1e-8, 3)
    bounds = [
        (None, None),     # p0
        (0.0, None),      # p1
        (1e-8, 3.0),      # a1
        (0.0, None),      # p2
        (1e-8, 3.0),      # a2
        (0.0, None),      # p3
        (1e-8, 3.0),      # a3
    ]

    # objective: mean squared error
    def _mse(params):
        pred = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
        return np.mean((pred - lossu_values) ** 2)

    # run bounded L-BFGS-B optimization
    result = minimize(
        _mse,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-12, 'maxiter':10000}
    )

    if result.success and len(result.x) == 7:
        return result.x
    else:
        # fallback to initial if optimization fails
        return initial_params

# attach metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END