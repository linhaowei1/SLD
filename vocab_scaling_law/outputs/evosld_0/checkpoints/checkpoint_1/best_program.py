# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu via a sum of power‐law terms:
      Lossu = A 
            - B * vocab_size^(−b) 
            - C * Non_vocab_parameters^(−c) 
            - D * num_characters^(−d)
    params = [A, B, b, C, c, D, d]  (7 parameters total)
    """
    A, B, b, Cc, c, D, d = params
    # prevent zero or negative inputs
    V = np.maximum(vocab_size, 1.0)
    P = np.maximum(Non_vocab_parameters, 1.0)
    N = np.maximum(num_characters, 1.0)
    return (A 
            - B * V**(-b) 
            - Cc * P**(-c) 
            - D * N**(-d))

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7‐parameter scaling law using multi‐start L-BFGS-B.
    Returns optimized params [A, B, b, C, c, D, d].
    """
    # Parameter bounds to ensure stability and interpretability
    bounds = [
        (-10, 10),  # A: base loss
        (0, 10),    # B: vocab coefficient
        (0, 2),     # b: vocab exponent
        (0, 10),    # C: non-vocab‐param coefficient
        (0, 2),     # c: non-vocab exponent
        (0, 10),    # D: char‐count coefficient
        (0, 2)      # d: char‐count exponent
    ]

    def mse_loss(params):
        pred = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
        return np.mean((pred - lossu_values)**2)

    best_loss = np.inf
    best_params = None
    rng = np.random.RandomState(0)

    # Multi-start to avoid local minima
    for _ in range(8):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(mse_loss, x0, method='L-BFGS-B', bounds=bounds)
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x.copy()

    # Fallback to midpoint of bounds if optimization fails
    if best_params is None:
        best_params = np.array([(lo + hi) / 2 for lo, hi in bounds])

    return best_params

# annotate expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END