# EVOLVE-BLOCK-START
"""
Improved vocab-based scaling law for LLM Lossu modeling.

We fit a 7-parameter law of the form:
    Lossu ≈ a 
           + b * (vocab_size_norm)^(-alpha) 
           + c * (non_vocab_params_norm)^(-beta) 
           + d * (num_chars_norm)^(-gamma)

Inputs are normalized to mitigate numerical issues and
we use L-BFGS-B with sensible bounds for robust fitting.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu from model size, vocab size, and training scale.

    Args:
        Non_vocab_parameters: array-like, model parameters excluding vocab embeddings
        vocab_size:          array-like, vocabulary sizes
        num_characters:      array-like, number of training characters
        params:              length-7 array [a, b, alpha, c, beta, d, gamma]

    Returns:
        np.ndarray of predicted Lossu
    """
    # Unpack parameters
    a, b, alpha, c, beta, d, gamma = params

    # Normalize inputs to O(1) range
    V = np.asarray(vocab_size, dtype=np.float64) / 1e4
    N = np.asarray(Non_vocab_parameters, dtype=np.float64) / 1e8
    C = np.asarray(num_characters, dtype=np.float64) / 1e10

    # Compute scaling-law prediction
    # We add small epsilon inside power if needed to avoid zero-division
    eps = 1e-12
    return (
        a
        + b * np.power(V + eps, -alpha)
        + c * np.power(N + eps, -beta)
        + d * np.power(C + eps, -gamma)
    )


def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter scaling law to observed Lossu.

    Args:
        Non_vocab_parameters: array-like, non-vocab model parameters
        vocab_size:          array-like, vocabulary sizes
        num_characters:      array-like, number of training characters
        lossu_values:        array-like, observed Lossu

    Returns:
        params: 1D np.ndarray of length 7
    """
    # Cast to numpy arrays
    N_arr = np.asarray(Non_vocab_parameters, dtype=np.float64)
    V_arr = np.asarray(vocab_size, dtype=np.float64)
    C_arr = np.asarray(num_characters, dtype=np.float64)
    y_arr = np.asarray(lossu_values, dtype=np.float64)

    # Initial guess: a=median(Lossu), b/c/d ~1, exponents ~0.5
    a0     = np.median(y_arr)
    b0, c0, d0 = 1.0, 0.5, 0.5
    alpha0, beta0, gamma0 = 0.5, 0.3, 0.2
    x0 = np.array([a0, b0, alpha0, c0, beta0, d0, gamma0], dtype=np.float64)

    # Bounds: a,b,c,d in [-20,20]; exponents in [0,5]
    bounds = [
        (-20.0, 20.0),  # a
        (-20.0, 20.0),  # b
        (0.0, 5.0),     # alpha
        (-20.0, 20.0),  # c
        (0.0, 5.0),     # beta
        (-20.0, 20.0),  # d
        (0.0, 5.0),     # gamma
    ]

    # Objective: mean squared error
    def objective(params):
        preds = scaling_law_func(N_arr, V_arr, C_arr, params)
        return np.mean((preds - y_arr) ** 2)

    # Run L-BFGS-B optimization
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    # Return fitted params or fallback to initial guess
    if result.success and result.x.shape[0] == 7:
        return result.x
    else:
        return x0

# Attach metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END