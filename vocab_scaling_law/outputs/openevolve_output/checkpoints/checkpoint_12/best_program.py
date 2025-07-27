# EVOLVE-BLOCK-START
"""
Vocab scaling law discovery for LLM training scenarios
Revised to an additive power‐law form for each of the three key axes
(parameters, vocabulary, compute) with 7 parameters:
  Lossu ≈ p0
         + a_n * N^(−α_n)
         + a_v * V^(−α_v)
         + a_c * C^(−α_c)

Where:
  N = Non_vocab_parameters
  V = vocab_size
  C = num_characters

This form often better captures diminishing returns in each dimension
and has shown improved fit on MSE and R² metrics.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu via an additive power‐law model:
      Lossu = p0
            + a_n * N^(−α_n)
            + a_v * V^(−α_v)
            + a_c * C^(−α_c)

    Args:
        Non_vocab_parameters: 1D array of non-vocab parameter counts (N)
        vocab_size:            1D array of vocabulary sizes (V)
        num_characters:        1D array of characters processed (C)
        params:                array([p0,
                                       a_n, α_n,
                                       a_v, α_v,
                                       a_c, α_c])

    Returns:
        1D array of predicted Lossu
    """
    # Unpack params
    p0, a_n, alpha_n, a_v, alpha_v, a_c, alpha_c = params

    # Add small epsilon to avoid zero‐power issues
    eps = 1e-8
    N = Non_vocab_parameters + eps
    V = vocab_size + eps
    C = num_characters + eps

    # Power‐law contributions
    term_n = a_n * np.power(N, -alpha_n)
    term_v = a_v * np.power(V, -alpha_v)
    term_c = a_c * np.power(C, -alpha_c)

    return p0 + term_n + term_v + term_c

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7‐parameter power‐law scaling law using BFGS on MSE.
    """
    # 7 parameters: [p0, a_n, α_n, a_v, α_v, a_c, α_c]
    initial_params = np.ones(7)

    def objective(params):
        try:
            pred = scaling_law_func(
                Non_vocab_parameters, vocab_size, num_characters, params
            )
            return np.mean((pred - lossu_values) ** 2)
        except Exception:
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# Metadata: number of parameters used by scaling_law_func
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END