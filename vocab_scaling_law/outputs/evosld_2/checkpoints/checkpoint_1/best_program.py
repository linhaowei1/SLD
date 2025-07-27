# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Two-term diminishing-returns scaling law for Lossu:
      Lossu ≈ p0 + p1 / (Non_vocab_parameters * num_characters)^p2
                    + p3 / (vocab_size)^p4

    Args:
        Non_vocab_parameters: Array of non-vocabulary parameter counts
        vocab_size:            Array of vocabulary sizes
        num_characters:        Array of number of characters processed
        params:                [p0, p1, p2, p3, p4]

    Returns:
        Array of predicted Lossu values
    """
    p0, p1, p2, p3, p4 = params
    # tiny constant to avoid division by zero
    eps = 1e-12
    # Combined compute-scale term
    K = Non_vocab_parameters * num_characters + eps
    # Apply diminishing-returns power laws
    term_compute = p1 * np.power(K, -p2)
    term_vocab   = p3 * np.power(vocab_size + eps, -p4)
    return p0 + term_compute + term_vocab

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the above scaling law to observed data via bounded L-BFGS-B.

    Returns:
        params: Optimized [p0, p1, p2, p3, p4]
    """
    # Initial parameter guesses:
    #   p0 ≈ median Lossu,
    #   p1, p3 ≈ std(Lossu),
    #   p2, p4 ≈ 0.5 (common diminishing-returns exponent)
    p0_init = np.median(lossu_values)
    amp     = np.std(lossu_values) + 1e-6
    x0      = np.array([p0_init, amp, 0.5, amp, 0.5])

    # Bounds: keep amplitudes ≥ 0, exponents in (1e-6, 5]
    bounds = [
        (None, None),    # p0 (intercept) unbounded
        (0.0, None),     # p1 ≥ 0
        (1e-6, 5.0),     # p2 in (0,5]
        (0.0, None),     # p3 ≥ 0
        (1e-6, 5.0),     # p4 in (0,5]
    ]

    def objective(params):
        pred = scaling_law_func(Non_vocab_parameters,
                                vocab_size,
                                num_characters,
                                params)
        # Mean squared error loss
        return np.mean((pred - lossu_values) ** 2)

    result = minimize(objective,
                      x0,
                      method='L-BFGS-B',
                      bounds=bounds)

    if result.success:
        return result.x
    else:
        # Fallback to initial guess on failure
        return x0

# Indicate how many parameters our scaling law uses
scaling_law_func.num_params = 5
# EVOLVE-BLOCK-END