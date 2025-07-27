# EVOLVE-BLOCK-START
"""
Evolved vocab-based scaling law for LLM training scenarios.

Key improvements:
- Input normalization for numerical stability
- Bounded exponents to avoid pathological fits
- Informed initialization from data statistics
- Simplified, readable power‐law form with exactly 7 parameters
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu from model and training scale.

    params (7):
      p0 : baseline offset
      p1 : vocab amplitude
      p2 : vocab exponent (>0)
      p3 : param amplitude
      p4 : param exponent (>0)
      p5 : char amplitude
      p6 : char exponent (>0)

    Lossu = p0
           + p1 * (vocab_size / V_max) ** (-p2)
           + p3 * (Non_vocab_parameters / P_max) ** (-p4)
           + p5 * (num_characters / C_max) ** (-p6)
    """
    p0, p1, p2, p3, p4, p5, p6 = params
    # normalize each input by its maximum to keep powers numerically stable
    V = vocab_size / np.max(vocab_size)
    P = Non_vocab_parameters / np.max(Non_vocab_parameters)
    C = num_characters / np.max(num_characters)

    return (
        p0
        + p1 * np.power(V, -p2)
        + p3 * np.power(P, -p4)
        + p5 * np.power(C, -p6)
    )

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter scaling law to observed Lossu values.

    Returns:
      params: array of length 7
    """
    # initialize parameters:
    #   p0 = mean of losses
    #   amplitudes p1, p3, p5 = |mean_loss|/3
    #   exponents p2, p4, p6 = 0.5
    mean_l = np.mean(lossu_values)
    init = np.array([
        mean_l,                    # p0
        abs(mean_l) / 3, 0.5,      # p1, p2
        abs(mean_l) / 3, 0.5,      # p3, p4
        abs(mean_l) / 3, 0.5       # p5, p6
    ])

    # bounds: keep exponents positive but not unreasonably large
    bounds = [
        (-np.inf, np.inf),     # p0
        (0, np.inf),           # p1
        (1e-6, 5),             # p2
        (0, np.inf),           # p3
        (1e-6, 5),             # p4
        (0, np.inf),           # p5
        (1e-6, 5),             # p6
    ]

    def mse_loss(params):
        pred = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
        return np.mean((pred - lossu_values) ** 2)

    result = minimize(
        mse_loss,
        init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-9, 'gtol':1e-9}
    )

    if result.success and result.x.shape[0] == 7:
        return result.x
    # fallback to initial guess on failure
    return init

# record expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END