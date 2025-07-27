import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Hybrid log‐reciprocal model with 7 parameters:
      Let x1 = log(N), x2 = log(V), x3 = log(C), where
        N = Non_vocab_parameters,
        V = vocab_size,
        C = num_characters.
      Then
        Lossu ≈ p0
               + p1·x1 + p2·x2 + p3·x3
               + p4·(1 / x1) + p5·(1 / x2) + p6·(1 / x3)
      
      This form captures both roughly linear log‐effects and
      saturating/diminishing‐returns via the reciprocal terms.
    """
    # small epsilon to avoid log(0) or division by zero
    eps = 1e-8
    N = Non_vocab_parameters + eps
    V = vocab_size + eps
    C = num_characters + eps

    # log‐features
    x1 = np.log(N)
    x2 = np.log(V)
    x3 = np.log(C)

    # unpack 7 parameters
    p0, p1, p2, p3, p4, p5, p6 = params

    # compute predicted Lossu
    lossu_pred = (
        p0
        + p1 * x1
        + p2 * x2
        + p3 * x3
        + p4 / (x1 + eps)
        + p5 / (x2 + eps)
        + p6 / (x3 + eps)
    )
    return lossu_pred

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the scaling law to vocabulary data and Lossu values.

    Uses BFGS with an all‐ones initialization.
    """
    initial_params = np.ones(scaling_law_func.num_params)

    def objective(params):
        try:
            pred = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
            return np.mean((pred - lossu_values) ** 2)
        except:
            # in case of invalid params, return large loss
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# declare expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END