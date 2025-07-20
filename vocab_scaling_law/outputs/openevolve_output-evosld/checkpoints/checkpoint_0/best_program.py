# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu given model size, vocab size, and training tokens.
    
    params: [a, b, c_exp, d, e_exp, f, g_exp] (7 total)
    """
    # Normalize inputs for numerical stability
    V = vocab_size.astype(np.float64) / 1e4
    P = Non_vocab_parameters.astype(np.float64) / 1e6
    C = num_characters.astype(np.float64) / 1e8

    a, b, c_exp, d, e_exp, f, g_exp = params
    # Enforce exponents >= 0 via absolute value
    c_exp = np.abs(c_exp)
    e_exp = np.abs(e_exp)
    g_exp = np.abs(g_exp)

    # Core scaling law: base + b*V^{-c} + d*P^{-e} + f*C^{-g}
    pred = a \
           + b * np.power(V, -c_exp) \
           + d * np.power(P, -e_exp) \
           + f * np.power(C, -g_exp)
    return pred

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter scaling law to observed Lossu data.
    Returns optimized params: [a, b, c, d, e, f, g].
    """
    # Initial guess: center around data median and moderate negative contributions
    a0 = np.median(lossu_values)
    init_params = np.array([a0,    # base offset
                            -1.0,  # b
                             0.5,  # c_exp
                            -1.0,  # d
                             0.5,  # e_exp
                            -1.0,  # f
                             0.5]) # g_exp

    # Bounds: a,b,d,f unbounded; exponents in [1e-3, 5]
    bounds = [
        (None, None),  # a
        (None, None),  # b
        (1e-3, 5.0),   # c_exp
        (None, None),  # d
        (1e-3, 5.0),   # e_exp
        (None, None),  # f
        (1e-3, 5.0),   # g_exp
    ]

    def objective(params):
        pred = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
        mse = np.mean((pred - lossu_values) ** 2)
        # lightweight L2 regularization to discourage extreme coefficients
        reg = 1e-4 * np.sum(params**2)
        return mse + reg

    result = minimize(
        objective,
        init_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if result.success and result.x.shape[0] == 7:
        return result.x
    else:
        # fallback to initial guess if optimization fails
        return init_params

# annotate expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END