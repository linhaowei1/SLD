# EVOLVE-BLOCK-START
"""
Evolved vocab scaling law discovery for LLM training scenarios

We model Lossu as a sum of three inverse‐power‐law terms
(vocabulary size, non‐vocabulary parameter count, and character count)
plus an additive constant (base).  To improve numerical stability
and fit robustness, we optimize only the three exponents in an
outer loop, and solve for the four linear coefficients (including
base) in closed form via least‐squares at each step.  This reduces
the nonconvex search dimensionality and yields a more accurate,
stable fit with the same 7 total parameters.

Scaling law:
    Lossu ≈ base
          + c1 * vocab_size^(−alpha)
          + c2 * Non_vocab_parameters^(−beta)
          + c3 * num_characters^(−gamma)
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu given model/training scales and 7 parameters.

    Lossu = base
          + c1 * vocab_size^(−alpha)
          + c2 * Non_vocab_parameters^(−beta)
          + c3 * num_characters^(−gamma)

    Args:
        Non_vocab_parameters: array_like, non‐vocab parameter counts
        vocab_size:           array_like, vocabulary sizes
        num_characters:       array_like, number of characters processed
        params:               array_like of length 7
                              [base, c1, alpha, c2, beta, c3, gamma]

    Returns:
        np.ndarray of predicted Lossu values
    """
    base, c1, alpha, c2, beta, c3, gamma = params
    # avoid zero‐division / negative exponents on zero
    P = np.maximum(Non_vocab_parameters, 1.0)
    V = np.maximum(vocab_size, 1.0)
    C = np.maximum(num_characters, 1.0)

    return (
        base
        + c1 * V ** ( - alpha )
        + c2 * P ** ( - beta )
        + c3 * C ** ( - gamma )
    )

# declare expected parameter count
scaling_law_func.num_params = 7

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7‐parameter scaling law to data.

    We optimize the three exponents (alpha, beta, gamma) via L‐BFGS‐B,
    and at each candidate exponent set we solve for the four linear
    coefficients [base, c1, c2, c3] by ordinary least squares.
    This two‐stage approach greatly improves convergence and
    numerical stability.

    Returns:
        params: np.ndarray of length 7
                [base, c1, alpha, c2, beta, c3, gamma]
    """
    # flatten / ensure numpy arrays
    P = np.asarray(Non_vocab_parameters).ravel()
    V = np.asarray(vocab_size).ravel()
    C = np.asarray(num_characters).ravel()
    Y = np.asarray(lossu_values).ravel()

    # pre‐check: at least 4 points needed
    if Y.size < 4:
        # fallback: return trivial zero model
        return np.array([np.mean(Y), 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    # objective: given exponents [alpha, beta, gamma], solve linear coeffs by LS
    def _mse_of_exponents(x):
        alpha, beta, gamma = x
        x1 = V ** ( - alpha )
        x2 = P ** ( - beta )
        x3 = C ** ( - gamma )
        # design matrix: [1, x1, x2, x3]
        M = np.vstack((np.ones_like(Y), x1, x2, x3)).T
        # least‐squares solve for [base, c1, c2, c3]
        # rcond=None uses default cutoff
        coeffs, *_ = np.linalg.lstsq(M, Y, rcond=None)
        Y_pred = M.dot(coeffs)
        return np.mean((Y_pred - Y) ** 2)

    # initial exponent guesses
    x0 = np.array([0.3, 0.3, 0.3], dtype=float)
    # bounds to keep exponents positive but not too large
    bounds = [(1e-6, 5.0), (1e-6, 5.0), (1e-6, 5.0)]

    res = minimize(
        _mse_of_exponents,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-12, 'maxiter':5000}
    )

    if res.success:
        alpha, beta, gamma = res.x
    else:
        alpha, beta, gamma = x0

    # final linear solve for base, c1, c2, c3
    x1 = V ** ( - alpha )
    x2 = P ** ( - beta )
    x3 = C ** ( - gamma )
    M_final = np.vstack((np.ones_like(Y), x1, x2, x3)).T
    coeffs, *_ = np.linalg.lstsq(M_final, Y, rcond=None)
    base, c1, c2, c3 = coeffs

    # assemble in the order expected by scaling_law_func
    # [base, c1, alpha, c2, beta, c3, gamma]
    return np.array([base, c1, alpha, c2, beta, c3, gamma], dtype=float)
# EVOLVE-BLOCK-END