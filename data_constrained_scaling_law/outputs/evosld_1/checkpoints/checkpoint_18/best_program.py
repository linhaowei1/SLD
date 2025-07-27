# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Data‐constrained scaling law with a three‐term structure:
      loss ≈ p0
           + p1 * M_norm^(−p2)
           + p3 * D_norm^(−p4)
           + p5 * S_norm^(−p6)
    where:
      M_norm = model_size / median(model_size)
      D_norm = tokens / median(tokens)
      R_norm = (tokens/unique_tokens) / median(tokens/unique_tokens)
      S_norm = M_norm * D_norm / R_norm
    This captures pure model scaling, pure data scaling, and their
    interaction modulated by data duplication.
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6 = params

    # Convert to numpy arrays
    M = np.asarray(model_size, dtype=float)
    T = np.asarray(tokens, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)

    # Small constant to avoid division by zero
    eps = 1e-12

    # Compute normalization constants
    M_med = np.median(M) + eps
    T_med = np.median(T) + eps
    R = T / (U + eps)
    R_med = np.median(R) + eps

    # Normalized inputs
    M_n = (M / M_med) + eps
    D_n = (T / T_med) + eps
    R_n = (R / R_med) + eps
    S_n = M_n * D_n / R_n + eps

    # Evaluate each term
    term_M = p1 * M_n**(-p2)
    term_D = p3 * D_n**(-p4)
    term_S = p5 * S_n**(-p6)

    # Predicted loss
    return p0 + term_M + term_D + term_S


def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the above scaling law to observed (tokens, model_size, unique_tokens, loss)
    via regularized least squares and multi‐start L-BFGS-B.
    Returns a length‐7 array of optimized parameters [p0…p6].
    """
    # Convert inputs to arrays
    T = np.asarray(tokens, dtype=float)
    M = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    eps = 1e-12
    # Precompute normalization constants once
    M_med = np.median(M) + eps
    T_med = np.median(T) + eps
    R = T / (U + eps)
    R_med = np.median(R) + eps

    # Pre‐normalized features
    M_n = (M / M_med) + eps
    D_n = (T / T_med) + eps
    R_n = (R / R_med) + eps
    S_n = M_n * D_n / R_n + eps

    # Objective: MSE + small L2 penalty on exponents to stabilize
    def objective(p):
        p0, p1, p2, p3, p4, p5, p6 = p
        pred = p0 + p1*M_n**(-p2) + p3*D_n**(-p4) + p5*S_n**(-p6)
        res  = pred - y
        mse  = np.mean(res**2)
        reg  = 1e-2 * (p2**2 + p4**2 + p6**2)
        return mse + reg

    # Parameter bounds: keep scale factors positive and exponents moderate
    bounds = [
        (0.0,    np.max(y)),  # p0: baseline floor
        (1e-6,   10.0),       # p1: model‐scale factor
        (0.01,   2.0),        # p2: model exponent
        (1e-6,   10.0),       # p3: data‐scale factor
        (0.01,   2.0),        # p4: data exponent
        (1e-6,   10.0),       # p5: synergy‐scale factor
        (0.01,   2.0)         # p6: synergy exponent
    ]

    # Multi‐start L-BFGS‐B
    best_p, best_obj = None, np.inf
    rng = np.random.RandomState(0)
    # central guess: mid‐points of each bound
    center = np.array([0.5*(lo + hi) for (lo, hi) in bounds])
    for i in range(10):
        if i < 2:
            init = center.copy()
        else:
            init = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(objective, init,
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-10})
        if res.success and res.fun < best_obj:
            best_obj, best_p = res.fun, res.x.copy()

    # Fallback to center if all starts fail
    if best_p is None:
        best_p = center

    return best_p

# Attach metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END