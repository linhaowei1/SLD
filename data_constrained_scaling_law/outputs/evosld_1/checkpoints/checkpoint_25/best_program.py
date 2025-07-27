# EVOLVE-BLOCK-START
"""
Evolved data-constrained scaling law for LLM loss with improved stability
and parsimony. We model loss as the sum of a model-size term and an
effective-data term (accounting for repetition saturation):

  loss ≈ A
       + B * (M / median(M))^(−α)
       + C * [ T / (median(T) * (1 + γ * (T/median(T) / (U/median(U)))^δ)) ]^(−β)

Parameters (7):
  p0 = A       : irreducible loss floor
  p1 = ln B    : log scale for model-size term
  p2 = α       : exponent for model-size
  p3 = ln C    : log scale for data term
  p4 = β       : exponent for effective-data
  p5 = γ       : repetition saturation scale
  p6 = δ       : repetition saturation exponent
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given tokens, model size, unique tokens and params.
    Args:
        tokens        : array-like, training tokens used
        model_size    : array-like, model parameter counts
        unique_tokens : array-like, unique tokens available
        params        : length-7 array [p0, lnB, α, lnC, β, γ, δ]
    Returns:
        loss_pred     : array of predicted losses
    """
    p0, lnB, α, lnC, β, γ, δ = params

    T = np.asarray(tokens, dtype=float)
    M = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)

    # median normalization for numerical stability
    M_med = np.median(M) + 1e-12
    T_med = np.median(T) + 1e-12
    U_med = np.median(U) + 1e-12

    Mn = M / M_med
    Tn = T / T_med
    Un = U / U_med

    # repetition ratio
    dup = Tn / (Un + 1e-12)
    # effective data with saturation: Tn / (1 + γ * dup^δ)
    eff_data = Tn / (1.0 + γ * np.power(dup, δ))

    # reconstruct positive coefficients
    B = np.exp(lnB)
    C = np.exp(lnC)

    # two-term scaling law
    loss = p0 + B * np.power(Mn, -α) + C * np.power(eff_data + 1e-12, -β)
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7-parameter scaling law to data by minimizing MSE.
    Returns optimized params array [p0..p6].
    """
    T = np.asarray(tokens, dtype=float)
    M = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    y_max = np.max(y)
    # parameter bounds
    bounds = [
        (0.0,        y_max),    # p0: A
        (-20.0,      20.0),     # p1: lnB
        (1e-3,       5.0),      # p2: α
        (-20.0,      20.0),     # p3: lnC
        (1e-3,       5.0),      # p4: β
        (1e-8,       100.0),    # p5: γ
        (1e-3,       5.0)       # p6: δ
    ]

    # objective: mean squared error
    def obj(p):
        pred = scaling_law_func(T, M, U, p)
        return np.mean((pred - y) ** 2)

    # multi-start L-BFGS-B
    best_p, best_val = None, np.inf
    rng = np.random.RandomState(42)
    for _ in range(8):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(obj, x0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success and res.fun < best_val:
            best_val, best_p = res.fun, res.x

    # fallback: midpoints
    if best_p is None:
        best_p = np.array([(lo + hi) * 0.5 for lo, hi in bounds])

    return best_p

# metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END