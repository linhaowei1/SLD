# EVOLVE-BLOCK-START
"""
Improved data-constrained scaling law discovery for LLM training scenarios.
7-parameter form with enhanced flexibility and log-space fitting for stability.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    7-parameter data-constrained scaling law:
      L = E + A * Nn^(-beta) * Tn^(-alpha) * (1 + C * (Tn/Un)^gamma)^delta

    Args:
        tokens       : array-like, training token counts
        model_size   : array-like, model parameter counts
        unique_tokens: array-like, unique tokens available
        params       : [A, alpha, beta, C, gamma, delta, E]
            A     : scale coefficient (>0)
            alpha : token scaling exponent (>0)
            beta  : model scaling exponent (>0)
            C     : constraint penalty coefficient (>=0)
            gamma : constraint penalty exponent (>=0)
            delta : penalty curvature exponent (>=0)
            E     : irreducible loss floor (>=0)
    Returns:
        loss_pred: numpy array of predicted losses
    """
    A, alpha, beta, C, gamma, delta, E = params
    # enforce positivity where needed
    A     = max(A,     1e-12)
    alpha = max(alpha, 1e-12)
    beta  = max(beta,  1e-12)
    C     = max(C,     0.0)
    gamma = max(gamma, 0.0)
    delta = max(delta, 0.0)
    E     = max(E,     0.0)

    # cast to arrays
    T = np.asarray(tokens,       dtype=float)
    N = np.asarray(model_size,   dtype=float)
    U = np.asarray(unique_tokens,dtype=float)

    # normalization by median to stabilize scales
    T_ref = max(np.median(T), 1.0)
    N_ref = max(np.median(N), 1.0)
    U_ref = max(np.median(U), 1.0)

    Tn = T / T_ref
    Nn = N / N_ref
    Un = U / U_ref

    # core scaling + penalty
    base    = (Nn ** (-beta)) * (Tn ** (-alpha))
    penalty = (1.0 + C * (Tn / Un) ** gamma) ** delta

    loss_pred = E + A * base * penalty
    return loss_pred

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values, initial_params=None):
    """
    Fit the 7-parameter scaling law via log-space MSE minimization.

    Args:
        tokens       : array-like, training tokens
        model_size   : array-like, model parameters
        unique_tokens: array-like, unique tokens available
        loss_values  : array-like, observed losses
        initial_params: optional list of 7 initial guesses
    Returns:
        fitted_params: array of 7 optimized parameters
    """
    T = np.asarray(tokens,        dtype=float)
    N = np.asarray(model_size,    dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    L = np.asarray(loss_values,   dtype=float)

    # default initial guess
    if initial_params is None:
        A0     = (np.max(L) - np.min(L)) * 0.5 + 1e-3
        alpha0 = 0.3
        beta0  = 0.07
        C0     = 1.0
        gamma0 = 0.5
        delta0 = 1.0
        E0     = max(np.min(L) * 0.2, 1e-3)
        x0 = [A0, alpha0, beta0, C0, gamma0, delta0, E0]
    else:
        # take up to 7, pad if needed
        x0 = list(initial_params)[:7]
        while len(x0) < 7:
            x0.append(1e-3)

    # bounds for stability
    Lmin = np.min(L)
    bounds = [
        (1e-8, 1e3),             # A
        (1e-6, 2.0),             # alpha
        (1e-6, 2.0),             # beta
        (0.0,   10.0),           # C
        (0.0,   3.0),            # gamma
        (0.0,   5.0),            # delta
        (0.0,   max(Lmin,1.0))   # E
    ]

    # objective: minimize MSE in log-space for robustness
    def objective(p):
        pred = scaling_law_func(T, N, U, p)
        # avoid log of non-positive
        pred = np.maximum(pred, 1e-12)
        return np.mean((np.log(pred) - np.log(L)) ** 2)

    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    if res.success:
        return res.x
    else:
        return np.array(x0, dtype=float)

# annotate number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import pandas as pd
    import os

    # Load data
    datapath = os.path.join("data", "data.csv")
    df = pd.read_csv(datapath)
    tokens        = df['tokens'].values
    model_size    = df['params'].values
    unique_tokens = df['unique_tokens'].values
    loss_values   = df['loss'].values

    print(f"Loaded {len(df)} data points")
    print(f"Token range: {tokens.min():.2e} - {tokens.max():.2e}")
    print(f"Model size range: {model_size.min():.2e} - {model_size.max():.2e}")
    print(f"Unique token range: {unique_tokens.min():.2e} - {unique_tokens.max():.2e}")
    print(f"Loss range: {loss_values.min():.4f} - {loss_values.max():.4f}")

    # Fit the improved scaling law
    print("\nFitting improved data-constrained scaling law...")
    params = fit_scaling_law(tokens, model_size, unique_tokens, loss_values)
    A, alpha, beta, C, gamma, delta, E = params

    print("\nOptimized Scaling Law Parameters:")
    print("=" * 50)
    print(f"A     (scale factor)           : {A:.4f}")
    print(f"α     (token exponent)         : {alpha:.4f}")
    print(f"β     (model size exponent)    : {beta:.4f}")
    print(f"C     (constraint coeff)       : {C:.4f}")
    print(f"γ     (constraint exponent)    : {gamma:.4f}")
    print(f"δ     (penalty curvature)      : {delta:.4f}")
    print(f"E     (irreducible loss floor) : {E:.4f}")

    # Evaluate fit
    pred = scaling_law_func(tokens, model_size, unique_tokens, params)
    mse  = np.mean((pred - loss_values) ** 2)
    rmse = np.sqrt(mse)
    r2   = 1 - np.sum((pred - loss_values) ** 2) / np.sum((loss_values - np.mean(loss_values)) ** 2)

    print("\nFit Quality:")
    print("=" * 30)
    print(f"R²   : {r2:.4f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.4f}")