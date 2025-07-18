# EVOLVE-BLOCK-START
"""
Data-constrained scaling law discovery for LLM training scenarios
Revised for simplicity, stability, and efficient parameter fitting
Uses a 6-parameter form:
    L = E + A * (N/N_ref)^(-beta) * (T/T_ref)^(-alpha) * (1 + C * (T/T_ref ÷ U/U_ref)^gamma)
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given training tokens, model size, and unique token budget.
    Uses a normalized, data-constrained scaling law with 6 parameters.
    
    Args:
        tokens: array-like, training token counts
        model_size: array-like, model parameter counts
        unique_tokens: array-like, unique tokens available
        params: [A, alpha, beta, C, gamma, E]
            A     : scale coefficient (>0)
            alpha : token scaling exponent (>0)
            beta  : model scaling exponent (>0)
            C     : constraint penalty coefficient (>=0)
            gamma : constraint penalty exponent (>=0)
            E     : irreducible loss floor (>=0)
    Returns:
        loss_pred: numpy array of predicted losses
    """
    # Unpack and enforce positivity where needed
    A, alpha, beta, C, gamma, E = params
    A     = max(A,     1e-12)
    alpha = max(alpha, 1e-12)
    beta  = max(beta,  1e-12)
    C     = max(C,     0.0)
    gamma = max(gamma, 0.0)
    E     = max(E,     0.0)
    
    # Convert to numpy arrays
    T = np.asarray(tokens, dtype=float)
    N = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    
    # Reference scales (median) for normalization
    T_ref = np.median(T)
    N_ref = np.median(N)
    U_ref = np.median(U)
    
    # Avoid division by zero
    T_ref = max(T_ref, 1.0)
    N_ref = max(N_ref, 1.0)
    U_ref = max(U_ref, 1.0)
    
    # Normalized variables
    Tn = T / T_ref
    Nn = N / N_ref
    Un = U / U_ref
    
    # Core scaling law with data-constraint penalty
    base = (Nn ** (-beta)) * (Tn ** (-alpha))
    penalty = 1.0 + C * (Tn / Un) ** gamma
    loss = E + A * base * penalty
    
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values, initial_params=None):
    """
    Fit the 6-parameter scaling law via MSE minimization.
    
    Args:
        tokens: array-like, training tokens
        model_size: array-like, model parameters
        unique_tokens: array-like, unique tokens available
        loss_values: array-like, observed losses
        initial_params: optional list of 6 initial guesses [A, alpha, beta, C, gamma, E]
    Returns:
        fitted_params: array of 6 optimized parameters
    """
    # Vectorize inputs
    T = np.asarray(tokens, dtype=float)
    N = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    L = np.asarray(loss_values, dtype=float)
    
    # Default initial guess if none provided
    if initial_params is None:
        A0     = 1.0
        alpha0 = 0.3
        beta0  = 0.07
        C0     = 1.0
        gamma0 = 0.5
        E0     = max(np.min(L) * 0.5, 1e-3)
        initial_params = [A0, alpha0, beta0, C0, gamma0, E0]
    else:
        initial_params = list(initial_params)[:6]
    
    # Bounds for stability
    bounds = [
        (1e-6, 1e2),   # A
        (1e-6, 2.0),   # alpha
        (1e-6, 2.0),   # beta
        (0.0, 10.0),   # C
        (0.0, 3.0),    # gamma
        (0.0, 10.0)    # E
    ]
    
    # Objective: mean squared error
    def objective(p):
        pred = scaling_law_func(T, N, U, p)
        return np.mean((pred - L) ** 2)
    
    # Optimize
    res = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    if res.success:
        fitted = res.x
    else:
        fitted = np.array(initial_params, dtype=float)
    
    return fitted

# Number of parameters expected
scaling_law_func.num_params = 6

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import pandas as pd
    import os
    
    # Load data
    datapath = os.path.join("data", "data.csv")
    df = pd.read_csv(datapath)
    tokens = df['tokens'].values
    model_size = df['params'].values
    unique_tokens = df['unique_tokens'].values
    loss = df['loss'].values
    
    print(f"Loaded {len(df)} data points")
    print(f"Token range: {tokens.min():.2e} - {tokens.max():.2e}")
    print(f"Model size range: {model_size.min():.2e} - {model_size.max():.2e}")
    print(f"Unique token range: {unique_tokens.min():.2e} - {unique_tokens.max():.2e}")
    print(f"Loss range: {loss.min():.4f} - {loss.max():.4f}")
    
    # Fit the revised scaling law
    print("\nFitting revised data-constrained scaling law...")
    params = fit_scaling_law(tokens, model_size, unique_tokens, loss)
    A, alpha, beta, C, gamma, E = params
    
    print("\nRevised Scaling Law Parameters:")
    print("=" * 50)
    print(f"A     (scale factor)           : {A:.4f}")
    print(f"α     (token exponent)         : {alpha:.4f}")
    print(f"β     (model size exponent)    : {beta:.4f}")
    print(f"C     (constraint coeff)       : {C:.4f}")
    print(f"γ     (constraint exponent)    : {gamma:.4f}")
    print(f"E     (irreducible loss floor) : {E:.4f}")
    
    # Evaluate fit
    pred = scaling_law_func(tokens, model_size, unique_tokens, params)
    mse = np.mean((pred - loss) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((pred - loss)**2) / np.sum((loss - np.mean(loss))**2)
    
    print("\nFit Quality:")
    print("=" * 30)
    print(f"R² score: {r2:.4f}")
    print(f"MSE     : {mse:.6f}")
    print(f"RMSE    : {rmse:.4f}")