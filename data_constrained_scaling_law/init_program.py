# EVOLVE-BLOCK-START
"""
Data-constrained scaling law discovery for LLM training scenarios
Initial program with a data-constrained scaling law form that can be evolved
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Data-constrained scaling law: L = A * (T/T0)^(-alpha) * (N/N0)^(-beta) * (1 + (T/U)^gamma) + E
    
    IMPORTANT: This function must use no more than 7 parameters.
    
    Args:
        tokens: np.ndarray, training tokens used
        model_size: np.ndarray, model parameter count
        unique_tokens: np.ndarray, unique tokens available
        params: Array of up to 7 parameters [A, alpha, beta, gamma, T0, N0, E]
        
    Returns:
        Predicted loss values
    """
    # Ensure we have up to 7 parameters, pad with defaults if needed
    A, alpha, beta, gamma, T0, N0, E = (list(params) + [1.0, 0.2, 0.2, 0.5, 1e9, 1e9, 0.1])[:7]
    
    # Convert inputs to numpy arrays and handle edge cases
    T = np.maximum(np.asarray(tokens, dtype=float), 1e3)
    N = np.maximum(np.asarray(model_size, dtype=float), 1e3)
    U = np.maximum(np.asarray(unique_tokens, dtype=float), 1e3)
    
    # Ensure reference values are reasonable
    T0 = max(abs(T0), 1e3)
    N0 = max(abs(N0), 1e3)
    
    # Data-constrained scaling law with constraint penalty
    loss = abs(A) * (T/T0)**(-abs(alpha)) * (N/N0)**(-abs(beta)) * (1 + (T/U)**abs(gamma)) + abs(E)
    
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values, initial_params=None):
    """
    Fit the data-constrained scaling law to the training data
    
    Args:
        tokens: array of training tokens used
        model_size: array of model parameter counts
        unique_tokens: array of unique tokens available
        loss_values: array of corresponding loss values
        initial_params: initial guess for up to 7 parameters
        
    Returns:
        Optimized parameters (up to 7 parameters)
    """
    # Convert inputs to numpy arrays
    T = np.asarray(tokens, dtype=float)
    N = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    L = np.asarray(loss_values, dtype=float)
    
    # Set initial parameters if not provided
    if initial_params is None:
        initial_params = [1.0, 0.2, 0.2, 0.5, np.median(T), np.median(N), np.min(L)*0.8]
    
    # Ensure up to 7 parameters
    if len(initial_params) < 7:
        initial_params = list(initial_params) + [1.0] * (7 - len(initial_params))
    else:
        initial_params = initial_params[:7]
    
    # Parameter bounds for optimization
    bounds = [
        (1e-5, 100.0),     # A: scale factor
        (0.01, 2.0),       # alpha: token scaling exponent
        (0.01, 2.0),       # beta: model scaling exponent
        (0.01, 2.0),       # gamma: constraint penalty exponent
        (1e3, 1e15),       # T0: reference token count
        (1e3, 1e15),       # N0: reference parameter count
        (0.0, 10.0)        # E: loss floor
    ]
    
    def objective(params):
        try:
            predicted = scaling_law_func(T, N, U, params)
            mse = np.mean((predicted - L) ** 2)
            return mse
        except Exception:
            return 1e10
    
    # Optimize parameters
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    
    # Return optimized parameters or initial guess if optimization failed
    fitted_params = result.x if result.success else initial_params
    
    return fitted_params

# Set the number of parameters this function expects (up to 7)
scaling_law_func.num_params = 7

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Test with the data-constrained scaling law
    df = pd.read_csv("data/data.csv")
    
    # Extract variables
    tokens = df['tokens'].values
    model_size = df['params'].values  
    unique_tokens = df['unique_tokens'].values
    loss = df['loss'].values
    
    print(f"Loaded {len(df)} data points")
    print(f"Token range: {tokens.min():.2e} - {tokens.max():.2e}")
    print(f"Model size range: {model_size.min():.2e} - {model_size.max():.2e}")
    print(f"Unique token range: {unique_tokens.min():.2e} - {unique_tokens.max():.2e}")
    print(f"Loss range: {loss.min():.4f} - {loss.max():.4f}")
    
    # Fit the scaling law
    print("\nFitting data-constrained scaling law...")
    params = fit_scaling_law(tokens, model_size, unique_tokens, loss)
    
    # Print results
    A, alpha, beta, gamma, T0, N0, E = params
    print("\nData-Constrained Scaling Law Parameters:")
    print("=" * 50)
    print(f"A (scale factor):        {A:.4f}")
    print(f"α (token scaling):       {alpha:.4f}")
    print(f"β (model scaling):       {beta:.4f}")  
    print(f"γ (constraint penalty):  {gamma:.4f}")
    print(f"T₀ (reference tokens):   {T0:.2e}")
    print(f"N₀ (reference params):   {N0:.2e}")
    print(f"E (loss floor):          {E:.4f}")
    
    # Evaluate fit quality
    pred = scaling_law_func(tokens, model_size, unique_tokens, params)
    mse = np.mean((pred - loss) ** 2)
    r2 = 1 - np.sum((pred - loss) ** 2) / np.sum((loss - np.mean(loss)) ** 2)
    
    print(f"\nFit Quality:")
    print("=" * 30)
    print(f"R² score:           {r2:.4f}")
    print(f"MSE:                {mse:.6f}")
    print(f"RMSE:               {np.sqrt(mse):.4f}")