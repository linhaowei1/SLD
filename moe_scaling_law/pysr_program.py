"""
PySR-based scaling law discovery for MoE scaling law
"""

import numpy as np
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not available. Install with: pip install pysr")
import warnings
warnings.filterwarnings('ignore')

def scaling_law_func(N, E, params):
    """
    Scaling law function using PySR discovered formula
    
    Args:
        N: Array of model parameters (N)
        E: Array of number of experts (E)
        params: Dictionary containing the fitted PySR model
        
    Returns:
        Predicted loss values
    """
    if params is None or 'model' not in params:
        return np.full_like(N, 2.7)  # Default fallback
    
    model = params['model']
    
    try:
        # Prepare feature matrix with log transforms
        X = np.column_stack([
            np.log(N + 1e-10),
            np.log(E + 1e-10)
        ])
        
        if PYSR_AVAILABLE and hasattr(model, 'predict'):
            predictions = model.predict(X)
        else:
            # Fallback linear model
            predictions = model.predict(X)
        # Ensure predictions are reasonable (loss values typically between 1-5)
        predictions = np.clip(predictions, 1.0, 5.0)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return np.full_like(N, 2.7)

def fit_scaling_law(N, E, loss_values):
    """
    Fit scaling law using PySR
    
    Args:
        N: Array of model parameters (N)
        E: Array of number of experts (E)
        loss_values: Array of loss values to fit
        
    Returns:
        Dictionary containing fitted PySR model
    """
    try:
        # Prepare feature matrix with log transforms
        X = np.column_stack([
            np.log(N + 1e-10),
            np.log(E + 1e-10)
        ])
        
        y = np.array(loss_values)
        
        if PYSR_AVAILABLE:
            # Configure PySR with 6-hour budget and increased complexity
            model = PySRRegressor(
                niterations=1000000,  # Increased iterations
                binary_operators=["+", "-", "*", "/", "pow"],
                unary_operators=["exp", "log", "sqrt", "abs", "sin", "cos"],
                populations=50,  # Increased populations
                population_size=100,  # Increased population size
                ncyclesperiteration=5500,  # Increased cycles
                timeout_in_seconds=21600,  # 6 hours
                maxsize=30,  # Increased max complexity
                maxdepth=10,  # Increased max depth
                parsimony=0.0001,  # Lower parsimony for more complex equations
                variable_names=["log_N", "log_E"],
                temp_equation_file=True,
                delete_tempfiles=False,
                verbosity=1,
                progress=True,
                multithreading=True,
                procs=0,  # Use all available cores
                random_state=42
            )
            
            # Fit the model
            model.fit(X, y)
            
            return {'model': model}
        else:
            # Fallback to linear regression if PySR not available
            from sklearn.linear_model import LinearRegression
            fallback_model = LinearRegression()
            fallback_model.fit(X, y)
            return {'model': fallback_model}
        
    except Exception as e:
        print(f"Fitting error: {e}")
        # Return a simple fallback model
        from sklearn.linear_model import LinearRegression
        fallback_model = LinearRegression()
        fallback_model.fit(X, y)
        return {'model': fallback_model}