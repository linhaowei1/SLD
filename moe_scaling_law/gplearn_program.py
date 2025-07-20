"""
GPlearn-based scaling law discovery for MoE scaling law
"""

import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def scaling_law_func(N, E, params):
    """
    Scaling law function using GPlearn discovered formula
    
    Args:
        N: Array of model parameters (N)
        E: Array of number of experts (E)
        params: Dictionary containing the fitted GPlearn model
        
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
        
        predictions = model.predict(X)
        # Ensure predictions are reasonable (loss values typically between 1-5)
        predictions = np.clip(predictions, 1.0, 5.0)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return np.full_like(N, 2.7)

def fit_scaling_law(N, E, loss_values):
    """
    Fit scaling law using GPlearn
    
    Args:
        N: Array of model parameters (N)
        E: Array of number of experts (E)
        loss_values: Array of loss values to fit
        
    Returns:
        Dictionary containing fitted GPlearn model
    """
    try:
        # Prepare feature matrix with log transforms
        X = np.column_stack([
            np.log(N + 1e-10),
            np.log(E + 1e-10)
        ])
        
        y = np.array(loss_values)

        est_gp = SymbolicRegressor(
             population_size=1000, generations=20, tournament_size=20, stopping_criteria=0.0, const_range=(-10.0, 10.0), init_depth=(2, 6), init_method='half and half', function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'), metric='mean absolute error', parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0, feature_names=['log_N', 'log_E'], warm_start=False, low_memory=False, n_jobs=20, verbose=1, random_state=42
        )
        # Fit the model
        est_gp.fit(X, y)
        
        return {'model': est_gp}
        
    except Exception as e:
        print(f"Fitting error: {e}")
        # Return a simple fallback model
        from sklearn.linear_model import LinearRegression
        fallback_model = LinearRegression()
        fallback_model.fit(X, y)
        return {'model': fallback_model}