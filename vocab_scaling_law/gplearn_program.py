"""
GPlearn-based scaling law discovery for vocab scaling law
"""

import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def scaling_law_func(vocab_size, model_size, params):
    """
    Scaling law function using GPlearn discovered formula
    
    Args:
        vocab_size: Array of vocabulary sizes
        model_size: Array of model sizes
        params: Dictionary containing the fitted GPlearn model
        
    Returns:
        Predicted loss values
    """
    if params is None or 'model' not in params:
        return np.full_like(vocab_size, 2.7)  # Default fallback
    
    model = params['model']
    
    try:
        # Prepare feature matrix with log transforms
        X = np.column_stack([
            np.log(vocab_size + 1e-10),
            np.log(model_size + 1e-10)
        ])
        
        predictions = model.predict(X)
        # Ensure predictions are reasonable (loss values typically between 1-5)
        predictions = np.clip(predictions, 1.0, 5.0)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return np.full_like(vocab_size, 2.7)

def fit_scaling_law(vocab_size, model_size, loss_values):
    """
    Fit scaling law using GPlearn
    
    Args:
        vocab_size: Array of vocabulary sizes
        model_size: Array of model sizes
        loss_values: Array of loss values to fit
        
    Returns:
        Dictionary containing fitted GPlearn model
    """
    try:
        # Prepare feature matrix with log transforms
        X = np.column_stack([
            np.log(vocab_size + 1e-10),
            np.log(model_size + 1e-10)
        ])
        
        y = np.array(loss_values)
        
        est_gp = SymbolicRegressor(
             population_size=1000, generations=20, tournament_size=20, stopping_criteria=0.0, const_range=(-10.0, 10.0), init_depth=(2, 6), init_method='half and half', function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'), metric='mean absolute error', parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0, feature_names=['log_vocab_size', 'log_model_size'], warm_start=False, low_memory=False, n_jobs=20, verbose=1, random_state=42
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