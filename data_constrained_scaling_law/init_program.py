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
    A scaling law function to model the relationship between data points and loss.
    
    This starts as a simple power law but can evolve into more complex forms.
    IMPORTANT: This function must use no more than 7 parameters, no more and no less.
    
    Args:
        data_points: Array of data points (training data size)
        params: Array of parameters for the scaling law
        
    Returns:
        Predicted loss values
    """
    
    loss = params[0] + params[1] / np.power(tokens + 1e07, params[2]) + params[3] / np.power(model_size + 1e07, params[4]) + params[5] / np.power(unique_tokens + 1e07, params[6])
    
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the scaling law to data points and loss values
    
    Args:
        # tokens: Array of training tokens used
        # model_size: Array of model parameter counts
        # unique_tokens: Array of unique tokens available
        # loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters (7 parameters)
    """
    initial_params = np.ones(7)
    
    def objective(params):
        try:
            predicted = scaling_law_func(tokens, model_size, unique_tokens, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6  # Return large error if computation fails
    
    result = minimize(objective, initial_params, method='BFGS')
    
    # Ensure result has exactly 4 parameters
    final_params = result.x if result.success else initial_params
    
    return final_params


# Set the number of parameters this function expects
scaling_law_func.num_params = 7

# EVOLVE-BLOCK-END
