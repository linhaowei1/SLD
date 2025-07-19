# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Initial program with a simple power law form that can be evolved
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize


def scaling_law_func(data_points, params):
    """
    A scaling law function to model the relationship between data points and loss.
    
    This starts as a simple power law but can evolve into more complex forms.
    IMPORTANT: This function must use no more than 4 parameters, no more and no less.
    
    Args:
        data_points: Array of data points (training data size)
        params: Array of parameters for the scaling law (must be exactly 4 parameters)
        
    Returns:
        Predicted loss values
    """
    # Ensure we have exactly 4 parameters
    if len(params) > 4:
        params = params[:4]
    
    # Convert data_points to numpy array and handle edge cases
    x = np.asarray(data_points, dtype=float)
    
    loss = np.power(params[0] / np.power(x + 1e07, params[1]) + params[2], params[3])
    return loss


def fit_scaling_law(data_points, loss_values):
    """
    Fit the scaling law to data points and loss values
    
    Args:
        data_points: Array of data points (training data size)
        loss_values: Array of corresponding loss values
        initial_params: Initial parameter guess
        
    Returns:
        Optimized parameters (4 parameters)
    """
    initial_params = np.ones(4)
    
    def objective(params):
        try:
            predicted = scaling_law_func(data_points, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6  # Return large error if computation fails
    
    result = minimize(objective, initial_params, method='BFGS')
    
    # Ensure result has exactly 4 parameters
    final_params = result.x if result.success else initial_params
    
    # Double check the length and truncate/pad if necessary
    if len(final_params) != 4:
        if len(final_params) > 4:
            final_params = final_params[:4]
        else:
            final_params = np.concatenate([final_params, np.ones(4 - len(final_params))])
    
    return final_params


# Set the number of parameters this function expects (MUST BE 4)
scaling_law_func.num_params = 4

# EVOLVE-BLOCK-END
