# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models
Initial program with a MoE scaling law form that can be evolved
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A scaling law function to model the relationship between number of experts, 
    total parameter count, and loss for MoE models.
    
    This starts as a simple power law but can evolve into more complex forms.
    IMPORTANT: This function must use no more than 6 parameters.
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        params: Array of parameters for the scaling law (up to 6 parameters)
        
    Returns:
        Predicted loss values
    """
    
    # Simple MoE scaling law with expert and parameter scaling
    loss =  params[0] + params[1] / np.power(num_experts + 1e-6, params[2]) + params[3] / np.power(total_parameter_count + 1e6, params[4])
    
    return loss

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the MoE scaling law to data
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters (up to 6 parameters)
    """
    initial_params = np.ones(5)
    
    def objective(params):
        try:
            predicted = scaling_law_func(num_experts, total_parameter_count, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6  # Return large error if computation fails
    
    result = minimize(objective, initial_params, method='BFGS')
    
    final_params = result.x if result.success else initial_params
    
    return final_params


# Set the number of parameters this function expects
scaling_law_func.num_params = 5

# EVOLVE-BLOCK-END