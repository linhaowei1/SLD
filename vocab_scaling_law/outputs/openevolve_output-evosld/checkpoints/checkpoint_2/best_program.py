# EVOLVE-BLOCK-START
"""
Vocab scaling law discovery for LLM training scenarios
Initial program with a vocab-based scaling law form that can be evolved
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    A scaling law function to model the relationship between vocabulary parameters and Lossu.
    
    This starts as a simple power law but can evolve into more complex forms.
    IMPORTANT: This function must use no more than 7 parameters.
    
    Args:
        Non_vocab_parameters: Array of non-vocabulary parameter counts
        vocab_size: Array of vocabulary sizes  
        num_characters: Array of number of characters processed
        params: Array of parameters for the scaling law (up to 7 parameters)
        
    Returns:
        Predicted Lossu values
    """
    
    # Initial scaling law: combining effects of vocab size, non-vocab params, and character count
    # Lossu = base + vocab_term + param_term + char_term
    lossu = (params[0] + 
             params[1] / np.power(vocab_size + 1e3, params[2]) + 
             params[3] / np.power(Non_vocab_parameters + 1e6, params[4]) + 
             params[5] / np.power(num_characters + 1e8, params[6]))
    
    return lossu

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the scaling law to vocabulary data and Lossu values
    
    Args:
        Non_vocab_parameters: Array of non-vocabulary parameter counts
        vocab_size: Array of vocabulary sizes
        num_characters: Array of number of characters processed
        lossu_values: Array of corresponding Lossu values
        
    Returns:
        Optimized parameters (7 parameters)
    """
    # Initialize parameters with reasonable values for vocab scaling
    initial_params = np.array([-2.0, -1.0, 0.1, -1.0, 0.1, -1.0, 0.1])
    
    def objective(params):
        try:
            predicted = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
            mse = np.mean((predicted - lossu_values) ** 2)
            return mse
        except:
            return 1e6  # Return large error if computation fails
    
    result = minimize(objective, initial_params, method='BFGS')
    
    # Ensure result has exactly 7 parameters
    final_params = result.x if result.success else initial_params
    
    return final_params


# Set the number of parameters this function expects
scaling_law_func.num_params = 7

# EVOLVE-BLOCK-END