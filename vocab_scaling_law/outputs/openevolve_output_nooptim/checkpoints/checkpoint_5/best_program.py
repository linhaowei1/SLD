# EVOLVE-BLOCK-START
"""
Vocab scaling law discovery for LLM training scenarios
Evolved scaling law using log-linear combination and inverse terms for improved fit.
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    A scaling law function to model the relationship between vocabulary parameters and Lossu.
    Uses a combination of log terms and inverse terms to capture scaling behavior.
    Args:
        Non_vocab_parameters: Array of non-vocabulary parameter counts
        vocab_size: Array of vocabulary sizes  
        num_characters: Array of number of characters processed
        params: Array of 7 parameters
    Returns:
        Predicted Lossu values
    """
    p0, p1, p2, p3, p4, p5, p6 = params
    # Offsets to avoid zero or negative inputs
    np_off = Non_vocab_parameters + 1e6
    v_off = vocab_size + 1e3
    c_off = num_characters + 1e8
    # Compute log-linear and inverse contributions
    log_term = p1 * np.log(np_off) + p2 * np.log(v_off) + p3 * np.log(c_off)
    inv_term = p4 / np_off + p5 / v_off + p6 / c_off
    # Total Lossu prediction
    return p0 + log_term + inv_term

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the scaling law to vocabulary data and Lossu values
    (Unchanged: uses BFGS with ones initialization)
    """
    initial_params = np.ones(7)
    def objective(params):
        try:
            predicted = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params)
            return np.mean((predicted - lossu_values) ** 2)
        except:
            return 1e6
    result = minimize(objective, initial_params, method='BFGS')
    final_params = result.x if result.success else initial_params
    return final_params

# Set the number of parameters this function expects
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END