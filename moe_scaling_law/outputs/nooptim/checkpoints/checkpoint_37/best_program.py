# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models

Revised scaling law uses a hybrid log‐linear + inverse‐power form:
  - Linear terms in log(num_experts) and log(total_parameter_count)
  - Inverse‐power terms for num_experts, total_parameter_count, and their product
This flexible yet simple form can capture diminishing returns and interaction effects.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A scaling law function to model the relationship between number of experts, 
    total parameter count, and loss for MoE models.
    
    Uses 6 parameters:
      p0: constant bias
      p1: coefficient for log(num_experts)
      p2: coefficient for log(total_parameter_count)
      p3: coefficient for 1/num_experts
      p4: coefficient for 1/total_parameter_count
      p5: coefficient for 1/(num_experts * total_parameter_count)
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        params: Array of 6 parameters for the scaling law
        
    Returns:
        Predicted loss values
    """
    p0, p1, p2, p3, p4, p5 = params
    
    # Ensure positive inputs for log/inverse operations
    ne = np.maximum(num_experts, 1.0)
    np_count = np.maximum(total_parameter_count, 1.0)
    
    # Log-linear contributions
    term_log_experts = p1 * np.log(ne)
    term_log_params  = p2 * np.log(np_count)
    
    # Inverse-power contributions
    term_inv_experts   = p3 / ne
    term_inv_params    = p4 / np_count
    term_inv_inter     = p5 / (ne * np_count)
    
    return p0 + term_log_experts + term_log_params + term_inv_experts + term_inv_params + term_inv_inter

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the MoE scaling law to data using BFGS optimization.
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized 6 parameters
    """
    # Initialize all 6 parameters to 1
    initial_params = np.ones(6)
    
    def objective(params):
        try:
            predicted = scaling_law_func(num_experts, total_parameter_count, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            # Penalty for invalid params
            return 1e6
    
    result = minimize(objective, initial_params, method='BFGS')
    final_params = result.x if result.success else initial_params
    return final_params

# Declare how many parameters the scaling law uses
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END