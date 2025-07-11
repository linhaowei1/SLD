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
    IMPORTANT: This function must use exactly 4 parameters, no more and no less.
    
    Args:
        data_points: Array of data points (training data size)
        params: Array of parameters for the scaling law (must be exactly 4 parameters)
        
    Returns:
        Predicted loss values
    """
    # Ensure we have exactly 4 parameters
    if len(params) != 4:
        # Truncate to 4 parameters if more, pad to 4 if less
        if len(params) > 4:
            params = params[:4]
        else:
            # Pad with default values if needed
            padded_params = np.concatenate([params, np.ones(4 - len(params))])
            params = padded_params
    
    # Convert data_points to numpy array and handle edge cases
    x = np.asarray(data_points, dtype=float)
    
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-6
    x_safe = np.maximum(x, epsilon)
    
    # Simple power law: loss = a * (data_points + b)^(-c) + d
    # This is a common form for scaling laws with exactly 4 parameters
    a = abs(params[0]) + 0.1  # Ensure positive scale factor
    b = abs(params[1]) + 1.0  # Ensure positive offset
    c = abs(params[2]) + 0.1  # Ensure positive exponent
    d = abs(params[3]) + 0.01 # Ensure positive baseline loss
    
    # Power law with offset: loss = a * (x + b)^(-c) + d
    loss = a * np.power(x_safe + b, -c) + d
    
    return loss


def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the scaling law to data points and loss values
    
    Args:
        data_points: Array of data points (training data size)
        loss_values: Array of corresponding loss values
        initial_params: Initial parameter guess (optional, must be exactly 4 parameters)
        
    Returns:
        Optimized parameters (exactly 4 parameters)
    """
    # Ensure initial parameters are exactly 4 elements
    if initial_params is None:
        initial_params = np.random.rand(4)
    else:
        if len(initial_params) != 4:
            if len(initial_params) > 4:
                initial_params = initial_params[:4]
            else:
                initial_params = np.concatenate([initial_params, np.ones(4 - len(initial_params))])
    
    def objective(params):
        try:
            # Ensure params has exactly 4 elements
            if len(params) != 4:
                return 1e6  # Return large error if wrong number of parameters
            
            predicted = scaling_law_func(data_points, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6  # Return large error if computation fails
    
    # Use bounds to constrain parameters and prevent numerical issues
    bounds = [(0.01, 10.0),   # a: scale factor
              (0.1, 1000.0),  # b: offset  
              (0.01, 3.0),    # c: exponent
              (0.001, 1.0)]   # d: baseline loss
    
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    
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


if __name__ == "__main__":
    # Use real data to test the scaling law function
    # Load CSV files from data folder
    
    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    
    # Training data sizes (corresponding to columns in CSV files)
    data_sizes = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600, 819200, 1638400])
    
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {csv_file}")
        print(f"{'='*50}")
        
        # Load CSV file
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # Get loss value columns (exclude first column for model name, last two columns for size and family)
        loss_columns = df.columns[1:-2]
        
        # Fit scaling law for each model
        for idx, row in df.iterrows():
            model_name = row['config name']
            
            # Extract loss values (skip first column's initial value since data size is 0)
            loss_values = []
            valid_data_sizes = []
            
            for i, col in enumerate(loss_columns[1:], 1):  # Skip column for data size 0
                loss_val = row[col]
                if pd.notna(loss_val) and loss_val > 0:  # Only use valid positive loss values
                    loss_values.append(float(loss_val))
                    valid_data_sizes.append(data_sizes[i-1])
            
            if len(loss_values) >= 4:  # Ensure enough data points for fitting
                loss_values = np.array(loss_values)
                valid_data_sizes = np.array(valid_data_sizes)
                
                print(f"\nModel: {model_name}")
                print(f"Number of data points: {len(valid_data_sizes)}")
                print(f"Data size range: {valid_data_sizes[0]} - {valid_data_sizes[-1]}")
                print(f"Loss value range: {loss_values[-1]:.3f} - {loss_values[0]:.3f}")
                
                # Fit scaling law
                fitted_params = fit_scaling_law(valid_data_sizes, loss_values)
                print(f"Fitted parameters: {fitted_params}")
                
                # Calculate fit quality (mean squared error)
                predicted_loss = scaling_law_func(valid_data_sizes, fitted_params)
                mse = np.mean((predicted_loss - loss_values) ** 2)
                print(f"Mean squared error: {mse:.6f}")
                
                # Display model information
                model_size = row['size']
                model_family = row['family']
                print(f"Model size: {model_size:,} parameters")
                print(f"Model family: {model_family}")
            else:
                print(f"\nModel {model_name}: Insufficient data points, skipping fit")
