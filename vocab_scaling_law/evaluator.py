"""
Evaluator for vocab scaling law discovery programs
"""

import importlib.util
import numpy as np
import pandas as pd
import os
import sys
import time
import traceback
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent directory to path for data_loader import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

def get_failure_result() -> Dict[str, float]:
    """
    Return standard result for failure cases, ensuring same key structure as success cases
    """
    # Use 100000 as worst values
    worst_mse = 100000.0
    worst_mae = 100000.0
    worst_nmse = 100000.0
    worst_r2 = -1.0  # Worst R2 score
    # Corresponding worst combined_score (close to 0)
    worst_score = 1.0 / (1.0 + worst_nmse)
    
    result = {
        "mse": worst_mse,
        "r2": worst_r2,
        "mae": worst_mae,
        "nmse": worst_nmse,
        "combined_score": worst_score,
    }
    
    return result

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run function with timeout
    
    Args:
        func: Function to run
        args: Function arguments
        kwargs: Keyword arguments
        timeout_seconds: Timeout in seconds
        
    Returns:
        Function result or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        result = future.result(timeout=timeout_seconds)
        return result

def safe_float(value):
    """Safely convert value to float"""
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0

def evaluate_fit_quality(predicted_values: np.ndarray, true_values: np.ndarray) -> Dict[str, float]:
    """
    Evaluate fit quality
    
    Args:
        predicted_values: Predicted values
        true_values: True values
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    # Ensure inputs are numpy arrays
    predicted = np.asarray(predicted_values, dtype=float)
    true = np.asarray(true_values, dtype=float)
    
    # Check shape matching
    if predicted.shape != true.shape:
        return {"error": "Predicted and true values shape mismatch"}
        
    # Filter out invalid values
    valid_mask = ~(np.isnan(predicted) | np.isnan(true) | np.isinf(predicted) | np.isinf(true))
    if not np.any(valid_mask):
        return {"error": "All predicted values are invalid"}
        
    pred_filtered = predicted[valid_mask]
    true_filtered = true[valid_mask]
    
    if len(pred_filtered) < 1:
        return {"error": "Insufficient valid data points"}
    
    # Calculate evaluation metrics
    mse = mean_squared_error(true_filtered, pred_filtered)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = mean_absolute_error(true_filtered, pred_filtered)
    
    # Normalized Mean Square Error (MSE normalized by variance of true values)
    true_var = np.var(true_filtered)
    nmse = mse / true_var if true_var > 0 else mse
    r2 = 1 - nmse
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
        "nmse": float(nmse),
        "valid_points": int(len(pred_filtered))
    }

def evaluate(program_path: str, use_test_data: bool = False, fitted_params: Dict = None, return_metrics: bool = True) -> Dict[str, float]:
    """
    Main function to evaluate vocab scaling law programs
    
    Args:
        program_path: Program file path
        use_test_data: If True, use test data; if False, use training data
        fitted_params: Pre-fitted parameters to use for prediction
        return_metrics: If True, return metrics; if False, return fitted parameters
        
    Returns:
        Dictionary containing evaluation metrics or fitted parameters
    """
    # Load program
    spec = importlib.util.spec_from_file_location("scaling_law_func", program_path)
    if spec is None or spec.loader is None:
        return get_failure_result()
        
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Check if required functions exist
    if not hasattr(program, "scaling_law_func"):
        return get_failure_result()
        
    if not hasattr(program, "fit_scaling_law"):
        return get_failure_result()
    
    scaling_law_func = program.scaling_law_func
    fit_scaling_law = program.fit_scaling_law
    
    # Load the data using unified data loader
    data_points = load_data(
        "vocab_scaling_law", 
        train=not use_test_data
    )
    
    if not data_points:
        return get_failure_result()
    
    # Aggregate all loss values and features from all data points
    all_lossu_values = []
    all_non_vocab_params = []
    all_vocab_size = []
    all_num_characters = []
    
    for point in data_points:
        # Each point has lossu_values (single element) and corresponding features
        all_lossu_values.extend(point["lossu_values"])
        all_non_vocab_params.extend(point["Non_vocab_parameters"])
        all_vocab_size.extend(point["vocab_size"])
        all_num_characters.extend(point["num_characters"])
        
    non_vocab_parameters = np.array(all_non_vocab_params)
    vocab_size = np.array(all_vocab_size)
    num_characters = np.array(all_num_characters)
    lossu_values = np.array(all_lossu_values)
    
    if fitted_params is None and not return_metrics:
        # Training mode: fit the scaling law on all training data and return parameters
        start_time = time.time()
        fitted_params = run_with_timeout(
            fit_scaling_law,
            args=(non_vocab_parameters, vocab_size, num_characters, lossu_values),
            timeout_seconds=600
        )
        fit_time = time.time() - start_time
        
        return {"fitted_params": fitted_params}
    elif fitted_params is not None or return_metrics:
        # Testing/evaluation mode: use fitted parameters or fit and return metrics
        if fitted_params is None:
            # Need to fit first
            fitted_params = run_with_timeout(
                fit_scaling_law,
                args=(non_vocab_parameters, vocab_size, num_characters, lossu_values),
                timeout_seconds=600
            )
        
        # Use parameters to predict and evaluate
        predicted_lossu = run_with_timeout(
            scaling_law_func,
            args=(non_vocab_parameters, vocab_size, num_characters, fitted_params),
            timeout_seconds=600
        )
        
        # Evaluate fit quality
        metrics = evaluate_fit_quality(predicted_lossu, lossu_values)
        
        if "error" in metrics:
            return get_failure_result()
        
        # Extract evaluation metrics
        mse_value = metrics["mse"]
        r2_value = metrics["r2"]
        mae_value = metrics["mae"]
        nmse_value = metrics["nmse"]
        
        # Calculate combined_score: use 1/(1+nmse) so that smaller nmse gives larger score (higher is better)
        combined_score = 1.0 / (1.0 + nmse_value)
        
        # Prepare return result
        result = {
            "mse": float(mse_value),
            "r2": float(r2_value),
            "mae": float(mae_value),
            "nmse": float(nmse_value),
            "combined_score": float(combined_score),
        }
        
        return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    
    # Load program
    spec = importlib.util.spec_from_file_location("scaling_law_func", program_path)
    if spec is None or spec.loader is None:
        print("Error: Could not load program")
        sys.exit(1)
        
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Check if required functions exist
    if not hasattr(program, "scaling_law_func") or not hasattr(program, "fit_scaling_law"):
        print("Error: Program must have 'scaling_law_func' and 'fit_scaling_law' functions")
        sys.exit(1)
        
    scaling_law_func = program.scaling_law_func
    fit_scaling_law = program.fit_scaling_law
    
    # Step 1: Use training data to fit one scaling law on all training data
    print("# Step 1: Fitting scaling law on ALL TRAINING data")
    train_result = evaluate(program_path, use_test_data=False, fitted_params=None, return_metrics=False)
    
    if "fitted_params" not in train_result:
        print("Error: Failed to fit parameters on training data")
        sys.exit(1)
    
    fitted_params = train_result["fitted_params"]
    print(f"Successfully fitted scaling law parameters")
    
    # Step 2: Use test data with fitted parameters for evaluation
    print("# Step 2: Evaluating fitted law on ALL TEST data")
    test_result = evaluate(program_path, use_test_data=True, fitted_params=fitted_params)
    
    if "mse" not in test_result:
        print("Error: Failed to evaluate on test data")
        sys.exit(1)
    
    # Print test results
    print(f"mse: {test_result['mse']}")
    print(f"r2: {test_result['r2']}")
    print(f"mae: {test_result['mae']}")
    print(f"nmse: {test_result['nmse']}")
    print(f"combined_score: {test_result['combined_score']}")