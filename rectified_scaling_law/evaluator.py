"""
Evaluator for scaling law discovery programs
Evaluator for assessing the performance of scaling law discovery programs
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
    # Dataset name list
    dataset_names = ["flan", "gigaword", "wikiword"]
    
    # Use 100000 as worst values
    worst_mse = 100000.0
    worst_mae = 100000.0
    worst_nmse = 100000.0
    worst_r2 = -1.0  # Worst R2 score
    # Corresponding worst overall_score (close to 0)
    worst_score = 1.0 / (1.0 + worst_nmse)
    
    result = {
        "mse": worst_mse,
        "r2": worst_r2,
        "mae": worst_mae,
        "nmse": worst_nmse,
        "combined_score": worst_score,
    }
    
    # Add failure scores for each possible dataset
    for dataset_name in dataset_names:
        result[f"mse_{dataset_name}"] = worst_mse
        result[f"r2_{dataset_name}"] = worst_r2
        result[f"mae_{dataset_name}"] = worst_mae
        result[f"nmse_{dataset_name}"] = worst_nmse
    
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


def evaluate(program_path: str, use_test_data: bool = False, fitted_params_dict: Dict = None, return_metrics: bool = True) -> Dict[str, float]:
    """
    Main function to evaluate scaling law programs
    
    Args:
        program_path: Program file path
        use_test_data: If True, use test data; if False, use training data
        fitted_params_dict: Dictionary mapping (model, dataset) to fitted parameters
        return_metrics: If True, return metrics; if False, return fitted parameters
        
    Returns:
        Dictionary containing evaluation metrics or fitted parameters dict
    """
    
    # Load program
    spec = importlib.util.spec_from_file_location("scaling_program", program_path)
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
    
    # Load datasets based on use_test_data parameter
    data_points = load_data("rectified_scaling_law", train=not use_test_data)
    
    # Check if data points are available
    if not data_points:
        return get_failure_result()
    
    if return_metrics:
        # Return metrics mode: calculate and return metrics regardless of training/test data
        if not use_test_data:
            # For training data, need to fit parameters first
            fitted_params_dict = {}
            
            for point in data_points:
                model = point["model"]
                dataset = point["dataset"]
                data_sizes = point["data_size"]
                true_loss = point["loss_values"]
                
                # Fit scaling law with timeout
                fitted_params = run_with_timeout(
                    fit_scaling_law, 
                    args=(data_sizes, true_loss),
                    timeout_seconds=600
                )
                
                # Store fitted parameters
                fitted_params_dict[(model, dataset)] = fitted_params
        
        # Calculate metrics using fitted parameters
        if fitted_params_dict is None:
            return get_failure_result()
        
        all_mse_scores = []
        all_r2_scores = []
        all_mae_scores = []
        all_nmse_scores = []
        
        for point in data_points:
            model = point["model"]
            dataset = point["dataset"]
            data_sizes = point["data_size"]
            true_loss = point["loss_values"]
            
            # Get fitted parameters for this model-dataset combination
            if (model, dataset) not in fitted_params_dict:
                continue
            
            fitted_params = fitted_params_dict[(model, dataset)]

            # Generate predictions using fitted parameters
            predicted_loss = run_with_timeout(
                scaling_law_func,
                args=(data_sizes, fitted_params),
                timeout_seconds=600
            )
            
            # Evaluate fit quality
            metrics = evaluate_fit_quality(predicted_loss, true_loss)
            
            if "error" in metrics:
                continue
            
            # Extract evaluation metrics
            all_mse_scores.append(metrics["mse"])
            all_r2_scores.append(metrics["r2"])
            all_mae_scores.append(metrics["mae"])
            all_nmse_scores.append(metrics["nmse"])
        
        # If all data points failed
        if not all_mse_scores:
            return get_failure_result()
        
        # Calculate overall statistics
        mean_mse = float(np.mean(all_mse_scores))
        mean_r2 = float(np.mean(all_r2_scores))
        mean_mae = float(np.mean(all_mae_scores))
        mean_nmse = float(np.mean(all_nmse_scores))
        
        # Calculate overall_score: use 1/(1+nmse) so that smaller nmse gives larger score (higher is better)
        overall_score = 1.0 / (1.0 + mean_nmse)
        
        # Prepare return result - ensure format matches get_failure_result()
        result = {
            "mse": mean_mse,
            "r2": mean_r2,
            "mae": mean_mae,
            "nmse": mean_nmse,
            "combined_score": overall_score,
        }
        
        return result
    
    else:
        # Return fitted parameters mode: only return fitted parameters
        fitted_params_dict = {}
        
        for point in data_points:
            model = point["model"]
            dataset = point["dataset"]
            data_sizes = point["data_size"]
            true_loss = point["loss_values"]
            
            # Fit scaling law with timeout
            fitted_params = run_with_timeout(
                fit_scaling_law, 
                args=(data_sizes, true_loss),
                timeout_seconds=600
            )
            
            # Store fitted parameters
            fitted_params_dict[(model, dataset)] = fitted_params
        
        return {"fitted_params": fitted_params_dict}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    
    # Load program
    spec = importlib.util.spec_from_file_location("scaling_program", program_path)
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
    
    # Step 1: Use training data to fit parameters for each data point
    print("# Step 1: Fitting parameters for each TRAINING data point")
    train_result = evaluate(program_path, use_test_data=False, return_metrics=False)
    
    if "fitted_params" not in train_result:
        print("Error: Failed to fit parameters on training data")
        sys.exit(1)
    
    fitted_params_dict = train_result["fitted_params"]
    print(f"Successfully fitted parameters for {len(fitted_params_dict)} model-dataset combinations")
    
    # Step 2: Use test data with fitted parameters for evaluation
    test_result = evaluate(program_path, use_test_data=True, fitted_params_dict=fitted_params_dict)
    
    if "mse" not in test_result:
        print("Error: Failed to evaluate on test data")
        sys.exit(1)
    
    # Print test results
    print(f"mse: {test_result['mse']}")
    print(f"r2: {test_result['r2']}")
    print(f"mae: {test_result['mae']}")
    print(f"nmse: {test_result['nmse']}")
    print(f"combined_score: {test_result['combined_score']}")
    