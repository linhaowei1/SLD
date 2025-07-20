"""
Evaluator for domain mixture scaling law discovery programs
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
    # Model size list
    model_sizes = ["70M", "160M", "305M", "410M"]
    
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
    
    # Add failure scores for each possible model size
    for model_size in model_sizes:
        result[f"mse_{model_size}"] = worst_mse
        result[f"r2_{model_size}"] = worst_r2
        result[f"mae_{model_size}"] = worst_mae
        result[f"nmse_{model_size}"] = worst_nmse
    
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
    Main function to evaluate domain mixture scaling law programs
    
    Args:
        program_path: Program file path
        use_test_data: If True, use test data; if False, use training data
        fitted_params_dict: Dictionary mapping model_size to fitted parameters
        return_metrics: If True, return metrics; if False, return fitted parameters
        
    Returns:
        Dictionary containing evaluation metrics or fitted parameters dict
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
        "domain_mixture_scaling_law", 
        train=not use_test_data
    )
    
    if not data_points:
        return get_failure_result()
    
    if return_metrics:
        # Return metrics mode: calculate and return metrics regardless of training/test data
        if not use_test_data:
            # For training data, need to fit parameters first for each model size
            fitted_params_dict = {}
            
            # Group data points by model size
            model_size_groups = {}
            for point in data_points:
                model_size = point["model_size"]
                if model_size not in model_size_groups:
                    model_size_groups[model_size] = []
                model_size_groups[model_size].append(point)
            
            # Fit parameters for each model size
            for model_size, points in model_size_groups.items():
                # Aggregate proportions and loss values for this model size
                proportions_list = []
                loss_values_list = []
                
                for point in points:
                    proportions_list.append(point["proportions"])
                    loss_values_list.append(point["loss_values"])
                
                # Stack into 2D arrays: [n_samples_for_this_model, 5]
                proportions = np.array(proportions_list)
                loss_values = np.array(loss_values_list)
                
                # Fit scaling law with timeout
                fitted_params = run_with_timeout(
                    fit_scaling_law,
                    args=(proportions, loss_values),
                    timeout_seconds=600
                )
                
                # Store fitted parameters for this model size
                fitted_params_dict[model_size] = fitted_params
        
        # Calculate metrics using fitted parameters
        if fitted_params_dict is None:
            return get_failure_result()
        
        all_mse_scores = []
        all_r2_scores = []
        all_mae_scores = []
        all_nmse_scores = []
        
        # Group data points by model size for evaluation
        model_size_groups = {}
        for point in data_points:
            model_size = point["model_size"]
            if model_size not in model_size_groups:
                model_size_groups[model_size] = []
            model_size_groups[model_size].append(point)
        
        # Evaluate each model size group
        for model_size, points in model_size_groups.items():
            # Get fitted parameters for this model size
            if model_size not in fitted_params_dict:
                continue
            
            fitted_params = fitted_params_dict[model_size]
            
            # Aggregate proportions and loss values for this model size
            # Each point has 5 proportions and 5 loss values, we need to flatten them properly
            proportions_list = []
            loss_values_list = []
            
            for point in points:
                # Each point represents one training configuration
                # Use the proportions as features and flatten loss values as targets
                proportions_list.append(point["proportions"])  # Shape: (5,)
                # Take the mean loss across the 5 domains for this configuration
                loss_values_list.append(np.mean(point["loss_values"]))
            
            # Stack into proper arrays
            proportions = np.array(proportions_list)  # [n_points, 5]
            loss_values = np.array(loss_values_list)  # [n_points]
            
            # Generate predictions using fitted parameters
            predicted_loss = run_with_timeout(
                scaling_law_func,
                args=(proportions, fitted_params),
                timeout_seconds=600
            )
            
            # Evaluate fit quality - both should be 1D arrays now
            predicted_flat = predicted_loss.flatten() if predicted_loss.ndim > 1 else predicted_loss
            true_flat = loss_values.flatten() if loss_values.ndim > 1 else loss_values
            
            metrics = evaluate_fit_quality(predicted_flat, true_flat)
            
            if "error" in metrics:
                continue
            
            # Extract evaluation metrics
            all_mse_scores.append(metrics["mse"])
            all_r2_scores.append(metrics["r2"])
            all_mae_scores.append(metrics["mae"])
            all_nmse_scores.append(metrics["nmse"])
        
        # If all model sizes failed
        if not all_mse_scores:
            return get_failure_result()
        
        # Calculate overall statistics
        mean_mse = float(np.mean(all_mse_scores))
        mean_r2 = float(np.mean(all_r2_scores))
        mean_mae = float(np.mean(all_mae_scores))
        mean_nmse = float(np.mean(all_nmse_scores))
        
        # Calculate combined_score: use 1/(1+nmse) so that smaller nmse gives larger score (higher is better)
        combined_score = 1.0 / (1.0 + mean_nmse)
        
        # Prepare return result - ensure format matches get_failure_result()
        result = {
            "mse": mean_mse,
            "r2": mean_r2,
            "mae": mean_mae,
            "nmse": mean_nmse,
            "combined_score": combined_score,
        }
        
        return result
    
    else:
        # Return fitted parameters mode: only return fitted parameters
        fitted_params_dict = {}
        
        # Group data points by model size
        model_size_groups = {}
        for point in data_points:
            model_size = point["model_size"]
            if model_size not in model_size_groups:
                model_size_groups[model_size] = []
            model_size_groups[model_size].append(point)
        
        # Fit parameters for each model size
        for model_size, points in model_size_groups.items():
            # Aggregate proportions and loss values for this model size
            # Each point has 5 proportions and 5 loss values, we need to flatten them properly
            proportions_list = []
            loss_values_list = []
            
            for point in points:
                # Each point represents one training configuration
                # Use the proportions as features and flatten loss values as targets
                proportions_list.append(point["proportions"])  # Shape: (5,)
                # Take the mean loss across the 5 domains for this configuration
                loss_values_list.append(np.mean(point["loss_values"]))
            
            # Stack into proper arrays
            proportions = np.array(proportions_list)  # [n_points, 5]
            loss_values = np.array(loss_values_list)  # [n_points]
            
            # Fit scaling law with timeout
            fitted_params = run_with_timeout(
                fit_scaling_law,
                args=(proportions, loss_values),
                timeout_seconds=600
            )
            
            # Store fitted parameters for this model size
            fitted_params_dict[model_size] = fitted_params
        
        return {"fitted_params": fitted_params_dict}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>")
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
    
    # Step 1: Use training data to fit parameters for each model size
    print("# Step 1: Fitting parameters for each TRAINING model size")
    train_result = evaluate(program_path, use_test_data=False, return_metrics=False)
    
    if "fitted_params" not in train_result:
        print("Error: Failed to fit parameters on training data")
        sys.exit(1)
    
    fitted_params_dict = train_result["fitted_params"]
    print(f"Successfully fitted parameters for {len(fitted_params_dict)} model sizes")
    
    # Extract and display the fitted equations
    if fitted_params_dict:
        print(f"\n# Fitted Equations:")
        for model_size, fitted_params in fitted_params_dict.items():
            if fitted_params and 'model' in fitted_params:
                model_obj = fitted_params['model']
                print(f"\nModel Size: {model_size}")
                try:
                    if hasattr(model_obj, '_program'):  # GPlearn model
                        print(f"GPlearn equation: {model_obj._program}")
                    elif hasattr(model_obj, 'equations_'):  # PySR model
                        if hasattr(model_obj.equations_, 'iloc') and len(model_obj.equations_) > 0:
                            best_eq = model_obj.equations_.iloc[-1]
                            print(f"PySR equation: {best_eq['equation']}")
                            print(f"Complexity: {best_eq['complexity']}")
                            print(f"Loss: {best_eq['loss']}")
                    elif hasattr(model_obj, 'coef_'):  # Linear regression fallback
                        print(f"Linear model coefficients: {model_obj.coef_}")
                        print(f"Linear model intercept: {model_obj.intercept_}")
                    else:
                        print("Could not extract equation from model")
                except Exception as e:
                    print(f"Error extracting equation: {e}")
    
    # Step 2: Use test data with fitted parameters for evaluation
    print("# Step 2: Evaluating fitted law on ALL TEST data")
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