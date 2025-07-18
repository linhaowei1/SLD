"""
Evaluator for data-constrained scaling law discovery programs
"""

import importlib.util
import numpy as np
import pandas as pd
import os
import time
import traceback
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

def get_failure_result() -> Dict[str, float]:
    """
    Return standard result for failure cases, ensuring same key structure as success cases
    """
    # Use 100000 as worst MSE score (very large MSE value)
    worst_mse = 100000.0
    # Corresponding worst combined_score (close to 0)
    worst_score = 1.0 / (1.0 + worst_mse)
    
    result = {
        "mse": worst_mse,
        "combined_score": worst_score,
    }
    
    return result

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=600):
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
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function execution timeout, exceeded {timeout_seconds} seconds")

def safe_float(value):
    """Safely convert value to float"""
    try:
        return float(value)
    except (TypeError, ValueError):
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
    try:
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
        
        if len(pred_filtered) < 2:
            return {"error": "Insufficient valid data points"}
        
        # Calculate evaluation metrics
        mse = mean_squared_error(true_filtered, pred_filtered)
        rmse = np.sqrt(mse)
        
        # R² score
        r2 = r2_score(true_filtered, pred_filtered)
        
        # Pearson correlation coefficient
        correlation, _ = pearsonr(true_filtered, pred_filtered)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((true_filtered - pred_filtered) / true_filtered)) * 100
        
        # Normalized Root Mean Square Error
        nrmse = rmse / (np.max(true_filtered) - np.min(true_filtered))
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "correlation": float(correlation),
            "mape": float(mape),
            "nrmse": float(nrmse),
            "valid_points": int(len(pred_filtered))
        }
        
    except Exception as e:
        return {"error": f"Error during evaluation: {str(e)}"}

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Main function to evaluate scaling law programs
    
    Args:
        program_path: Program file path
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
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
        
        # Load the data
        try:
            df = pd.read_csv("data/data.csv")
        except FileNotFoundError:
            return get_failure_result()
        
        tokens = df['tokens'].values
        model_size = df['params'].values
        unique_tokens = df['unique_tokens'].values
        loss_values = df['loss'].values
        
        try:
            # Fit the scaling law with timeout
            start_time = time.time()
            fitted_params = run_with_timeout(
                fit_scaling_law,
                args=(tokens, model_size, unique_tokens, loss_values),
                timeout_seconds=600
            )
            fit_time = time.time() - start_time
            
            # Generate predictions
            predicted_loss = run_with_timeout(
                scaling_law_func,
                args=(tokens, model_size, unique_tokens, fitted_params),
                timeout_seconds=600
            )
            
            # Evaluate fit quality
            metrics = evaluate_fit_quality(predicted_loss, loss_values)
            
            if "error" in metrics:
                return get_failure_result()
            
            # Only use MSE as evaluation metric
            mse_value = metrics["mse"]
            
            # Calculate combined_score: use 1/(1+mse) so that smaller mse gives larger score (higher is better)
            combined_score = 1.0 / (1.0 + mse_value)
            
            # Prepare return result
            result = {
                "mse": float(mse_value),
                "combined_score": float(combined_score),
            }
            
            return result
            
        except TimeoutError as e:
            return get_failure_result()
        except Exception as e:
            return get_failure_result()
        
    except Exception as e:
        return get_failure_result()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        sys.exit(1)
    
    program_path = sys.argv[1]
    
    result = evaluate(program_path)
    
    for key, value in result.items():
        print(f"{key}: {value}")