"""
Evaluator for scaling law discovery programs
Evaluator for assessing the performance of scaling law discovery programs
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

# Data size list (skip first column's 0, start from 200)
DATA_SIZES = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600, 819200, 1638400])

def get_failure_result() -> Dict[str, float]:
    """
    Return standard result for failure cases, ensuring same key structure as success cases
    """
    # Dataset name list
    dataset_names = ["flan", "gigaword", "wmt19"]
    
    # Use 100000 as worst MSE score (very large MSE value)
    worst_mse = 100000.0
    # Corresponding worst overall_score (close to 0)
    worst_score = 1.0 / (1.0 + worst_mse)
    
    result = {
        "mse": worst_mse,
        "combined_score": worst_score,
    }
    
    # Add failure scores for each possible dataset
    for dataset_name in dataset_names:
        result[f"mse_{dataset_name}"] = worst_mse
    
    return result

def load_real_datasets(data_dir="data"):
    """
    Load real datasets from CSV files
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Dictionary containing all dataset and model data
    """
    datasets = {}
    
    # CSV file list
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    
    for csv_file in csv_files:
        dataset_name = csv_file.replace(".csv", "")
        file_path = os.path.join(data_dir, csv_file)
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Get loss value columns (skip first column config name, skip data size 0 column, exclude last two columns size and family)
            loss_columns = df.columns[2:-2]  # Start from 3rd column (skip config name and 0 column), to third-to-last column
            
            # Initialize dataset
            datasets[dataset_name] = {}
            
            # Create data for each model
            for idx, row in df.iterrows():
                model_name = row['config name']
                model_size = row['size'] 
                model_family = row['family']
                
                # Extract loss values
                loss_values = []
                valid_data_sizes = []
                
                for i, col in enumerate(loss_columns):
                    loss_val = row[col]
                    if pd.notna(loss_val) and loss_val > 0:  # Only use valid positive loss values
                        loss_values.append(float(loss_val))
                        valid_data_sizes.append(DATA_SIZES[i])
                
                if len(loss_values) >= 4:  # Ensure enough data points for fitting
                    datasets[dataset_name][model_name] = {
                        "data_points": np.array(valid_data_sizes),
                        "loss_values": np.array(loss_values),
                        "model_size": model_size,
                        "model_family": model_family
                    }
            
        except Exception as e:
            continue
    
    return datasets

# Dynamically determine data directory path
def get_data_dir():
    """Get correct path for data directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    # If data folder exists in current directory, use it first
    if os.path.exists("data"):
        return "data"
    # Otherwise use data folder in same directory as script
    elif os.path.exists(data_dir):
        return data_dir
    else:
        # Try parent directory
        parent_data_dir = os.path.join(os.path.dirname(script_dir), "data")
        if os.path.exists(parent_data_dir):
            return parent_data_dir
        else:
            raise FileNotFoundError("Cannot find data directory, please ensure data folder exists")

# Load real datasets
try:
    data_directory = get_data_dir()
    TEST_DATASETS = load_real_datasets(data_directory)
except Exception as e:
    TEST_DATASETS = {}


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
        
        # Check if test datasets are available
        if not TEST_DATASETS:
            return get_failure_result()
        
        # Evaluate on multiple test datasets and models
        all_scores = []
        dataset_scores = {}
        model_count = 0
        total_models = sum(len(models) for models in TEST_DATASETS.values())
        
        for dataset_name, models in TEST_DATASETS.items():
            dataset_model_scores = []
            
            for model_name, model_data in models.items():
                model_count += 1
                try:
                    data_points = model_data["data_points"]
                    true_loss = model_data["loss_values"]
                    
                    # Fit scaling law with timeout
                    start_time = time.time()
                    fitted_params = run_with_timeout(
                        fit_scaling_law, 
                        args=(data_points, true_loss),
                        timeout_seconds=600
                    )
                    fit_time = time.time() - start_time
                    
                    # Generate predictions
                    predicted_loss = run_with_timeout(
                        scaling_law_func,
                        args=(data_points, fitted_params),
                        timeout_seconds=600
                    )
                    
                    # Evaluate fit quality
                    metrics = evaluate_fit_quality(predicted_loss, true_loss)
                    
                    if "error" in metrics:
                        continue
                    
                    # Only use MSE as evaluation metric
                    mse_value = metrics["mse"]
                    
                    dataset_model_scores.append(mse_value)
                    all_scores.append(mse_value)
                    
                except TimeoutError as e:
                    pass
                except Exception as e:
                    pass
            
            # Calculate dataset average MSE
            if dataset_model_scores:
                dataset_avg_mse = np.mean(dataset_model_scores)
                dataset_scores[dataset_name] = float(dataset_avg_mse)
            else:
                dataset_scores[dataset_name] = 100000.0  # Set to worst score when failed (finite value)
        
        # If all datasets failed
        if not all_scores:
            return get_failure_result()
        
        # Calculate MSE statistics
        mean_mse = float(np.mean(all_scores))
        # std_mse = float(np.std(all_scores))
        # min_mse = float(np.min(all_scores))
        # max_mse = float(np.max(all_scores))
        
        # Calculate overall_score: use 1/(1+mse) so that smaller mse gives larger score (higher is better)
        overall_score = 1.0 / (1.0 + mean_mse)
        
        # Prepare return result - ensure format matches get_failure_result()
        result = {
            "mse": mean_mse,
            "combined_score": overall_score,
        }
        
        # Add MSE scores for all possible datasets
        dataset_names = ["flan", "gigaword", "wmt19"]
        for dataset_name in dataset_names:
            if dataset_name in dataset_scores:
                result[f"mse_{dataset_name}"] = dataset_scores[dataset_name]
            else:
                # If dataset doesn't exist, use worst score
                result[f"mse_{dataset_name}"] = 100000.0
        
        return result
        
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
