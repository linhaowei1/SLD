"""
Evaluator for data-constrained scaling law discovery programs
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
import json
import logging
from datetime import datetime
from multiprocessing import Pool, Manager

# Try to import PySR and GPLearn
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not available")

try:
    from gplearn.genetic import SymbolicRegressor
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    print("Warning: GPLearn not available")

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

def fit_scaling_law_with_pysr_gplearn(tokens, model_size, unique_tokens, loss_values, timeout_hours=6):
    """
    Fit scaling law using both PySR and GPLearn in parallel with increased budgets
    
    Args:
        tokens: Array of training tokens used
        model_size: Array of model parameter counts
        unique_tokens: Array of unique tokens available
        loss_values: Array of corresponding loss values
        timeout_hours: Hours to run each method
        
    Returns:
        Dictionary with both PySR and GPLearn results
    """
    results = {}
    
    # Prepare input data
    X = np.column_stack([tokens, model_size, unique_tokens])
    y = loss_values
    
    def run_pysr():
        if not PYSR_AVAILABLE:
            return None
        try:
            # Configure PySR with 6-hour budget and increased complexity
            model = PySRRegressor(
                niterations=1000000,  # Increased iterations
                binary_operators=["+", "-", "*", "/", "pow"],
                unary_operators=["exp", "log", "sqrt", "abs", "sin", "cos"],
                populations=50,  # Increased populations
                population_size=100,  # Increased population size
                ncyclesperiteration=5500,  # Increased cycles
                timeout_in_seconds=timeout_hours * 3600,  # 6 hours
                maxsize=30,  # Increased max complexity
                maxdepth=10,  # Increased max depth
                parsimony=0.0001,  # Lower parsimony for more complex equations
                variable_names=["tokens", "model_size", "unique_tokens"],
                equation_file="pysr_equations.csv",
                temp_equation_file=True,
                delete_tempfiles=False,
                verbosity=1,
                progress=True,
                multithreading=True,
                procs=0,  # Use all available cores
            )
            
            print(f"Starting PySR with {timeout_hours} hour timeout...")
            start_time = time.time()
            model.fit(X, y)
            fit_time = time.time() - start_time
            
            # Log the best equation found
            if hasattr(model, 'equations_') and len(model.equations_) > 0:
                best_eq = model.equations_.iloc[-1]
                equation_info = {
                    'method': 'PySR',
                    'equation': str(best_eq['equation']),
                    'complexity': int(best_eq['complexity']),
                    'loss': float(best_eq['loss']),
                    'fit_time': fit_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save equation info
                with open('pysr_best_equation.json', 'w') as f:
                    json.dump(equation_info, f, indent=2)
                    
                print(f"\\nPySR Best Equation: {equation_info['equation']}")
                print(f"Complexity: {equation_info['complexity']}, Loss: {equation_info['loss']}")
                
                return {'model': model, 'equation_info': equation_info}
            else:
                return {'model': model, 'equation_info': None}
                
        except Exception as e:
            print(f"PySR failed: {e}")
            return None
    
    def run_gplearn():
        if not GPLEARN_AVAILABLE:
            return None
        try:
            # Configure GPLearn with increased budget
            model = SymbolicRegressor(
                population_size=2000,  # Increased population
                generations=1000,  # Increased generations
                stopping_criteria=0.001,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=1.0,
                verbose=1,
                parsimony_coefficient=0.001,  # Lower parsimony for complex equations
                function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'),
                init_depth=(2, 6),
                init_method='half and half',
                const_range=(-10.0, 10.0),
                metric='mean absolute error',
                tournament_size=20,
                n_jobs=-1,  # Use all available cores
                random_state=42
            )
            
            print(f"Starting GPLearn with increased budget...")
            start_time = time.time()
            model.fit(X, y)
            fit_time = time.time() - start_time
            
            # Log the best equation found
            equation_info = {
                'method': 'GPLearn',
                'equation': str(model._program),
                'fitness': float(model.fitness_),
                'fit_time': fit_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save equation info
            with open('gplearn_best_equation.json', 'w') as f:
                json.dump(equation_info, f, indent=2)
                
            print(f"\\nGPLearn Best Equation: {equation_info['equation']}")
            print(f"Fitness: {equation_info['fitness']}")
            
            return {'model': model, 'equation_info': equation_info}
            
        except Exception as e:
            print(f"GPLearn failed: {e}")
            return None
    
    # Run both methods in parallel
    print("Starting parallel symbolic regression with PySR and GPLearn...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        pysr_future = executor.submit(run_pysr)
        gplearn_future = executor.submit(run_gplearn)
        
        # Get results as they complete
        pysr_result = None
        gplearn_result = None
        
        try:
            pysr_result = pysr_future.result(timeout=timeout_hours * 3600 + 300)  # 5 min buffer
        except Exception as e:
            print(f"PySR execution failed or timed out: {e}")
            
        try:
            gplearn_result = gplearn_future.result(timeout=timeout_hours * 3600 + 300)  # 5 min buffer
        except Exception as e:
            print(f"GPLearn execution failed or timed out: {e}")
    
    results['pysr'] = pysr_result
    results['gplearn'] = gplearn_result
    
    # Save combined results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'timeout_hours': timeout_hours,
        'pysr_success': pysr_result is not None,
        'gplearn_success': gplearn_result is not None,
        'pysr_equation': pysr_result['equation_info'] if pysr_result and pysr_result.get('equation_info') else None,
        'gplearn_equation': gplearn_result['equation_info'] if gplearn_result and gplearn_result.get('equation_info') else None
    }
    
    with open('symbolic_regression_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\n=== Symbolic Regression Summary ===")
    print(f"PySR Success: {summary['pysr_success']}")
    print(f"GPLearn Success: {summary['gplearn_success']}")
    
    return results

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
    Main function to evaluate scaling law programs
    
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
        "data_constrained_scaling_law", 
        train=not use_test_data
    )
    
    if not data_points:
        return get_failure_result()
    
    # Aggregate all loss values and data sizes from all data points
    all_loss_values = []
    all_tokens = []
    all_model_size = []
    all_unique_tokens = []
    
    for point in data_points:
        # Each point has loss_values (single element) and data_size (tokens)
        all_loss_values.extend(point["loss_values"])
        all_tokens.extend(point["data_size"])
        all_model_size.extend(point["model_size"])
        all_unique_tokens.extend(point['unique_tokens'])
        
    tokens = np.array(all_tokens)
    loss_values = np.array(all_loss_values)
    model_size = np.array(all_model_size)  
    unique_tokens = np.array(all_unique_tokens)
    
    if fitted_params is None and not return_metrics:
        # Training mode: fit the scaling law on all training data and return parameters
        start_time = time.time()
        fitted_params = run_with_timeout(
            fit_scaling_law,
            args=(tokens, model_size, unique_tokens, loss_values),
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
                args=(tokens, model_size, unique_tokens, loss_values),
                timeout_seconds=600
            )
        
        # Use parameters to predict and evaluate
        predicted_loss = run_with_timeout(
            scaling_law_func,
            args=(tokens, model_size, unique_tokens, fitted_params),
            timeout_seconds=600
        )
        
        # Evaluate fit quality
        metrics = evaluate_fit_quality(predicted_loss, loss_values)
        
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
    
    # Extract and display the fitted equation
    if fitted_params and 'model' in fitted_params:
        model = fitted_params['model']
        print(f"\n# Fitted Equation:")
        try:
            if hasattr(model, '_program'):  # GPlearn model
                print(f"GPlearn equation: {model._program}")
            elif hasattr(model, 'equations_'):  # PySR model
                if hasattr(model.equations_, 'iloc') and len(model.equations_) > 0:
                    best_eq = model.equations_.iloc[-1]
                    print(f"PySR equation: {best_eq['equation']}")
                    print(f"Complexity: {best_eq['complexity']}")
                    print(f"Loss: {best_eq['loss']}")
            elif hasattr(model, 'coef_'):  # Linear regression fallback
                print(f"Linear model coefficients: {model.coef_}")
                print(f"Linear model intercept: {model.intercept_}")
            else:
                print("Could not extract equation from model")
        except Exception as e:
            print(f"Error extracting equation: {e}")
    
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