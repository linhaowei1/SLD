"""
Unified data loading interface for scaling law discovery.

Dynamically loads data from task-specific loader files located in the 'data/' subdirectory.
This approach keeps the data loading logic for each task isolated and clean.
"""
import numpy as np
import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple

def get_data_dir(app_name: str) -> Path:
    """
    Get the data directory path for a specific application.
    
    Args:
        app_name: The name of the scaling law task.
    
    Returns:
        Path object to the data directory.
    """
    base_dir = Path(__file__).parent
    app_data_dir = base_dir / "data" / app_name
    
    if not app_data_dir.exists():
        raise FileNotFoundError(f"Data directory not found for {app_name}: {app_data_dir}")
    return app_data_dir

def load_data(
    app_name: str,
    train: bool = True,
    random_seed: int = 42
) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    """
    Unified data loading interface. Imports and runs the appropriate data loader.

    Each loader is expected to return a dictionary mapping a control group key
    to a tuple of (features, labels), where:
    - features (X): A numpy array of shape (n_samples, n_features).
    - labels (y): A numpy array of shape (n_samples,).

    Args:
        app_name: The name of the task (e.g., 'rectified_scaling_law').
        train: If True, load training data; otherwise, load test data.
        random_seed: Seed for reproducible data splitting.

    Returns:
        A dictionary containing the prepared data for each group.
    """
    loader_file_name = f"{app_name}_loader.py"
    loader_path = Path(__file__).parent / "data" / loader_file_name

    if not loader_path.exists():
        raise FileNotFoundError(f"Data loader not found for task '{app_name}' at {loader_path}")

    # Dynamically import the loader module
    spec = importlib.util.spec_from_file_location(f"data.{app_name}_loader", loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    # The data directory for the specific task's CSV/JSON files
    data_dir = get_data_dir(app_name)

    # Call the loader function within the imported module
    return loader_module.load_data_for_task(data_dir, train, random_seed)

if __name__ == '__main__':
    # Example of how to use the new loader
    ALL_TASKS = [
        "rectified_scaling_law",
        "data_constrained_scaling_law",
        "moe_scaling_law",
        "vocab_scaling_law",
        "domain_mixture_scaling_law",
        "lr_scaling_law",
        "vanilla_scaling_law",
    ]

    for task in ALL_TASKS:
        print(f"\n--- Testing '{task}' ---")
        try:
            train_data = load_data(task, train=True)
            test_data = load_data(task, train=False)
            
            # Get the first group to inspect its shape
            first_group_key = next(iter(train_data))
            X_train, y_train = train_data[first_group_key]
            
            print(f"Train groups: {len(train_data)}. First group '{first_group_key}' shape: X={X_train.shape}, y={y_train.shape}")
            
            if test_data:
                first_test_key = next(iter(test_data))
                X_test, y_test = test_data[first_test_key]
                print(f"Test groups: {len(test_data)}. First group '{first_test_key}' shape: X={X_test.shape}, y={y_test.shape}")
            else:
                print("Test data is empty (as expected for some tasks).")

        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"Error loading data for task '{task}': {e}")