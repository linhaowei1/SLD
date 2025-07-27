"""
Unified data loading interface for both rectified_scaling_law and data_constrained_scaling_law
Provides train/test splits with 8:2 ratio using deterministic random seed
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path


def get_data_dir(app_name: str) -> str:
    """
    Get the data directory path for a specific application
    
    Args:
        app_name: Either 'rectified_scaling_law', 'data_constrained_scaling_law', 'moe_scaling_law', 'vocab_scaling_law', or 'domain_mixture_scaling_law'
    
    Returns:
        Path to the data directory
    """
    base_dir = Path(__file__).parent
    app_data_dir = base_dir / app_name / "data"
    
    if app_data_dir.exists():
        return str(app_data_dir)
    else:
        raise FileNotFoundError(f"Data directory not found for {app_name}: {app_data_dir}")


def load_rectified_scaling_law_data(data_dir: str, train: bool = True, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load rectified scaling law data with specific train/test split
    
    Args:
        data_dir: Path to data directory
        train: If True, return training data (200-409600 sizes); if False, return test data (819200 size)
        random_seed: Random seed for reproducible splits
        
    Returns:
        List of data points with model, dataset, loss_values, and data_size
    """
    np.random.seed(random_seed)
    
    # Data size array (columns after model name and 0 column)
    DATA_SIZES = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600, 819200])
    
    # Collect all data points from all datasets
    all_data_points = []
    csv_files = ["flan.csv", "gigaword.csv", "wikiword.csv"]
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        dataset_name = csv_file.replace('.csv', '')
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Get loss value columns (skip first column model name, skip data size 0 column)
            loss_columns = df.columns[2:]  # Start from 3rd column (skip model name and 0 column)
            
            # Create data points for each row (model)
            for idx, row in df.iterrows():
                model_name = row[df.columns[0]]  # First column is model name
                
                # For test data: only use 819200 data size
                if not train:
                    # Find 819200 column (last column)
                    if len(loss_columns) > 0:
                        last_col = loss_columns[-1]  # 819200 column
                        loss_val = row[last_col]
                        if pd.notna(loss_val) and loss_val > 0:
                            all_data_points.append({
                                "model": str(model_name),
                                "dataset": dataset_name,
                                "loss_values": np.array([float(loss_val)]),
                                "data_size": np.array([819200])
                            })
                else:
                    # For train data: use 200-409600 data sizes (exclude 819200)
                    loss_values = []
                    valid_data_sizes = []
                    
                    # Only use first 12 columns (exclude 819200)
                    for i, col in enumerate(loss_columns[:-1]):  # Exclude last column (819200)
                        loss_val = row[col]
                        if pd.notna(loss_val) and loss_val > 0:
                            loss_values.append(float(loss_val))
                            valid_data_sizes.append(DATA_SIZES[i])
                    
                    if len(loss_values) >= 4:  # Ensure enough data points for fitting
                        all_data_points.append({
                            "model": str(model_name),
                            "dataset": dataset_name,
                            "loss_values": np.array(loss_values),
                            "data_size": np.array(valid_data_sizes)
                        })
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    return all_data_points


def load_data_constrained_scaling_law_data(data_dir: str, train: bool = True, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load data-constrained scaling law data with train/test split
    
    Args:
        data_dir: Path to data directory
        train: If True, return training data; if False, return test data
        random_seed: Random seed for reproducible splits
        
    Returns:
        List of row-level data points with loss_values and data_size
    """
    np.random.seed(random_seed)
    
    file_path = os.path.join(data_dir, "data.csv")
    
    df = pd.read_csv(file_path)
    
    # Create data points for each row
    all_data_points = []
    
    for idx, row in df.iterrows():
        data_point = {
            "loss_values": np.array([row['loss']]),  # Single loss value per row
            "data_size": np.array([row['tokens']]),
            "unique_tokens": np.array([row['unique_tokens']]),
            "model_size": np.array([row['params']]),
        }
        all_data_points.append(data_point)
    
    # Split all data points into train/test (8:2 ratio)
    n_points = len(all_data_points)
    n_train = int(0.8 * n_points)
    
    # Shuffle indices for random split
    shuffled_indices = np.random.permutation(n_points)
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]
    
    # Select appropriate subset based on train parameter
    selected_indices = train_indices if train else test_indices
    return [all_data_points[i] for i in selected_indices]


def load_vocab_scaling_law_data(data_dir: str, train: bool = True, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load vocab scaling law data with train/test split
    
    Args:
        data_dir: Path to data directory
        train: If True, return training data; if False, return test data
        random_seed: Random seed for reproducible splits
        
    Returns:
        List of data points with vocab scaling features and Lossu values
    """
    np.random.seed(random_seed)
    
    file_path = os.path.join(data_dir, "data.csv")
    
    df = pd.read_csv(file_path)
    
    # Create data points for each row
    all_data_points = []
    
    for idx, row in df.iterrows():
        data_point = {
            "lossu_values": np.array([row['Lossu']]),  # Single Lossu value per row
            "vocab_size": np.array([row['vocab_size']]),
            "Non_vocab_parameters": np.array([row['Non_vocab_parameters']]),
            "num_characters": np.array([row['num_characters']]),
        }
        all_data_points.append(data_point)
    
    # Split all data points into train/test (8:2 ratio)
    n_points = len(all_data_points)
    n_train = int(0.8 * n_points)
    
    # Shuffle indices for random split
    shuffled_indices = np.random.permutation(n_points)
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]
    
    # Select appropriate subset based on train parameter
    selected_indices = train_indices if train else test_indices
    return [all_data_points[i] for i in selected_indices]


def load_moe_scaling_law_data(data_dir: str, train: bool = True, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load MoE scaling law data with train/test split (filtered for step = 249000)
    
    Args:
        data_dir: Path to data directory
        train: If True, return training data; if False, return test data
        random_seed: Random seed for reproducible splits
        
    Returns:
        List of data points with loss_validation, num_experts, and total_parameter_count
    """
    np.random.seed(random_seed)
    
    file_path = os.path.join(data_dir, "data.csv")
    
    df = pd.read_csv(file_path)
    
    # Filter for step = 249000 only
    df = df[df['step'] == 249000]
    
    # Create data points for each row
    all_data_points = []
    
    for idx, row in df.iterrows():
        data_point = {
            "loss_values": np.array([row['loss_validation']]),  # Single loss value per row
            "num_experts": np.array([row['num_experts']]),
            "total_parameter_count": np.array([row['dense_parameter_count']]),
        }
        all_data_points.append(data_point)
    
    # Split all data points into train/test (8:2 ratio)
    n_points = len(all_data_points)
    n_train = int(0.8 * n_points)
    
    # Shuffle indices for random split
    shuffled_indices = np.random.permutation(n_points)
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]
    
    # Select appropriate subset based on train parameter
    selected_indices = train_indices if train else test_indices
    return [all_data_points[i] for i in selected_indices]


def load_domain_mixture_scaling_law_data(data_dir: str, train: bool = True, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load domain mixture scaling law data with train/test split
    
    Args:
        data_dir: Path to data directory
        train: If True, return training data; if False, return test data
        random_seed: Random seed for reproducible splits
        
    Returns:
        List of data points with proportions and loss values for all 5 domains
    """
    import json
    
    np.random.seed(random_seed)
    
    file_path = os.path.join(data_dir, "all_models_data.json")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Collect all data points from all model sizes
    all_data_points = []
    
    for model_size, model_data in data.items():
        if model_size == "1B":  # Skip empty 1B data
            continue
            
        # Use train or eval data based on train parameter
        split_data = model_data["train" if train else "eval"]
        
        for point in split_data:
            data_point = {
                "model_size": model_size,
                "proportions": np.array(point["proportions"]),  # 5 domain proportions
                "loss_values": np.array(point["loss"]),  # 5 domain loss values
                "ratio": point["ratio"]  # String identifier
            }
            all_data_points.append(data_point)
    
    return all_data_points


def load_data(app_name: str, train: bool = True, random_seed: int = 42) -> Any:
    """
    Unified data loading interface for all applications
    
    Args:
        app_name: Either 'rectified_scaling_law', 'data_constrained_scaling_law', 'moe_scaling_law', 'vocab_scaling_law', or 'domain_mixture_scaling_law'
        train: If True, return training data; if False, return test data
        random_seed: Random seed for reproducible splits
        
    Returns:
        Data in the format expected by the respective application
    """
    data_dir = get_data_dir(app_name)
    
    if app_name == "rectified_scaling_law":
        return load_rectified_scaling_law_data(data_dir, train, random_seed)
    elif app_name == "data_constrained_scaling_law":
        return load_data_constrained_scaling_law_data(data_dir, train, random_seed)
    elif app_name == "moe_scaling_law":
        return load_moe_scaling_law_data(data_dir, train, random_seed)
    elif app_name == "vocab_scaling_law":
        return load_vocab_scaling_law_data(data_dir, train, random_seed)
    elif app_name == "domain_mixture_scaling_law":
        return load_domain_mixture_scaling_law_data(data_dir, train, random_seed)
    else:
        raise ValueError(f"Unknown application name: {app_name}. Must be 'rectified_scaling_law', 'data_constrained_scaling_law', 'moe_scaling_law', 'vocab_scaling_law', or 'domain_mixture_scaling_law'")


if __name__ == "__main__":
    # Test the unified data loader
    print("Testing rectified_scaling_law data loader:")
    print("=" * 50)
    
    train_data = load_data("rectified_scaling_law", train=True)
    test_data = load_data("rectified_scaling_law", train=False)
    
    for dataset_name, models in train_data.items():
        print(f"Dataset: {dataset_name}")
        print(f"  Train models: {len(models)}")
        print(f"  Test models: {len(test_data.get(dataset_name, {}))}")
    
    print("\nTesting data_constrained_scaling_law data loader:")
    print("=" * 50)
    
    train_tokens, train_model_size, train_unique_tokens, train_loss = load_data("data_constrained_scaling_law", train=True)
    test_tokens, test_model_size, test_unique_tokens, test_loss = load_data("data_constrained_scaling_law", train=False)
    
    print(f"Train samples: {len(train_tokens)}")
    print(f"Test samples: {len(test_tokens)}")
    print(f"Total samples: {len(train_tokens) + len(test_tokens)}")