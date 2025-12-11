"""
Unified data loading interface for scaling law discovery.

Dynamically loads data from the Hugging Face Hub repository 'pkuHaowei/sldbench'.
This approach centralizes data access and ensures consistency.
"""
import numpy as np
import datasets
from typing import Dict, Any, Tuple

# --- Configuration ---

HUB_REPO_ID = "pkuHaowei/sldbench"

# Defines the schema for each task, mapping feature/target names from the Hub
# to the columns in the dataset.
TASK_SCHEMA_MAP = {
    "data_constrained_scaling_law": {
        "feature_names": ["unique_tokens", "params", "tokens"],
        "target_name": "loss",
    },
    "domain_mixture_scaling_law": {
        "feature_names": [f"proportion_domain_{i+1}" for i in range(5)],
        "target_name": [f"loss_domain_{i+1}" for i in range(5)],
    },
    "lr_bsz_scaling_law": {
        "feature_names": ["lr", "bsz", "data_size", "non_embedding_param_size"],
        "target_name": "lm_loss",
    },
    "moe_scaling_law": {
        "feature_names": ["num_experts", "dense_parameter_count"],
        "target_name": "loss_validation",
    },
    "sft_scaling_law": {
        "feature_names": ["sft_data_size"],
        "target_name": "sft_loss",
    },
    "vocab_scaling_law": {
        "feature_names": ["non_vocab_parameters", "vocab_size", "num_characters"],
        "target_name": "unigram_normalized_loss",
    },
    "parallel_scaling_law": {
        "feature_names": ["num_params", "parallel_size"],
        "target_name": "loss"
    },
    "easy_question_scaling_law": {
        "feature_names": ["log_flops"],
        "target_name": "brier_score",
    }
}

def load_data(
    app_name: str,
    train: bool = True,
) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    """
    Unified data loading interface. Loads and processes data from Hugging Face Hub.

    Each task's dataset is grouped by a 'group' key. The function returns a 
    dictionary mapping each group key to a tuple of (features, labels).
    - features (X): A numpy array of shape (n_samples, n_features).
    - labels (y): A numpy array of shape (n_samples,) or (n_samples, n_targets).

    Args:
        app_name: The name of the task (e.g., 'sft_scaling_law').
        train: If True, load training data; otherwise, load test data.

    Returns:
        A dictionary containing the prepared data, structured by group.
    """
    if app_name not in TASK_SCHEMA_MAP:
        raise ValueError(f"Task '{app_name}' not found in TASK_SCHEMA_MAP. Available tasks: {list(TASK_SCHEMA_MAP.keys())}")

    split = 'train' if train else 'test'
    schema = TASK_SCHEMA_MAP[app_name]
    
    try:
        # Load the specific task dataset from the Hugging Face Hub
        dataset = datasets.load_dataset(HUB_REPO_ID, name=app_name, split=split)
    except Exception as e:
        raise IOError(f"Failed to load dataset '{app_name}' with split '{split}' from '{HUB_REPO_ID}'. Reason: {e}")

    # Ensure target_name is a list for consistent processing
    feature_names = schema["feature_names"]
    target_names = schema["target_name"]
    if not isinstance(target_names, list):
        target_names = [target_names]

    processed_data = {}
    
    # The datasets are partitioned by a 'group' column
    unique_groups = sorted(list(set(dataset['group'])))

    for group_key in unique_groups:
        # Filter the dataset for the current group
        group_data = dataset.filter(lambda example: example['group'] == group_key)
        
        # Extract features (X) and stack them into a single numpy array
        X_list = [np.array(group_data[fname]) for fname in feature_names]
        X = np.stack(X_list, axis=1)
        
        # Extract targets (y)
        y_list = [np.array(group_data[tname]) for tname in target_names]
        y_stacked = np.stack(y_list, axis=1)
        
        # Squeeze the last dimension if there is only one target
        y = y_stacked.squeeze(axis=1) if y_stacked.shape[1] == 1 else y_stacked
        
        processed_data[group_key] = (X, y)

    return processed_data

if __name__ == '__main__':
    # Example of how to use the new loader
    # The list of tasks is now derived directly from the schema map
    ALL_TASKS = list(TASK_SCHEMA_MAP.keys())

    for task in ALL_TASKS:
        print(f"\n--- Testing '{task}' ---")
        try:
            # Load training data
            train_data = load_data(task, train=True)
            print(f"Successfully loaded training data from Hugging Face repo '{HUB_REPO_ID}'.")
            
            # Inspect the first group's shape
            first_group_key = next(iter(train_data))
            X_train, y_train = train_data[first_group_key]
            print(f"Train groups: {len(train_data)}. First group '{first_group_key}' shape: X={X_train.shape}, y={y_train.shape}")
            
            # Load test data
            test_data = load_data(task, train=False)
            if test_data:
                first_test_key = next(iter(test_data))
                X_test, y_test = test_data[first_test_key]
                print(f"Test groups: {len(test_data)}. First group '{first_test_key}' shape: X={X_test.shape}, y={y_test.shape}")
            else:
                print("Test data is empty.")

        except (ValueError, IOError, KeyError) as e:
            print(f"Error loading data for task '{task}': {e}")