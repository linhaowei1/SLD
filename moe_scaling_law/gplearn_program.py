"""
GPlearn-based scaling law discovery for MoE scaling law
"""

import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def discover_scaling_law(train_total_params, train_num_experts, train_loss,
                         test_total_params, test_num_experts):
    """
    Uses gplearn to discover an MoE scaling law from training data and predicts loss on test data.

    Args:
        train_total_params (np.ndarray): Array of total parameter counts from training data (N).
        train_num_experts (np.ndarray): Array of number of experts from training data (E).
        train_loss (np.ndarray): Array of loss values from training data.
        test_total_params (np.ndarray): Array of total parameter counts from test data (N).
        test_num_experts (np.ndarray): Array of number of experts from test data (E).

    Returns:
        tuple[np.ndarray, str]: Tuple containing predicted loss values for test set and string representation of discovered law.
    """
    try:
        # 1. Prepare feature matrices using log transform.
        epsilon = 1e-10
        X_train = np.column_stack([
            np.log(train_total_params + epsilon),
            np.log(train_num_experts + epsilon)
        ])
        y_train = np.array(train_loss)

        X_test = np.column_stack([
            np.log(test_total_params + epsilon),
            np.log(test_num_experts + epsilon)
        ])

        # 2. Initialize and fit SymbolicRegressor
        est_gp = SymbolicRegressor(
            population_size=1000, generations=20, tournament_size=20, stopping_criteria=0.0,
            const_range=(-10.0, 10.0), init_depth=(2, 6), init_method='half and half',
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'),
            metric='mean absolute error', parsimony_coefficient=0.001,
            p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01,
            p_point_replace=0.05, max_samples=1.0, feature_names=['log_N', 'log_E'], # N: Total Params, E: Experts
            warm_start=False, low_memory=False, n_jobs=20, verbose=1, random_state=42
        )
        est_gp.fit(X_train, y_train)

        # 3. Make predictions
        predicted_loss = est_gp.predict(X_test)
        predicted_loss = np.clip(predicted_loss, 0.1, 10.0)

        # 4. Get the discovered equation
        equation_info = str(est_gp._program)

        return predicted_loss, equation_info

    except Exception as e:
        print(f"An error occurred in discover_scaling_law: {e}")
        fallback_prediction = np.full(test_total_params.shape, 3.0) 
        error_message = f"Error during GP fitting: {e}"
        return fallback_prediction, error_message