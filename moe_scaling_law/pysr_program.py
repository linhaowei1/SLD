"""
PySR-based scaling law discovery for MoE scaling law
"""

import numpy as np
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def discover_scaling_law(train_total_params, train_num_experts, train_loss,
                         test_total_params, test_num_experts):
    """
    Uses PySR (or Linear Regression as a fallback) to discover an MoE scaling law from training data 
    and predicts loss on test data.

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

        # 2. Initialize and fit the model.
        model = None
        if PYSR_AVAILABLE:
            print("Info: PySR is available. Starting symbolic regression.")
            model = PySRRegressor(
                niterations=20,  # Match GPlearn generations  
                binary_operators=["+", "-", "*", "/", "min", "max"],  # Basic operators
                unary_operators=["sqrt", "log", "abs", "neg", "inv"],  # Simplified unary operators
                populations=31,  # Default value
                population_size=27,  # Default value (close to 27)
                ncycles_per_iteration=550,  # Default value (close to 380)
                timeout_in_seconds=300,  # Reasonable timeout
                maxsize=30,  # Default value (was 30)
                maxdepth=None,  # Use default (no depth limit)
                variable_names=["log_N", "log_E"], # N: Total Params, E: Experts
                verbosity=1,
                progress=True,
                random_state=42,
                elementwise_loss="L1DistLoss()"
            )
            model.fit(X_train, y_train)
        else:
            print("Warning: PySR not available. Using Linear Regression fallback.")
            model = LinearRegression()
            model.fit(X_train, y_train)

        # 3. Predict on the test data.
        predicted_loss = model.predict(X_test)
        predicted_loss = np.clip(predicted_loss, 0.1, 10.0)

        # 4. Get the discovered equation information.
        equation_info = "Equation could not be retrieved."
        if PYSR_AVAILABLE and hasattr(model, 'get_best') and len(model.equations) > 0:
            equation_info = str(model.get_best()["equation"])
        elif isinstance(model, LinearRegression):
            coeffs = model.coef_
            intercept = model.intercept_
            equation_info = (f"LinearRegression(Loss = {coeffs[0]:.4f}*log_N + "
                             f"{coeffs[1]:.4f}*log_E + {intercept:.4f})")

        return predicted_loss, equation_info

    except Exception as e:
        print(f"An error occurred in discover_scaling_law: {e}")
        fallback_prediction = np.full(test_total_params.shape, 3.0)
        error_message = f"Error during PySR/fallback fitting: {e}"
        return fallback_prediction, error_message