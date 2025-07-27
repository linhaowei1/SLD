"""
PySR-based scaling law discovery for vocab scaling law
"""

import numpy as np
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
from sklearn.linear_model import LinearRegression
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')


def discover_scaling_law(train_non_vocab_params, train_vocab_size, train_num_chars, train_lossu,
                         test_non_vocab_params, test_vocab_size, test_num_chars):
    """
    Uses PySR (or Linear Regression as a fallback) to discover a vocabulary scaling law from training data 
    and predicts loss on test data.

    Args:
        train_non_vocab_params (np.ndarray): Array of non-vocabulary parameter counts from training data.
        train_vocab_size (np.ndarray): Array of vocabulary sizes from training data.
        train_num_chars (np.ndarray): Array of character set sizes from training data.
        train_lossu (np.ndarray): Array of loss values from training data.
        test_non_vocab_params (np.ndarray): Array of non-vocabulary parameter counts from test data.
        test_vocab_size (np.ndarray): Array of vocabulary sizes from test data.
        test_num_chars (np.ndarray): Array of character set sizes from test data.

    Returns:
        tuple[np.ndarray, str]: Tuple containing predicted loss values for test set and string representation of discovered law.
    """
    try:
        # 1. Prepare feature matrices using log transform.
        epsilon = 1e-10
        X_train = np.column_stack([
            np.log(train_non_vocab_params + epsilon),
            np.log(train_vocab_size + epsilon),
            np.log(train_num_chars + epsilon)
        ])
        y_train = np.array(train_lossu)

        X_test = np.column_stack([
            np.log(test_non_vocab_params + epsilon),
            np.log(test_vocab_size + epsilon),
            np.log(test_num_chars + epsilon)
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
                variable_names=["log_N", "log_V", "log_C"], # N: Non-vocab, V: Vocab, C: Chars
                temp_equation_file=True, # Recommended for parallel computation
                delete_tempfiles=True,
                verbosity=1,  # Set to 0 to reduce unnecessary output
                progress=True,
                random_state=42,
                elementwise_loss="L1DistLoss()"  # L1 loss is MAE
            )
            model.fit(X_train, y_train)
        else:
            # Use Linear Regression as a fallback if PySR is not installed.
            print("Warning: PySR not available. Using Linear Regression fallback.")
            model = LinearRegression()
            model.fit(X_train, y_train)

        # 3. Predict on the test data.
        predicted_loss = model.predict(X_test)
        predicted_loss = np.clip(predicted_loss, 0.1, 10.0)

        # 4. Get the discovered equation information.
        equation_info = "Equation could not be retrieved."
        if PYSR_AVAILABLE and hasattr(model, 'get_best') and len(model.equations) > 0:
            # Get the best equation from the hall of fame.
            equation_info = str(model.get_best()["equation"])
        elif isinstance(model, LinearRegression):
            coeffs = model.coef_
            intercept = model.intercept_
            equation_info = (f"LinearRegression(Loss = {coeffs[0]:.4f}*log_N + "
                             f"{coeffs[1]:.4f}*log_V + {coeffs[2]:.4f}*log_C + {intercept:.4f})")

        return predicted_loss, equation_info

    except Exception as e:
        print(f"An error occurred in discover_scaling_law: {e}")
        # Fallback logic for any error during the process.
        fallback_prediction = np.full(test_vocab_size.shape, 3.0) 
        error_message = f"Error during PySR/fallback fitting: {e}"
        return fallback_prediction, error_message