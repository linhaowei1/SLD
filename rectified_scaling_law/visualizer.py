"""
Visualizer for comparing different scaling law models
Compares two scaling law programs by fitting them to the same data and plotting results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import importlib.util
import sys
from typing import Tuple, List, Dict


class ScalingLawComparator:
    """
    Class to compare two different scaling law implementations
    """
    
    def __init__(self, program1_path: str, program2_path: str):
        """
        Initialize with paths to two scaling law programs
        
        Args:
            program1_path: Path to first scaling law program
            program2_path: Path to second scaling law program  
        """
        self.program1_path = program1_path
        self.program2_path = program2_path
        self.model1 = self._load_program(program1_path, "model1")
        self.model2 = self._load_program(program2_path, "model2")
        
        # Standard data sizes used in the datasets
        self.data_sizes = np.array([
            200, 400, 800, 1600, 3200, 6400, 12800, 25600,
            51200, 102400, 204800, 409600, 819200, 1638400
        ], dtype=float)
        
        # Colors for different models  
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
    def _load_program(self, program_path: str, module_name: str):
        """Load a scaling law program as a module"""
        if not os.path.exists(program_path):
            raise FileNotFoundError(f"Program file not found: {program_path}")
            
        spec = importlib.util.spec_from_file_location(module_name, program_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def load_data(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """Load all CSV data files"""
        csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
        datasets = {}
        
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            if os.path.exists(file_path):
                datasets[csv_file.replace('.csv', '')] = pd.read_csv(file_path)
            else:
                print(f"Warning: Data file not found: {file_path}")
                
        return datasets
    
    def extract_model_data(self, row: pd.Series, loss_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract valid data points and loss values from a model row"""
        valid_data_sizes = []
        loss_values = []
        
        for i, col in enumerate(loss_columns[1:], 1):  # Skip first column (data size 0)
            loss_val = row[col]
            if pd.notna(loss_val) and loss_val > 0:
                loss_values.append(float(loss_val))
                valid_data_sizes.append(self.data_sizes[i-1])
                
        return np.array(valid_data_sizes), np.array(loss_values)
    
    def fit_and_predict(self, data_points: np.ndarray, loss_values: np.ndarray, 
                       model, model_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit a model and generate predictions"""
        try:
            # Fit the model
            fitted_params = model.fit_scaling_law(data_points, loss_values)
            
            # Generate smooth curve for plotting
            x_smooth = np.logspace(np.log10(data_points.min()), 
                                 np.log10(data_points.max()), 100)
            y_smooth = model.scaling_law_func(x_smooth, fitted_params)
            
            # Calculate MSE on original data points
            y_pred = model.scaling_law_func(data_points, fitted_params)
            mse = np.mean((y_pred - loss_values) ** 2)
            
            return x_smooth, y_smooth, mse
            
        except Exception as e:
            print(f"Error fitting {model_name}: {e}")
            return None, None, float('inf')

    def plot_dataset_comparison(self, dataset_name: str, dataset_results: List[Dict],
                              save_plots: bool = True, show_plots: bool = True):
        """Plot all models for a single dataset on one figure"""
        
        if not dataset_results:
            print(f"No results to plot for dataset: {dataset_name}")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot each model on both linear and log scale
        for i, result in enumerate(dataset_results):
            model_name = result['model_name']
            data_points = result['data_points']
            loss_values = result['loss_values']
            x1, y1, mse1 = result['model1_fit']
            x2, y2, mse2 = result['model2_fit']
            
            color = self.colors[i % len(self.colors)]
            
            # Linear scale plot
            ax1.scatter(data_points, loss_values, color=color, s=50, zorder=5, 
                       alpha=0.8, label=f'{model_name} (Data)')
            if x1 is not None:
                ax1.plot(x1, y1, color=color, linestyle='-', linewidth=2, alpha=0.8,
                        label=f'{model_name} - Model 1 (MSE: {mse1:.2e})')
            if x2 is not None:
                ax1.plot(x2, y2, color=color, linestyle='--', linewidth=2, alpha=0.8,
                        label=f'{model_name} - Model 2 (MSE: {mse2:.2e})')
            
            # Log scale plot
            ax2.scatter(data_points, loss_values, color=color, s=50, zorder=5, 
                       alpha=0.8, label=f'{model_name} (Data)')
            if x1 is not None:
                ax2.plot(x1, y1, color=color, linestyle='-', linewidth=2, alpha=0.8,
                        label=f'{model_name} - Model 1 (MSE: {mse1:.2e})')
            if x2 is not None:
                ax2.plot(x2, y2, color=color, linestyle='--', linewidth=2, alpha=0.8,
                        label=f'{model_name} - Model 2 (MSE: {mse2:.2e})')
        
        # Configure linear scale plot
        ax1.set_xlabel('Training Data Size')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{dataset_name} Dataset\nLinear Scale')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Configure log scale plot
        ax2.set_xlabel('Training Data Size')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{dataset_name} Dataset\nLog Scale')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            safe_name = f"{dataset_name}".replace(' ', '_').replace('/', '_')
            plt.savefig(f'plots/dataset_comparison_{safe_name}.png', dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def compare_all_datasets(self, data_dir: str = "data", save_plots: bool = True, 
                           show_plots: bool = False):
        """Compare models on all datasets and models"""
        datasets = self.load_data(data_dir)
        
        if not datasets:
            print("No datasets found!")
            return
            
        all_results = []
        
        print(f"Comparing models:")
        print(f"Model 1: {self.program1_path}")
        print(f"Model 2: {self.program2_path}")
        print("="*60)
        
        for dataset_name, df in datasets.items():
            print(f"\nProcessing dataset: {dataset_name}")
            print("-" * 40)
            
            # Get loss columns (exclude first column and last two columns)
            loss_columns = df.columns[1:-2]
            
            dataset_results = []
            
            for _, row in df.iterrows():
                model_name = row['config name']
                data_points, loss_values = self.extract_model_data(row, loss_columns)
                
                if len(data_points) >= 4:
                    # Fit both models
                    x1, y1, mse1 = self.fit_and_predict(data_points, loss_values, self.model1, "Model 1")
                    x2, y2, mse2 = self.fit_and_predict(data_points, loss_values, self.model2, "Model 2")
                    
                    # Store results for this model
                    result = {
                        'model_name': model_name,
                        'data_points': data_points,
                        'loss_values': loss_values,
                        'model1_fit': (x1, y1, mse1),
                        'model2_fit': (x2, y2, mse2)
                    }
                    dataset_results.append(result)
                    
                    # Print individual model comparison
                    print(f"\n{model_name}")
                    print(f"Data points: {len(data_points)}")
                    print(f"Data size range: {data_points.min():.0f} - {data_points.max():.0f}")
                    print(f"Loss range: {loss_values.min():.4f} - {loss_values.max():.4f}")
                    if x1 is not None and x2 is not None:
                        print(f"Model 1 MSE: {mse1:.6e}")
                        print(f"Model 2 MSE: {mse2:.6e}")
                        print(f"Better model: {'Model 1' if mse1 < mse2 else 'Model 2'} (ratio: {max(mse1, mse2) / min(mse1, mse2):.2f}x)")
                        
                        all_results.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'model_size': row['size'],
                            'model_family': row['family'],
                            'mse1': mse1,
                            'mse2': mse2,
                            'better_model': 1 if mse1 < mse2 else 2
                        })
            
            # Plot all models for this dataset on one figure
            if dataset_results:
                self.plot_dataset_comparison(dataset_name, dataset_results, save_plots, show_plots)
        
        # Summary statistics
        if all_results:
            results_df = pd.DataFrame(all_results)
            print(f"\n{'='*60}")
            print("SUMMARY STATISTICS")
            print(f"{'='*60}")
            print(f"Total comparisons: {len(results_df)}")
            print(f"Model 1 wins: {(results_df['better_model'] == 1).sum()}")
            print(f"Model 2 wins: {(results_df['better_model'] == 2).sum()}")
            print(f"Model 1 win rate: {(results_df['better_model'] == 1).mean():.1%}")
            print(f"Model 2 win rate: {(results_df['better_model'] == 2).mean():.1%}")
            
            # Save results
            if save_plots:
                results_df.to_csv('plots/comparison_results.csv', index=False)
                print(f"\nResults saved to plots/comparison_results.csv")
            
            return results_df
        else:
            print("No successful comparison results")
            return None


def main():
    """
    Main function to demonstrate usage
    """
    # Example usage
    program1_path = "init_program.py"
    program2_path = "openevolve_output/best/best_program.py"
    
    try:
        comparator = ScalingLawComparator(program1_path, program2_path)
        results = comparator.compare_all_datasets(save_plots=True, show_plots=False)
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 