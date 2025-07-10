# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Initial program with a simple power law form that can be evolved
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize


def scaling_law_func(data_points, params):
    """
    A scaling law function to model the relationship between data points and loss.
    
    This starts as a simple power law but can evolve into more complex forms.
    
    Args:
        data_points: Array of data points (training data size)
        params: Array of parameters for the scaling law
        
    Returns:
        Predicted loss values
    """
    # Ensure we have enough parameters
    if len(params) < 4:
        # Pad with default values if needed
        padded_params = np.concatenate([params, np.ones(4 - len(params))])
        params = padded_params
    
    # Convert data_points to numpy array and handle edge cases
    x = np.asarray(data_points, dtype=float)
    
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-6
    x_safe = np.maximum(x, epsilon)
    
    # Simple power law: loss = a * (data_points + b)^(-c)
    # This is a common form for scaling laws
    a = abs(params[0]) + 0.1  # Ensure positive scale factor
    b = abs(params[1]) + 1.0  # Ensure positive offset
    c = abs(params[2]) + 0.1  # Ensure positive exponent
    d = abs(params[3]) + 0.1  # Ensure positive offset
    # Power law with offset
    loss = np.power(a * x_safe ** (-b) + c, d)
    
    return loss


def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the scaling law to data points and loss values
    
    Args:
        data_points: Array of data points (training data size)
        loss_values: Array of corresponding loss values
        initial_params: Initial parameter guess (optional)
        
    Returns:
        Optimized parameters
    """
    if initial_params is None:
        initial_params = np.random.rand(4)
    
    def objective(params):
        try:
            predicted = scaling_law_func(data_points, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6  # Return large error if computation fails
    
    result = minimize(objective, initial_params, method='Nelder-Mead')
    return result.x if result.success else initial_params


# Set the number of parameters this function expects
scaling_law_func.num_params = 4

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # 使用真实数据来测试缩放律函数
    # 从data文件夹加载CSV文件
    
    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    
    # 训练数据大小（对应CSV文件中的列）
    data_sizes = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600, 819200, 1638400])
    
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"处理数据集: {csv_file}")
        print(f"{'='*50}")
        
        # 加载CSV文件
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # 获取损失值列（排除第一列模型名、最后两列size和family）
        loss_columns = df.columns[1:-2]
        
        # 为每个模型拟合缩放律
        for idx, row in df.iterrows():
            model_name = row['config name']
            
            # 提取损失值（跳过第一列的初始值，因为数据大小为0）
            loss_values = []
            valid_data_sizes = []
            
            for i, col in enumerate(loss_columns[1:], 1):  # 跳过0大小的列
                loss_val = row[col]
                if pd.notna(loss_val) and loss_val > 0:  # 只使用有效的正数损失值
                    loss_values.append(float(loss_val))
                    valid_data_sizes.append(data_sizes[i-1])
            
            if len(loss_values) >= 4:  # 确保有足够的数据点进行拟合
                loss_values = np.array(loss_values)
                valid_data_sizes = np.array(valid_data_sizes)
                
                print(f"\n模型: {model_name}")
                print(f"数据点数量: {len(valid_data_sizes)}")
                print(f"数据大小范围: {valid_data_sizes[0]} - {valid_data_sizes[-1]}")
                print(f"损失值范围: {loss_values[-1]:.3f} - {loss_values[0]:.3f}")
                
                # 拟合缩放律
                fitted_params = fit_scaling_law(valid_data_sizes, loss_values)
                print(f"拟合参数: {fitted_params}")
                
                # 计算拟合质量（均方误差）
                predicted_loss = scaling_law_func(valid_data_sizes, fitted_params)
                mse = np.mean((predicted_loss - loss_values) ** 2)
                print(f"均方误差: {mse:.6f}")
                
                # 显示模型信息
                model_size = row['size']
                model_family = row['family']
                print(f"模型大小: {model_size:,} 参数")
                print(f"模型家族: {model_family}")
            else:
                print(f"\n模型 {model_name}: 数据点不足，跳过拟合")
