"""
Evaluator for scaling law discovery programs
用于评估缩放律发现程序性能的评估器
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

# 数据大小列表（跳过第一列的0，从200开始）
DATA_SIZES = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600, 819200, 1638400])

def get_failure_result() -> Dict[str, float]:
    """
    返回失败情况下的标准结果，确保与成功情况有相同的key结构
    """
    # 数据集名称列表
    dataset_names = ["flan", "gigaword", "wmt19"]
    
    # 使用100000作为最差MSE分数（很大的MSE值）
    worst_mse = 100000.0
    # 对应的最差overall_score（接近0）
    worst_score = 1.0 / (1.0 + worst_mse)
    
    result = {
        "mse": worst_mse,
        "combined_score": worst_score,
    }
    
    # 为每个可能的数据集添加失败分数
    for dataset_name in dataset_names:
        result[f"mse_{dataset_name}"] = worst_mse
    
    return result

def load_real_datasets(data_dir="data"):
    """
    从CSV文件加载真实数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        包含所有数据集和模型数据的字典
    """
    datasets = {}
    
    # CSV文件列表
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    
    for csv_file in csv_files:
        dataset_name = csv_file.replace(".csv", "")
        file_path = os.path.join(data_dir, csv_file)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 获取损失值列（跳过第一列config name，跳过数据大小0的列，排除最后两列size和family）
            loss_columns = df.columns[2:-2]  # 从第3列开始（跳过config name和0列），到倒数第3列
            
            # 初始化数据集
            datasets[dataset_name] = {}
            
            # 为每个模型创建数据
            for idx, row in df.iterrows():
                model_name = row['config name']
                model_size = row['size'] 
                model_family = row['family']
                
                # 提取损失值
                loss_values = []
                valid_data_sizes = []
                
                for i, col in enumerate(loss_columns):
                    loss_val = row[col]
                    if pd.notna(loss_val) and loss_val > 0:  # 只使用有效的正数损失值
                        loss_values.append(float(loss_val))
                        valid_data_sizes.append(DATA_SIZES[i])
                
                if len(loss_values) >= 4:  # 确保有足够的数据点进行拟合
                    datasets[dataset_name][model_name] = {
                        "data_points": np.array(valid_data_sizes),
                        "loss_values": np.array(loss_values),
                        "model_size": model_size,
                        "model_family": model_family
                    }
            
        except Exception as e:
            continue
    
    return datasets

# 动态确定数据目录路径
def get_data_dir():
    """获取数据目录的正确路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    # 如果当前目录下有data文件夹，优先使用
    if os.path.exists("data"):
        return "data"
    # 否则使用脚本同目录下的data文件夹
    elif os.path.exists(data_dir):
        return data_dir
    else:
        # 尝试上级目录
        parent_data_dir = os.path.join(os.path.dirname(script_dir), "data")
        if os.path.exists(parent_data_dir):
            return parent_data_dir
        else:
            raise FileNotFoundError("找不到data目录，请确保data文件夹存在")

# 加载真实数据集
try:
    data_directory = get_data_dir()
    TEST_DATASETS = load_real_datasets(data_directory)
except Exception as e:
    TEST_DATASETS = {}


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    使用超时运行函数
    
    Args:
        func: 要运行的函数
        args: 函数参数
        kwargs: 关键字参数
        timeout_seconds: 超时时间（秒）
        
    Returns:
        函数结果或抛出 TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"函数运行超时，超过 {timeout_seconds} 秒")


def safe_float(value):
    """安全地将值转换为浮点数"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate_fit_quality(predicted_values: np.ndarray, true_values: np.ndarray) -> Dict[str, float]:
    """
    评估拟合质量
    
    Args:
        predicted_values: 预测值
        true_values: 真实值
        
    Returns:
        包含各种评估指标的字典
    """
    try:
        # 确保输入为numpy数组
        predicted = np.asarray(predicted_values, dtype=float)
        true = np.asarray(true_values, dtype=float)
        
        # 检查形状匹配
        if predicted.shape != true.shape:
            return {"error": "预测值和真实值形状不匹配"}
            
        # 过滤掉无效值
        valid_mask = ~(np.isnan(predicted) | np.isnan(true) | np.isinf(predicted) | np.isinf(true))
        if not np.any(valid_mask):
            return {"error": "所有预测值都无效"}
            
        pred_filtered = predicted[valid_mask]
        true_filtered = true[valid_mask]
        
        if len(pred_filtered) < 2:
            return {"error": "有效数据点不足"}
        
        # 计算评估指标
        mse = mean_squared_error(true_filtered, pred_filtered)
        rmse = np.sqrt(mse)
        
        # R² 分数
        r2 = r2_score(true_filtered, pred_filtered)
        
        # 皮尔逊相关系数
        correlation, _ = pearsonr(true_filtered, pred_filtered)
        
        # 平均绝对百分比误差
        mape = np.mean(np.abs((true_filtered - pred_filtered) / true_filtered)) * 100
        
        # 归一化均方根误差
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
        return {"error": f"评估过程中出错: {str(e)}"}


def evaluate(program_path: str) -> Dict[str, float]:
    """
    评估缩放律程序的主函数
    
    Args:
        program_path: 程序文件路径
        
    Returns:
        包含评估指标的字典
    """
    try:
        # 加载程序
        spec = importlib.util.spec_from_file_location("scaling_program", program_path)
        if spec is None or spec.loader is None:
            return get_failure_result()
            
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # 检查必需的函数是否存在
        if not hasattr(program, "scaling_law_func"):
            return get_failure_result()
            
        if not hasattr(program, "fit_scaling_law"):
            return get_failure_result()
        
        scaling_law_func = program.scaling_law_func
        fit_scaling_law = program.fit_scaling_law
        
        # 检查是否有可用的测试数据集
        if not TEST_DATASETS:
            return get_failure_result()
        
        # 在多个测试数据集和模型上评估
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
                    
                    # 使用超时拟合缩放律
                    start_time = time.time()
                    fitted_params = run_with_timeout(
                        fit_scaling_law, 
                        args=(data_points, true_loss),
                        timeout_seconds=600
                    )
                    fit_time = time.time() - start_time
                    
                    # 生成预测
                    predicted_loss = run_with_timeout(
                        scaling_law_func,
                        args=(data_points, fitted_params),
                        timeout_seconds=600
                    )
                    
                    # 评估拟合质量
                    metrics = evaluate_fit_quality(predicted_loss, true_loss)
                    
                    if "error" in metrics:
                        continue
                    
                    # 只使用MSE作为评估指标
                    mse_value = metrics["mse"]
                    
                    dataset_model_scores.append(mse_value)
                    all_scores.append(mse_value)
                    
                except TimeoutError as e:
                    pass
                except Exception as e:
                    pass
            
            # 计算数据集平均MSE
            if dataset_model_scores:
                dataset_avg_mse = np.mean(dataset_model_scores)
                dataset_scores[dataset_name] = float(dataset_avg_mse)
            else:
                dataset_scores[dataset_name] = 100000.0  # 失败时设为最差分数（有限值）
        
        # 如果所有数据集都失败了
        if not all_scores:
            return get_failure_result()
        
        # 计算MSE统计信息
        mean_mse = float(np.mean(all_scores))
        # std_mse = float(np.std(all_scores))
        # min_mse = float(np.min(all_scores))
        # max_mse = float(np.max(all_scores))
        
        # 计算overall_score：使用1/(1+mse)使得mse越小，分数越大（越大越好）
        overall_score = 1.0 / (1.0 + mean_mse)
        
        # 准备返回结果 - 确保格式与get_failure_result()一致
        result = {
            "mse": mean_mse,
            "combined_score": overall_score,
        }
        
        # 为所有可能的数据集添加MSE分数
        dataset_names = ["flan", "gigaword", "wmt19"]
        for dataset_name in dataset_names:
            if dataset_name in dataset_scores:
                result[f"mse_{dataset_name}"] = dataset_scores[dataset_name]
            else:
                # 如果数据集不存在，使用最差分数
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
