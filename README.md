# SLD - Scaling Law Discovery

基于AI的缩放律发现框架，使用OpenEvolve进化式编程来自动发现和优化LLM微调的缩放律函数。

## 📋 项目概述

这个项目结合了两个核心组件：
- **rectified_scaling_law**: 缩放律发现的主要实现，包含初始程序、评估器和配置
- **openevolve**: 强大的进化式编程框架，用于自动优化和发现算法

通过OpenEvolve的进化算法，项目可以自动发现更准确的缩放律函数，用于预测LLM在不同训练数据规模下的性能表现。

## 🎯 主要功能

- **自动缩放律发现**: 使用进化算法自动优化缩放律函数的数学形式和参数
- **多数据集评估**: 在FLAN、Gigaword、WMT19等真实LLM微调数据集上验证性能
- **鲁棒性优化**: 确保发现的缩放律在不同模型家族（T5、GPT等）和数据规模上都有良好表现
- **实时监控**: 支持检查点保存、可视化进化过程和性能追踪

## 🏗️ 项目结构

```
SLD/
├── README.md                    # 项目说明文档
├── rectified_scaling_law/       # 主要项目代码
│   ├── config.yaml             # OpenEvolve配置文件
│   ├── evaluator.py            # 缩放律性能评估器
│   ├── init_program.py         # 初始缩放律实现
│   ├── data/                   # 真实LLM微调数据集
│   │   ├── flan.csv           # FLAN数据集的训练曲线
│   │   ├── gigaword.csv       # Gigaword数据集的训练曲线
│   │   └── wmt19.csv          # WMT19数据集的训练曲线
│   └── openevolve_output/      # OpenEvolve输出结果（运行后生成）
└── openevolve/                 # OpenEvolve进化式编程框架
    ├── openevolve/             # 核心框架代码
    ├── examples/               # 使用示例
    ├── configs/                # 配置模板
    └── ...
```

## 🚀 快速开始

### 环境准备

1. **安装OpenEvolve**:
```bash
cd openevolve
pip install -e .
```

2. **配置API密钥**:
```bash
export OPENAI_API_KEY=your-api-key-here
```
支持所有OpenAI兼容的API端点（OpenAI、Anthropic、本地模型等）

### 运行缩放律发现

1. **进入项目目录**:
```bash
cd rectified_scaling_law
```

2. **启动进化过程**:
```bash
python ../openevolve/openevolve-run.py init_program.py evaluator.py --config config.yaml --iterations 100
```

3. **监控进度**:
```bash
# 查看生成的结果
ls openevolve_output/

# 查看最佳程序
cat openevolve_output/best_program.py
```
## 📊 数据集说明

项目使用三个真实的LLM微调数据集：

- **FLAN**: 指令微调数据集，包含多种NLP任务
- **Gigaword**: 文本摘要数据集
- **WMT19**: 机器翻译数据集

每个数据集包含不同模型大小和训练数据规模下的loss值，用于验证缩放律的准确性。

数据格式:
- 训练数据规模：200 到 1,638,400 个样本
- 模型参数：从数百万到数十亿参数
- 模型家族：T5、GPT等主流架构

## ⚙️ 配置详解

### 主要配置参数

```yaml
# 进化参数
max_iterations: 50              # 最大进化迭代次数
random_seed: 42                 # 随机种子，确保可重复性

# LLM配置
llm:
  models:
    - name: "o4-mini"          # 使用的语言模型
      weight: 1.0              # 模型权重
  max_tokens: 16384            # 最大token数
  
# 进化算法配置
database:
  population_size: 100         # 种群大小
  archive_size: 50            # 存档大小
  num_islands: 3              # 岛屿数量（并行进化）
  
# 评估配置
evaluator:
  timeout: 30                 # 评估超时时间
  parallel_evaluations: 4     # 并行评估数量
```

## 🔬 评估指标

项目使用以下指标评估缩放律的性能：

- **MSE (均方误差)**: 主要评估指标，越小越好
- **R²分数**: 拟合质量评估
- **皮尔逊相关系数**: 预测与真实值的相关性
- **交叉数据集泛化**: 在不同数据集间的性能一致性

## 📈 初始缩放律

项目从一个基础的幂律函数开始：

```python
def scaling_law_func(data_points, params):
    """
    初始的幂律缩放函数: loss = (a * x^(-b) + c)^d
    """
    a = abs(params[0]) + 0.1  # 尺度因子
    b = abs(params[1]) + 1.0  # 幂指数
    c = abs(params[2]) + 0.1  # 偏移量
    d = abs(params[3]) + 0.1  # 输出变换
    
    loss = np.power(a * x_safe ** (-b) + c, d)
    return loss
```

OpenEvolve会自动优化这个函数的数学形式和参数拟合算法。

## 📁 输出结果

运行完成后，`openevolve_output/`目录将包含：

- `best_program.py`: 发现的最佳缩放律函数
- `checkpoints/`: 各个迭代的检查点
- `evolution_log.txt`: 详细的进化日志
- `performance_metrics.json`: 性能指标历史

## 🔧 高级使用

### 从检查点恢复

```bash
python ../openevolve/openevolve-run.py init_program.py evaluator.py \
  --config config.yaml \
  --checkpoint openevolve_output/checkpoints/checkpoint_50 \
  --iterations 50
```

### 自定义评估器

可以修改`evaluator.py`来：
- 添加新的数据集
- 引入额外的评估指标
- 调整评估权重

### 配置优化

根据计算资源调整配置：
- 增加`population_size`以获得更好的结果
- 调整`parallel_evaluations`匹配CPU核心数
- 设置合适的`timeout`避免卡住

## 🤝 贡献指南

欢迎贡献代码和想法：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📝 许可证

本项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [OpenEvolve](https://github.com/codelion/openevolve) - 强大的进化式编程框架
- 真实LLM数据集的提供者们
- 开源社区的支持和贡献
