# Chronos-2: 时间序列预测基础模型

[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb)

**Chronos-2** 是一个用于时间序列预测的基础模型，基于 [Chronos](https://arxiv.org/abs/2403.07815) 和 [Chronos-Bolt](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/) 构建。它提供了显著的能力改进，可以处理早期模型不支持的多种预测场景。

## 功能对比

| 功能 | Chronos | Chronos-Bolt | Chronos-2 |
|------|---------|--------------|-----------|
| 单变量预测 | ✅ | ✅ | ✅ |
| 跨项目学习 | ❌ | ❌ | ✅ |
| 多变量预测 | ❌ | ❌ | ✅ |
| 仅过去协变量（实数/分类） | ❌ | ❌ | ✅ |
| 已知未来协变量（实数/分类） | 🧩 | 🧩 | ✅ |
| 微调支持 | ✅ | ✅ | ✅ |
| 最大上下文长度 | 512 | 2048 | 8192 |

> 🧩 Chronos/Chronos-Bolt 不原生支持未来协变量，但可以与外部协变量回归器结合使用（参见 [AutoGluon 教程](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html#incorporating-the-covariates)）。这仅对每个时间步的效果进行建模，而不是跨时间的效果。相比之下，Chronos-2 原生支持所有协变量类型。

更多关于 Chronos-2 的详细信息，请参阅 [技术报告](https://www.arxiv.org/abs/2510.15821)。

## 安装

```bash
pip install 'chronos-forecasting[extras]>=2.2' 'matplotlib'
```

## 快速开始

### 加载模型

```python
import os

# 使用仅 1 个 GPU（如果可用）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline

# 加载 Chronos-2 管道
# 推荐使用 GPU 加速推理，CPU 也支持（使用 device_map="cpu"）
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2", 
    device_map="cuda"
)
```

### 单变量预测

```python
# 加载数据（长格式 pandas DataFrame）
context_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")
print("输入数据框形状:", context_df.shape)
display(context_df.head())

# 执行预测
pred_df = pipeline.predict_df(
    context_df, 
    prediction_length=24, 
    quantile_levels=[0.1, 0.5, 0.9]
)

print("输出数据框形状:", pred_df.shape)
display(pred_df.head())
```

### predict_df 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `df` | 长格式 DataFrame，包含 id、timestamp 和目标列 | - |
| `future_df` | 可选的未来协变量 DataFrame（同时存在于 df 和 future_df 中的列被视为已知未来协变量） | None |
| `id_column` | 时间序列标识符列 | "item_id" |
| `timestamp_column` | 时间戳列 | "timestamp" |
| `target` | 要预测的目标列名 | "target" |
| `prediction_length` | 预测步数 | - |
| `quantile_levels` | 要计算的分位数 | [0.1, 0.2, ..., 0.9] |

返回包含预测值（包括点预测和分位数）的 DataFrame。

## 带协变量的预测

Chronos-2 可以利用协变量来提高预测准确性。以下是一个实际的能源价格预测示例：

```python
# 能源价格预测配置
target = "target"  # 包含要预测的值的列名（能源价格）
prediction_length = 24  # 要预测的小时数
id_column = "id"  # 标识不同时间序列的列（国家/地区）
timestamp_column = "timestamp"  # 包含日期时间信息的列
timeseries_id = "DE"  # 要可视化的特定时间序列（德国）

# 加载历史能源价格和协变量的过去值
energy_context_df = pd.read_parquet(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
)
energy_context_df[timestamp_column] = pd.to_datetime(energy_context_df[timestamp_column])

# 加载协变量的未来值
energy_test_df = pd.read_parquet(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet"
)
energy_test_df[timestamp_column] = pd.to_datetime(energy_test_df[timestamp_column])
energy_future_df = energy_test_df.drop(columns=target)

# 使用协变量进行预测
pred_df = pipeline.predict_df(
    df=energy_context_df,
    future_df=energy_future_df,
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
)
```

## 可视化预测结果

```python
def plot_forecast(
    context_df,
    test_df,
    pred_df,
    timeseries_id,
    target,
    timestamp_column="timestamp",
    id_column="item_id",
):
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # 筛选特定时间序列的数据
    ts_context = context_df[context_df[id_column] == timeseries_id].set_index(timestamp_column)[target]
    ts_ground_truth = test_df[test_df[id_column] == timeseries_id].set_index(timestamp_column)[target]
    ts_pred = pred_df[pred_df[id_column] == timeseries_id].set_index(timestamp_column)
    
    # 绘制历史数据、真实值和预测值
    ts_context.plot(ax=ax, label=f"历史 {target}", color="xkcd:azure")
    ts_ground_truth.plot(ax=ax, label=f"未来 {target} (真实值)", color="xkcd:grass green")
    ts_pred["predictions"].plot(ax=ax, label="预测", color="xkcd:violet")
    
    # 绘制预测区间
    ax.fill_between(
        ts_pred.index,
        ts_pred["0.1"],
        ts_pred["0.9"],
        alpha=0.7,
        label="预测区间",
        color="xkcd:light lavender",
    )
    
    ax.axvline(x=ts_context.index[-1], color="black", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_title(f"{target} 预测 - {timeseries_id}")
    fig.show()
```

## 支持的场景

1. **单变量时间序列预测**：经典的时间序列预测任务
2. **跨项目学习**：利用多个相关时间序列的信息进行预测
3. **多变量预测**：同时预测多个相关目标变量
4. **带协变量的预测**：
   - **仅过去协变量**：在预测时已知的历史协变量（如过去的天气数据）
   - **已知未来协变量**：预测期间已知的协变量（如节假日、计划事件）

## 硬件要求

- **GPU**（推荐）：支持 CUDA 的 NVIDIA GPU，可显著加速推理
- **CPU**（支持）：可在 CPU 上运行，但推理速度较慢

## 参考

- [Chronos 论文](https://arxiv.org/abs/2403.07815)
- [Chronos-2 技术报告](https://www.arxiv.org/abs/2510.15821)
- [AutoGluon 时间序列教程](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)