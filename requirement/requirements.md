# Chronos-2 Quickstart 依赖项文档

## 文档概述

本文档详细列出了运行 `chronos-2-quickstart.ipynb` 笔记本所需的所有依赖项，包括库名称、版本要求、安装命令以及在项目中的具体用途。

---

## 依赖项分类总览

| 类别 | 核心库 | 主要用途 |
|------|--------|----------|
| **核心预测库** | chronos-forecasting | 时间序列预测基础模型 |
| **数据处理** | pandas, numpy | DataFrame操作和数值计算 |
| **可视化** | matplotlib | 预测结果可视化 |
| **深度学习** | torch, transformers, accelerate | 模型加载和推理 |
| **优化技术** | peft | 参数高效微调 |
| **数据访问** | boto3, datasets | AWS S3和数据集处理 |
| **评估工具** | fev | 预测评估指标 |

---

## 1. 核心预测库

### chronos-forecasting[extras]

| 属性 | 详情 |
|------|------|
| **名称** | chronos-forecasting |
| **版本要求** | >= 2.2 |
| **安装命令** | `pip install 'chronos-forecasting[extras]>=2.2'` |
| **用途** | 提供 Chronos-2 时间序列预测基础模型的完整功能 |
| **在文件中的使用** | - 加载预训练模型 `BaseChronosPipeline.from_pretrained()`<br>- 执行预测 `pipeline.predict_df()`<br>- 支持单变量、多变量和协变量预测 |

---

## 2. 数据处理库

### pandas

| 属性 | 详情 |
|------|------|
| **名称** | pandas |
| **版本要求** | >= 2.0 |
| **安装命令** | `pip install 'pandas>=2.0'` |
| **用途** | 数据处理和分析，提供 DataFrame 数据结构 |
| **在文件中的使用** | - 读取 CSV 和 Parquet 数据文件<br>- 数据预处理和转换<br>- 时间序列数据格式处理 |

### numpy

| 属性 | 详情 |
|------|------|
| **名称** | numpy |
| **版本要求** | >= 1.21 |
| **安装命令** | `pip install 'numpy>=1.21'` |
| **用途** | 数值计算基础库 |
| **在文件中的使用** | 作为 pandas 和 torch 的底层数值支持 |

---

## 3. 可视化库

### matplotlib

| 属性 | 详情 |
|------|------|
| **名称** | matplotlib |
| **版本要求** | >= 3.10.0 |
| **安装命令** | `pip install 'matplotlib>=3.10.0'` |
| **用途** | 数据可视化和绘图 |
| **在文件中的使用** | - 绘制历史数据、预测值和预测区间<br>- 创建时间序列图表<br>- 自定义图表样式和颜色 |

---

## 4. 深度学习框架

### torch

| 属性 | 详情 |
|------|------|
| **名称** | torch |
| **版本要求** | >= 2.2 |
| **安装命令** | `pip install 'torch>=2.2'` |
| **用途** | PyTorch 深度学习框架 |
| **在文件中的使用** | - 模型加载和推理<br>- GPU 加速支持<br>- 张量运算 |

### transformers

| 属性 | 详情 |
|------|------|
| **名称** | transformers |
| **版本要求** | >= 4.41 |
| **安装命令** | `pip install 'transformers>=4.41'` |
| **用途** | Hugging Face 预训练模型库 |
| **在文件中的使用** | 提供 Transformer 架构支持和模型加载能力 |

### accelerate

| 属性 | 详情 |
|------|------|
| **名称** | accelerate |
| **版本要求** | >= 1.1.0 |
| **安装命令** | `pip install 'accelerate>=1.1.0'` |
| **用途** | GPU/TPU 分布式训练和推理加速 |
| **在文件中的使用** | 支持多 GPU 推理和自动设备映射 |

### peft

| 属性 | 详情 |
|------|------|
| **名称** | peft |
| **版本要求** | >= 0.18.1 |
| **安装命令** | `pip install 'peft>=0.18.1'` |
| **用途** | 参数高效微调（Parameter-Efficient Fine-Tuning） |
| **在文件中的使用** | 支持 LoRA 等微调技术（如需要） |

### einops

| 属性 | 详情 |
|------|------|
| **名称** | einops |
| **版本要求** | >= 0.7.0 |
| **安装命令** | `pip install 'einops>=0.7.0'` |
| **用途** | 灵活的张量操作和重排 |
| **在文件中的使用** | 支持模型内部的张量形状变换 |

---

## 5. 数据访问和云服务

### boto3

| 属性 | 详情 |
|------|------|
| **名称** | boto3 |
| **版本要求** | >= 1.10 |
| **安装命令** | `pip install 'boto3>=1.10'` |
| **用途** | AWS SDK for Python |
| **在文件中的使用** | 访问 AWS S3 存储桶下载示例数据集 |

### datasets

| 属性 | 详情 |
|------|------|
| **名称** | datasets |
| **版本要求** | >= 2.15 |
| **安装命令** | `pip install 'datasets>=2.15'` |
| **用途** | Hugging Face 数据集库 |
| **在文件中的使用** | 数据集加载和处理工具（由 fev 库使用） |

### fev

| 属性 | 详情 |
|------|------|
| **名称** | fev |
| **版本要求** | >= 0.8.0 |
| **安装命令** | `pip install 'fev>=0.8.0'` |
| **用途** | 预测评估和可视化工具 |
| **在文件中的使用** | 提供预测评估指标计算 |

### pydantic

| 属性 | 详情 |
|------|------|
| **名称** | pydantic |
| **版本要求** | ~= 2.0 |
| **安装命令** | `pip install 'pydantic~=2.0'` |
| **用途** | 数据验证和设置管理 |
| **在文件中的使用** | 配置和数据结构验证（由 fev 库使用） |

### scipy

| 属性 | 详情 |
|------|------|
| **名称** | scipy |
| **版本要求** | 无特定版本（由依赖项自动安装） |
| **安装命令** | `pip install scipy` |
| **用途** | 科学计算和统计分析 |
| **在文件中的使用** | 提供统计函数支持 |

---

## 安装说明

### 方法一：使用 requirements.txt

```bash
# 安装所有依赖项
pip install -r requirements.txt
```

### 方法二：最小安装（推荐）

```bash
# 安装核心依赖项
pip install 'chronos-forecasting[extras]>=2.2' 'matplotlib'
```

### 方法三：CPU 专用安装

```bash
# 无 GPU 环境下安装（使用 CPU 版本 PyTorch）
pip install 'chronos-forecasting[extras]>=2.2' 'matplotlib' --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 硬件要求

| 配置 | 推荐 | 最低 |
|------|------|------|
| **GPU** | NVIDIA CUDA-enabled GPU（如 NVIDIA A100, RTX 3090+） | 无（CPU 模式支持） |
| **显存** | 至少 8GB | N/A（CPU） |
| **内存** | 至少 16GB | 至少 8GB |

---

## 使用注意事项

1. **GPU 加速**：推荐使用 GPU 进行推理以获得更好的性能
2. **设备配置**：在加载模型时通过 `device_map` 参数指定设备
   - GPU：`device_map="cuda"`
   - CPU：`device_map="cpu"`
3. **Hugging Face Token**：建议设置 `HF_TOKEN` 环境变量以获得更高的下载速率限制
4. **模型下载**：首次运行时会自动下载约 478MB 的模型权重文件

---

## 依赖项关系图

```
chronos-forecasting[extras]>=2.2
├── torch>=2.2
│   ├── numpy>=1.21
│   └── cuda-toolkit (可选，GPU支持)
├── transformers>=4.41
│   └── huggingface_hub
├── accelerate>=1.1.0
├── peft>=0.18.1
├── einops>=0.7.0
├── pandas>=2.0
├── boto3>=1.10
├── fev>=0.8.0
│   ├── datasets>=2.15
│   ├── pydantic~=2.0
│   └── scipy
└── matplotlib>=3.10.0
```

---

## 版本兼容性说明

| 依赖项 | 最低版本 | 推荐版本 | 兼容性说明 |
|--------|----------|----------|------------|
| chronos-forecasting | 2.2 | 2.3.0 | 必须 >=2.2 以支持 Chronos-2 |
| torch | 2.2 | 2.11.0+ | 与 transformers 版本兼容 |
| transformers | 4.41 | 5.10.2 | 支持最新模型格式 |
| pandas | 2.0 | 2.2.2 | 支持 pyarrow 后端 |

---

**文档版本**: 1.0  
**生成日期**: 2026-06-22  
**来源文件**: notebooks/chronos-2-quickstart.ipynb