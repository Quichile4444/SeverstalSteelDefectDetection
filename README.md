# Severstal 钢材缺陷检测：ResNet18 与 ResNet18+ViT 对比

基于 **ResNet18 基线模型** 与 **ResNet18\+ViT 混合模型** 的钢材表面缺陷多标签分类检测，适配 Severstal 钢铁产线真实长条图像数据，支持训练、评估、推理、ONNX 部署全流程。

## 项目简介

本项目针对 [**Kaggle 2019 Severstal: Steel Defect Detection**](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) 数据集，完成钢材缺陷的**4 分类多标签检测**。

- 解决 1600×256 长条钢材图像的缺陷识别问题

- 对比纯 CNN 基线与 CNN\+Transformer 混合模型性能

- 支持数据切片、RLE 掩码解码、模型训练、评估、可视化、ONNX 导出与推理部署

## 数据集概述

|项目|内容|
|:-:|---|
|**数据集名称**|Severstal: Steel Defect Detection|
|**总图像数**|12,568 张训练图 \+ 5,506 张测试图（测试集标签不公开）|
|**分辨率**|1600 × 256 像素（高分辨率长条图）|
|**缺陷类别**|4 类缺陷（多标签分类）|
|**标注格式**|像素级分割掩码（Run\-Length Encoding）|
|**数据来源**|Severstal 钢铁公司真实产线|
|**竞赛**|Kaggle 2019 经典竞赛数据集|

## 项目结构

```bash
steel_defect_detection/
├── data/
│   ├── raw/                    # 存放最原始的 Kaggle 数据
│   │   ├── train.csv           # 原始标注（包含 RLE 编码）
│   │   └── train_images/       # 原始 1600x256 长图
│   └── processed/              # 存放预处理后的数据
│       ├── train_sliced.csv    # 切片后的新标注
│       └── train_images_sliced/# 切好的 256x256 小图
│
├── preprocess/                 # 数据加工
│   ├── __init__.py
│   ├── rle_decode.py           # 将RLE转Mask
│   └── slice_images.py         # 将数据raw切片保存到processed
│
├── checkpoints/				# 存放训练过程中保存的Pytorch权重 (.pth)
├── runs/						# 存放训练日志、损失曲线图等
│
├── utils/						# 存放通用工具脚本
│   ├── __init__.py
│   ├── dataset.py				# 数据加载与预处理
│   └── metrics.py				# 评估指标计算（如Accuracy、Precision、Recall、ROC、AUC、F1-Score）
│
├── models/						# 模型定义
│   ├── __init__.py
│   ├── resnet18_baseline.py	# ResNet18 基线模型
│   └── resnet_vit_hybrid.py	# ResNet18 + ViT 融合模型
│
├── data_analysis.py			# 分析数据集
├── config.py					# 全局配置文件（超参数、路径等）
├── train.py					# 模型训练
├── evaluate.py					# 模型评估
├── predict.py					# Pytorch推理预测
├── compare_models.py			# 模型比对
│
├── export_onnx.py				# PyTorch（.pth）→ONNX（.onnx）导出
└── infer_onnx.py				# ONNX Runtime推理
```

## 环境配置

直接复制以下依赖到 `requirements\.txt` 并安装：

```txt
matplotlib==3.9.4
numpy==1.26.2
onnx==1.19.1
onnxruntime-gpu==1.19.2
opencv-python==4.12.0.88
pandas==2.3.3
scikit-learn==1.5.1
torch==2.7.1+cu118
torchvision==0.22.1+cu118
```

安装命令：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1\. 数据准备

1. 下载 Kaggle 数据集并放入 `data/raw/`

2. 执行数据切片与 RLE 解码：

```bash
python preprocess/slice_images.py
```

### 2\. 模型训练

```bash
# 训练双模型（ResNet18 基线 + ResNet18-ViT 混合）
python train.py
```

### 3\. 模型评估

```bash
python evaluate.py
```

### 4\. 模型对比

```bash
python compare_models.py
```

### 5\. 模型推理

```bash
# PyTorch 推理
python predict.py

# ONNX 推理
python infer_onnx.py
```

### 6\. 导出 ONNX

```bash
python export_onnx.py
```

## 训练日志与结果

### ResNet18\_ViT\_hybrid 训练结果

```Plain Text
ResNet18_ViT_hybrid 训练结束，最佳Macro F1为：0.7947
=======================ResNet18_ViT_hybrid 模型评估结果========================
Exact Match Accuracy: 0.9107
Hamming Loss: 0.0239
Macro Precision: 0.7007
Macro Recall: 0.9222
Macro F1: 0.7951
```

### ResNet18\_baseline 训练结果

```Plain Text
ResNet18_baseline 训练结束，最佳Macro F1为：0.7698
=======================ResNet18_baseline 模型评估结果========================
Exact Match Accuracy: 0.8965
Hamming Loss: 0.0278
Macro Precision: 0.6915
Macro Recall: 0.8892
Macro F1: 0.7701
```

## 模型性能对比

|评估指标|ResNet18|ResNet18\+ViT|混合模型提升 Δ|
|:-:|:-:|:-:|:-:|
|Exact Match|0\.8965|0\.9107|\+0\.0142|
|Hamming Loss|0\.0278|0\.0239|\-0\.0039|
|Macro Precision|0\.6915|0\.7007|\+0\.0092|
|Macro Recall|0\.8892|0\.9222|\+0\.0331|
|Macro F1|0\.7701|0\.7951|\+0\.0250|

## 核心功能

- **数据预处理**：RLE 掩码解码、长条图像切片、数据集统计分析

- **双模型实现**：ResNet18 基线 / ResNet18\+ViT 混合特征融合

- **多标签训练**：带类别平衡权重、早停策略、余弦退火学习率、混合精度训练

- **完整评估**：Exact Match、Hamming Loss、Macro F1、各类别 F1、混淆矩阵

- **模型对比**：自动生成指标对比表、类别 F1 柱状图、混淆矩阵热力图

- **工业部署**：支持 PyTorch → ONNX 导出，ONNX Runtime 快速推理

## 推理与部署

- 支持**单张切片推理**、**整张大图端到端推理**

- 自动在图像上标注缺陷类别与置信度

- 导出 ONNX 模型可直接用于 C\+\+/Python 工业部署

## 许可证

本项目采用 **MIT License** 开源，详见 [LICENSE](LICENSE) 文件。

## 致谢

- Kaggle \&amp; Severstal 提供公开钢材缺陷数据集

- PyTorch、ONNX、OpenCV 等开源社区
