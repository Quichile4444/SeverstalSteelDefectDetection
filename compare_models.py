# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : compare_models.py
@Author  : Quichile
@Time    : 2026/5/11 17:49
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from models.resnet18_baseline import ResNet18Baseline
from models.resnet_vit_hybrid import ResNetVitHybrid
from evaluate import evaluate_model

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False


def compare(model, model_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = evaluate_model(model, model_name, device)

    # ====================绘制各类别F1对比柱状图=========================
    classes = ['Class1', 'Class2', 'Class3', "Class4"]
    plt.figure(figsize=(8, 5))
    plt.bar(classes, metrics['Class F1s'], color=['red', 'blue', 'green', 'yellow'])
    plt.title('{} 模型——各类别的F1分数'.format(model_name))
    plt.ylabel('F1 score')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics['Class F1s']):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.savefig('./runs/{}_class_f1.png'.format(model_name), dpi=150)
    plt.show()

    # ==================绘制四个类别的混淆矩阵=======================
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (cm, cls_name) in enumerate(zip(metrics['Confusion Matrices'], classes)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        axes[i].set_title(cls_name)
    plt.suptitle('{}模型——二分类混淆矩阵'.format(model_name), y=1.05)
    plt.tight_layout()
    plt.savefig('./runs/{}_confusion_matrices.png'.format(model_name), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet = ResNet18Baseline().to(device=device)
    hybrid = ResNetVitHybrid().to(device=device)

    # compare(resnet, 'ResNet18_baseline')
    # compare(hybrid, 'ResNet18_ViT_hybrid')

    metrics_base = evaluate_model(resnet, 'ResNet18_baseline', device)
    metrics_hybrid = evaluate_model(hybrid, 'ResNet18_ViT_hybrid', device)

    # 整理数据到表格并计算差值
    indicator_names = ['Exact Match', 'Hamming Loss', 'Macro Precision', 'Macro Recall', 'Macro F1']
    resnet_values = [metrics_base[k] for k in indicator_names]
    hybrid_values = [metrics_hybrid[k] for k in indicator_names]
    delta_values = [h - r for h, r in zip(hybrid_values, resnet_values)]

    # 创建DataFrame展示
    df = pd.DataFrame({
        '评估指标': indicator_names,
        'ResNet18': resnet_values,
        'ResNet18+ViT': hybrid_values,
        '混合模型提升Δ': delta_values
    })
    print('模型综合指标对比：\n{}'.format(df.to_string(index=False, col_space=[12, 12, 12, 12])))
