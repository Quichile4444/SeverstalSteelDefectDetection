# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : evaluate.py
@Author  : Quichile
@Time    : 2026/5/5 20:33
"""

import torch
import matplotlib.pyplot as plt
from utils.dataset import create_dataloaders
from utils.metrics import calculate_metrics
from models.resnet18_baseline import ResNet18Baseline
from models.resnet_vit_hybrid import ResNetVitHybrid

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(model, model_name, device, p=False):
    _, val_loader = create_dataloaders()
    model.load_state_dict(torch.load(f'./checkpoints/{model_name}_best.pth'))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu()
            all_preds.append(preds)
            all_labels.append(labels.int().cpu())

    # 拼接所有预测和真实标签，计算评估指标
    y_true = torch.cat(all_labels).numpy()
    y_preds = torch.cat(all_preds).numpy()

    metrics = calculate_metrics(y_true=y_true, y_pred=y_preds)

    if p:
        print('\n======================={} 模型评估结果========================'.format(model_name))
        print('Exact Match Accuracy: {:.4f}'.format(metrics['Exact Match']))
        print('Hamming Loss: {:.4f}'.format(metrics['Hamming Loss']))
        print('Macro Precision: {:.4f}'.format(metrics['Macro Precision']))
        print('Macro Recall: {:.4f}'.format(metrics['Macro Recall']))
        print('Macro F1: {:.4f}\n\n'.format(metrics['Macro F1']))

    return metrics


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_resnet = ResNet18Baseline().to(device=device)
    metrics_resnet = evaluate_model(model=model_resnet, model_name='ResNet18_baseline', device=device, p=True)

    model_hybrid = ResNetVitHybrid().to(device=device)
    metrics_hybrid = evaluate_model(model=model_hybrid, model_name='ResNet18_ViT_hybrid', device=device, p=True)
