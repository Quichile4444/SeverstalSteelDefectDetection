# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : metrics.py
@Author  : Quichile
@Time    : 2026/5/5 19:27
"""

from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred):
    """计算多标签分类要求指标"""
    metrics = {}
    # 1、Exact Match Accuracy（4个标签全对才算对）：准确率
    metrics['Exact Match'] = accuracy_score(y_true=y_true, y_pred=y_pred)

    # 2、 Hamming Loss（所有标签位中预测错误的比例）：汉明损失
    metrics['Hamming Loss'] = hamming_loss(y_true=y_true, y_pred=y_pred)

    # 3、每个类别的Precision/Recall/F1-Score以及Macro F1
    metrics['Macro Precision'] = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    metrics['Macro Recall'] = recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    metrics['Macro F1'] = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

    # 每个类别的F1
    class_f1s = f1_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
    metrics['Class F1s'] = class_f1s

    # 每个类别的混淆矩阵
    cms = []
    for i in range(y_true.shape[1]):
        cms.append(confusion_matrix(y_true=y_true[:, i], y_pred=y_pred[:, i], labels=[0, 1]))
    metrics['Confusion Matrices'] = cms

    return metrics
