# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : resnet18_baseline.py
@Author  : Quichile
@Time    : 2026/5/4 21:09
"""

import torch.nn as nn
import torchvision.models as models


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(ResNet18Baseline, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 加载预训练模型，作迁移学习

        # 冻结浅层layer1及之前
        for name, param in backbone.named_parameters():
            if name.startswith(('conv1', 'bn1')):
                param.requires_grad = False

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # 去除最后一层Linear
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)                                    # 尺寸减半五次后：(B, 512, 8, 8)，池化后为(B, 512, 1, 1)
        x = self.flatten(x)                                     # (B, 512)
        x = self.dropout(x)
        y = self.fc(x)                                          # (B, 512) -> (B, 4)
        return y
