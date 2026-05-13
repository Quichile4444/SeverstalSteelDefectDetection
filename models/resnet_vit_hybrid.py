# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : resnet_vit_hybrid.py
@Author  : Quichile
@Time    : 2026/5/5 15:16
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetVitHybrid(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6, vit_layers=2, vit_heads=8):
        super(ResNetVitHybrid, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 冻结
        for name, param in backbone.named_parameters():
            if name.startswith(('conv1', 'bn1')):
                param.requires_grad = False

        # resnet完整流程
        self.resnet_backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1                           # 64*64*64
        self.layer2 = backbone.layer2                           # 128*32*32
        self.layer3 = backbone.layer3                           # 256*16*16
        self.layer4 = backbone.layer4                           # 512*8*8

        # ========================ViT模块=============================
        embed_dim = 64                                          # layer2输出维度为64, 64, 64
        patch_size = 8                                          # 设置patch大小
        num_patches = (64 // patch_size)**2                     # 8*8=64

        # Patch Embedding层，64×64×64的特征图切成64个64×8×8的patch
        self.patch_embed = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # 添加LayerNorm层
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

        # 添加类别标记[class] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 第一个维度是Batch，以作广播

        # 添加位置编码Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 设计Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=vit_heads,
            dim_feedforward=embed_dim * 2,                      # MLP节点个数/神经元数量
            batch_first=True,                                   # 输入格式是(Batch, Sequence_len, embed_dim)
            dropout=0.3,
            # norm_first=True,                                     # Pre-LayerNorm
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=vit_layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)                  # resnet全局特征提取
        # ====================特征融合（ViT+layer1234）====================
        resnet_dim = 128 + 256 + 512                            # ResNet18的输出通道896
        vit_dim = embed_dim                                     # 64

        self.cat = nn.Sequential(
            nn.Linear(resnet_dim + vit_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # resnet完整流程
        x_backbone = self.resnet_backbone(x)                    # (64, 64, 64)
        f1 = self.layer1(x_backbone)                            # (64, 64, 64)
        f2 = self.layer2(f1)                                    # (128, 32, 32)
        f3 = self.layer3(f2)                                    # (256, 16, 16)
        f4 = self.layer4(f3)                                    # (512, 8, 8)

        # ViT流程
        B = x.shape[0]
        x_vit = self.patch_embed(f1)                            # (B, 64, 8, 8)

        # transformer只能处理序列（Sequence），不能处理2D图片，同时需要（B，S，E）序列格式
        x_vit = x_vit.flatten(2).transpose(1, 2)                # (B, 64, 8, 8) -> (B, 64, 64) -> (B, 64, 64)

        x_vit = self.layer_norm(x_vit)

        # 拼接[CLS] Token
        cls_token = self.cls_token.expand(B, -1, -1)            # 扩展到B，后面两维不变
        x_vit = torch.cat((cls_token, x_vit), dim=1)            # (B, 65, 64)

        x_vit = x_vit + self.pos_embed                          # 与位置编码逐元素相加
        x_vit = self.vit_encoder(x_vit)                         # 送入transformer(B, 65. 64)
        x_vit = x_vit[:, 0]                                     # 提取Cls_token输出，所有行，第0列(B, 64)为全局特征

        resnet_f2 = self.avgpool(f2).flatten(1)                 # (B, 128)
        resnet_f3 = self.avgpool(f3).flatten(1)                 # (B, 256)
        resnet_f4 = self.avgpool(f4).flatten(1)                 # (B, 512)
        resnet_out = torch.cat([resnet_f2, resnet_f3, resnet_f4], dim=1)    # (B, 896)

        combine = torch.cat([x_vit, resnet_out], dim=1)         # (B, 960)
        cat = self.cat(combine)
        logits = self.fc(cat)
        return logits
