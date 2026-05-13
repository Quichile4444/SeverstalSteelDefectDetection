# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : train.py.py
@Author  : Quichile
@Time    : 2026/5/5 16:31
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import time
from utils.dataset import create_dataloaders
from sklearn.metrics import f1_score
from models.resnet18_baseline import ResNet18Baseline
from models.resnet_vit_hybrid import ResNetVitHybrid


def calculate_pos_weight(train_loader):
    """计算多标签分类的正负样本权重"""
    total_pos = torch.zeros(4)                                  # 正样本
    total_neg = torch.zeros(4)                                  # 负样本
    for _, labels in train_loader:
        total_pos += labels.sum(dim=0)
        total_neg += (labels == 0).sum(dim=0)
    # 避免除0，限制最大权重导致梯度爆炸
    pos_weight = (total_neg / (total_pos + 1e-6)).clamp(max=60)     # max若设置100，容易导致验证损失以及F1剧烈振荡，不稳定
    return pos_weight


def train_model(model, model_name, device, epochs=100):
    train_loader, val_loader = create_dataloaders()

    pos_weight = calculate_pos_weight(train_loader=train_loader).to(device=device)
    print('{} 类别权重pos_weight为：{}'.format(model_name, pos_weight.cpu().numpy()))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)    # 损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)   # 优化器（原先为-3-4）
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-6)  # 学习率调度器
    scaler = torch.amp.GradScaler('cuda')                       # 用于自动混合精度训练（AMP），节省显存并加速

    # -----------早停相关的参数--------------
    best_macro_f1 = 0.0                                         # 预设最好的Macro F1分数
    f1_patience_counter = 0                                     # 记录f1没升反降的轮次
    f1_patience = 20                                            # F1不上升的早停轮数
    loss_patience_counter = 0
    loss_patience = 5

    os.makedirs('./checkpoints', exist_ok=True)                 # 存放训练日志

    # -------------计时参数-------------
    data_load_time = 0.0
    batch_train_time = 0.0

    for epoch in range(epochs):
        # =================模型训练=================
        epoch_start_time = time.time()                          # 记录起始一个epoch时间
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            start_data = time.time()                            # 开始读取数据时间

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                               # 清空梯度

            data_load_time += time.time() - start_data          # 记录数据加载结束时间
            start_batch = time.time()                           # 开始训练时间

            with torch.amp.autocast('cuda'):                    # 混合精度训练
                logits = model(images)
                batch_train_loss = criterion(logits, labels)
                train_loss += batch_train_loss.item()

            scaler.scale(batch_train_loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer=optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer=optimizer)
            scaler.update()

            batch_train_time += time.time() - start_batch       # 记录训练结束时间

        avg_train_loss = train_loss / len(train_loader)

        # ===================模型验证==========================
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    batch_val_loss = criterion(logits, labels)
                    val_loss += batch_val_loss.item()

                preds = (torch.sigmoid(logits) > 0.5).int().cpu()   # 将logits转为概率，再按0.5阈值生成预测标签
                all_preds.append(preds)
                all_labels.append(labels.int().cpu())

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()                                        # 学习率更新

        # 拼接所有预测和真实标签，计算评估指标
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算Macro F1分数
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        macro_f1 = np.mean(f1_per_class)

        # 打印当前epoch的训练、验证损失和验证集的Macro F1分数
        print('-' * 40)
        print(f'Epoch {epoch + 1} / {epochs}\nTrain Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val Macro F1: {macro_f1:.4f}')

        # 每个epoch打印耗时
        avg_data_time = data_load_time / len(train_loader)
        avg_batch_time = batch_train_time / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        epoch_min = int(epoch_time // 60)
        epoch_sec = int(epoch_time % 60)
        print(f'平均数据加载耗时：{avg_data_time:.4f}s | 平均训练batch耗时：{avg_batch_time:.4f}s | '
              f'Epoch{epoch + 1}总耗时:{epoch_min}m {epoch_sec}s')

        data_load_time = batch_train_time = 0.0

        # =============== 设置早停 ===================
        val_loss_ratio = avg_val_loss / avg_train_loss

        if macro_f1 > best_macro_f1:
            f1_patience_counter = 0
            if val_loss_ratio <= 1.5:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), f'./checkpoints/{model_name}_best.pth')
                print('模型已保存（最佳的Macro F1为：{:.4f}）'.format(best_macro_f1))

        elif macro_f1 < best_macro_f1:
            f1_patience_counter += 1
            if f1_patience_counter >= f1_patience:
                print('！！！触发早停！连续{}个Epoch的Macro F1未提升。'.format(f1_patience))
                break

        if val_loss_ratio <= 1.5:
            loss_patience_counter = 0
        else:
            loss_patience_counter += 1
            print(f'  ⚠️ 过拟合警告: Val Loss 是 Train Loss 的 {val_loss_ratio:.1f}倍')
            if loss_patience_counter >= loss_patience:
                print('！！！触发早停！连续{}个Epoch出现过拟合现象。'.format(loss_patience))
                break

    print('{} 训练结束，最佳Macro F1为：{:.4f}\n\n'.format(model_name, best_macro_f1))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('现在使用的是：{}'.format(device))

    model_resnet18 = ResNet18Baseline().to(device=device)
    model_resnet18_vit_hybrid = ResNetVitHybrid().to(device=device)

    print('================ResNet18_baseline开始训练、验证====================')
    train_model(model=model_resnet18, model_name='ResNet18_baseline', device=device)

    print('================ResNet18_ViT_hybrid开始训练、验证====================')
    train_model(model=model_resnet18_vit_hybrid, model_name='ResNet18_ViT_hybrid', device=device)
