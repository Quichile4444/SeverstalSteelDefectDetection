# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : dataset.py
@Author  : Quichile
@Time    : 2026/5/4 16:52
"""

import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from config import *


class SteelDefectDataset(Dataset):
    """
    自定义Dataset类
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        """
        初始化

        :param df: 包含图片名和标签，即image_name和四个缺陷defect1234
        :param img_dir: 图片路径
        :param transform: 图像增强操作
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        :param idx: 索引值
        :return: image: 图像的Tensor形式；label: 四个缺陷标签值（0/1），Tensor形式
        """
        # 获取当前行
        row = self.df.iloc[idx]
        img_name = row['image_name']
        # 将四个标签转化成数组
        label = np.array([row['defect1'], row['defect2'], row['defect3'], row['defect4']], dtype=np.float32)

        # 读取图片
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)   # cv2读取的顺序是BGR，不能被transforms读

        # 数据增强
        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(label)                         # 标签从数组转成Tensor格式

        return image, label


def get_transforms():
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),                                # 数组需转成PIL
        transforms.RandomHorizontalFlip(p=0.5),                 # 50%概率水平翻转
        transforms.RandomVerticalFlip(p=0.3),                   # 垂直
        transforms.RandomRotation(30),                          # 旋转
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        ),   # 亮度、对比度、饱和度、色相
        transforms.ToTensor(),                                  # 自动归一化
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),     # 随机擦除
        transforms.Normalize(                                   # 使用ImageNet数据集的均值和标准差
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集数据
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(                                   # 使用ImageNet数据集的均值和标准差
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def create_dataloaders():
    """
    创建并返回训练集验证集Dataset的DataLoader
    """
    df = pd.read_csv(PROCESSED_CSV_PATH)
    df['origin_id'] = df['image_name'].apply(lambda x: x.split('_patch')[0])    # 将切片名称切割，取第一个元素
    image_ids = df['origin_id'].unique()

    # 划分数据集：使用原始Id是为了防止同一张图片的六个切片，有的在训练集，有的验证集，数据泄露
    train_ids, val_ids = train_test_split(image_ids, test_size=VAL_RATIO, random_state=RANDOM_SEED, shuffle=True)

    # 根据划分好的Id，把大df拆分成两个小df
    train_df = df[df['origin_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['origin_id'].isin(val_ids)].reset_index(drop=True)

    print('训练集切片数量：{}，验证集切片数量：{}'.format(len(train_df), len(val_df)))

    train_transform, val_transform = get_transforms()

    train_dataset = SteelDefectDataset(df=train_df, img_dir=PROCESSED_IMG_DIR, transform=train_transform)
    val_dataset = SteelDefectDataset(df=val_df, img_dir=PROCESSED_IMG_DIR, transform=val_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader
