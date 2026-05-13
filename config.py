# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : config.py
@Author  : Quichile
@Time    : 2026/5/4 15:59
"""

# 数据路径
RAW_DATA_DIR = "./data/raw"
RAW_CSV_PATH = './data/raw/train.csv'
RAW_IMG_DIR = './data/raw/train_images'
PROCESSED_DATA_DIR = "./data/processed"
PROCESSED_CSV_PATH = "./data/processed/train_sliced.csv"
PROCESSED_IMG_DIR = "./data/processed/train_images_sliced"

# 数据参数
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PATCH_WIDTH = 256
NUM_CLASSES = 4                                                 # 4 种缺陷类型
VAL_RATIO = 0.15                                                # 验证集比例 (15%)
RANDOM_SEED = 42                                                # 随机种子，保证每次划分结果一样

# 训练参数
BATCH_SIZE = 128
NUM_WORKERS = 4
