# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : slice_images.py
@Author  : Quichile
@Time    : 2026/4/30 20:46
"""

import pandas as pd
import cv2
import os
from tqdm import tqdm
from config import *
from preprocess.rle_utils import rle_decode                     # 获取解码代码


# 配置路径
RAW_CSV_PATH = '../data/raw/train.csv'
RAW_IMG_DIR = '../data/raw/train_images'


def main():
    # 创建文件夹
    os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)

    df = pd.read_csv(RAW_CSV_PATH)
    # 获取每张图片的id（不能去掉.jpg，与源CSV文件对齐）
    all_images = [f for f in os.listdir(RAW_IMG_DIR) if f.endswith('.jpg')]
    print('文件夹{}中有{}张图片'.format(RAW_IMG_DIR, len(all_images)))

    sliced_records = []                                         # 存放切割后的图片信息（图片名，四类分别是否缺陷1/0）
    for img_id in tqdm(all_images):
        img_path = os.path.join(RAW_IMG_DIR, img_id)            # 获取图片路径
        img = cv2.imread(img_path)                              # 获取图片

        if img is None:
            continue

        h, w = img.shape[:2]
        n_patches = w // PATCH_WIDTH                            # 计算能切出多少patch

        # 提取当前图片4个类别的RLE字符串
        rles = []
        for c in range(1, NUM_CLASSES + 1):
            row = df[(df['ImageId'] == img_id) & (df['ClassId'] == c)]  # 无外层df得出的结果是1/0，多条件索引用&，而不是and
            if row.empty:
                rle_str = ''
            else:
                rle_str = row.iloc[0]['EncodedPixels']
            rles.append(rle_str)

        # 将当前图片每类缺陷的RLE字符串解码为4个掩码矩阵
        masks = []                                              # 存放二维矩阵
        for rle in rles:
            mask = rle_decode(rle, (h, w))
            masks.append(mask)

        # 切割
        for i in range(n_patches):
            x_start = i * PATCH_WIDTH
            x_end = x_start + PATCH_WIDTH
            patch_img = img[:, x_start:x_end]

            patch_name = f'{img_id}_patch{i}.jpg'
            save_path = os.path.join(PROCESSED_IMG_DIR, patch_name)
            cv2.imwrite(save_path, patch_img)

            # 判断模块是否有缺陷（有一个就为1）
            labels = []
            for c in range(NUM_CLASSES):
                # print(masks[c].ndim):->rle_decode原先对空缺陷的处理结果是一维shape，只有一个维度
                patch_mask = masks[c][:, x_start:x_end]
                label = 1 if patch_mask.any() else 0
                labels.append(label)

            sliced_records.append({
                'image_name': patch_name,
                'defect1': labels[0],
                'defect2': labels[1],
                'defect3': labels[2],
                'defect4': labels[3],
            })

    # 保存CSV文件
    sliced_df = pd.DataFrame(sliced_records)
    sliced_df.to_csv(PROCESSED_CSV_PATH, index=False)

    # 验证
    total_patches = len(sliced_df)
    no_defect_patches = len(sliced_df[
                                (sliced_df['defect1'] == 0) & (sliced_df['defect2'] == 0) &
                                (sliced_df['defect3'] == 0) & (sliced_df['defect4'] == 0)
                            ])
    has_defect_patches = total_patches - no_defect_patches

    print('共生成{}个切片图片，每个切片的大小为{}'.format(total_patches, PATCH_WIDTH))
    print('其中无缺陷切片数量：{}，占比{:.2f}%'.format(no_defect_patches, no_defect_patches/total_patches*100))
    print('有缺陷的切片数量：{}，占比{:.2f}%'.format(has_defect_patches, has_defect_patches/total_patches*100))


if __name__ == '__main__':
    main()
