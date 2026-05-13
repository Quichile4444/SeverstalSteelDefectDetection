# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : rle_utils.py
@Author  : Quichile
@Time    : 2026/4/29 16:33
"""

import numpy as np
import pandas as pd


def rle_decode(rle_string: str, shape: tuple) -> np.ndarray:
    """
    将RLE字符串解码为二值掩码矩阵，0为正常，1表示有缺陷

    :param rle_string: 标注文件中的EncodedPixels，如果为空说明没缺陷
    :param shape: 图像的尺寸：Height，Width
    :return: 一个二维numpy数组，形状为256*6000，代表是否缺陷，值为0或1
    """
    # 如果字符串为NaN，则无缺陷，返回全零数组
    if pd.isna(rle_string) or str(rle_string).strip() == '':    # .strip去掉str两边的空格、换行/制表符
        return np.zeros(shape, dtype=np.uint8)

    numbers = list(map(int, str(rle_string).strip().split()))   # 把字符串根据空格切开，批量变成整数，再合并成列表
    # 两两数字第一位为起始位置，第二位为缺陷长度
    starts = numbers[0::2]
    lengths = numbers[1::2]

    # 开始解码，0为正常，EncodedPixels为空，1代表有缺陷，不为空。先置0.后设1
    mask_flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts,lengths):                   # zip将一对一捆绑起来
        mask_flat[start: start + length] = 1

    mask_2d = mask_flat.reshape(shape, order='F')               # 重新转化为二维矩阵，F代表列优先

    return mask_2d


def calculate_ratio(rle_string: str, shape: tuple) -> float:
    """
    用于计算某个缺陷在整张图像的大想比例，查看缺陷是否小

    :param rle_string: Run-Length Encoding编码
    :param shape: 图像尺寸
    :return: 占比
    """
    if pd.isna(rle_string) or str(rle_string).strip() == '':
        return 0.0

    numbers = list(map(int, str(rle_string).strip().split()))
    lengths = numbers[1::2]

    defect_pixels = sum(lengths)                                # 计算图像的缺陷像素数
    total_pixels = shape[0] * shape[1]                          # 计算整张图像素数

    return defect_pixels / total_pixels
