# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : data_analysis.py
@Author  : Quichile
@Time    : 2026/4/29 18:34
"""

# ================= 步骤一 获取数据并分析 ====================
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import *
from preprocess.rle_utils import calculate_ratio                # 导入缺陷面积比例的计算函数

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False


def main():
    df = pd.read_csv(RAW_CSV_PATH)
    total_csv = len(df)

    # 检索所有图片名，同时取消后缀名。获取图片数和缺陷行数
    all_images = [f.replace('.jpg', '') for f in os.listdir(RAW_IMG_DIR) if f.endswith('.jpg')]
    total_images = len(all_images)
    print('CSV文件中记录缺陷的行数：{}'.format(total_csv))
    print('train_images文件夹中的图片总数：{}\n'.format(total_images))

    # 从CSV中提取有缺陷记录的图片数量
    defect_images = df['ImageId'].unique()
    defect_count = len(defect_images)
    no_defect_count = total_images - defect_count
    print('有缺陷的图片数量：{}'.format(defect_count))
    print('完全没有缺陷的图片数量：{}\n'.format(no_defect_count))

    class_counts = []                                           # 统计4类缺陷各自多少张图
    class_ratios = []                                           # 各自类别的缺陷占比
    for i in range(1, 5):
        class_df = df[df['ClassId'] == i]
        count = len(class_df)
        class_counts.append(count)                              # 当前缺陷的图像数量

        # 计算当前缺陷类别的面积占比
        ratios = class_df['EncodedPixels'].apply(lambda x: calculate_ratio(x, (256, 6000)))
        mean_ratio = ratios.mean()                              # 每类缺陷面积占比取该类别的平均
        class_ratios.append(mean_ratio)

        print('类别{}：共{}张图片有缺陷（占全部图片的{:.2f}），平均缺陷面积占比：{:.2f}%'
              .format(i, count, count/total_images * 100, mean_ratio * 100))

    # 画图
    os.makedirs('./runs', exist_ok=True)                        # 创建文件夹

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图记录各类别有缺陷的图像数量
    classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    colors = ['red', 'blue', 'yellow', 'green']
    axes[0].bar(x=classes, height=class_counts, color=colors)
    axes[0].set_title('每类缺陷的图片数量')
    axes[0].set_ylabel('图片数量')
    for i, v in enumerate(class_counts):
        # 在柱状图上添加文字，xy表示XY轴坐标，v表示柱子高度（原本缺陷图片数），+65表示往上65个单位显示文字，
        # s代表要显示的文字内容，ha表示水平对齐方式，即水平居中
        axes[0].text(x=i, y=v + 10, s=str(v), ha='center')

    # 右图记录各类别缺陷的面积占比
    axes[1].bar(x=classes, height=[r * 100 for r in class_ratios], color=colors)
    axes[1].set_title('每类缺陷平均面积占比')
    axes[1].set_ylabel('面积占比（%）')
    for i, v in enumerate(class_ratios):
        # 注意！v原本就是百分数，结果平均为 1%，如果和上面一样＋10会爆界限，如果+小一点的数字（2.2401，加多了，全在上；0.01，加少了，都在下）
        # 究其原因是因为基数太小了，得让基数变大（乘100），让值平均为1，此时值也需小，否则超过2.4
        axes[1].text(x=i, y=v * 100 + 0.01, s=f'{v * 100:.4f}', ha='center')

    # 标注总图片数、缺陷数和无缺陷数
    fig.suptitle('Total images: {}, with defects: {}, without defects: {}'
                 .format(total_images, defect_count, no_defect_count))

    plt.tight_layout()
    plt.savefig('./runs/data_analysis.png', dpi=150)
    print('你已成功将图片data_analysis.png保存到./runs文件夹了')
    plt.show()


if __name__ == '__main__':
    main()
