# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : predict.py
@Author  : Quichile
@Time    : 2026/5/12 10:26
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from config import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SteelDefectDetector:
    def __init__(self, onnx_model_path):
        """初始化ONNX"""
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = ['Defect 1', 'Defect 2', 'Defect 3', 'Defect 4']

    def preprocess(self, img):
        """预处理单张切片"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def predict_long_image(self, img_path):
        """端到端预测长图"""
        img = cv2.imread(img_path)
        if img is None:
            print('图片读取失败！')
            return

        h, w = img.shape[:2]
        n_patches = w // 256

        all_probs = []                                          # 存放所有patches的概率，最后取最大值

        for i in range(n_patches):
            patch = img[:, i * PATCH_WIDTH:(i + 1) * PATCH_WIDTH]

            # 预处理并推理
            input_img = self.preprocess(patch)
            logits = self.session.run(None, {self.input_name: input_img})[0]
            probs = sigmoid(logits)[0]                          # 取四个类别的概率
            all_probs.append(probs)

            # 在原图上标注该patch的预测结果
            for cls_id in range(4):
                if probs[cls_id] > 0.5:
                    label_text = f'{self.class_names[cls_id]}: {probs[cls_id]*100:.1f}({os.path.basename(onnx_path)})'
                    cv2.putText(img=img, text=label_text, org=(i * 256 + 10, 30 + cls_id * 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=2)

        # 对所有patch的置信度取最大值
        final_probs = np.max(all_probs, axis=0)
        final_preds = (final_probs > 0.5).astype(int)           # 根据阈值确定标签

        print('图片{}检测结果：'.format(os.path.basename(img_path)))
        for i in range(4):
            status = '有缺陷❌' if final_preds[i] == 1 else '该图片正常✔'
            print('{}：最高置信度{:.2f}% -> {}'.format(self.class_names[i], final_probs[i]*100, status))

        save_dir = './runs/predict_results'
        os.makedirs(save_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(onnx_path))[0]
        save_path = os.path.join(save_dir, f'result_{name}_{os.path.basename(img_path)}')
        cv2.imwrite(save_path, img)
        print('可视化结果已保存至{}'.format(save_path))


if __name__ == '__main__':
    onnx_path = './checkpoints/ResNet18_ViT_hybrid.onnx'
    detector = SteelDefectDetector(onnx_model_path=onnx_path)

    test_long_img = './data/raw/train_images/0a9aaba9a.jpg'
    detector.predict_long_image(test_long_img)
