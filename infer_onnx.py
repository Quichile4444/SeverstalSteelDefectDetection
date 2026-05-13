# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : infer_onnx.py
@Author  : Quichile
@Time    : 2026/5/12 8:46
"""

import time
import numpy as np
import cv2
import onnxruntime as ort


def sigmoid(x):                                 # ONNX导出的是logits（原始输出），需转成0-1的概率
    return 1 / (1 + np.exp(-x))


def preprocess_image_opencv(img_path: str):
    """OpenCV预处理，对齐pytorch的transform"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))           # 设置大小，防止传入的图片大小错误
    img = img.astype(np.float32) / 255.0        # 归一化到0-1

    # ImageNet标准化/std标准化（和训练时的Normalize对应）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW，增加batch维度(256, 256, 3)->(3, 256, 256)->(1, 3, 256, 256)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)           # ONNX要求批量输入
    return img


def infer_single_patch(onnx_path: str, img_path: str):
    # 加载ONNX，创建推理会话
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name   # 获取输入节点名称

    # 预处理并计时
    start_preprocess = time.time()
    img_input = preprocess_image_opencv(img_path)
    time_preprocess = (time.time() - start_preprocess) * 1000  # ms

    # 模型推理并计时
    start_infer = time.time()
    logits = session.run(None, {input_name: img_input})[0]      # (1, 4)
    time_infer = (time.time() - start_infer) * 1000

    # 后处理并计时
    start_postprocess = time.time()
    probs = sigmoid(logits)                                     # logits转化为概率
    preds = (probs > 0.5).astype(int)                           # 按阈值0.5划分0/1标签
    time_postprocess = (time.time() - start_postprocess) * 1000

    print('预处理耗时：{:.2f} ms'.format(time_preprocess))
    print('模型推理耗时：{:.2f} ms'.format(time_infer))
    print('后处理耗时：{:.2f} ms'.format(time_postprocess))
    print('预测概率：{}'.format(probs[0]))
    print('预测标签：{}'.format(preds[0]))


if __name__ == '__main__':
    test_img = './data/processed/train_images_sliced/602bfaba5.jpg_patch0.jpg'
    onnx_model = './checkpoints/ResNet18_ViT_hybrid.onnx'
    infer_single_patch(onnx_model, test_img)
