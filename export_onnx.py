# -*- coding: UTF-8 -*-
"""
@Project : SeverstalSteelDefect 
@File    : export_onnx.py
@Author  : Quichile
@Time    : 2026/5/11 21:24
"""

import torch
import numpy as np
import onnx                                                     # ONNX核心库，用于校验模型结构
import onnxruntime as ort                                       # ONNX推理引擎，用于验证推理结果
from models.resnet18_baseline import ResNet18Baseline
from models.resnet_vit_hybrid import ResNetVitHybrid


def export_to_onnx(model_class, pth_path: str, onnx_path: str):
    # 加载模型结构和权重
    model = model_class()
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)                   # 构造虚拟输入，形状需和模型实际输入一致

    # 执行导出
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,                                              # onnx文件保存路径
        input_names=['input'],                                  # 输入节点的名字
        output_names=['output'],
        opset_version=14,                                       # ONNX算子版本
        dynamic_axes={                                          # 设置动态维度，使得推理时可以输入不同batch_size
            'input': {0: 'batch_size'},                         # input的第0维（batch_size）动态
            'output': {0: 'batch_size'}                         # output的第0维动态
        }
    )
    print('ONNX文件已保存至：{}'.format(onnx_path))

    # 检查ONNX模型是否合法
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)                         # 检查计算图是否有错误（如算子不支持、维度不匹配）
    print('1、ONNX模型结构验证通过')

    # 对比PyTroch和ONNX Runtime的输出，确保精度不丢失
    with torch.no_grad():                                       # PyTroch推理
        pytorch_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(onnx_path)               # 初始化ONNX推理会话
    # ONNX推理：run(输出节点名(None表示所有), {输入名: 输入数据})。返回输出列表（取第0个为结果）
    ort_output = ort_session.run(None, {'input': dummy_input.numpy()})[0]
    # 数值对比：要求相对误差(rtol)和绝对误差(atol)均小于1e-5
    np.testing.assert_allclose(pytorch_output, ort_output, rtol=1e-5, atol=1e-5)
    print('2、精度验证通过，输出误差小于1e-5')


if __name__ == '__main__':
    export_to_onnx(
        model_class=ResNet18Baseline,
        pth_path='./checkpoints/ResNet18_baseline_best.pth',
        onnx_path='./checkpoints/ResNet18_baseline.onnx'
    )

    export_to_onnx(
        model_class=ResNetVitHybrid,
        pth_path='./checkpoints/ResNet18_ViT_hybrid_best.pth',
        onnx_path='./checkpoints/ResNet18_ViT_hybrid.onnx'
    )
