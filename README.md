# 3layernetforcifar

## CIFAR-10 图像分类 - 三层神经网络

## 项目简介

本项目实现了一个基于 NumPy 的三层神经网络，用于 CIFAR-10 数据集的图像分类任务。项目包括模型训练、测试和超参数优化。代码结构清晰，便于理解和扩展。

## 目录结构

CIFAR10-ThreeLayerNet/
├── ThreeLayerNet_CIFAR10.ipynb       # 主文件，包含训练和测试代码
├── tools/
│   ├── functions.py                  # 工具函数（ReLU、Sigmoid、Softmax 等）
│   ├── layers.py                     # 神经网络层（Affine、Relu、Sigmoid 等）
│   ├── optimizer.py                  # 优化器（SGD、Adam 等）
│   └── multi_layer_net.py            # 多层神经网络实现
├── dataset/                          # CIFAR-10 数据集（自动下载）
├── best_model_params.pkl             # 训练后的模型参数文件
└── README.md                         # 本文件

## 训练与测试

1. 数据加载与预处理：
- 使用 torchvision 加载 CIFAR-10 数据集。
- 将图像数据展平为一维数组，并归一化到 [0, 1]。
- 将标签转换为 one-hot 编码。
- 划分训练集和验证集。

2. 模型初始化：
- 构建一个三层神经网络，包括一个输入层、两个隐藏层和一个输出层。
- 隐藏层使用 ReLU 激活函数，输出层使用 Softmax 激活函数。
- 支持 L2 正则化、Dropout 和批量归一化。

3. 训练过程：
- 使用 SGD 优化器进行训练，支持学习率下降。
- 每个 epoch 计算训练损失和验证损失，并记录验证准确率。
- 保存验证准确率最高的模型参数。

4. 训练完成后，将看到以下结果：
- 训练损失和验证损失的变化曲线。
- 验证准确率的变化曲线。
- 最佳验证准确率和对应的测试准确率。
