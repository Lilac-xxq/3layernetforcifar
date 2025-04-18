# 3layernetforcifar

## CIFAR-10 图像分类 - 三层神经网络

## 项目简介

本项目实现了一个基于 NumPy 的三层神经网络，用于 CIFAR-10 数据集的图像分类任务。项目包括模型训练、测试和超参数优化。代码结构清晰，便于理解和扩展。

## 目录结构

- ThreeLayerNet_CIFAR10.ipynb   - 主文件，包含用于训练和测试三层神经网络的代码。
- params_visualize.py           - 参数可视化
- tools/functions.py            - 包含神经网络中使用的工具函数，如ReLU、Sigmoid、Softmax等激活函数。
- tools/layers.py               - 定义了神经网络中的各种层，包括Affine层、Relu层、Sigmoid层等。
- tools/optimizer.py            - 包含了不同的优化算法，如SGD（随机梯度下降）、Adam等
- tools/multi_layer_net.py      - 实现了多层神经网络的结构和前向传播、反向传播等核心功能。
- tools/visualize.py            - 包含用于参数可视化的工具

- visualizations/               - 参数可视化结果
- dataset/                      - CIFAR-10 数据集存放目录，数据集会在代码运行时自动下载。
- best_model_params.pkl         - 存储训练完成后的最优模型参数，用于后续的模型测试或部署。
- README.md                     - 项目说明文件，包含项目的介绍、运行指南和其他必要信息。
- 三层神经网络分类器实验报告.pdf         - 实验报告，包含模型介绍、训练与测试、超参数寻优、参数可视化等环节 

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
