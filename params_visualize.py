import numpy as np
import os
import pickle
from tools.visualize import visualize_all
from torchvision import datasets, transforms

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 下载并加载数据集
train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

# 提取测试数据和标签
x_test = test_set.data  # 测试集数据
y_test = np.array(test_set.targets)  # 测试集标签

# 数据预处理
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0  # 展平并归一化

# 加载训练好的模型参数
with open('best_model_params.pkl', 'rb') as f:
    params = pickle.load(f)

# 选择一个测试样本进行可视化
sample_image = x_test[0].reshape(1, -1)  # 选择第一个测试样本

# 执行可视化
visualize_all(params, sample_image, save_dir='visualizations')