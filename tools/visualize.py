import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

def visualize_weights(params, save_dir='visualizations'):
    """
    可视化网络权重
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取权重
    W1 = params['W1']
    W2 = params['W2']
    
    # 可视化第一层权重（滤波器）
    plt.figure(figsize=(15, 10))
    for i in range(16):  # 显示前16个滤波器
        plt.subplot(4, 4, i+1)
        # 将权重重塑为图像形式 (32x32x3)
        filter_img = W1[:, i].reshape(32, 32, 3)
        # 归一化到[0,1]范围
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        plt.imshow(filter_img)
        plt.axis('off')
    plt.suptitle('First Layer Filters')
    plt.savefig(os.path.join(save_dir, 'first_layer_filters.png'))
    plt.close()
    
    # 绘制权重直方图
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(W1.flatten(), bins=50)
    plt.title('W1 Distribution')
    plt.subplot(1, 2, 2)
    plt.hist(W2.flatten(), bins=50)
    plt.title('W2 Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_distributions.png'))
    plt.close()

def visualize_activations(params, sample_image, save_dir='visualizations'):
    os.makedirs(save_dir, exist_ok=True)

    # 第一层激活值
    W1 = params['W1']
    b1 = params['b1']
    z1 = np.dot(sample_image, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU激活函数

    # 第二层激活值
    W2 = params['W2']
    b2 = params['b2']
    z2 = np.dot(a1, W2) + b2
    a2 = np.maximum(0, z2)  # ReLU激活函数

    # 可视化第一层激活值（a1）
    plt.figure(figsize=(10, 2))
    sns.heatmap(a1, cmap='viridis', cbar=True)
    plt.title('First Layer Activations (a1)')
    plt.xlabel('Hidden Units')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'first_layer_activations.png'))
    plt.close()

    # 可视化第二层激活值（a2）
    plt.figure(figsize=(10, 2))
    sns.heatmap(a2, cmap='viridis', cbar=True)
    plt.title('Second Layer Activations (a2)')
    plt.xlabel('Classes')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'second_layer_activations.png'))
    plt.close()

def visualize_all(params, sample_image, save_dir='visualizations'):
    """
    执行所有可视化
    """
    visualize_weights(params, save_dir)
    visualize_activations(params, sample_image, save_dir)