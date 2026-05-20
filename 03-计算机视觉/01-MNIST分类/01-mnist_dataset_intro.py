# -*- coding: utf-8 -*-
"""
课程 1.4 · 练习 1：MNIST 数据集初探

本节目标：
- 使用 torchvision 加载 MNIST 数据集
- 查看数据的结构和形状
- 可视化几张图片，看看长什么样

运行: python 01-mnist_dataset_intro.py
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 第一步：设置数据预处理
# ============================================================
print("=" * 60)
print("第一步：设置数据预处理")
print("=" * 60)

# ToTensor: 把 PIL Image (0-255) 转成 PyTorch 张量 (0-1)
# Normalize: 用 MNIST 的均值和标准差做标准化
# 注意：MNIST 是灰度图，所以均值和标准差只有一个值（元组里只有一个数字）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("预处理流水线：ToTensor → Normalize(mean=0.1307, std=0.3081)")

# ============================================================
# 第二步：加载 MNIST 数据集
# ============================================================
print("\n" + "=" * 60)
print("第二步：加载 MNIST 数据集")
print("=" * 60)

# 下载训练集（第一次运行会自动下载）
train_dataset = datasets.MNIST(
    root='E:/Harry AI 学习/datasets/MNIST',
    train=True,
    transform=transform,
    download=True
)

# 下载测试集
test_dataset = datasets.MNIST(
    root='E:/Harry AI 学习/datasets/MNIST',
    train=False,
    transform=transform,
    download=True
)

print(f"训练集：{len(train_dataset)} 张图片")
print(f"测试集：{len(test_dataset)} 张图片")

# ============================================================
# 第三步：探索数据结构
# ============================================================
print("\n" + "=" * 60)
print("第三步：探索数据结构")
print("=" * 60)

# 取一张图片看看
img, label = train_dataset[0]
print(f"第一张图片的张量形状: {img.shape}")
print(f"  - img.shape = (1, 28, 28)：1个通道，28行，28列")
print(f"  - 标签（label）: {label}，表示数字 {label}")
print(f"  - 张量数值范围: [{img.min():.3f}, {img.max():.3f}]")

# 看10个不同标签的例子
print("\n训练集中 10 个类别的示例：")
for digit in range(10):
    for idx in range(len(train_dataset)):
        _, lbl = train_dataset[idx]
        if lbl == digit:
            img, _ = train_dataset[idx]
            print(f"  数字 {digit}: 数据集索引 {idx}，形状 {img.shape}")
            break

# ============================================================
# 第四步：可视化（去掉标准化，还原回原始像素）
# ============================================================
print("\n" + "=" * 60)
print("第四步：可视化图片")
print("=" * 60)

# 注意：Normalization 会把数值范围从 (0,1) 变成大致 (-1,1)
# 可视化时需要反标准化才能看到正常图片
inv_transform = transforms.Compose([
    transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
])

# 画 2 行 5 列，共 10 张图片
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(10):
    img, label = train_dataset[i]
    # 反标准化，还原到 0-1 范围
    img_original = inv_transform(img).squeeze()
    axes[i].imshow(img_original, cmap='gray')
    axes[i].set_title(f'标签: {label}', fontsize=14)
    axes[i].axis('off')

plt.suptitle('MNIST 手写数字示例（前10张）', fontsize=16)
plt.tight_layout()
plt.savefig('E:/Harry AI 学习/03-计算机视觉/01-MNIST分类/mnist_samples.png', dpi=150)
print("图片已保存: mnist_samples.png")

# 另一种可视化：不反标准化，直接看
# 但此时图片偏白，因为 Normalize 把均值附近的数据拉到 0
fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
axes2 = axes2.flatten()
for i in range(10):
    img, label = test_dataset[i]
    axes2[i].imshow(img.squeeze(), cmap='gray')
    axes2[i].set_title(f'标签: {label}', fontsize=14)
    axes2[i].axis('off')
plt.suptitle('测试集前10张（标准化后）', fontsize=16)
plt.tight_layout()
plt.savefig('E:/Harry AI 学习/03-计算机视觉/01-MNIST分类/mnist_test_samples.png', dpi=150)
print("测试集图片已保存: mnist_test_samples.png")

# ============================================================
# 第五步：统计信息
# ============================================================
print("\n" + "=" * 60)
print("第五步：统计信息")
print("=" * 60)

# 统计训练集中每个数字的数量
from collections import Counter
train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]

print("训练集标签分布：")
for digit, count in sorted(Counter(train_labels).items()):
    print(f"  数字 {digit}: {count} 张 ({100*count/len(train_dataset):.1f}%)")

print("\n测试集标签分布：")
for digit, count in sorted(Counter(test_labels).items()):
    print(f"  数字 {digit}: {count} 张 ({100*count/len(test_dataset):.1f}%)")

print("\n" + "=" * 60)
print("练习 1 完成！")
print("=" * 60)
print("本节要点：")
print("1. MNIST 图片形状是 (1, 28, 28)：1通道灰度图")
print("2. transforms.ToTensor() 把像素从 0-255 转成 0-1")
print("3. transforms.Normalize 用 (0.1307,) 和 (0.3081,) 做标准化")
print("4. 训练集 60000 张，测试集 10000 张，每个类别大致均匀分布")
