# -*- coding: utf-8 -*-
"""
课程 1.4 · 练习 3：MNIST CNN 模型定义与前向传播演示

本节目标：
- 深入理解 CNN 模型的每个组成部分
- 通过一个批次的数据，手动追踪张量形状的变化
- 理解卷积层、池化层如何改变图片尺寸
- 不做完整训练，只演示模型如何工作

运行: python 03-mnist_cnn_model.py

前置知识：本节不需要训练，只需要看懂模型结构即可。
"""

import torch
import torch.nn as nn

# ============================================================
# 第一步：定义 CNN 模型
# ============================================================
print("=" * 60)
print("第一步：定义 SimpleCNN 模型")
print("=" * 60)

class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络，用于 MNIST 手写数字分类

    数据流（重点！建议手动画一遍）：

    输入:    (batch, 1, 28, 28)
      ↓      Conv1 (1→32, 3x3, padding=1)
    特征1:   (batch, 32, 28, 28)   ← 尺寸不变（padding=1）
      ↓      ReLU
      ↓      MaxPool (2x2, stride=2)
    特征2:   (batch, 32, 14, 14)   ← 尺寸减半
      ↓      Conv2 (32→64, 3x3, padding=1)
    特征3:   (batch, 64, 14, 14)   ← 尺寸不变
      ↓      ReLU
      ↓      MaxPool (2x2, stride=2)
    特征4:   (batch, 64, 7, 7)    ← 尺寸再减半
      ↓      Flatten（展平）
    向量:    (batch, 3136)         ← 7x7x64 = 3136
      ↓      FC1 (3136→128)
    特征5:   (batch, 128)
      ↓      ReLU + Dropout(0.25)
      ↓      FC2 (128→10)
    输出:    (batch, 10)          ← 10个数字的分数
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积层1: 1通道 → 32通道
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 卷积层2: 32通道 → 64通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # 其他层
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 卷积层1 + 激活 + 池化: 28→14
        x = self.conv1(x)
        print(f"  Conv1 输出: {x.shape}")   # (batch, 32, 28, 28)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f"  MaxPool1 输出: {x.shape}") # (batch, 32, 14, 14)

        # 卷积层2 + 激活 + 池化: 14→7
        x = self.conv2(x)
        print(f"  Conv2 输出: {x.shape}")   # (batch, 64, 14, 14)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f"  MaxPool2 输出: {x.shape}") # (batch, 64, 7, 7)

        # 展平: 7x7x64 → 3136
        x = x.view(x.size(0), -1)
        print(f"  Flatten 后: {x.shape}")    # (batch, 3136)

        # 全连接层
        x = self.fc1(x)
        print(f"  FC1 输出: {x.shape}")     # (batch, 128)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        print(f"  FC2 输出: {x.shape}")     # (batch, 10)
        return x

# ============================================================
# 第二步：打印模型结构
# ============================================================
print("\n模型结构：")
model = SimpleCNN()
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

# ============================================================
# 第三步：追踪一个批次的数据流
# ============================================================
print("\n" + "=" * 60)
print("第三步：追踪数据流（手把手走一遍）")
print("=" * 60)

# 创建一个假的批次：1张图片，1通道，28x28
fake_batch = torch.rand(1, 1, 28, 28)
print(f"\n输入: {fake_batch.shape}")  # (1, 1, 28, 28)

output = model(fake_batch)
print(f"\n最终输出: {output.shape}")  # (1, 10)
print(f"输出内容（10个类别的分数）: {output[0].tolist()}")

# ============================================================
# 第四步：理解卷积核的工作方式
# ============================================================
print("\n" + "=" * 60)
print("第四步：理解卷积核的工作方式")
print("=" * 60)

# 用一个特定的卷积核（边缘检测）
edge_kernel = torch.tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=torch.float32).view(1, 1, 3, 3)

# 应用到一张 MNIST 图片上
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

temp_dataset = datasets.MNIST(
    root='E:/Harry AI 学习/datasets/MNIST',
    train=True,
    transform=transform,
    download=False
)

img, label = temp_dataset[0]  # 第一张图
img_batch = img.unsqueeze(0)  # 加批次维度: (1, 1, 28, 28)

# 用卷积核处理（不 padding，所以尺寸变小）
conv_edge = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
conv_edge.weight.data = edge_kernel

result = conv_edge(img_batch)
print(f"原图形状: {img_batch.shape}")          # (1, 1, 28, 28)
print(f"边缘检测后: {result.shape}")           # (1, 1, 26, 26) ← 28-3+1=26

# ============================================================
# 第五步：比较 MLP 和 CNN 的维度差异
# ============================================================
print("\n" + "=" * 60)
print("第五步：MLP vs CNN 维度对比")
print("=" * 60)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp = SimpleMLP()
mlp_params = sum(p.numel() for p in mlp.parameters())
cnn_params = sum(p.numel() for p in model.parameters())

print(f"MLP 参数量: {mlp_params:,}")
print(f"CNN 参数量: {cnn_params:,}")
print(f"CNN 比 MLP 节省: {100*(mlp_params-cnn_params)/mlp_params:.1f}% 的参数")

print("\n" + "=" * 60)
print("练习 3 完成！")
print("=" * 60)
print("本节要点：")
print("1. CNN 通过局部感受野提取空间特征")
print("2. MaxPool 让特征图尺寸减半：28→14→7")
print("3. Flatten 把 7×7×64=3136 个值展成一维向量")
print("4. FC 层做最终分类：128→10")
print("5. Dropout(p=0.25) 训练时随机丢弃 25% 神经元，防止过拟合")
