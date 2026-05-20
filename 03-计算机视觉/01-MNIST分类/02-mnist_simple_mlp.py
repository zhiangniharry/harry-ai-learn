# -*- coding: utf-8 -*-
"""
课程 1.4 · 练习 2：MNIST 简单 MLP（多层感知机）基线

本节目标：
- 用最简单的 MLP（没有卷积）做 MNIST 分类
- 理解为什么 MLP 在图像任务上不如 CNN
- 感受一下准确率能到多少

运行: python 02-mnist_simple_mlp.py

对比结果：
- MLP（练习2）：约 97-98%
- CNN（练习3）：约 98-99%
差距不大，但 CNN 更高效，参数量更少
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ============================================================
# 第一步：超参数
# ============================================================
print("=" * 60)
print("第一步：超参数设置")
print("=" * 60)

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"批次大小: {BATCH_SIZE}")
print(f"训练轮数: {EPOCHS}")
print(f"学习率: {LEARNING_RATE}")
print(f"计算设备: {DEVICE}")

# ============================================================
# 第二步：加载数据
# ============================================================
print("\n" + "=" * 60)
print("第二步：加载 MNIST 数据")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='E:/Harry AI 学习/datasets/MNIST',
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root='E:/Harry AI 学习/datasets/MNIST',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"训练集: {len(train_dataset)} 张 | 每轮 {len(train_loader)} 批")
print(f"测试集: {len(test_dataset)} 张 | 每轮 {len(test_loader)} 批")

# ============================================================
# 第三步：定义 MLP 模型
# ============================================================
print("\n" + "=" * 60)
print("第三步：定义 MLP 模型（多层感知机）")
print("=" * 60)

class SimpleMLP(nn.Module):
    """
    最简单的多层感知机（MLP）

    MLP 把图片（28x28=784个像素）当作一维向量处理：
    没有任何空间结构信息，所有像素平等对待。

    结构：
    输入层:   784 个神经元（28x28 展平）
    隐藏层1:  256 个神经元 + ReLU
    隐藏层2:  128 个神经元 + ReLU
    输出层:   10  个神经元（0-9 十个类别）
    """

    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 三层全连接网络
        self.fc1 = nn.Linear(28 * 28, 256)   # 输入 → 隐藏1
        self.fc2 = nn.Linear(256, 128)       # 隐藏1 → 隐藏2
        self.fc3 = nn.Linear(128, 10)        # 隐藏2 → 输出
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. 展平: (batch, 1, 28, 28) → (batch, 784)
        x = x.view(x.size(0), -1)

        # 2. 全连接层 + 激活
        x = self.relu(self.fc1(x))  # 784 → 256
        x = self.relu(self.fc2(x))  # 256 → 128
        x = self.fc3(x)             # 128 → 10（输出logit，不做激活，后面交给CrossEntropyLoss）
        return x

model = SimpleMLP().to(DEVICE)

# 打印模型结构
print("MLP 模型结构：")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")
print(f"  对比 CNN 约 42 万参数，MLP 反而更多（~23 万），但效果更差")

# ============================================================
# 第四步：训练和测试
# ============================================================
print("\n" + "=" * 60)
print("第四步：开始训练")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_test_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # ---- 训练 ----
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    epoch_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_acc = 100. * correct / total
    epoch_time = time.time() - epoch_start

    # ---- 测试 ----
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_acc = 100. * correct / total
    test_loss = test_loss / len(test_loader)

    print(f"第 {epoch}/{EPOCHS} 轮 | "
          f"训练准确率 {train_acc:.2f}% | "
          f"测试准确率 {test_acc:.2f}% | "
          f"用时 {epoch_time:.1f}s")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'E:/Harry AI 学习/models/mnist_mlp_best.pth')
        print(f"  ★ 保存最佳 MLP 模型: {best_test_acc:.2f}%")

# ============================================================
# 第五步：总结
# ============================================================
print("\n" + "=" * 60)
print("第五步：结果总结")
print("=" * 60)
print(f"最佳测试准确率: {best_test_acc:.2f}%")
print("\nMLP 的问题：")
print("  1. 把图片展平成一维向量，丢失了空间结构（像素位置关系）")
print("  2. 每个像素独立看待，忽略了相邻像素的相关性")
print("  3. 参数量并不少，但提取特征的能力弱")
print("\nCNN 的优势：")
print("  1. 局部感受野，只看相邻像素")
print("  2. 权重共享，大幅减少参数量")
print("  3. 平移不变性，识别任意位置的同一特征")
print("\n练习 2 完成！接下来做练习 3，体验 CNN 的威力。")
