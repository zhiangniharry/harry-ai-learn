# -*- coding: utf-8 -*-
"""
课程 1.4: MNIST 手写数字分类（CNN 从头训练）

本代码演示：
1. 使用 torchvision 加载 MNIST 数据集
2. 定义一个简单的卷积神经网络（CNN）
3. 完整的训练循环 + 验证
4. 在测试集上评估准确率
5. 保存训练好的模型

数据集: MNIST (手写数字 0-9, 28x28 灰度图)
训练集: 60000 张, 测试集: 10000 张
目标: 识别手写数字（0-9）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os

# ============================================================
# 第一步：设置（超参数）
# ============================================================
print("=" * 60)
print("第一步：设置（超参数）")
print("=" * 60)

# 超参数（hyperparameters）：控制训练过程的参数
BATCH_SIZE = 64      # 每批处理多少张图片。64 是常用的默认值
EPOCHS = 5           # 训练几轮（一轮 = 看一遍整个训练集）
LEARNING_RATE = 0.001  # 学习率：每次参数更新的步长大小
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"批次大小 (batch_size): {BATCH_SIZE}")
print(f"训练轮数 (epochs): {EPOCHS}")
print(f"学习率 (learning_rate): {LEARNING_RATE}")
print(f"计算设备: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("注意: 没有检测到 GPU，使用 CPU 训练（会慢很多）")

print()

# ============================================================
# 第二步：准备数据
# ============================================================
print("=" * 60)
print("第二步：准备数据（下载 MNIST）")
print("=" * 60)

# 数据预处理：把图片转换成 PyTorch 能用的张量格式
# ToTensor: 把 PIL 图片 (0-255) 转成 (0-1) 的张量
# Normalize: 标准化，让图片数值在 (-1, 1) 区间（训练更稳定）
transform = transforms.Compose([
    transforms.ToTensor(),           # 转换成张量 (1, 28, 28)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

# 下载训练集和测试集
# train=True 表示下载训练集，train=False 表示测试集
# transform 应用上面的预处理
# download=True 如果本地没有就自动下载
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

# DataLoader：把数据打包成批次，每批 BATCH_SIZE 张
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,      # 打乱顺序（训练时 shuffle=True 让模型学得更好）
    num_workers=0      # Windows 下建议用 0，避免多进程问题
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,     # 测试时不需要打乱
    num_workers=0
)

print(f"训练集: {len(train_dataset)} 张图片")
print(f"测试集: {len(test_dataset)} 张图片")
print(f"图片尺寸: 28 x 28 灰度图")
print(f"类别数: 10 (数字 0-9)")
print(f"每轮训练批次数: {len(train_loader)}")
print(f"每轮测试批次数: {len(test_loader)}")
print()

# 看一张图片验证
sample_data, sample_label = train_dataset[0]
print(f"示例图片：张量形状 {sample_data.shape}，标签 {sample_label}（数字 {sample_label}）")
print()

# ============================================================
# 第三步：定义 CNN 模型
# ============================================================
print("=" * 60)
print("第三步：定义卷积神经网络（CNN）")
print("=" * 60)

class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络，用于 MNIST 手写数字分类

    结构：
    Conv1 (卷积层1): 1通道输入 → 32通道，3x3卷积核
    ReLU: 激活函数，引入非线性
    MaxPool: 2x2 池化，图片缩小一半 (28→14)

    Conv2 (卷积层2): 32通道 → 64通道，3x3卷积核
    ReLU: 激活函数
    MaxPool: 图片再缩小一半 (14→7)

    Flatten: 把 7x7x64 = 3136 个值展成一维向量
    FC1 (全连接层1): 3136 → 128
    ReLU: 激活函数
    Dropout: 随机丢弃 25% 神经元，防止过拟合
    FC2 (全连接层2): 128 → 10（对应 10 个数字类别）
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积层 1: 输入 1 通道（灰度图），输出 32 通道
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 卷积层 2: 输入 32 通道，输出 64 通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 全连接层 1: 输入 7*7*64=3136，输出 128
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=128)
        # 全连接层 2: 输入 128，输出 10（数字 0-9）
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # Dropout: 训练时随机丢弃 25% 的神经元
        self.dropout = nn.Dropout(p=0.25)
        # ReLU 激活函数
        self.relu = nn.ReLU()
        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 卷积层 1 + 激活 + 池化: (batch, 1, 28, 28) → (batch, 32, 14, 14)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 卷积层 2 + 激活 + 池化: (batch, 32, 14, 14) → (batch, 64, 7, 7)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 展平: (batch, 64, 7, 7) → (batch, 3136)
        x = x.view(x.size(0), -1)

        # 全连接层 1 + 激活 + Dropout: (batch, 3136) → (batch, 128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 全连接层 2（输出层）: (batch, 128) → (batch, 10)
        x = self.fc2(x)
        return x

# 创建模型实例，移动到 GPU（如果可用）
model = SimpleCNN().to(DEVICE)

# 打印模型结构
print("模型结构:")
print(model)
print()

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print()

# ============================================================
# 第四步：定义损失函数和优化器
# ============================================================
print("=" * 60)
print("第四步：定义损失函数和优化器")
print("=" * 60)

# CrossEntropyLoss（交叉熵损失）：多分类问题的标准损失函数
# 它会自动做 Softmax，把输出转成概率分布
criterion = nn.CrossEntropyLoss()

# Adam 优化器：自动调节每个参数的学习率（比 SGD 更傻瓜、更常用）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"损失函数: CrossEntropyLoss（交叉熵损失）")
print(f"优化器: Adam (lr={LEARNING_RATE})")
print()

# ============================================================
# 第五步：训练函数
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一轮的函数

    对每个 batch:
    1. 前向传播：模型预测
    2. 计算损失：预测 vs 真实标签
    3. 反向传播：计算梯度
    4. 优化器更新：用梯度调整参数
    """
    model.train()  # 切换到训练模式（启用 Dropout 和 BatchNorm）

    running_loss = 0.0      # 累计损失
    correct = 0             # 累计正确预测数
    total = 0               # 累计总图片数

    for batch_idx, (data, target) in enumerate(train_loader):
        # 把数据移动到 GPU（如果可用）
        data, target = data.to(device), target.to(device)

        # 1. 前向传播：模型预测
        output = model(data)

        # 2. 计算损失
        loss = criterion(output, target)

        # 3. 清零梯度（否则会累加）
        optimizer.zero_grad()

        # 4. 反向传播：计算梯度
        loss.backward()

        # 5. 优化器更新：调整参数
        optimizer.step()

        # 累计统计
        running_loss += loss.item()
        _, predicted = output.max(1)  # 取概率最大的类别作为预测
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 每 100 个批次打印一次进度
        if (batch_idx + 1) % 100 == 0:
            current_acc = 100. * correct / total
            print(f"  批次 {batch_idx+1:3d}/{len(train_loader)} | "
                  f"损失 {loss.item():.4f} | 当前准确率 {current_acc:.2f}%")

    # 这一轮的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================
# 第六步：测试函数
# ============================================================
def test(model, test_loader, criterion, device):
    """
    在测试集上评估模型（不看标签，只预测）
    """
    model.eval()  # 切换到评估模式（禁用 Dropout）

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不需要计算梯度（节省显存和计算）
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

# ============================================================
# 第七步：开始训练
# ============================================================
print("=" * 60)
print("第七步：开始训练")
print("=" * 60)
print(f"开始时间: {time.strftime('%H:%M:%S')}")
print()

best_test_acc = 0.0  # 记录最佳测试准确率

for epoch in range(1, EPOCHS + 1):
    print(f"----- 第 {epoch} / {EPOCHS} 轮训练 -----")
    epoch_start = time.time()

    # 训练一轮
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

    # 在测试集上评估
    test_loss, test_acc = test(model, test_loader, criterion, DEVICE)

    epoch_time = time.time() - epoch_start

    print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
    print(f"  测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2f}%")
    print(f"  用时: {epoch_time:.1f} 秒")
    print()

    # 保存最佳模型
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        save_path = 'E:/Harry AI 学习/models/mnist_cnn_best.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
        }, save_path)
        print(f"  ★ 保存最佳模型到: {save_path}")
        print(f"  ★ 最佳测试准确率: {best_test_acc:.2f}%")
    print()

print("=" * 60)
print("训练完成！")
print("=" * 60)
print(f"最佳测试准确率: {best_test_acc:.2f}%")
print(f"结束时间: {time.strftime('%H:%M:%S')}")

# ============================================================
# 第八步：加载最佳模型，做最终评估
# ============================================================
print()
print("=" * 60)
print("第八步：加载最佳模型，最终评估")
print("=" * 60)

# 加载最佳模型
checkpoint = torch.load('E:/Harry AI 学习/models/mnist_cnn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"从 {checkpoint} 加载最佳模型（第 {checkpoint['epoch']} 轮）")

# 最终测试
final_loss, final_acc = test(model, test_loader, criterion, DEVICE)
print(f"最终测试准确率: {final_acc:.2f}%")
print()

# 展示一些预测结果
print("一些预测示例（前 10 张测试图片）:")
model.eval()
with torch.no_grad():
    for i in range(10):
        img, label = test_dataset[i]
        img_batch = img.unsqueeze(0).to(DEVICE)
        output = model(img_batch)
        prob = torch.softmax(output, dim=1)[0]
        pred = output.argmax(1).item()
        confidence = prob[pred].item() * 100
        print(f"  图片 {i+1}: 真实标签={label}，预测={pred}，置信度={confidence:.1f}%")

print()
print("代码结束！模型已保存到 E:/Harry AI 学习/models/mnist_cnn_best.pth")
