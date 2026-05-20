# -*- coding: utf-8 -*-
"""
课程 1.3: PyTorch 训练循环
===========================================
本代码教你:
1. 前向传播 - 模型预测
2. 计算损失 - 预测有多差
3. 反向传播 - 找出谁的错
4. 更新权重 - 改正错误
5. 保存模型 - 留住成果

运行方式:
    python 03-training_loop.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# 第一部分: 准备数据
# ============================================================
print("=" * 60)
print("第一部分: 准备数据")
print("=" * 60)

print("\n[我们用上一课学的 Dataset 和 DataLoader 准备数据]")
print("[任务: 给定 x，预测 y = 2*x + 1]")
print("[这是一个简单的线性回归问题，适合理解训练流程]")


class NumberDataset(Dataset):
    """数字数据集"""
    def __init__(self, start=0, end=100):
        self.x = torch.arange(start, end, dtype=torch.float32).unsqueeze(1)
        self.y = 2 * self.x + 1
        print(f"   [OK] 生成 {len(self.x)} 个数据点")
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 创建训练集和测试集
train_dataset = NumberDataset(0, 80)   # 80% 训练
test_dataset = NumberDataset(80, 100)  # 20% 测试

# 保持原始数据，不做归一化
# 模型直接学习 y = 2*x + 1
print("\n[保持原始数据，模型直接学习 y = 2*x + 1]")
print("[weight 目标值: 2.0, bias 目标值: 1.0]")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"\n[数据划分:]")
print(f"   训练集: {len(train_dataset)} 个样本")
print(f"   测试集: {len(test_dataset)} 个样本")
print(f"   批次大小: 16")
print(f"   训练批次数: {len(train_loader)}")


# ============================================================
# 第二部分: 定义模型
# ============================================================
print("\n" + "=" * 60)
print("第二部分: 定义模型")
print("=" * 60)

print("\n[什么是模型?]")
print("   模型就是一个数学函数: y = f(x)")
print("   我们的目标是找到正确的 f，让预测尽量准确")
print("   对于 y = 2x + 1，正确的 f 就是: f(x) = 2*x + 1")
print("   但我们不知道 2 和 1，要让模型自己学出来")

print("\n[神经网络的基本单元: 线性层]")
print("   线性层的公式: y = weight * x + bias")
print("   - weight: 权重，对应我们要学的 '2'")
print("   - bias: 偏置，对应我们要学的 '1'")
print("   一开始 weight 和 bias 是随机值，训练后接近真实值")


class SimpleModel(nn.Module):
    """
    最简单的神经网络模型
    只有一个线性层: y = weight * x + bias
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 定义一个线性层: 输入1维, 输出1维
        self.linear = nn.Linear(in_features=1, out_features=1)
        print("   [OK] 模型创建完成")
        print(f"   初始 weight: {self.linear.weight.item():.4f}")
        print(f"   初始 bias: {self.linear.bias.item():.4f}")
    
    def forward(self, x):
        """
        前向传播: 输入 x，输出预测值
        """
        return self.linear(x)


# 创建模型
model = SimpleModel()

# 把模型移到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\n[模型运行在: {device}]")


# ============================================================
# 第三部分: 定义损失函数和优化器
# ============================================================
print("\n" + "=" * 60)
print("第三部分: 损失函数和优化器")
print("=" * 60)

print("\n[损失函数 (Loss Function)]")
print("   作用: 衡量预测值和真实值之间的差距")
print("   差距越大，损失越大，模型越差")
print("   差距越小，损失越小，模型越好")
print("\n   常用损失函数:")
print("   - MSELoss: 均方误差，用于回归问题（预测连续值）")
print("   - CrossEntropyLoss: 交叉熵，用于分类问题")
print("   我们用 MSELoss，因为 y = 2x + 1 是连续值")

criterion = nn.MSELoss()
print("\n   [OK] 使用 MSELoss (均方误差)")

print("\n[优化器 (Optimizer)]")
print("   作用: 根据损失的梯度，更新模型的参数")
print("   类比: 爬山时，梯度告诉你哪个方向最陡")
print("   优化器就是沿着最陡的反方向走一步")
print("\n   常用优化器:")
print("   - SGD: 随机梯度下降，最基础")
print("   - Adam: 自适应学习率，最常用，效果通常最好")
print("   我们用 Adam，lr=0.01 (学习率)")

optimizer = optim.Adam(model.parameters(), lr=0.01)
print("\n   [OK] 使用 Adam 优化器，学习率 0.01")


# ============================================================
# 第四部分: 训练循环（核心！）
# ============================================================
print("\n" + "=" * 60)
print("第四部分: 训练循环（核心！）")
print("=" * 60)

print("\n[训练循环的 5 个步骤:]")
print("   1. 前向传播: 输入数据，得到预测")
print("   2. 计算损失: 预测 vs 真实值")
print("   3. 反向传播: 计算梯度（谁的错）")
print("   4. 更新权重: 用优化器调整参数")
print("   5. 清零梯度: 防止梯度累积")

print("\n[开始训练...]")
print("-" * 50)

num_epochs = 100  # 训练轮数增加

for epoch in range(num_epochs):
    # ===== 训练阶段 =====
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        # 把数据移到 GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # 步骤 1: 前向传播
        predictions = model(batch_x)
        
        # 步骤 2: 计算损失
        loss = criterion(predictions, batch_y)
        
        # 步骤 3: 反向传播（计算梯度）
        optimizer.zero_grad()  # 先清零旧梯度
        loss.backward()        # 计算新梯度
        
        # 步骤 4: 更新权重
        optimizer.step()
        
        # 累加损失
        train_loss += loss.item()
    
    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader)
    
    # ===== 测试阶段 =====
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    
    with torch.no_grad():  # 测试时不计算梯度，节省内存
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    
    # 打印进度 (每 20 轮打印一次)
    if (epoch + 1) % 20 == 0:
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        print(f"   Epoch [{epoch+1:3d}/{num_epochs}]  "
              f"训练损失: {avg_train_loss:.6f}  "
              f"测试损失: {avg_test_loss:.6f}  "
              f"weight: {weight:.4f}  "
              f"bias: {bias:.4f}")

print("-" * 50)
print("[训练完成!]")


# ============================================================
# 第五部分: 查看训练结果
# ============================================================
print("\n" + "=" * 60)
print("第五部分: 查看训练结果")
print("=" * 60)

final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()

print(f"\n[模型学到的参数:]")
print(f"   weight (权重): {final_weight:.4f}  (目标值: 2.0000)")
print(f"   bias (偏置):   {final_bias:.4f}  (目标值: 1.0000)")
print(f"\n   学到的函数: y = {final_weight:.4f} * x + {final_bias:.4f}")
print(f"   真实函数:   y = 2.0000 * x + 1.0000")

print(f"\n[验证预测:]")
test_x = torch.tensor([[5.0], [10.0], [20.0]]).to(device)
model.eval()
with torch.no_grad():
    predictions = model(test_x)

for i in range(len(test_x)):
    x_val = test_x[i].item()
    pred = predictions[i].item()
    true = 2 * x_val + 1
    print(f"   x = {x_val:5.1f}  预测: {pred:7.4f}  真实: {true:7.4f}  误差: {abs(pred-true):.4f}")


# ============================================================
# 第六部分: 保存和加载模型
# ============================================================
print("\n" + "=" * 60)
print("第六部分: 保存和加载模型")
print("=" * 60)

print("\n[为什么要保存模型?]")
print("   训练一次可能要几小时甚至几天")
print("   保存后可以直接加载使用，不用重新训练")

# 保存模型
save_path = r'E:\Harry AI 学习\models\simple_model.pth'
torch.save(model.state_dict(), save_path)
print(f"\n   [OK] 模型已保存: {save_path}")

print("\n[保存的内容:]")
print("   model.state_dict() 包含所有可学习的参数")
print("   - weight: 线性层的权重")
print("   - bias: 线性层的偏置")

print("\n[加载模型的方法:]")
print("   model = SimpleModel()")
print("   model.load_state_dict(torch.load('simple_model.pth'))")
print("   model.eval()  # 设置为评估模式")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print("""
[本课核心概念:]

1. 前向传播 (Forward)
   - 输入数据 → 模型计算 → 得到预测
   - 类比: 学生做题，写出答案

2. 损失函数 (Loss)
   - 衡量预测和真实的差距
   - MSELoss 用于回归，CrossEntropyLoss 用于分类
   - 类比: 老师批改作业，计算错了多少分

3. 反向传播 (Backward)
   - 从损失往回算，每个参数的梯度
   - 梯度 = 损失对这个参数的变化率
   - 类比: 老师告诉学生，哪道题错最多，哪部分知识最薄弱

4. 优化器 (Optimizer)
   - 根据梯度更新参数
   - Adam 是最常用的优化器
   - 类比: 学生根据老师的反馈，调整学习方法

5. 训练循环 (Training Loop)
   for epoch in range(epochs):
       for batch in dataloader:
           1. 前向传播 → 预测
           2. 计算损失
           3. 反向传播 → 梯度
           4. 更新参数
           5. 清零梯度

6. 保存模型
   - torch.save(model.state_dict(), '文件名.pth')
   - 只保存参数，不保存模型结构

[下一课 (1.4): MNIST 手写数字分类]
   用今天学的训练循环，训练一个真正的神经网络！
   识别 0-9 的手写数字，CNN 卷积神经网络入门。
""")

print("\n[OK] 课程 1.3 完成！")
