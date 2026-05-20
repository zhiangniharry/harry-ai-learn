# -*- coding: utf-8 -*-
"""
课程 1.2: PyTorch 数据管道 (Dataset / DataLoader)
===========================================
本代码教你:
1. 什么是 Dataset - 数据的"说明书"
2. 什么是 DataLoader - 数据的"搬运工"  
3. 数据预处理 - 让数据变干净
4. 数据增强 - 让数据变多

运行方式:
    python 02-dataloader_tutorial.py

你会看到:
- 每一步详细的打印输出，告诉你现在在做什么
- 数据的样子、形状、类型
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# 第一部分: 理解 Dataset（数据集）
# ============================================================
print("=" * 60)
print("第一部分: 理解 Dataset（数据集）")
print("=" * 60)

print("\n[什么是 Dataset?]")
print("   Dataset 就像一本食谱书，告诉 PyTorch:")
print("   - 你一共有多少数据? (len)")
print("   - 怎么拿到第 i 个数据? (getitem)")
print("   - 每个数据长什么样? (返回什么)")
print("   类比: 就像去餐厅点菜，菜单告诉你有什么菜，怎么做")


# 我们创建一个简单的数字数据集
# 假设我们要预测: 给定一个数字 x，预测 y = 2*x + 1
class SimpleNumberDataset(Dataset):
    """
    最简单的 Dataset 示例
    数据: 数字 x，标签: y = 2*x + 1
    """
    
    def __init__(self, start=0, end=100):
        """
        初始化数据集
        类比: 厨师准备食材，把材料都摆好
        """
        print("\n   [Dataset.__init__] 正在准备数据...")
        
        # 生成数据: x 从 start 到 end
        self.x = torch.arange(start, end, dtype=torch.float32)
        # 标签: y = 2*x + 1
        self.y = 2 * self.x + 1
        
        print(f"   [OK] 生成了 {len(self.x)} 个数据点")
        print(f"   [OK] x 范围: [{self.x.min():.0f}, {self.x.max():.0f}]")
        print(f"   [OK] y 范围: [{self.y.min():.0f}, {self.y.max():.0f}]")
    
    def __len__(self):
        """
        返回数据集的总大小
        类比: 菜单上有多少道菜
        """
        return len(self.x)
    
    def __getitem__(self, idx):
        """
        返回第 idx 个数据
        类比: 点第 idx 道菜，厨师把菜端上来
        
        参数:
            idx: 数据的索引（第几个）
        返回:
            (x, y): 一个数据对（输入，标签）
        """
        return self.x[idx], self.y[idx]


# 创建数据集
print("\n[创建 SimpleNumberDataset...]")
dataset = SimpleNumberDataset(start=0, end=10)

print(f"\n[数据集信息:]")
print(f"   总数据量: {len(dataset)} 个")

print(f"\n[查看前 3 个数据:]")
for i in range(3):
    x, y = dataset[i]
    print(f"   第 {i} 个: x = {x:.0f}, y = {y:.0f}  (验证: 2*{x:.0f}+1 = {2*x+1:.0f})")


# ============================================================
# 第二部分: 理解 DataLoader（数据加载器）
# ============================================================
print("\n" + "=" * 60)
print("第二部分: 理解 DataLoader（数据加载器）")
print("=" * 60)

print("\n[什么是 DataLoader?]")
print("   DataLoader 就像一个智能服务员:")
print("   - 自动把数据分成小批次 (batch)")
print("   - 每 epoch 自动打乱顺序 (shuffle)")
print("   - 多进程加载数据 (num_workers)")
print("   - 自动把数据整理成张量")
print("   类比: 餐厅服务员把菜分成几桌送，而不是一次全端上来")

print("\n[创建 DataLoader...]")
print("   参数说明:")
print("   - dataset: 要加载的数据集")
print("   - batch_size: 每批多少数据（一次喂给模型多少）")
print("   - shuffle: 是否打乱顺序（训练时打混，测试时不打混）")
print("   - num_workers: 用几个进程加载数据（0=主进程自己加载）")

dataloader = DataLoader(
    dataset=dataset,        # 数据集
    batch_size=3,           # 每批 3 个数据
    shuffle=True,           # 打乱顺序
    num_workers=0           # 单进程加载（Windows 建议 0）
)

print(f"\n[DataLoader 信息:]")
print(f"   批次大小 (batch_size): 3")
print(f"   总批次数: {len(dataloader)} (10 个数据 / 3 = 4 批，最后一批只有 1 个)")

print(f"\n[第一轮遍历 (Epoch 1):]")
for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    print(f"   批次 {batch_idx}: x = {batch_x.tolist()}, y = {batch_y.tolist()}")

print(f"\n[第二轮遍历 (Epoch 2, shuffle=True 所以顺序变了):]")
for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    print(f"   批次 {batch_idx}: x = {batch_x.tolist()}, y = {batch_y.tolist()}")


# ============================================================
# 第三部分: 图像数据预处理（重点！）
# ============================================================
print("\n" + "=" * 60)
print("第三部分: 图像数据预处理")
print("=" * 60)

print("\n[为什么需要预处理?]")
print("   原始图像数据有各种问题:")
print("   - 尺寸不一样大 (有的 1000x800, 有的 600x400)")
print("   - 像素值范围不一样 (0-255 或 0-1)")
print("   - 格式不一样 (JPG, PNG, BMP)")
print("   - 颜色通道顺序不一样 (RGB vs BGR)")
print("   预处理就是让所有数据变成统一的格式，模型才能处理")

print("\n[模拟图像数据预处理...]")

# 模拟 3 张不同尺寸的"图像"
# 实际中你会用 PIL.Image.open() 或 cv2.imread() 读取真实图片
print("\n   假设我们有 3 张图片:")
print("   - 图片 0: 4x4 像素, 随机值")
print("   - 图片 1: 3x3 像素, 随机值")  
print("   - 图片 2: 5x5 像素, 随机值")

# 创建模拟图像数据
image_0 = torch.rand(4, 4, 3)  # 4x4, 3 通道 (RGB)
image_1 = torch.rand(3, 3, 3)  # 3x3
image_2 = torch.rand(5, 5, 3)  # 5x5

images = [image_0, image_1, image_2]
labels = [0, 1, 0]  # 标签: 0=猫, 1=狗

print(f"\n   原始图像尺寸:")
for i, img in enumerate(images):
    print(f"   - 图片 {i}: {img.shape} (高x宽x通道)")


class ImageDataset(Dataset):
    """
    图像数据集示例
    包含预处理: 调整大小 + 归一化
    """
    
    def __init__(self, images, labels, target_size=(4, 4)):
        """
        初始化
        
        参数:
            images: 图像列表
            labels: 标签列表
            target_size: 目标尺寸 (高, 宽)，所有图片调整到这个大小
        """
        print("\n   [ImageDataset.__init__] 初始化图像数据集...")
        self.images = images
        self.labels = labels
        self.target_size = target_size
        print(f"   [OK] 目标尺寸: {target_size}")
        print(f"   [OK] 数据量: {len(images)} 张")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        获取第 idx 个图像，并进行预处理
        
        预处理步骤:
        1. 调整大小 (Resize): 把所有图片变成一样大
        2. 归一化 (Normalize): 把像素值从 [0, 255] 变成 [0, 1] 或 [-1, 1]
        3. 调整维度: 从 (H, W, C) 变成 (C, H, W) -- PyTorch 要求通道在前
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # 步骤 1: 调整大小 (简化版: 直接切片或插值)
        # 实际中: transforms.Resize(target_size)
        h, w = self.target_size
        # 这里简单处理: 如果太大就切片，太小就重复
        image = image[:h, :w, :]  # 切片到目标大小
        if image.shape[0] < h or image.shape[1] < w:
            # 如果不够大，用 0 填充
            new_image = torch.zeros(h, w, 3)
            new_image[:image.shape[0], :image.shape[1], :] = image
            image = new_image
        
        # 步骤 2: 归一化 (把值缩放到 0-1 之间)
        # 实际中像素值是 0-255，除以 255
        # 这里已经是 0-1 了，所以不变
        image = image.float()
        
        # 步骤 3: 调整维度顺序 (H, W, C) → (C, H, W)
        # PyTorch 的卷积层要求输入是 (Batch, Channel, Height, Width)
        image = image.permute(2, 0, 1)
        
        return image, label


# 创建图像数据集
image_dataset = ImageDataset(images, labels, target_size=(4, 4))

print(f"\n[查看预处理后的数据:]")
for i in range(len(image_dataset)):
    img, lbl = image_dataset[i]
    print(f"   图片 {i}: 形状 {img.shape}, 标签 {lbl}")
    print(f"   - 像素值范围: [{img.min():.3f}, {img.max():.3f}]")


# 创建 DataLoader
image_loader = DataLoader(image_dataset, batch_size=2, shuffle=False)

print(f"\n[用 DataLoader 加载图像批次:]")
for batch_idx, (batch_images, batch_labels) in enumerate(image_loader):
    print(f"   批次 {batch_idx}:")
    print(f"   - 图像形状: {batch_images.shape} (Batch x Channel x Height x Width)")
    print(f"   - 标签: {batch_labels.tolist()}")
    print(f"   - 像素值范围: [{batch_images.min():.3f}, {batch_images.max():.3f}]")


# ============================================================
# 第四部分: 数据增强 (Data Augmentation)
# ============================================================
print("\n" + "=" * 60)
print("第四部分: 数据增强 (Data Augmentation)")
print("=" * 60)

print("\n[什么是数据增强?]")
print("   数据增强 = 用现有数据生成'新'数据")
print("   就像同一个东西，从不同角度拍照，还是同一个东西")
print("   好处:")
print("   - 数据量变多了（不用花钱收集新数据）")
print("   - 模型更鲁棒（见过各种变形，不容易被骗）")
print("   - 防止过拟合（不会只记住训练数据的特定样子）")

print("\n[常见的图像增强方法:]")
print("   1. 随机裁剪 (RandomCrop): 从大图切一小块")
print("   2. 随机翻转 (RandomHorizontalFlip): 左右镜像")
print("   3. 颜色抖动 (ColorJitter): 改亮度、对比度、饱和度")
print("   4. 旋转 (RandomRotation): 转个角度")
print("   5. 归一化 (Normalize): 减均值除标准差")

print("\n[模拟数据增强效果...]")

# 模拟一张图片
sample_image = torch.rand(4, 4, 3)
print(f"\n   原始图片: {sample_image.shape}")
print(f"   像素值范围: [{sample_image.min():.3f}, {sample_image.max():.3f}]")

# 增强 1: 水平翻转
flipped = torch.flip(sample_image, dims=[1])  # 沿宽度维度翻转
print(f"\n   水平翻转后: {flipped.shape}")
print(f"   像素值范围: [{flipped.min():.3f}, {flipped.max():.3f}]")
print("   [OK] 形状不变，内容镜像了")

# 增强 2: 随机裁剪 (从 4x4 crop 到 3x3)
crop_h, crop_w = 3, 3
start_h = np.random.randint(0, sample_image.shape[0] - crop_h + 1)
start_w = np.random.randint(0, sample_image.shape[1] - crop_w + 1)
cropped = sample_image[start_h:start_h+crop_h, start_w:start_w+crop_w, :]
print(f"\n   随机裁剪后: {cropped.shape}")
print(f"   裁剪位置: 从 ({start_h}, {start_w}) 开始")
print("   [OK] 变小了，但内容还是原图的一部分")

# 增强 3: 颜色抖动 (改亮度)
brightness_factor = 0.8 + torch.rand(1).item() * 0.4  # 0.8 ~ 1.2
jittered = sample_image * brightness_factor
jittered = torch.clamp(jittered, 0, 1)  # 限制在 0-1
print(f"\n   亮度调整 (x{brightness_factor:.2f}) 后: {jittered.shape}")
print(f"   像素值范围: [{jittered.min():.3f}, {jittered.max():.3f}]")
print("   [OK] 亮了一点或暗了一点")


# ============================================================
# 第五部分: 实际使用 torchvision.transforms
# ============================================================
print("\n" + "=" * 60)
print("第五部分: 实际使用 torchvision.transforms")
print("=" * 60)

try:
    from torchvision import transforms
    
    print("\n[torchvision.transforms 是 PyTorch 官方提供的数据预处理工具]")
    print("   它把各种预处理操作封装成可以链式调用的模块")
    
    # 定义预处理流程
    print("\n   定义预处理流程 (transform):")
    print("   " + "-" * 50)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),           # 先转成 PIL Image（某些操作需要）
        transforms.Resize((224, 224)),     # 调整大小为 224x224（标准输入尺寸）
        transforms.RandomHorizontalFlip(), # 50% 概率水平翻转
        transforms.RandomRotation(10),     # 随机旋转 +/-10 度
        transforms.ToTensor(),             # 转成张量，并归一化到 [0, 1]
        transforms.Normalize(              # 标准化：减均值，除标准差
            mean=[0.485, 0.456, 0.406],    # ImageNet 数据集的均值
            std=[0.229, 0.224, 0.225]      # ImageNet 数据集的标准差
        )
    ])
    
    print("   1. ToPILImage()       -> 转成 PIL 图像格式")
    print("   2. Resize(224, 224)   -> 调整大小为 224x224")
    print("   3. RandomHorizontalFlip() -> 随机水平翻转")
    print("   4. RandomRotation(10) -> 随机旋转 +/-10 度")
    print("   5. ToTensor()         -> 转成 PyTorch 张量")
    print("   6. Normalize()        -> 标准化（用 ImageNet 的均值和标准差）")
    print("   " + "-" * 50)
    
    print("\n[为什么用 ImageNet 的均值和标准差?]")
    print("   ImageNet 是一个巨大的图像数据集（1000 类，120万张图）")
    print("   大多数预训练模型（如 ResNet）都是在 ImageNet 上训练的")
    print("   用同样的归一化参数，可以让新数据和预训练数据分布一致")
    print("   这样迁移学习效果更好")
    
    print("\n[OK] transforms 定义完成！可以传给 Dataset 使用")
    print("   实际用法:")
    print("   dataset = MyDataset(transform=transform)")
    print("   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)")
    
except ImportError:
    print("\n[!] torchvision 未安装，跳过 transforms 演示")
    print("   安装命令: pip install torchvision")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print("""
[本课核心概念:]

1. Dataset (数据集)
   - 作用: 告诉 PyTorch 你的数据在哪里，长什么样
   - 必须实现: __len__() 和 __getitem__()
   - 类比: 菜单（告诉你有什么菜）

2. DataLoader (数据加载器)
   - 作用: 自动分批、打乱、加载数据
   - 关键参数: batch_size, shuffle, num_workers
   - 类比: 服务员（把菜分成几桌端上来）

3. 预处理 (Preprocessing)
   - 作用: 让数据格式统一
   - 常见操作: Resize, Normalize, ToTensor
   - 类比: 厨师切菜（把食材切成统一大小）

4. 数据增强 (Augmentation)
   - 作用: 用现有数据生成新数据
   - 常见操作: Flip, Rotation, Crop, ColorJitter
   - 类比: 拍照时换角度（同一个东西，不同视角）

5. transforms.Compose
   - 作用: 把多个预处理操作串起来
   - 好处: 代码简洁，可复用
   - 类比: 流水线（一步接一步）

[下一课 (1.3): 训练循环]
   我们将用今天学的 DataLoader，写一个完整的训练脚本！
""")

print("\n[OK] 课程 1.2 完成！")
print("   建议: 修改上面的代码，试试不同的 batch_size 和 shuffle 值")
print("   看看输出有什么变化！")
