# Harry AI 学习项目

> 从 0 到 1 的 AI 学习路线，涵盖 PyTorch 基础、计算机视觉、自然语言处理、生成式 AI 与 MLOps 部署。

---

## 环境配置

### 硬件

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 10 Pro (Build 19045) |
| CPU | Intel Core i5-9400 @ 2.90GHz（6 核 6 线程） |
| 内存 | 16 GB RAM |
| 显卡 | NVIDIA GeForce RTX 2070（8 GB VRAM） |
| CUDA 驱动 | 596.49（CUDA 13.2） |

### 软件

| 项目 | 版本 |
|------|------|
| Python | 3.11.9 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| CUDA | 12.1 |
| cuDNN | 9.1.0 |
| numpy | 2.4.4 |
| Pillow | 12.2.0 |

### 虚拟环境

```bash
# 激活虚拟环境（PowerShell）
E:\Harry AI 学习\.venv\Scripts\Activate.ps1

# 运行脚本
E:\Harry AI 学习\.venv\Scripts\python.exe your_script.py
```

---

## 课程体系

> **每门课程 = 代码实现 + 课程笔记（.docx）+ 教学 PPT（.pptx）**
>
> 学习目标必须理解透彻，PPT 教学必须详尽，每个知识点都要讲清楚"是什么、为什么、怎么用"

---

## 第 1 步：环境配置

### 课程 1.1：PyTorch + CUDA 环境验证

- [x] 已完成
- [x] 课程笔记：已生成
- [ ] PPT：待制作

**必须掌握的内容：**

- PyTorch 张量（Tensor）的基本创建、运算与 GPU 迁移（`.to(device)`）
- `torch.cuda.is_available()` 的作用与返回值含义
- 为什么需要 GPU 加速：CUDA 核心与 Tensor Core 的区别
- RTX 2070 的计算能力（Compute Capability）是否支持当前 PyTorch 版本
- 如何验证 cuDNN 可用：`torch.backends.cudnn.is_available()`
- CPU vs GPU 训练速度的数量级差异（10x~100x）

**验证方法：**
```python
import torch
print(torch.cuda.is_available())  # True = 可用
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 2070
```

---

## 第 2 步：PyTorch 基础

---

### 课程 1.2：数据管道（Dataset / DataLoader）

- [x] 已完成
- [x] 课程笔记：`课程1.2-数据管道详解.docx`
- [x] 代码：`02-dataloader_tutorial.py`
- [ ] PPT：待制作

**必须掌握的内容：**

1. **Dataset 的本质**
   - `__len__()`：告诉 PyTorch 数据集有多大
   - `__getitem__(idx)`：根据索引取出一个样本（数据 + 标签）
   - 为什么必须同时返回 `(x, y)`：输入与标签的配对关系

2. **DataLoader 的作用**
   - `batch_size`：每批喂给模型多少样本；过大显存爆炸，过小梯度不稳定
   - `shuffle=True`：训练时为什么要打乱；测试时为什么 `shuffle=False`
   - `num_workers`：多进程加载提速；Windows 下建议用 0 避免问题
   - `drop_last`：当样本数不能被 batch_size 整除时，最后一个不完整的 batch 是否丢弃

3. **数据预处理（Preprocessing）**
   - `transforms.ToTensor()`：为什么把 PIL Image 转成张量；像素值如何从 [0,255] 变成 [0,1]
   - `transforms.Normalize(mean, std)`：减去均值再除标准差的物理意义；为什么能让训练更稳定
   - `transforms.Resize((H, W))`：统一输入尺寸的意义；为什么 CNN 要求固定输入大小
   - 为什么 PyTorch 要求维度顺序是 `(C, H, W)` 而不是 `(H, W, C)`

4. **数据增强（Data Augmentation）**
   - 为什么增强能防止过拟合：让模型见过更多变体
   - `RandomHorizontalFlip`：哪些场景适合水平翻转；哪些场景不适合（数字识别可以）
   - `RandomRotation`：旋转角度过大为什么会破坏语义
   - `ColorJitter`：亮度、对比度、饱和度调整的数值范围
   - `RandomCrop`：先放大再裁剪 vs 直接裁剪的区别
   - 为什么数据增强只用于训练集，测试集要保持原始分布

5. **torchvision.transforms.Compose**
   - 链式调用的执行顺序（从左到右）
   - 每个 transform 的输入输出形状变化要能徒手推导

---

### 课程 1.3：训练循环（Forward / Backward / Optimizer）

- [x] 已完成
- [x] 课程笔记：`课程1.3-训练循环详解.docx`
- [x] 代码：`03-training_loop.py`
- [ ] PPT：待制作

**必须掌握的内容：**

1. **前向传播（Forward Propagation）**
   - 输入数据如何一步步经过每一层网络
   - 为什么要 `model.train()`：Dropout 和 BatchNorm 在训练/评估时的不同行为
   - `model.eval()` 的作用：BatchNorm 用全局均值/方差，Dropout 不丢弃

2. **损失函数（Loss Function）**
   - `nn.MSELoss`：均方误差的数学公式 $\frac{1}{n}\sum(\hat{y}-y)^2$；为什么适合回归
   - `nn.CrossEntropyLoss`：交叉熵的物理意义；它内部自动做 Softmax
   - 为什么分类不用 MSE：梯度消失问题；Softmax + CrossEntropy 才是正确的梯度流
   - 多分类 vs 二分类的损失函数选择

3. **反向传播（Back Propagation）**
   - `loss.backward()` 做了什么：计算每个参数对损失的梯度 $\frac{\partial L}{\partial w}$
   - 梯度从哪里来：链式法则（Chain Rule）在计算图中的应用
   - 什么是计算图：PyTorch 的动态图机制（define-by-run）
   - 梯度为什么要清零：`optimizer.zero_grad()` 的位置（在 `backward()` 之前）

4. **优化器（Optimizer）**
   - `SGD`（随机梯度下降）：公式 $w = w - lr \cdot \nabla L$；"随机"是什么意思
   - `Adam`：一阶矩估计（动量）+ 二阶矩估计（自适应学习率）；为什么比 SGD 更常用
   - `lr`（学习率）：过大震荡不收敛，过小收敛极慢；典型值 0.001
   - 梯度消失与梯度爆炸：现象、成因、解决方式（梯度裁剪、归一化初始化）

5. **训练循环的完整流程（5步循环）**
   ```
   for epoch in range(EPOCHS):
       model.train()
       for batch in train_loader:
           1. 前向传播 → output = model(batch_x)
           2. 计算损失 → loss = criterion(output, batch_y)
           3. 清零梯度 → optimizer.zero_grad()
           4. 反向传播 → loss.backward()
           5. 更新权重 → optimizer.step()
   ```
   - 每个 epoch 遍历整个训练集；一个 epoch 里的 batch 数量 = 训练集大小 / batch_size
   - 训练集上评估 vs 测试集上评估的区别：防止信息泄露

6. **模型的保存与加载**
   - `torch.save(model.state_dict(), 'model.pth')`：只保存参数，不保存结构
   - 加载时必须先创建模型实例：`model = SimpleModel(); model.load_state_dict(torch.load(...))`
   - `.pth` vs `.pt` vs `.ckpt`：文件扩展名的约定俗成，无本质区别
   - 为什么 checkpoint 里要保存 optimizer 的 state_dict：恢复训练继续

---

### 课程 1.4：MNIST 手写数字分类（CNN）

- [x] 已完成
- [x] 课程笔记：`课程1.4-MNIST手写数字分类.docx`
- [x] 代码：`04-mnist_cnn_train.py` + 4个分步脚本
- [ ] PPT：待制作

**必须掌握的内容：**

1. **MNIST 数据集**
   - 28×28 灰度图，10 类（数字 0-9）
   - 训练集 60000，测试集 10000
   - `transforms.Normalize((0.1307,), (0.3081,))`：这两个 magic number 是怎么算出来的（MNIST 全体像素的均值和标准差）

2. **卷积层（Conv2d）**
   - 卷积核（kernel / filter）的物理意义：模板匹配；边缘检测/纹理提取
   - `in_channels`：彩色图=3（RGB），灰度图=1
   - `out_channels`：提取多少种不同的特征
   - `kernel_size=3`：卷积核边长；越大感受野越大，参数越多
   - `padding`：为什么需要填充；same padding vs valid padding
   - 每次卷积后尺寸变化公式：$H_{out} = \lfloor \frac{H_{in} + 2p - k}{s} \rfloor + 1$
   - 28×28 输入 → conv1(p=1,k=3) → 还是 28×28 → maxpool → 14×14

3. **激活函数（ReLU）**
   - $f(x) = \max(0, x)$：为什么引入非线性；没有激活函数多层等价一层
   - ReLU 的问题：Dead ReLU Problem（梯度为0的神经元永远不更新）
   - Leaky ReLU / GELU 作为替代

4. **池化层（MaxPool2d）**
   - 2×2 池化核，stride=2：图片尺寸减半，信息压缩
   - 为什么用 MaxPool 而不是 AveragePool：保留最显著的特征
   - 池化层没有可学习参数

5. **全连接层（Linear / FC）**
   - CNN 特征图为什么要 flatten：把空间信息展平成一维向量
   - 7×7×64 = 3136 个值 → FC1 → 128：全连接层承担分类器的角色
   - Dropout(p=0.25)：随机丢弃 25% 神经元；训练时生效，评估时不生效
   - 为什么 dropout 能防过拟合：减少神经元共适应（co-adaptation）

6. **CNN 完整前向传播路径**
   ```
   Input (1, 28, 28)
   → Conv2d(1→32) + ReLU + MaxPool → (32, 14, 14)
   → Conv2d(32→64) + ReLU + MaxPool → (64, 7, 7)
   → Flatten → 3136
   → Linear(3136→128) + ReLU + Dropout → 128
   → Linear(128→10) → 10
   → CrossEntropyLoss ← 真实标签
   ```

7. **准确率评估**
   - `output.argmax(1)`：在第1维度（类别维度）取最大值的索引
   - 训练准确率 vs 测试准确率的 Gap：过拟合的判断标准
   - 最佳模型保存策略：只保留测试集上表现最好的 checkpoint

---

## 第 3 步：计算机视觉（CV）

---

### 课程 2.1：通用图像分类

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **迁移学习（Transfer Learning）**
   - 什么是预训练模型（Pre-trained Model）：在大规模数据集（ImageNet 120万张图）上训练好的模型
   - 为什么迁移学习有效：底层特征（边缘、纹理）通用，高层特征（语义）任务相关
   - `torchvision.models`：ResNet、VGG、EfficientNet、MobileNet 等预训练模型的使用方法
   - 冻结（freeze）vs 微调（fine-tune）：什么时候冻结，什么时候解冻

2. **ResNet 结构**
   - 残差连接（Skip Connection）：$y = F(x) + x$；解决深层网络退化问题
   - 为什么残差能解决梯度消失：梯度有直接通道回传
   - ResNet-18 / ResNet-50 / ResNet-101 的区别： bottleneck 模块设计

3. **数据准备**
   - ImageFolder 数据集格式：目录结构 = 根目录/类别名/图片.jpg
   - 如何自定义数据路径：Dataset 的 `__init__` 中记录文件列表
   - 训练集/验证集/测试集的正确划分：不要让模型在训练时看到测试数据

4. **模型微调流程**
   ```
   1. 加载预训练模型（weights=models.ResNet18_Weights.DEFAULT）
   2. 替换最后一层全连接层：model.fc = nn.Linear(512, num_classes)
   3a. 冻结特征提取层：只训练新加的 FC 层（适合小数据集）
   3b. 解冻全部层：所有参数一起训练（适合大数据集）
   4. 学习率设置：特征层用小lr，全新层用大lr（差异10~100倍）
   5. 训练 + 验证
   ```

5. **学习率调度（LR Scheduler）**
   - `StepLR`：每隔固定 epoch 下降一次
   - `CosineAnnealingLR`：余弦曲线下降，更平滑
   - `ReduceLROnPlateau`：验证集loss不降时自动降低学习率
   - Warmup（预热）：训练初期学习率从0慢慢升上去，防止梯度爆炸

6. **模型评估指标**
   - Top-1 准确率 vs Top-5 准确率
   - Precision / Recall / F1-score：对不平衡数据集尤其重要
   - Confusion Matrix（混淆矩阵）：哪些类别最容易混淆

---

### 课程 2.2：目标检测（Object Detection）

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **目标检测 vs 图像分类的区别**
   - 图像分类：整张图 → 一个标签
   - 目标检测：图像 → 多个物体框 + 每个框的类别 + 置信度
   - 难点：物体数量不定、位置不定、大小不定

2. **两阶段检测器（Two-Stage）**
   - R-CNN → Fast R-CNN → Faster R-CNN 的演进路线
   - Region Proposal Network（RPN）：Faster R-CNN 的核心；提取2000个候选框
   - RoI Pooling / RoI Align：将不同大小的候选框变成固定尺寸特征
   - 分类头 + 回归头：分别负责类别预测和边界框坐标回归

3. **单阶段检测器（One-Stage）**
   - YOLO（You Only Look Once）：直接把图像划分成 S×S 网格，每个网格预测多个边界框
   - SSD（Single Shot Detector）：多尺度特征图预测，从小特征图检测大物体
   - 为什么单阶段比两阶段快：不需要单独的候选框提取阶段
   - YOLO 的置信度分数：物体存在的概率 × IOU

4. **锚框（Anchor Boxes）**
   - 预定义不同尺寸和长宽比的边界框模板
   - 每个 anchor 与真实框计算 IOU，超过阈值标为正样本
   - 多尺度 Anchor：在不同层的特征图上用不同大小的 anchor

5. **非极大值抑制（NMS）**
   - 问题：同一个物体可能被多个框检测到（重复检测）
   - 步骤：按置信度排序 → 保留最高置信度框 → 删除与它 IOU 过高的其他框 → 重复
   - soft-NMS：用衰减替代直接删除，解决密集物体问题

6. **评估指标**
   - IOU（Intersection over Union）：预测框与真实框的交并比
   - mAP（mean Average Precision）：所有类别的平均精度的均值；目标检测的核心指标
   - Precision / Recall 在目标检测中的计算：TP / FP / FN 的定义

7. **常用检测框架**
   - torchvision.models.detection：Faster R-CNN、SSD、YOLOv8（官方实现）
   - mmdetection：学术框架，支持大量模型
   - Ultralytics YOLOv8：工程友好，训练推理一条龙

---

### 课程 2.3：项目 — 自定义目标检测器

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **数据集构建**
   - 标注工具：LabelImg / CVAT 的使用
   - COCO 格式 vs VOC 格式：JSON 标注 vs XML 标注
   - 自定义数据集的数据增强：mosaic、mixup、copy-paste

2. **模型选型与训练**
   - 根据数据集大小选择模型：数据少用小模型，数据多用大模型
   - 预训练权重从哪里来：在大数据集（如 COCO）上预训练过
   - 训练参数调试：batch size、learning rate、epoch 数与数据集规模的关系
   - 如何判断模型是否收敛：loss 曲线、mAP 曲线

3. **模型调优**
   - 数据不足时的策略：数据增强、预训练微调、迁移学习
   - 过拟合的表现：训练 loss 下降但验证 loss 上升
   - 如何改进：增加数据、加正则化（Dropout/Weight Decay）、简化模型

4. **模型部署**
   - PyTorch → ONNX 导出：`torch.onnx.export()`
   - ONNX Runtime 推理：跨平台、高性能
   - 模型量化和剪枝：INT8 量化减少模型体积和推理时间

---

## 第 4 步：自然语言处理（NLP）

---

### 课程 3.1：Transformer 从零实现

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **注意力机制（Attention）的核心思想**
   - 为什么需要注意力：RNN 的长距离依赖问题（梯度消失）；注意力让模型"关注"相关词
   - Query / Key / Value 的物理含义：Query=问什么，Key=答什么，Value=答案内容
   - 注意力分数的计算：$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
   - 为什么要除 $\sqrt{d_k}$：防止点积值过大导致 Softmax 梯度消失

2. **自注意力（Self-Attention）**
   - Query、Key、Value 都来自同一个输入（自己关注自己）
   - 如何捕捉词与词之间的依赖关系：任意两个位置直接计算，路径长度为1
   - 与 RNN 的对比：RNN 需要 O(n) 步才能建立远距离依赖，注意力 O(1)

3. **多头注意力（Multi-Head Attention）**
   - 为什么分成多个头：每个头学习不同的注意力模式（句法、语义、位置等）
   - 公式：$MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O$
   - 每个头的 $d_k = d_{model} / h$；典型值 $h=8, d_{model}=512$

4. **位置编码（Positional Encoding）**
   - 为什么需要位置信息：注意力机制本身对位置不敏感（Permutation Invariant）
   - 三角函数位置编码：$PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$；为什么用 sin/cos 交替
   - 旋转位置编码（RoPE）和相对位置编码（ALiBi）作为扩展

5. **编码器（Encoder）结构**
   ```
   Input Embedding + Positional Encoding
   → Multi-Head Self-Attention + Residual + LayerNorm
   → Feed Forward Network + Residual + LayerNorm
   （重复 N 次，N=6 for Base）
   ```
   - 残差连接的作用：梯度直接回传，训练深层网络
   - LayerNorm vs BatchNorm：LayerNorm 不依赖 batch 统计量，更适合 NLP

6. **解码器（Decoder）结构**
   - Masked Multi-Head Attention：遮住未来位置（不能"偷看"答案）
   - Cross-Attention：Query 来自解码器，Key/Value 来自编码器输出（翻译任务的关键）
   - 训练时用 Teacher Forcing：真实前缀做输入；推理时只能用自己预测的前缀

7. **从零实现一个 Transformer**
   - 如何用 PyTorch 实现：MultiHeadAttention 类、FFN 类、EncoderLayer 类、DecoderLayer 类
   - 代码调试技巧：打印注意力权重矩阵，可视化模型关注了哪些词

---

### 课程 3.2：BERT 预训练模型微调

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **BERT 的预训练任务**
   - MLM（Masked Language Model）：随机遮住 15% 的词，让模型预测被遮的词
   - NSP（Next Sentence Prediction）：判断句子对是否是相邻关系；句子A后面是否跟句子B
   - 为什么用这两个任务：MLM 学语义，NSP 学句子间关系

2. **BERT 的输入表示**
   - Token Embedding + Segment Embedding + Position Embedding 三者相加
   - `[CLS]`：分类任务的输出 token，放在句首
   - `[SEP]`：分隔两个句子
   - `[PAD]`：补齐到固定长度
   - WordPiece / BPE 分词：为什么不用字符级或词级分词

3. **HuggingFace Transformers 库**
   - `AutoModel.from_pretrained("bert-base-chinese")`：一行代码加载预训练模型
   - `AutoTokenizer.from_pretrained`：加载对应分词器
   - 分词、编码、Padding、Attention Mask 的完整流程
   - `model(input_ids, attention_mask, token_type_ids, labels)`

4. **微调（Fine-tuning）策略**
   - 全参数微调：所有层都训练，计算量大但效果最好
   - 冻结策略：只训练顶层；适合数据极少的场景
   - 差分学习率：底层小lr，高层大lr

5. **常见下游任务**
   - 文本分类（Text Classification）：取 `[CLS]` 的输出向量做分类
   - 命名实体识别（NER）：每个 token 输出一个标签（PER/LOC/ORG）
   - 问答系统（QA）：预测答案的起始和结束位置
   - 句子对匹配：判断两个句子是否构成 entailment/contradiction

6. **GPT 系列对比 BERT**
   - GPT：单向语言模型（只看左边），适合生成任务；用 Decoder
   - BERT：双向语言模型（看左右），适合理解任务；用 Encoder
   - ChatGPT / GPT-4 的演进：GPT-1 → GPT-2 → GPT-3 → RLHF → GPT-4

---

### 课程 3.3：项目 — 自定义 Chatbot

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **对话数据集**
   - 常用数据集：Cornell Movie Dialogs、DailyDialog、LCCC
   - 数据格式：`(input_sequence, response_sequence)` 对
   - 数据清洗：去除噪音、处理特殊字符、长度截断

2. **Seq2Seq 模型**
   - Encoder-Decoder 架构：输入序列 → 编码 → 解码生成响应
   - 编码器输出的是上下文向量（Context Vector）：这是信息的瓶颈
   - 注意力在 Seq2Seq 中的作用：解码时"回头看"输入的相关部分

3. **Beam Search 解码**
   - Greedy 解码的问题：一步错步步错
   - Beam Search：保留 top-k 条候选路径，在更多可能中搜索最优序列
   - temperature 采样：控制生成的多样性

4. **对话系统的评估**
   - BLEU / ROUGE：基于 n-gram 重叠的指标
   - Perplexity：语言模型的困惑度，越低越好
   - 人工评估的重要性：自动指标不能完全反映对话质量

5. **检索式 vs 生成式对话**
   - 检索式：给定用户输入，从候选库中检索最相似的回复；可控、安全
   - 生成式：端到端生成回复；灵活但可能产生无意义回复（hallucination）
   - 工业界做法：检索+生成融合（Retrieval Augmented Generation, RAG）

---

## 第 5 步：生成式 AI

---

### 课程 4.1：Stable Diffusion 微调

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **扩散模型（Diffusion Model）原理**
   - 前向过程（Forward Process）：逐步给图像加噪声，最终变成纯噪声
   - 反向过程（Reverse Process）：从纯噪声开始，逐步去噪，恢复出图像
   - 为什么叫"扩散"：噪声扩散的物理类比；热力学第二定律的熵增过程
   - 训练目标：预测每一步加了多少噪声（Noise Predictor）

2. **Stable Diffusion 架构**
   - VAE（Variational Autoencoder）：图像 → 压缩潜空间 → 重建；降低计算量
   - 潜空间（Latent Space）：为什么在潜空间做扩散比直接对像素做快49倍
   - U-Net：去噪网络；输入是噪声图 + 条件（文本embedding）；输出是预测的噪声
   - CLIP Text Encoder：把文字转成向量，作为条件注入 U-Net

3. **条件引导（Conditional Generation）**
   - 文字到图像：CLIP 编码的文字向量作为条件
   - 图生图（img2img）：原始图像加噪后去噪，保留结构换风格
   - Inpainting：只重绘指定区域

4. **微调方法**
   - DreamBooth：给概念（如某人的脸）分配一个特殊 token（如 "Sks"），微调整个模型
   - LoRA（Low-Rank Adaptation）：只训练低秩矩阵，冻结原模型；参数量减少100倍+
   - Textual Inversion：学习一个伪词（pseudo-word）来代表新概念；不改变模型权重
   - Checkpoint 合并：将 LoRA 权重合并回原始模型

5. **训练技巧**
   - 学习率设置：通常比 CV/NLP 模型低（1e-4 ~ 1e-5）
   - 数据集规模：DreamBooth 3~5张图即可；LoRA 通常需要几十到几百张
   - 步数（Steps）：一般 500~2000 步；过多过拟合，过少欠拟合
   - 分辨率：768×768 是 SD 的标准；更高分辨率需要超分

6. **推理生成**
   - `prompt engineering`：权重语法（`(word:1.2)`）、负面提示词（negative prompt）
   - CFG（Classifier-Free Guidance）：引导强度（guidance scale）越高越符合提示词，越低越有创意
   - 采样器（Sampler）：Euler、Euler A、DPM++ 2M Karras 等；收敛速度和质量不同

---

## 第 6 步：MLOps 与部署

- [ ] 未完成
- [ ] 课程笔记：待生成
- [ ] 代码：待编写
- [ ] PPT：待制作

**必须掌握的内容：**

1. **模型导出（Export）**
   - TorchScript：`torch.jit.trace()` 和 `torch.jit.script()` 的区别
   - ONNX：`torch.onnx.export()`；跨框架、跨平台部署
   - ONNX Runtime：高性能推理引擎；比原生 PyTorch 快 2~5 倍

2. **模型量化（Quantization）**
   - FP32 → FP16：半精度，显存减半，速度提升1.5~2x，精度几乎不变
   - FP16 → INT8：8位整数，显存再减半；需要校准（Calibration）避免精度损失
   - 动态量化 vs 静态量化：动态量化不需重训练，静态量化需要校准数据
   - GPTQ / AWQ：针对大语言模型的量化方法

3. **模型剪枝（Pruning）**
   - 结构化剪枝：直接移除整个神经元/卷积核
   - 非结构化剪枝：只移除权重中接近0的值，需要硬件支持稀疏计算

4. **Web API 部署**
   - FastAPI + TorchServe：RESTful API 推理接口
   - 请求流：HTTP请求 → 反序列化 → GPU推理 → 序列化返回
   - 并发处理：批处理（batching）提高吞吐量；单请求低延迟 vs 高吞吐的矛盾

5. **Docker 部署**
   - 多阶段构建：减小镜像体积
   - CUDA 容器化：`nvidia-docker` / `docker --gpus` 开启 GPU 支持
   - 镜像体积优化：从 10GB+ 减到 2~3GB

6. **监控与维护**
   - 模型漂移（Model Drift）：生产数据分布变化导致精度下降
   - A/B 测试：新旧模型在线对比
   - 日志与告警：请求延迟、错误率、GPU 利用率

---

## 目录结构（含 PPT 子目录）

```
E:\Harry AI 学习\
├── 01-环境配置/
│   ├── ppt/                      # PPT 教学幻灯片
│   │   └── 1.1-环境验证.pptx
│   └── verify_pytorch.py
│
├── 02-PyTorch基础/
│   ├── ppt/                      # PPT 教学幻灯片
│   │   ├── 1.2-数据管道.pptx
│   │   ├── 1.3-训练循环.pptx
│   │   └── 1.4-MNIST分类.pptx
│   ├── 02-dataloader_tutorial.py
│   ├── 03-training_loop.py
│   ├── 04-mnist_cnn_train.py
│   ├── generate_docx_1.2.py
│   ├── generate_docx_1.3.py
│   ├── generate_docx_1.4.py
│   ├── 课程1.2-数据管道详解.docx
│   ├── 课程1.3-训练循环详解.docx
│   └── 课程1.4-MNIST手写数字分类.docx
│
├── 03-计算机视觉/
│   ├── ppt/                      # PPT 教学幻灯片
│   │   ├── 2.1-通用图像分类.pptx
│   │   ├── 2.2-目标检测.pptx
│   │   └── 2.3-自定义检测器项目.pptx
│   ├── 01-MNIST分类/
│   │   ├── ppt/                  # MNIST 单独 PPT
│   │   └── *.py
│   ├── 01-图像分类/              # 空，待完成
│   ├── 02-目标检测/              # 空，待完成
│   └── 03-项目-自定义检测器/     # 空，待完成
│
├── 04-自然语言处理/
│   ├── ppt/                      # PPT 教学幻灯片
│   │   ├── 3.1-Transformer从零实现.pptx
│   │   ├── 3.2-BERT微调.pptx
│   │   └── 3.3-自定义Chatbot项目.pptx
│   ├── 01-Transformer实现/        # 空，待完成
│   ├── 02-BERT微调/              # 空，待完成
│   ├── 03-项目-自定义Chatbot/     # 空，待完成
│   ├── datasets/                 # NLP 语料
│   └── outputs/                  # 模型输出
│
├── 05-生成式AI/
│   ├── ppt/                      # PPT 教学幻灯片
│   │   └── 4.1-StableDiffusion微调.pptx
│   └── 01-StableDiffusion微调/   # 空，待完成
│
├── 06-MLOps与部署/
│   ├── ppt/                      # PPT 教学幻灯片
│   │   └── 5.1-模型导出与部署.pptx
│   └── （待完成）
│
├── 99-论文与阅读/
│   ├── Karpathy-AutoResearch/
│   └── Dimension-TheMeltingPoint/
│
├── datasets/                     # 共享数据集
├── models/                       # 模型权重（.pth / .onnx）
├── utils/                        # 工具脚本
├── .venv/                        # Python 虚拟环境
├── README.md
└── .gitignore
```

---

## PPT 制作要求

> 每个课程目录下的 `ppt/` 文件夹存放对应的教学幻灯片，使用 `pptx-generator` 技能制作。

### PPT 内容规范

每份 PPT 必须包含以下板块：

| 序号 | 板块 | 内容要求 |
|------|------|---------|
| 1 | 封面 | 课程编号 + 课程名称 + 项目名称 |
| 2 | 学习目标 | 本课结束后学生能做什么（3~5条可量化目标） |
| 3 | 核心概念 | 每个概念 1~2 页，格式：定义 + 公式 + 图示 + 例子 |
| 4 | 代码演示 | 分步讲解，每个关键代码块有注释和运行结果截图 |
| 5 | 流程图 | 数据流 / 模型结构 / 训练流程的可视化 |
| 6 | 对比与总结 | 与其他方法的对比表格；本课核心公式汇总 |
| 7 | 思考题 | 3~5 道思考题，考察深度理解（不是记忆） |
| 8 | 下节预告 | 承上启下，引入下一课内容 |

### PPT 设计风格

- 字体：中文使用思源黑体/微软雅黑；英文使用 Helvetica / Arial
- 配色：主色 #2563EB（蓝），辅色 #10B981（绿），背景白色或浅灰
- 代码块：深色背景（#1E1E1E），等宽字体（Fira Code / Consolas）
- 每页不超过 6 行文字，文字不超过 60 字

### 使用 pptx-generator 制作

```bash
# 在 Claude Code 中调用 pptx-generator 技能
/skill pptx-generator
```

---

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `utils/gen_karpathy_docs.py` | 自动生成 Karpathy AutoResearch 论文阅读文档 |
| `utils/gen_melting_docs.py` | 自动生成熔点论文阅读文档 |
| `utils/reorg_files.py` | 批量文件整理 |
| `utils/reorg.ps1` | PowerShell 文件整理 |

---

## 论文阅读

- **Karpathy AutoResearch** — 自动驾驶研究方向
- **Dimension-TheMeltingPoint** — 维度熔点相关研究

---

## 学习进度

| 步骤 | 模块 | 课程数 | 已完成 | 待完成 |
|------|------|--------|--------|--------|
| 1 | 环境配置 | 1 | 1 | 0 |
| 2 | PyTorch 基础 | 3 | 3 | 0 |
| 3 | 计算机视觉 | 3 | 1 | 2 |
| 4 | 自然语言处理 | 3 | 0 | 3 |
| 5 | 生成式 AI | 1 | 0 | 1 |
| 6 | MLOps 与部署 | 1 | 0 | 1 |
| **合计** | | **12** | **5** | **7** |

**完成度：5/12 ≈ 42%**
