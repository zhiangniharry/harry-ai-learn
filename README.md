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
| torchaudio | 2.5.1+cu121 |
| CUDA | 12.1 |
| cuDNN | 9.1.0 |
| numpy | 2.4.4 |
| Pillow | 12.2.0 |

> **GPU 训练确认正常**：CUDA 可用，cuDNN 可用，PyTorch 成功识别 RTX 2070。

### 虚拟环境

```bash
# 激活虚拟环境（PowerShell）
E:\Harry AI 学习\.venv\Scripts\Activate.ps1

# 或通过绝对路径运行 Python
E:\Harry AI 学习\.venv\Scripts\python.exe your_script.py
```

依赖包列表（`.venv\Scripts\pip list`）：

```
Jinja2            3.1.6
Pillow           12.2.0
filelock          3.29.0
fsspec          2026.4.0
lxml              6.1.0
mpmath            1.3.0
networkx          3.6.1
numpy             2.4.4
python-docx       1.2.0
sympy             1.13.1
torch             2.5.1+cu121
torchaudio        2.5.1+cu121
torchvision       0.20.1+cu121
typing_extensions  4.15.0
```

---

## 目录结构

```
E:\Harry AI 学习\
├── 01-环境配置/          # 环境验证脚本
├── 02-PyTorch基础/       # DataLoader、训练循环、课程笔记（docx）
├── 03-计算机视觉/        # 图像分类、目标检测、MNIST 项目
│   ├── 01-MNIST分类/
│   ├── 01-图像分类/
│   ├── 02-目标检测/
│   └── 03-项目-自定义检测器/
├── 04-自然语言处理/      # Transformer、BERT、Chatbot 项目
│   ├── 01-Transformer实现/
│   ├── 02-BERT微调/
│   ├── 03-项目-自定义Chatbot/
│   ├── datasets/         # NLP 语料
│   └── outputs/         # 模型输出
├── 05-生成式AI/          # Stable Diffusion 微调
│   └── 01-StableDiffusion微调/
├── 06-MLOps与部署/       # 待补充
├── 99-论文与阅读/        # 论文精读笔记
│   ├── Karpathy-AutoResearch/
│   └── Dimension-TheMeltingPoint/
├── datasets/             # 共享数据集
├── models/               # 模型权重（.pth 文件）
├── utils/                # 工具脚本
├── .venv/                # Python 虚拟环境
└── README.md
```

---

## 学习路线

```
第 1 步  环境配置
         └─ 验证 PyTorch + CUDA 可用

第 2 步  PyTorch 基础
         ├─ DataLoader 数据管道
         ├─ 训练循环（正向传播 / 反向传播 / 优化）
         └─ 课程笔记自动生成脚本（docx）

第 3 步  计算机视觉（CV）
         ├─ MNIST 手写数字分类（CNN）
         ├─ 通用图像分类
         ├─ 目标检测（YOLO / Faster R-CNN 等）
         └─ 项目：自定义目标检测器

第 4 步  自然语言处理（NLP）
         ├─ Transformer 从零实现
         ├─ BERT 预训练模型微调
         └─ 项目：自定义 Chatbot

第 5 步  生成式 AI
         └─ Stable Diffusion 微调（LoRA / DreamBooth）

第 6 步  MLOps 与部署
         ├─ ONNX / TorchScript 导出
         ├─ 模型量化与加速
         └─ Web API / Docker 部署
```

---

## 快速开始

### 1. 验证环境

```bash
E:\Harry AI 学习\.venv\Scripts\python.exe 01-环境配置\verify_pytorch.py
```

### 2. 运行 CV 示例（MNIST）

```bash
E:\Harry AI 学习\.venv\Scripts\python.exe 03-计算机视觉\01-MNIST分类\*.py
```

### 3. 生成课程笔记

```bash
E:\Harry AI 学习\.venv\Scripts\python.exe 02-PyTorch基础\generate_docx_1.2.py
E:\Harry AI 学习\.venv\Scripts\python.exe 02-PyTorch基础\generate_docx_1.3.py
```

---

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `utils/gen_karpathy_docs.py` | 自动生成 Karpathy AutoResearch 论文阅读文档 |
| `utils/gen_melting_docs.py` | 自动生成熔点论文阅读文档 |
| `utils/reorg_files.py` | 批量文件整理 |
| `utils/reorg.ps1` | PowerShell 文件整理 |
| `check_chrome.ps1` | 检查 Chrome 版本 |
| `check_gateway.ps1` | 检查网关连通性 |

---

## 论文阅读

- **Karpathy AutoResearch** — 自动驾驶研究方向
- **Dimension-TheMeltingPoint** — 维度熔点相关研究

---

## TODO

- [ ] 补充 06-MLOps与部署 内容
- [ ] 添加 Transformer 可视化演示
- [ ] 完成自定义 Chatbot 项目
- [ ] 完善 Stable Diffusion 微调流程
- [ ] 添加项目单元测试
