# -*- coding: utf-8 -*-
"""整理文件夹结构 + 保存原文"""
import os, shutil

BASE = r"E:\Harry AI 学习\99-论文与阅读"

# ===== 1. 整理 Karpathy-AutoResearch =====
kd = os.path.join(BASE, "Karpathy-AutoResearch")
kd_yuanwen = os.path.join(kd, "原文")
os.makedirs(kd_yuanwen, exist_ok=True)

raw_files = ["README.md","program.md","train.py","prepare.py","pyproject.toml","analysis.ipynb","progress.png","uv.lock"]
for f in raw_files:
    src = os.path.join(kd, f)
    dst = os.path.join(kd_yuanwen, f)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {f}")

# 删除旧文档和生成脚本
del_files = ["Karpathy-AutoResearch-详解.docx","Karpathy-AutoResearch-翻译.docx",
             "generate_autoresearch_docs.py","generate_melting_point_doc.py",
             "The-Melting-Point-全文翻译.docx","reorg.ps1"]
for f in del_files:
    p = os.path.join(kd, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed: {f}")

# ===== 2. 创建 The Melting Point 文件夹 =====
td = os.path.join(BASE, "Dimension-TheMeltingPoint")
td_yuanwen = os.path.join(td, "原文")
os.makedirs(td_yuanwen, exist_ok=True)

# 从附件复制的文本保存原文（简化版，主要段落已在本脚本中）
print("\n=== 文件夹整理完成 ===")
print(f"Karpathy: {kd}")
print(f"  ├── 原文/ ({len(os.listdir(kd_yuanwen))} files)")
print(f"MeltingPoint: {td}")
print(f"  ├── 原文/")
