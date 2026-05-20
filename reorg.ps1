# 整理文件夹结构
$ErrorActionPreference = "Stop"

$kd = "E:\Harry AI 学习\99-论文与阅读\Karpathy-AutoResearch"
$td = "E:\Harry AI 学习\99-论文与阅读\Dimension-TheMeltingPoint"

# Karpathy: 创建原文文件夹
New-Item -ItemType Directory -Force -Path "$kd\原文" | Out-Null

# 移动原始文件到原文/
$raw_files = @("README.md","program.md","train.py","prepare.py","pyproject.toml","analysis.ipynb","progress.png","uv.lock")
foreach ($f in $raw_files) {
    $src = Join-Path $kd $f
    $dst = Join-Path "$kd\原文" $f
    if (Test-Path $src) {
        Move-Item $src $dst -Force
        Write-Output "Moved: $f"
    }
}

# 删除旧文档和生成脚本
$del_files = @("Karpathy-AutoResearch-详解.docx","Karpathy-AutoResearch-翻译.docx","generate_autoresearch_docs.py","generate_melting_point_doc.py","The-Melting-Point-全文翻译.docx")
foreach ($f in $del_files) {
    $p = Join-Path $kd $f
    if (Test-Path $p) {
        Remove-Item $p -Force
        Write-Output "Removed: $f"
    }
}

# The Melting Point: 创建文件夹
New-Item -ItemType Directory -Force -Path "$td\原文" | Out-Null

Write-Output "DONE"
