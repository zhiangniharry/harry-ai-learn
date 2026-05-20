import torch

print('=' * 50)
print('PyTorch 环境验证')
print('=' * 50)

print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 是否可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f'GPU 矩阵乘法测试: 成功 (结果形状 {z.shape})')
else:
    print('警告: CUDA 不可用，只能使用 CPU')

print('=' * 50)
print('验证完成')
print('=' * 50)
