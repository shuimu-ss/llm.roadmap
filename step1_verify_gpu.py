# -*- coding: utf-8 -*-
"""第一步验收脚本：确认 GPU 真能用于训练，而不只是"显示可用"。
运行：c:/Users/admin/pytorch-env/Scripts/python.exe step1_verify_gpu.py
"""
import torch

print("=" * 52)
print("PyTorch :", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("显卡型号:", p.name)
    print("显存总量: %.2f GB" % (p.total_memory / 1024**3))
    print("计算能力: %d.%d  (≥8.0 才支持 bf16)" % (p.major, p.minor))
    print("bf16 支持:", torch.cuda.is_bf16_supported())

    # 1) 真的在 GPU 上算一次矩阵乘 + 反向，确认训练链路通
    w = torch.randn(512, 512, device="cuda", requires_grad=True)
    x = torch.randn(32, 512, device="cuda")
    loss = (x @ w).square().mean()
    loss.backward()
    print("GPU 前向+反向: OK, loss=%.4f, grad=%.4f" % (loss.item(), w.grad.abs().mean().item()))

    # 2) 看实际还剩多少可用显存（决定能跑多大的模型）
    free, total = torch.cuda.mem_get_info()
    print("当前空闲显存: %.2f GB / %.2f GB" % (free / 1024**3, total / 1024**3))
print("=" * 52)
