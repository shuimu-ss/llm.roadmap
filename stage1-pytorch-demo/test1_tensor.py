# test1_tensor.py
# 目标：认识 PyTorch 的“张量（tensor）”——就是能被 GPU 加速的专用数字数组。
# 普通 Python 列表不能丢给显卡，张量可以。

import torch

# 1. 造一个张量（一维向量）
x = torch.tensor([1, 2, 3, 4])
print("张量 x：", x)
print("数据类型：", x.dtype)          # 元素类型，比如 int64
print("形状 shape：", x.shape)        # 几行几列

# 2. 常用创建方式
zeros = torch.zeros(2, 3)            # 2x3 全 0
ones = torch.ones(3)                 # 长度3 全 1
rand = torch.randn(2, 2)             # 2x2 正态分布随机数
print("\nzeros:\n", zeros)
print("randn:\n", rand)

# 3. 形状变换与设备迁移
flat = rand.flatten()                # 拉平成一维
print("\nflatten 后：", flat)

# 看能不能上显卡（有 NVIDIA 显卡且装了 CUDA 版 torch 才会 True）
if torch.cuda.is_available():
    x = x.cuda()
    print("已放到 GPU：", x.device)
else:
    print("没有可用 GPU，用 CPU 跑（训练慢一些，但逻辑完全一样）")

# 4. 张量也能做数学
y = x * 2 + 1
print("\nx*2+1 =", y)
