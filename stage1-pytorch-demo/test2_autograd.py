# test2_autograd.py
# 目标：理解“自动求导（autograd）”——这是训练 AI 的核心机制。
# 大白话：模型猜错了，靠反向传播自动算出“每个参数该往哪边调、调多少”。

import torch

# requires_grad=True 表示：我要跟踪这个值的计算过程，方便后面自动算梯度
x = torch.tensor(3.0, requires_grad=True)

# 假设一个简单函数 y = x^2
# 当 x=3 时 y=9
y = x ** 2

# 反向传播：从 y 出发，算出 y 对 x 的导数（梯度）
# 数学上 dy/dx = 2x，x=3 时梯度 = 6
y.backward()

print("x 当前值：", x.item())      # 3.0
print("y 值：", y.item())          # 9.0
print("x 的梯度 (dy/dx)：", x.grad.item())  # 6.0

# 怎么用？如果我想让 y 变小：
# 梯度告诉我“x 增大 y 会增大”，所以应该把 x 往小调。
# 以后训练大模型，就是 optimizer 拿着这个梯度去改上亿个参数，原理完全一样。

# 注意：默认梯度会累加，下一次反向前要清零（后面 train_demo 里会看到 optimizer.zero_grad()）
