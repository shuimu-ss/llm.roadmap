# train_demo.py
# 目标：跑通“训练固定五步曲”。这是所有深度学习训练（包括大模型）的底层模板。
#
# 我们要让模型学会：输入 2.0，输出 4.0（也就是学一个 y = 2x 的关系）
#
# 五步曲：
#   ① optimizer.zero_grad()   清空上一步的梯度
#   ② model(x)                前向：让模型猜一个值
#   ③ loss = criterion(...)   算“猜错多少”
#   ④ loss.backward()         反向：算出每个参数该怎么改
#   ⑤ optimizer.step()        动手改参数

import torch
import torch.nn as nn
import torch.optim as optim

# ---------- 1. 定义模型 ----------
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # 输入1维，输出1维

    def forward(self, x):
        return self.linear(x)

model = TinyNet()

# ---------- 2. 定义损失函数和优化器 ----------
# 损失：用均方误差（MSE）衡量“预测值和目标值差多少”
criterion = nn.MSELoss()
# 优化器：SGD，负责按梯度去改参数；lr 是每次改多大（学习率）
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---------- 3. 训练循环 ----------
for i in range(100):
    x = torch.tensor([[2.0]])      # 输入
    target = torch.tensor([[4.0]]) # 希望模型学会的输出

    optimizer.zero_grad()          # ① 清零梯度
    pred = model(x)                # ② 前向预测
    loss = criterion(pred, target) # ③ 算损失
    loss.backward()                # ④ 反向求导
    optimizer.step()               # ⑤ 更新参数

    if i % 10 == 0:
        print(f"第 {i:3d} 轮  损失={loss.item():.4f}")

# ---------- 4. 验证 ----------
with torch.no_grad():   # 推理阶段不记录梯度，省内存
    test_x = torch.tensor([[2.0]])
    res = model(test_x)
    print("\n训练完成，输入 2.0，模型预测：", res.item())

# ---------- 5. 保存 & 加载（简历里“保存与推理流程”就靠这个）----------
torch.save(model.state_dict(), "tinynet.pth")
print("模型已保存为 tinynet.pth")

loaded = TinyNet()
loaded.load_state_dict(torch.load("tinynet.pth"))
with torch.no_grad():
    print("重新加载后推理：", loaded(test_x).item())
