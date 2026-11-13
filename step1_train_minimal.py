# -*- coding: utf-8 -*-
"""第一步交付物：在 GPU 上跑通「训练 → 保存 → 加载推理」最小闭环。
不用看懂代码，只要运行、看输出。跑完你就有了第一个能写进笔记的实验结果。

运行：
  c:/Users/admin/pytorch-env/Scripts/python.exe step1_train_minimal.py
"""
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 学一个简单关系：y = 2x + 1
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

model = TinyNet().to(dev)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# 造 64 组训练数据，放到 GPU
x = torch.rand(64, 1, device=dev) * 10
y = 2 * x + 1

print("设备:", dev)
print("起始 loss: %.4f" % loss_fn(model(x), y).item())

for step in range(300):
    opt.zero_grad()                      # ① 清零梯度
    pred = model(x)                      # ② 前向预测
    loss = loss_fn(pred, y)              # ③ 算损失
    loss.backward()                      # ④ 反向求导
    opt.step()                           # ⑤ 更新参数
    if step in (0, 50, 100, 200, 299):
        print("step %3d  loss = %.4f" % (step, loss.item()))

print("最终 loss: %.4f" % loss.item())
w = model.fc.weight.item(); b = model.fc.bias.item()
print("学到的参数: w=%.3f (目标2.0), b=%.3f (目标1.0)" % (w, b))

# 保存 / 加载 / 推理
torch.save(model.state_dict(), "step1_model.pth")
reloaded = TinyNet().to(dev)
reloaded.load_state_dict(torch.load("step1_model.pth", map_location=dev))
with torch.no_grad():
    print("加载后推理 x=5.0 -> %.3f (期望 11.0)" % reloaded(torch.tensor([[5.0]], device=dev)).item())
print("模型已保存: step1_model.pth")
