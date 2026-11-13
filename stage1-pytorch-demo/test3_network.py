# test3_network.py
# 目标：搭一个最小的神经网络“零件”，理解模型的基本结构。
# 关键点：
#   - 所有模型（包括后面要写的 Transformer、LLaMA）都继承 nn.Module
#   - __init__ 里摆零件（层），forward 里规定数据怎么流过这些零件
#   - forward 不能改名，PyTorch 会固定调用它

import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 一个线性层：输入 2 个数 -> 输出 1 个数
        # 内部就是 y = x·W + b，W、b 就是模型要学习的“参数”
        self.layer = nn.Linear(2, 1)

    def forward(self, input_data):
        return self.layer(input_data)

# 实例化网络
net = TinyNet()
print(net)  # 打印网络结构

# 随便给一组输入：[1.0, 2.0]
data = torch.tensor([[1.0, 2.0]])
out = net(data)
print("网络输出：", out)

# 看看模型里有哪些可训练参数（这就是训练时要改的数字）
for name, param in net.named_parameters():
    print(name, param.shape)
