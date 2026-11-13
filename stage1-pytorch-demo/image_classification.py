# image_classification.py
# 目标：跑通一个“完整可交付”的图像分类流程——训练、保存、加载、推理。
# 这是阶段1最能写进简历的小项目（对应 JD：能独立完成数据加载、训练、保存、推理）。
#
# 为了不依赖外网下载数据集，这里用“合成数据”演示：
#   class 0 = 左上角有亮块的图像
#   class 1 = 右下角有亮块的图像
# 模型是一个小 CNN，会真正学会这个空间规律（损失会下降、准确率会上升）。
#
# 想换成真实“猫狗分类”？把下面 build_dataset() 里标 [换成真实数据] 的部分
# 改成用 torchvision.datasets.ImageFolder 读你本地的
#   data/train/cat/*.jpg
#   data/train/dog/*.jpg
# 即可，其余训练/保存/推理代码完全不用动。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. 数据集 ----------
class PatternDataset(Dataset):
    """合成二分类数据集：左上亮块 vs 右下亮块（28x28 单通道）。"""
    def __init__(self, n=400):
        self.samples = []
        for _ in range(n):
            img = torch.zeros(1, 28, 28)
            label = torch.randint(0, 2, (1,)).item()
            if label == 0:
                img[0, 4:12, 4:12] = 1.0      # 左上亮块
            else:
                img[0, 16:24, 16:24] = 1.0    # 右下亮块
            self.samples.append((img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# [换成真实数据] 示例（取消注释并放好图片后即可用）：
# from torchvision.datasets import ImageFolder
# train_ds = ImageFolder("data/train", transform=transforms.ToTensor())

train_ds = PatternDataset(n=400)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# ---------- 2. 小 CNN 模型 ----------
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 卷积提取特征
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 28->14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 14->7
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 2),                           # 二分类输出
        )

    def forward(self, x):
        return self.net(x)

model = MiniCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- 3. 训练 ----------
print(f"训练设备：{DEVICE}")
for epoch in range(5):
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    print(f"Epoch {epoch+1}  损失={total_loss/len(train_loader):.4f}  准确率={correct/total:.2%}")

# ---------- 4. 保存 & 推理 ----------
torch.save(model.state_dict(), "minicnn.pth")
print("模型已保存为 minicnn.pth")

# 取一张测试图推理
model.eval()
test_img, test_label = train_ds[0]
with torch.no_grad():
    pred = model(test_img.unsqueeze(0).to(DEVICE)).argmax(1).item()
print(f"样本真实标签={test_label}，模型预测={pred}，{'✓' if pred==test_label else '✗'}")
