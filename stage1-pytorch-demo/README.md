# pytorch-demo

PyTorch 入门练习项目，对应「大模型算法」求职路线的 **阶段1：打底**。

包含：张量操作、自动求导、简单神经网络、完整训练流程，以及一个端到端的图像分类小项目（训练→保存→加载→推理）。

## 环境

```bash
pip install torch torchvision
# 仅 CPU 也可运行；有 NVIDIA 显卡建议装 CUDA 版以加速训练
```

Python 3.10+，PyTorch 2.x

## 运行

```bash
python test1_tensor.py        # 张量：创建、形状、CPU/GPU 迁移
python test2_autograd.py      # 自动求导：理解梯度就是“参数该往哪边调”
python test3_network.py       # 最小神经网络零件（继承 nn.Module）
python train_demo.py          # 训练五步曲：清零→前向→算损失→反向→更新
python image_classification.py # 完整图像分类：训练/保存/加载/推理
```

## 内容说明

| 文件 | 学到的点 | 对应简历话术 |
|------|----------|--------------|
| test1_tensor.py | 张量创建、形状变换、设备迁移 | 张量运算 |
| test2_autograd.py | requires_grad / backward 自动求导 | 自动求导机制 |
| test3_network.py | nn.Module / Linear / forward | 能搭基础网络 |
| train_demo.py | 训练循环固定模板 | 独立完成训练流程 |
| image_classification.py | CNN + DataLoader + 保存/推理 | 数据加载、模型训练、保存与推理 |

`image_classification.py` 默认用合成数据演示完整流程；代码中已标注如何替换成真实
`data/train/{cat,dog}` 图片目录（用 `torchvision.datasets.ImageFolder`），训练/保存/推理代码无需改动。

## 简历可写

> 熟练使用 Python、PyTorch，掌握张量运算与自动求导机制；能够独立完成数据集加载、
> 模型训练、保存与推理流程；熟悉 Git 版本管理。
