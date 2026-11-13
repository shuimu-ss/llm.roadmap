# transformer-from-scratch

大模型求职路线 **阶段2：Transformer 核心**。不依赖现成高层封装，从零实现关键模块。

## 文件

- `transformer_from_scratch.py`
  - `PositionalEncoding`：正弦余弦位置编码
  - `scaled_dot_product_attention`：缩放点积注意力
  - `MultiHeadAttention`：多头注意力（Q/K/V 投影、分头、拼接、输出投影）
  - `EncoderLayer`：注意力 + 前馈 + 残差 + LayerNorm
  - `TinyTransformerLM` + `demo()`：字符级“下一个字预测”小任务

## 运行

```bash
python transformer_from_scratch.py
```

输出训练 loss 下降，并给出一个“给前文预测下一个字”的样例。

## 重点理解（面试常问）

- **QKV 是什么**：Q=当前词想找什么；K=每个词用来被匹配的标签；V=真正携带的信息，按 QK 相似度加权求和。
- **为什么需要位置编码**：自注意力本身对顺序无感，必须显式注入“第几位”。
- **残差 + LayerNorm**：让深层网络梯度稳定、好训练。

## 简历可写

> 理解 Transformer 架构，独立实现多头注意力、正弦位置编码、残差连接等核心模块；
> 基于 PyTorch 搭建简易序列预测模型，掌握自注意力与位置编码原理。
