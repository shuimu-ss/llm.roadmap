# transformer_from_scratch.py
# 阶段2 核心：不调现成库，自己实现 Transformer 的关键模块。
# 对应 JD：掌握注意力优化、位置编码等核心模块实现。
#
# 这个文件实现：
#   1) 正弦余弦位置编码 (PositionalEncoding)
#   2) 缩放点积注意力 (Scaled Dot-Product Attention)
#   3) 多头注意力 (MultiHeadAttention)
#   4) 一个极简编码器层 (EncoderLayer：注意力 + 前馈 + 残差 + LayerNorm)
#   5) 用它做个字符级“下一个字预测”小任务，训练几步并采样
#
# 关键直觉（面试必考）：
#   Q (Query)：当前这个词“想找什么信息”
#   K (Key)  ：每个词“能提供什么信息的标签”，用来和 Q 匹配
#   V (Value)：每个词“真正携带的信息”，按 QK 匹配度加权求和后得到输出
#   位置编码：Attention 本身没有顺序概念，必须额外告诉模型“词在第几位”

import torch
import torch.nn as nn
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- 1. 正弦余弦位置编码 ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)          # (max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)          # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)          # 奇数维用 cos
        self.pe = pe.unsqueeze(0)                   # (1, max_len, d_model) 方便广播

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)].to(x.device)


# ---------- 2. 缩放点积注意力 ----------
def scaled_dot_product_attention(q, k, v):
    # q,k,v: (batch, heads, seq_len, d_k)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (b,h,seq,seq)
    attn = torch.softmax(scores, dim=-1)                            # 每个词对其他词的注意力权重
    out = torch.matmul(attn, v)                                     # 按权重汇总 V
    return out, attn


# ---------- 3. 多头注意力 ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 用三个线性层把输入分别投影成 Q/K/V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)   # 多头拼接后再投影回 d_model

    def forward(self, x):
        b, n, _ = x.shape
        # 投影并拆成多头：(b, n, d_model) -> (b, h, n, d_k)
        q = self.w_q(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)

        attn_out, attn = scaled_dot_product_attention(q, k, v)      # (b, h, n, d_k)
        # 多头拼回：(b, h, n, d_k) -> (b, n, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.w_o(attn_out), attn


# ---------- 4. 编码器层 ----------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=128):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 残差连接：输出 = 层归一化( x + 注意力(x) )
        attn_out, _ = self.self_attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ---------- 5. 小任务：字符级下一个字预测 ----------
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, num_heads=4, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.enc = EncoderLayer(d_model, num_heads)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)              # (b, n) -> (b, n, d_model)
        x = self.pos(x)                 # 加位置信息
        x = self.enc(x)                 # 过一层编码器
        return self.head(x)             # 每个位置预测下一个字的概率分布


def demo():
    text = "人工智能正在改变世界人工智能让机器学习成为可能人工"
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)

    # 构造训练样本：(上下文) -> 下一个字
    seq_len = 6
    xs, ys = [], []
    for i in range(len(text) - seq_len):
        xs.append([stoi[c] for c in text[i : i + seq_len]])
        ys.append(stoi[text[i + seq_len]])
    X = torch.tensor(xs)
    Y = torch.tensor(ys)

    model = TinyTransformerLM(vocab).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(200):
        logits = model(X.to(DEVICE))            # (N, seq_len, vocab)
        # 用每个位置的输出去预测“下一个字”，这里取最后一个位置的输出做简化演示
        loss = criterion(logits[:, -1, :], Y.to(DEVICE))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 40 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    # 采样：给前 seq_len 个字，预测下一个
    model.eval()
    with torch.no_grad():
        ctx = torch.tensor([[stoi[c] for c in text[:seq_len]]]).to(DEVICE)
        out = model(ctx)[:, -1, :].argmax(-1).item()
        print(f"输入“{text[:seq_len]}” -> 模型预测下一个字：“{itos[out]}”")


if __name__ == "__main__":
    demo()
