# llm-roadmap

大模型算法岗求职向的完整实践仓库：从 PyTorch 打底、手写 Transformer，到 **QLoRA 指令微调 + 前后对比评估**，全部在单张消费级显卡上真实跑通。

> 所有数字均为本机实测，脚本可一键复现，不是示例代码或转述的教程结果。

---

## 一句话结果

在 RTX 4060 Laptop **8GB** 上，用 QLoRA（4-bit NF4）微调 Qwen2.5-0.5B-Instruct，
**39.5 秒**完成 3 epoch，显存峰值仅 **2.47GB**，留出集关键词命中率 **20% → 65%（+45pp）**。

| 指标 | 数值 |
|---|---|
| 基座模型 | Qwen2.5-0.5B-Instruct |
| 微调方式 | QLoRA · 4-bit NF4 + 双重量化 |
| 可训练参数 | 8,798,208 / 323,917,696（**2.716%**） |
| 训练数据 | 自建领域指令集，清洗后 100 条（训练 80 / 验证 20） |
| 训练配置 | r=16, alpha=32, lr=2e-4, max_len=256, bf16, gradient checkpointing, grad_accum=8 |
| 训练耗时 | **39.5 秒**（3 epoch / 30 step） |
| Loss | **4.8958 → 0.1257** |
| 显存峰值 | **2.47 GB** / 8 GB |
| 留出集命中率 | **20.0% → 65.0%**（4/20 → 13/20） |
| 平均回答长度 | 137.4 → 47.5 字 |

---

## 环境

| 项 | 版本 |
|---|---|
| Python | 3.12.10 |
| torch | 2.9.1+cu126 |
| CUDA | 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop（8GB，计算能力 8.9，支持 bf16） |
| transformers / peft / bitsandbytes | 5.22.2 / 0.21.2 / 0.50.2 |

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# torch 请按 CUDA 版本单独安装，例如 cu126：
# pip install torch --index-url https://download.pytorch.org/whl/cu126
```

---

## 三步复现

模型权重不入库（体积约 1GB），首次运行需下载；已配置国内镜像。

```bash
# 1) 下载基座模型（约 34 秒，走 hf-mirror）
python download_model.py

# 2) 生成并清洗指令数据集
python make_dataset.py

# 3) QLoRA 微调
python finetune.py

# 4) 微调前后对比评估
python evaluate.py
```

> 国内网络若访问 HuggingFace 失败，脚本已内置 `HF_ENDPOINT=https://hf-mirror.com`；
> 也可改用 ModelScope 下载 `qwen/Qwen2.5-0.5B-Instruct` 到 `models/` 目录。

---

## 目录结构

```
llm-roadmap/
├── README.md                  # 本文件
├── requirements.txt           # 实测可用依赖版本
├── .gitignore                 # 忽略模型权重/日志/缓存
│
├── step1_verify_gpu.py        # 环境验收：GPU 前向/反向/bf16 检查
├── step1_train_minimal.py     # 最小「训练→保存→加载推理」闭环
├── make_dataset.py            # 数据生成 + 清洗（去噪/去重/长度过滤/防泄漏）
├── finetune.py                # QLoRA 微调（4-bit 失败自动降级 bf16 LoRA）
├── evaluate.py                # 微调前后同 prompt 生成对比
├── download_model.py          # 基座模型下载（国内镜像）
│
├── data/                      # 数据集（生成物，可复现）
│   ├── train.jsonl  valid.jsonl
│   └── report.txt             # 清洗日志
├── results/                   # ★ 实验结果证据（可核查）
│   ├── train_log.json         # 训练全部参数与结果
│   ├── eval_result.json       # 前后评估数字
│   ├── eval_examples.md       # 8 个真实问答前后对比
│   └── data_cleaning_report.txt
├── stage1-pytorch-demo/       # 阶段1：PyTorch 打底 5 个练习
├── stage2-transformer/        # 阶段2：手写注意力 + 位置编码
└── docs/                      # 面试速查手册、操作清单、简历话术（HTML）
```

---

## 数据清洗做了什么

`make_dataset.py` 里演示了完整链路，日志见 `results/data_cleaning_report.txt`：

| 步骤 | 结果 |
|---|---|
| 原始条数 | 117 |
| 去空值与乱码 | 115（删除 2） |
| 精确去重（MD5） | 100（删除重复 15） |
| 长度过滤 | 100 |
| 切分 | 训练 80 / 验证 20 |
| 训练-验证交叉泄漏 | **0 条** |

**先切分再校验泄漏**，避免评测虚高。

---

## 训练中的关键处理

- `pad_token` 未设置会直接报错 → 用 `eos_token` 补齐。
- **prompt 部分 label 置为 -100**，只对回答计算损失，否则模型会去背问题。
- `apply_chat_template` 训练/推理保持一致，避免生成复读或不终止。
- 梯度检查点 + `batch=1` + 梯度累积 8 + bf16，把显存压到 2.47GB。
- transformers 5.x 的 `apply_chat_template` 返回 `BatchEncoding` 而非 `list`，脚本已做兼容。

---

## 诚实说明（实验局限）

这份仓库证明的是**完整链路可跑通 + 结果可复述**，不是「效果很好的模型」：

1. 数据集由知识点模板组合生成，同一知识点有 4 种问法，模型可能靠记忆而非泛化取胜；不能证明学到通用能力。
2. 命中率是**关键词匹配**口径，不是人工盲评或 LLM-as-judge，存在误判；请勿简写成「准确率」。
3. 仅 80 条训练数据 / 3 epoch / 0.5B 基座，NF4、GQA 等抽象概念上模型仍会编造术语（见 `results/eval_examples.md`）。
4. 后续改进方向：换公开中文指令集、增加通用数据回放缓解遗忘、人工盲评或 pairwise 胜率、更大基座与更长上下文。

---

## 效果直观对比（摘自 results/eval_examples.md）

**Q：OpenHarmony 轻量系统用的内核是什么？**
- 微调前：*"在OpenHarmony轻量系统中，使用的内核是基于Linux的……具有良好的兼容性和可扩展性……"*（错误且冗长）
- 微调后：*"OpenHarmony 轻量系统使用 LiteOS-M 内核，面向资源受限的 MCU 设备，典型如 Hi3861。"*（准确、简洁）

**Q：RoPE 位置编码作用在哪个矩阵上？**
- 微调前：*"RoPE（Rapidly Oriented Planar Epipolar Relationships）是一种用于快速匹配相机之间姿态关系的算法……"*（完全编造）
- 微调后：*"RoPE 通过旋转变换直接作用于 Query 和 Key……只保留相对位置信息。"*

---

## License

MIT
