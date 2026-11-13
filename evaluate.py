# -*- coding: utf-8 -*-
"""评估：同一个问题分别问「微调前」和「微调后」，量化对比。
指标：关键词命中率（标准答案里的关键术语是否被覆盖）+ 平均回答长度。
产出：output/eval_result.json、output/eval_examples.md（前后对比截图素材）
运行：c:/Users/admin/pytorch-env/Scripts/python.exe evaluate.py
"""
import os, json, torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models", "Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.path.join(BASE, "output", "adapter")
OUT = os.path.join(BASE, "output")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 每个问题对应的判定关键词（命中即算答到点上）
KEYWORDS = {
    "LiteOS-M": ["LiteOS", "轻量"], "2.4GHz": ["2.4"], "921600": ["921600"],
    "full build": ["全量", "full"], "1883": ["1883"], "QoS2": ["恰好一次", "QoS2"],
    "router": ["router", "pushUrl"], "Column": ["Column", "Row", "纵向"],
    "根号 d_k": ["softmax", "梯度", "方差"], "位置编码": ["顺序", "位置"],
    "RoPE": ["Query", "Key", "旋转", "相对"], "LoRA": ["梯度", "优化器", "冻结"],
    "NF4": ["4", "量化"], "PagedAttention": ["分页", "碎片", "块"],
    "KV Cache": ["缓存", "重复", "Key"], "bf16": ["动态范围", "溢出", "fp32"],
    "梯度检查点": ["重算", "换显存", "激活"], "灾难性遗忘": ["学习率", "回放", "冻结"],
    "label": ["-100", "prompt", "回答部分"], "loss": ["不能", "质量", "留出集"],
    "LayerNorm": ["batch", "单样本", "变长"], "Decoder-only": ["一致", "简单", "规模"],
    "GQA": ["共享", "KV", "Head", "头"], "CLIP": ["对比学习", "向量空间"],
    "SFT": ["无标注", "指令", "对齐"],
}

def load_rows():
    with open(os.path.join(BASE, "data", "valid.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def pick_keywords(output):
    keys = []
    for k in KEYWORDS:
        if k in output or k in " ".join(KEYWORDS.keys()):
            pass
    # 直接按答案文本匹配
    for k, v in KEYWORDS.items():
        if k.lower() in output.lower():
            keys.extend(v)
            return keys
    return list(KEYWORDS.values())[0]  # 兜底

def get_kw(output):
    """返回该答案的判定关键词列表"""
    for key, vals in KEYWORDS.items():
        if key.lower() in output.lower():
            return vals
    # 用答案里出现的前 2 个中文术语兜底
    return []

def generate(model, tok, question, dev):
    msgs = [{"role": "user", "content": question}]
    p = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    # transformers 5.x 返回 BatchEncoding（dict），旧版返回 list —— 统一转 tensor
    ids = p["input_ids"] if isinstance(p, dict) else (p.input_ids if hasattr(p, "input_ids") else p)
    ids = torch.tensor([list(ids)], dtype=torch.long, device=dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return text

def score(answers, rows):
    hits, lens = 0, []
    for r, a in zip(rows, answers):
        kw = get_kw(r["output"])
        # 命中判定：答案里出现关键词中至少一个，或答案与标准答案有实质重合
        ok = (any(k.lower() in a.lower() for k in kw) if kw else False)
        if not ok:
            # 退一步：标准答案的前 6 个字是否出现在生成里
            ok = r["output"][:6] in a
        hits += int(ok)
        lens.append(len(a))
    return dict(hit_rate=round(hits / len(rows), 3), hits=hits, total=len(rows),
                avg_len=round(sum(lens) / len(lens), 1))

def main():
    from peft import PeftModel
    random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows()
    print("验证集: %d 条" % len(rows), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                trust_remote_code=True).to(dev)
    base.eval()
    before = [generate(base, tok, r["instruction"], dev) for r in rows]
    s_before = score(before, rows)
    print("微调前:", s_before, flush=True)

    del base
    torch.cuda.empty_cache()

    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                             trust_remote_code=True).to(dev),
        ADAPTER_DIR).eval()
    after = [generate(tuned, tok, r["instruction"], dev) for r in rows]
    s_after = score(after, rows)
    print("微调后:", s_after, flush=True)

    result = dict(before=s_before, after=s_after,
                  delta=round(s_after["hit_rate"] - s_before["hit_rate"], 3),
                  method="留出集 20 条，关键词命中率 + 人工可核查的固定 prompt 生成")
    with open(os.path.join(OUT, "eval_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "eval_examples.md"), "w", encoding="utf-8") as f:
        f.write("# 微调前后对比（同一 prompt，贪心解码）\n\n")
        f.write("整体：命中率 %.1f%% -> %.1f%% (Δ %+.1f 个百分点)\n\n" % (
            s_before["hit_rate"] * 100, s_after["hit_rate"] * 100, result["delta"] * 100))
        for i, r in enumerate(rows[:8]):
            f.write("## %d. %s\n\n" % (i + 1, r["instruction"]))
            f.write("- **标准答案**：%s\n" % r["output"])
            f.write("- **微调前**：%s\n" % before[i][:180])
            f.write("- **微调后**：%s\n\n" % after[i][:180])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("对比素材: output/eval_examples.md", flush=True)

if __name__ == "__main__":
    main()
