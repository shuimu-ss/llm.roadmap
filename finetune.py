# -*- coding: utf-8 -*-
"""QLoRA / LoRA 指令微调 —— 完整可跑，全部自动。
优先尝试 4-bit QLoRA；bitsandbytes 不可用时自动降级为 bf16 LoRA，不影响出结果。

运行：c:/Users/admin/pytorch-env/Scripts/python.exe finetune.py
产出：output/adapter/ (微调权重)、output/train_log.json (7 个数字里的训练部分)
"""
import os, json, time, random, sys
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models", "Qwen2.5-0.5B-Instruct")
OUT_DIR = os.path.join(BASE, "output")
ADAPTER_DIR = os.path.join(OUT_DIR, "adapter")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 超参（面试要报的第 5 项） ----------
CFG = dict(
    r=16, lora_alpha=32, lora_dropout=0.05,
    lr=2e-4, epochs=3, batch_size=1, grad_accum=8,
    max_len=256, bf16=True, gradient_checkpointing=True,
)

def log(*a):
    print(*a, flush=True)

def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def build_prompt(row):
    """用模型自带的 chat template，保证训练和推理格式一致（最常见的坑）。"""
    return [{"role": "user", "content": row["instruction"]}]

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(42); torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("=" * 60)
    log("设备:", dev)
    if dev == "cuda":
        log("显卡:", torch.cuda.get_device_name(0), "显存 %.2f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token        # 坑①：不设 pad_token 会报错
    tok.padding_side = "right"

    # ---------- 尝试 4-bit QLoRA，失败则降级 ----------
    mode = "QLoRA (4-bit NF4)"
    try:
        import bitsandbytes as bnb
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, quantization_config=quant, device_map={"": 0},
            trust_remote_code=True, dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        log("量化模式: 4-bit NF4 + 双重量化 (QLoRA)")
    except Exception as e:
        log("!! 4-bit 量化不可用，自动降级为 bf16 LoRA")
        log("   原因:", str(e)[:200])
        mode = "LoRA (bf16, 无量化)"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        ).to(dev)
        log("量化模式: bf16 LoRA（链路完全相同，只是不量化）")

    model.config.use_cache = False
    if CFG["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    lcfg = LoraConfig(
        r=CFG["r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log("可训练参数: %d / 总参数 %d (%.3f%%)" % (trainable, total, 100 * trainable / total))

    # ---------- 数据：只对回答算 loss（坑②） ----------
    train_rows = load_rows(os.path.join(BASE, "data", "train.jsonl"))
    def encode(row):
        p = tok.apply_chat_template(build_prompt(row), tokenize=True, add_generation_prompt=True)
        # transformers 5.x 返回 BatchEncoding，旧版返回 list —— 两种都兼容
        p_ids = list(p["input_ids"]) if isinstance(p, dict) else list(p.input_ids) if hasattr(p, "input_ids") else list(p)
        a_ids = tok(row["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + a_ids)[: CFG["max_len"]]
        labels = ([-100] * len(p_ids) + a_ids)[: CFG["max_len"]]
        return ids, labels

    samples = [encode(r) for r in train_rows]
    log("训练样本: %d 条, 最大长度 %d" % (len(samples), CFG["max_len"]))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"])
    model.train()
    steps, losses, peak_mem = 0, [], 0.0
    n = len(samples)
    accum = CFG["grad_accum"]

    for ep in range(CFG["epochs"]):
        random.shuffle(samples)
        for i in range(0, n, accum):
            batch = samples[i: i + accum]
            L = max(len(x[0]) for x in batch)
            input_ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            labels = torch.full((len(batch), L), -100, dtype=torch.long)
            attn = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (ids, lab) in enumerate(batch):
                input_ids[j, :len(ids)] = torch.tensor(ids)
                labels[j, :len(lab)] = torch.tensor(lab)
                attn[j, :len(ids)] = 1
            input_ids, labels, attn = input_ids.to(dev), labels.to(dev), attn.to(dev)

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / accum).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            steps += 1
            losses.append(out.loss.item())
            if dev == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024**3)
            if steps % 5 == 0 or steps == 1:
                log("epoch %d  step %3d  loss %.4f" % (ep + 1, steps, out.loss.item()))

    elapsed = time.time() - t0
    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)

    result = dict(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        mode=mode,
        train_samples=len(samples),
        epochs=CFG["epochs"], steps=steps,
        loss_start=round(losses[0], 4), loss_end=round(losses[-1], 4),
        r=CFG["r"], lora_alpha=CFG["lora_alpha"], lr=CFG["lr"],
        max_len=CFG["max_len"], bf16=CFG["bf16"],
        gradient_checkpointing=CFG["gradient_checkpointing"],
        peak_gpu_mem_gb=round(peak_mem, 2),
        gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        train_seconds=round(elapsed, 1),
        trainable_params=trainable,
    )
    with open(os.path.join(OUT_DIR, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log("=" * 60)
    log("训练完成，耗时 %.1f 秒" % elapsed)
    log(json.dumps(result, ensure_ascii=False, indent=2))
    log("adapter 保存于:", ADAPTER_DIR)

if __name__ == "__main__":
    main()
