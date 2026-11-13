# -*- coding: utf-8 -*-
"""生成一份小型中文指令数据集，并演示完整数据清洗流程。
主题贴合你的专业（嵌入式 / 通信 / 鸿蒙），简历上可写"领域指令微调"。
输出：data/train.jsonl, data/valid.jsonl, data/report.txt
"""
import json, os, random, hashlib, re

random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# 知识点池：问题模板 + 标准答案，组合生成，保证答案确定、可评估
TOPICS = [
    ("OpenHarmony 轻量系统用的内核是什么", "OpenHarmony 轻量系统使用 LiteOS-M 内核，面向资源受限的 MCU 设备，典型如 Hi3861。"),
    ("Hi3861 开发板支持哪个 Wi-Fi 频段", "Hi3861 只支持 2.4GHz 频段，不支持 5GHz，连接手机热点时需选择 2.4G。"),
    ("HiBurn 烧录时波特率一般设为多少", "HiBurn 烧录 Hi3861 时串口波特率通常设为 921600，并需勾选 Auto burn 后按复位键。"),
    ("hb build -f 中的 -f 表示什么", "-f 表示 full build 全量编译，会清理旧产物并重新构建整个工程。"),
    ("MQTT 默认端口是多少", "MQTT 基于 TCP 的默认端口是 1883；启用 TLS 加密时常用 8883。"),
    ("MQTT 中 QoS2 的含义是什么", "QoS2 表示恰好一次投递，通过四次握手保证消息不丢不重，开销最大。"),
    ("ArkUI 中页面跳转用哪个 API", "ArkUI 使用 router.pushUrl({url:'pages/Second'}) 跳转，router.back() 返回上一页。"),
    ("ArkUI 里 Column 和 Row 的区别", "Column 是纵向布局容器，子组件从上到下排列；Row 是横向布局，从左到右排列。"),
    ("Transformer 为什么要除以根号 d_k", "防止 d_k 较大时点积数值过大，使 softmax 进入饱和区、梯度趋近于零，除以根号 d_k 把方差拉回 1。"),
    ("自注意力为什么需要位置编码", "自注意力对输入顺序不敏感（置换等变），必须显式注入位置信息才能区分词序。"),
    ("RoPE 位置编码作用在哪个矩阵上", "RoPE 通过旋转变换直接作用于 Query 和 Key，使二者内积只依赖相对位置。"),
    ("LoRA 为什么能省显存", "LoRA 冻结基座权重，只训练低秩增量矩阵，因此无需保存全量梯度和 Adam 优化器状态。"),
    ("QLoRA 的 NF4 是什么", "NF4 是面向正态分布权重设计的 4-bit 量化数据类型，配合双重量化进一步压缩基座。"),
    ("vLLM 的 PagedAttention 解决什么问题", "把 KV Cache 按块分页管理，消除按最大长度预留造成的显存碎片与浪费。"),
    ("KV Cache 为什么能加速推理", "缓存已生成 token 的 Key/Value，每步只计算新 token，避免重复计算历史。"),
    ("bf16 相比 fp16 的优势", "bf16 指数位更多、动态范围接近 fp32，训练时不易溢出，通常无需 loss scaling。"),
    ("梯度检查点的原理", "前向只保存少量检查点，反向时重算中间激活，用计算时间换显存。"),
    ("灾难性遗忘怎么缓解", "降低学习率、减少 epoch、混入通用数据回放，或用 LoRA 冻结基座。"),
    ("指令微调时 label 要怎么处理", "把 prompt 部分的 label 置为 -100，只对回答部分计算损失，避免模型学着背问题。"),
    ("微调效果能不能只看 loss", "不能。loss 下降只说明拟合更好，不等于回答质量提升，需留出集人工或模型打分评估。"),
    ("LayerNorm 和 BatchNorm 的区别", "LayerNorm 对单样本自身的特征维归一化，不依赖 batch 统计，更适合变长序列。"),
    ("Decoder-only 为什么成为主流", "结构简单、训练与推理目标一致、参数效率高，且易于堆叠规模获得强泛化能力。"),
    ("GQA 如何降低显存", "多个 Query 头共享一组 Key/Value 头，直接缩小 KV Cache 数倍。"),
    ("CLIP 是怎么对齐图文的", "用对比学习把图像与文本编码到同一向量空间，拉近配对样本、推远不配对样本。"),
    ("SFT 和预训练的目标差异", "预训练在海量无标注语料上学通用语言知识，SFT 用少量指令对学按要求回答的格式与对齐。"),
]

QTMPL = [
    "请简要回答：{q}？",
    "请问{q}？",
    "{q}，一句话说明。",
    "解释一下：{q}。",
]

def build():
    raw = []
    for q, a in TOPICS:
        for tpl in QTMPL:
            raw.append({"instruction": tpl.format(q=q), "input": "", "output": a})
    # 故意混入重复和脏数据，用来演示清洗真实生效
    raw += raw[:15]
    raw.append({"instruction": "请简要回答：", "input": "", "output": ""})
    raw.append({"instruction": "<html>乱码%%%", "input": "", "output": "???"})
    return raw

def clean(raw):
    log = []
    log.append("原始条数: %d" % len(raw))

    # 1) 去空 / 去噪
    def ok(x):
        ins, out = x["instruction"], x["output"]
        if len(out.strip()) < 5:
            return False
        if re.search(r"<html>|乱码|%%%|\?\?\?", ins + out):
            return False
        return True
    step = [x for x in raw if ok(x)]
    log.append("去空值与乱码后: %d (删除 %d)" % (len(step), len(raw) - len(step)))

    # 2) 精确去重（按 instruction+output 哈希）
    seen, uniq = set(), []
    for x in step:
        h = hashlib.md5((x["instruction"] + "||" + x["output"]).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(x)
    log.append("精确去重后: %d (删除重复 %d)" % (len(uniq), len(step) - len(uniq)))

    # 3) 长度过滤
    step = [x for x in uniq if 4 <= len(x["instruction"]) <= 120 and 5 <= len(x["output"]) <= 300]
    log.append("长度过滤后: %d" % len(step))

    # 4) 先切分再去重校验（防测试集泄漏）
    random.shuffle(step)
    n_valid = max(20, int(len(step) * 0.2))
    valid, train = step[:n_valid], step[n_valid:]
    vh = {hashlib.md5((x["instruction"] + x["output"]).encode()).hexdigest() for x in valid}
    leak = sum(1 for x in train if hashlib.md5((x["instruction"] + x["output"]).encode()).hexdigest() in vh)
    log.append("切分: 训练 %d / 验证 %d" % (len(train), len(valid)))
    log.append("训练-验证交叉泄漏检查: %d 条 (应为 0)" % leak)
    return train, valid, log

def main():
    train, valid, log = clean(build())
    for name, rows in (("train", train), ("valid", valid)):
        with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    txt = "\n".join(log)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print("\n样例:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2))
    print("\n输出目录:", OUT)

if __name__ == "__main__":
    main()
