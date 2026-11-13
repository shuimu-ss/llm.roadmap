# -*- coding: utf-8 -*-
"""生成可编辑的 Word 简历（一页 A4），内容与 HTML 版一致。
运行：c:/Users/admin/.workbuddy/binaries/python/envs/default/Scripts/python.exe gen_resume_docx.py
"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.abspath(__file__))
# 个人信息用占位符，避免真实身份进入公开仓库；填好后再运行
NAME = "你的姓名"
STUDENT_ID = "（请填写学号）"
SCHOOL = "（学校 · 学院 · 专业 · 年级）"
CITY = "（意向城市）"
OUT = os.path.join(BASE, "大模型算法实习生_简历.docx")

BLUE = RGBColor(0x1F, 0x5F, 0xD6)
FONT = "微软雅黑"


def set_font(run, size=10.5, bold=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), FONT)
    rFonts.set(qn("w:ascii"), FONT)
    rFonts.set(qn("w:hAnsi"), FONT)


def add_bottom_border(p, color="1F5FD6", sz="8"):
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), sz)
    bottom.set(qn("w:space"), "3")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    pPr.append(pbdr)


def para(doc, text="", size=9.5, bold=False, color=None, space_after=1,
         align=None, space_before=0, line=1.1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = line
    if align is not None:
        p.alignment = align
    if text:
        set_font(p.add_run(text), size, bold, color)
    return p


def rich(doc, parts, size=9.5, space_after=1, indent=None, line=1.1, space_before=0):
    """parts: list of (text, bold)"""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = line
    if indent is not None:
        p.paragraph_format.left_indent = Cm(indent)
    for t, b in parts:
        set_font(p.add_run(t), size, b)
    return p


def heading(doc, text):
    p = para(doc, text, size=11, bold=True, color=BLUE, space_before=5, space_after=3)
    add_bottom_border(p)
    return p


def bullet(doc, parts, size=9):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.left_indent = Cm(0.55)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    p.paragraph_format.line_spacing = 1.1
    set_font(p.add_run("• "), size)
    for t, b in parts:
        set_font(p.add_run(t), size, b)
    return p


def main():
    d = Document()
    sec = d.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(0.95)
    sec.bottom_margin = Cm(0.95)
    sec.left_margin = Cm(1.45)
    sec.right_margin = Cm(1.45)

    style = d.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(9.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    # ---------- 抬头 ----------
    p = para(d, NAME, size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_after=2)
    add_bottom_border(p, sz="12")
    para(d, "求职意向：大模型算法实习生 / AI 应用开发实习生", size=10, bold=True,
         color=BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(d, SCHOOL + "　|　学号 " + STUDENT_ID + "　|　意向城市：" + CITY,
         size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
    para(d, "电话：____________　邮箱：____________　GitHub：____________",
         size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)

    # ---------- 教育背景 ----------
    heading(d, "教育背景")
    rich(d, [("（学校　专业　本科）", True),
             ("　　　　　　　　　　2024.09 – 2028.06（预计）", False)], size=10.5)
    para(d, "相关课程：嵌入式系统软件开发、通信电路与系统、概率论、数字信号处理", size=9.5)

    # ---------- 项目经历 ----------
    heading(d, "项目经历")
    rich(d, [("QLoRA 领域指令微调与效果评估", True),
             ("　　　　　　　　　　　　　　2026.11 · 个人项目", False)], size=9.5, space_after=1)
    para(d, "技术栈：PyTorch / HuggingFace Transformers / PEFT / bitsandbytes / 数据清洗",
         size=8.5, color=RGBColor(0x4A, 0x52, 0x62), space_after=2)
    bullet(d, [("基于 Transformers / PEFT，在 ", False), ("RTX 4060 Laptop 8GB", True),
               (" 单卡上使用 ", False), ("QLoRA（4-bit NF4 + 双重量化）", True),
               (" 对 Qwen2.5-0.5B-Instruct 做领域指令微调，独立跑通数据准备、训练、保存与评估全流程。", False)])
    bullet(d, [("完成指令数据清洗：", False), ("去噪、精确去重、长度过滤、先切分再校验训练/验证集泄漏", True),
               ("，117 条原始数据得到 100 条有效样本（训练 80 / 验证 20，交叉泄漏 0）。", False)])
    bullet(d, [("使用 chat template 对齐训练与推理格式，将 prompt 部分 label 置为 -100 只对回答计算损失；通过 bf16、梯度检查点与梯度累积把 ", False),
               ("显存峰值控制在 2.47GB", True), ("，仅训练 ", False), ("2.716%", True), (" 参数。", False)])
    bullet(d, [("3 个 epoch、30 step 内 loss 由 ", False), ("4.8958 降至 0.1257", True),
               ("（训练耗时 39.5 秒）；在 20 条留出集上做同 prompt 生成对比，", False),
               ("关键词命中率由 20% 提升至 65%（+45pp）", True),
               ("，平均回答长度由 137 字降至 47 字。", False)])
    p = para(d, "产出：完整可复现仓库（含清洗/微调/评估脚本、训练日志、前后对比样例），所有数字可核查。",
             size=8.5, color=RGBColor(0x4A, 0x52, 0x62), space_after=2)
    p.paragraph_format.left_indent = Cm(0.55)

    rich(d, [("OpenHarmony 嵌入式开发与设备端通信实验", True),
             ("　　　　　　　课程项目 · 2026.09", False)], size=10.5, space_before=2, space_after=1)
    para(d, "技术栈：OpenHarmony / Hi3861 / HiBurn / MQTT / DevEco Studio / ArkUI",
         size=8.5, color=RGBColor(0x4A, 0x52, 0x62), space_after=2)
    bullet(d, [("使用 hb set / hb build -f 构建 wifiiot_hispark_pegasus 工程，经 HiBurn（921600 波特率）烧录 Hi3861 开发板并通过串口验证运行状态。", False)])
    bullet(d, [("实现 Wi-Fi 热点接入与 MQTT 客户端通信（broker.emqx.io:1883、主题 MQTT_PC），完成消息订阅与收发验证。", False)])
    bullet(d, [("基于 DevEco Studio 与 ArkUI 实现多页面导航（router.pushUrl / back）及账号密码输入校验交互。", False)])

    # ---------- 专业技能 ----------
    heading(d, "专业技能")
    for k, v in [
        ("能讲清细节", "Python、PyTorch 张量与自动求导、完整训练闭环、DataLoader、模型保存与推理；HuggingFace Transformers / PEFT；LoRA 与 QLoRA 原理及显存构成；指令数据清洗与防泄漏；留出集评估设计"),
        ("理解原理", "Transformer 架构、多头注意力 QKV、缩放点积与 √d_k、正弦/RoPE 位置编码、残差与 LayerNorm、Decoder-only 结构；KV Cache 与 vLLM PagedAttention / 连续批处理；bf16 与梯度检查点"),
        ("了解、未实操", "DeepSpeed / ZeRO 分布式训练、TensorRT-LLM、CLIP / LLaVA 多模态、RLHF / DPO 对齐"),
        ("工具", "Git、Linux 基础、DevEco Studio、HiBurn、串口调试"),
    ]:
        rich(d, [(k + "　", True), (v, False)], size=9, space_after=1)

    # ---------- 其他 ----------
    heading(d, "其他")
    bullet(d, [("本科阶段有鸿蒙应用开发实践；具备嵌入式端与上位机联调经验。", False)], size=9.5)

    d.save(OUT)
    print("saved:", OUT)
    print("size:", os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
