#!/usr/bin/env python3
"""
AI/ML 概念可视化图表生成器
产出 12 张高质量图表，覆盖具身智能面试核心概念
用法: python generate_ai_ml_figures.py
输出: data/figures/ai_ml/
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, FancyArrow
from matplotlib.font_manager import FontProperties
import numpy as np
import os, sys

# ── 全局配置 ────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "figures", "ai_ml")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
})

# 中文字体
CN_FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", size=12)
CN_TITLE = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", size=18)
CN_SMALL = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", size=10)

COLORS = {
    "blue":    "#2563EB", "cyan":   "#06B6D4", "purple": "#7C3AED",
    "green":   "#10B981", "orange": "#F59E0B", "red":    "#EF4444",
    "pink":    "#EC4899", "indigo": "#6366F1", "teal":   "#14B8A6",
    "gray":    "#6B7280", "slate":  "#475569", "amber":  "#D97706",
    "dark":    "#1E293B", "light":  "#F1F5F9",
}

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  ✅ {name}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 1: AI → ML → DL → Embodied AI 层级关系图
# ═══════════════════════════════════════════════════════════
def fig1_ai_hierarchy():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    # 同心但偏移的圆角矩形表示包含关系
    layers = [
        (1.0, 1.5, 12.0, 5.0, "人工智能 (AI)", "推理、规划、感知、学习、控制、NLP、CV...", COLORS["blue"], 0.15),
        (1.8, 2.0, 10.4, 4.0, "机器学习 (ML)", "监督学习 · 无监督学习 · 强化学习 · 模仿学习", COLORS["cyan"], 0.18),
        (2.6, 2.5, 8.8, 3.0, "深度学习 (DL)", "CNN · RNN · Transformer · Diffusion Model · GAN", COLORS["purple"], 0.22),
        (3.4, 3.0, 7.2, 2.0, "具身智能\n(Embodied AI)", "VLA · World Model · Diffusion Policy · ACT", COLORS["green"], 0.28),
    ]
    for x, y, w, h, title, subtitle, color, alpha in layers:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                               facecolor=color, edgecolor="white", linewidth=2, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.55, title, ha="center", va="top",
                fontproperties=CN_FONT, fontsize=13, color="white", fontweight="bold")
        ax.text(x + w/2, y + h - 1.2, subtitle, ha="center", va="top",
                fontproperties=CN_SMALL, fontsize=8, color="white", alpha=0.9)

    # 右侧标注关键要素
    annotations = [
        (13.0, 6.2, "🧠 知识表示\n  专家系统\n  搜索算法", COLORS["blue"]),
        (13.0, 4.6, "📊 SVM · XGBoost\n  随机森林 · GMM\n  贝叶斯推断", COLORS["cyan"]),
        (13.0, 3.2, "🔬 ResNet · ViT\n  GPT · BERT\n  Diffusion · NeRF", COLORS["purple"]),
        (13.0, 2.0, "🤖 OpenVLA · π₀\n  RT-2 · Octo\n  Diffusion Policy", COLORS["green"]),
    ]
    for x, y, text, color in annotations:
        ax.text(x, y, text, ha="left", va="top", fontproperties=CN_SMALL, fontsize=7,
                color=color, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                       edgecolor=color, alpha=0.8))

    ax.set_title("人工智能技术栈层级关系", fontproperties=CN_TITLE, pad=20, color=COLORS["dark"])
    save(fig, "01_ai_hierarchy.png")


# ═══════════════════════════════════════════════════════════
# 图 2: 传统控制 vs 端到端控制架构对比
# ═══════════════════════════════════════════════════════════
def fig2_traditional_vs_e2e():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("传统级联控制 vs 端到端 (End-to-End) 控制", fontproperties=CN_TITLE, y=1.01)

    # ── 传统流水线 ──
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("传统「感知-规划-控制」流水线", fontproperties=CN_FONT, color=COLORS["red"], pad=10)

    boxes_traditional = [
        (1, 5.5, "📷 摄像头", "RGB-D 传感器", COLORS["gray"]),
        (3.5, 5.5, "👁️ YOLO/检测", "目标识别\n6D 位姿估计", COLORS["orange"]),
        (6, 5.5, "🗺️ MoveIt/规划", "运动规划\n碰撞检测 · IK", COLORS["amber"]),
        (8.5, 5.5, "⚙️ MCU/控制", "关节控制器\nPID · 力矩环", COLORS["red"]),
    ]
    for x, y, title, sub, color in boxes_traditional:
        rect = FancyBboxPatch((x-0.8, y-0.7), 2.2, 1.6, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor="white", alpha=0.2)
        ax.add_patch(rect)
        ax.text(x, y+0.2, title, ha="center", fontsize=11, fontweight="bold", color=color)
        ax.text(x, y-0.5, sub, ha="center", fontsize=8, color=COLORS["slate"])

    # 箭头
    for x in [2.5, 5.0, 7.5]:
        ax.annotate("", xy=(x+0.6, 5.8), xytext=(x-0.6, 5.8),
                     arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=2))

    # 误差累积标注
    ax.annotate("误差累积 ⇢", xy=(5, 3.5), fontsize=10, color=COLORS["red"], ha="center",
                fontweight="bold")
    ax.text(5, 2.8, "每个环节独立优化\n中间表示丢失信息\n级联误差逐级放大",
            ha="center", fontsize=8, color=COLORS["slate"])

    # ── 端到端 ──
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("端到端 VLA (Vision-Language-Action)", fontproperties=CN_FONT, color=COLORS["green"], pad=10)

    # 输入
    rect_in = FancyBboxPatch((0.3, 4.8), 3, 2.2, boxstyle="round,pad=0.3",
                              facecolor=COLORS["blue"], edgecolor="white", alpha=0.2)
    ax.add_patch(rect_in)
    ax.text(1.8, 6.4, "📥 多模态输入", ha="center", fontweight="bold", fontsize=11)
    ax.text(1.8, 5.8, "• RGB 图像 (128×128)\n• 自然语言指令\n• 关节状态 (9维)", ha="center", fontsize=8, color=COLORS["slate"])

    # 大模型
    rect_model = FancyBboxPatch((3.8, 4.8), 2.4, 2.2, boxstyle="round,pad=0.3",
                                 facecolor=COLORS["purple"], edgecolor="white", alpha=0.3)
    ax.add_patch(rect_model)
    ax.text(5.0, 6.4, "🧠 大模型", ha="center", fontweight="bold", fontsize=11)
    ax.text(5.0, 5.8, "Transformer\nCross-Attention\nDiffusion Head", ha="center", fontsize=8, color=COLORS["slate"])

    # 输出
    rect_out = FancyBboxPatch((6.7, 4.8), 3, 2.2, boxstyle="round,pad=0.3",
                               facecolor=COLORS["green"], edgecolor="white", alpha=0.2)
    ax.add_patch(rect_out)
    ax.text(8.2, 6.4, "📤 动作输出", ha="center", fontweight="bold", fontsize=11)
    ax.text(8.2, 5.8, "• 关节增量 (7维)\n• 夹爪开合 (1维)\n• 未来 32 步动作序列", ha="center", fontsize=8, color=COLORS["slate"])

    ax.annotate("", xy=(3.8, 5.9), xytext=(3.3, 5.9),
                 arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=2))
    ax.annotate("", xy=(6.7, 5.9), xytext=(6.2, 5.9),
                 arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=2))

    # 优势
    ax.text(5, 3.2, "✨ 关键优势", ha="center", fontweight="bold", fontsize=10, color=COLORS["green"])
    ax.text(5, 2.5, "✓ 单一损失函数端到端优化     ✓ 消除中间表示误差\n✓ 隐式学习物理直觉            ✓ 可泛化到未见场景",
            ha="center", fontsize=8, color=COLORS["slate"])

    # 安全沙箱
    rect_safety = FancyBboxPatch((6.8, 1.2), 2.8, 1.5, boxstyle="round,pad=0.2",
                                  facecolor=COLORS["red"], edgecolor="white", alpha=0.15)
    ax.add_patch(rect_safety)
    ax.text(8.2, 2.2, "🛡️ 安全沙箱\n(Safety Sandbox)", ha="center", fontsize=9, fontweight="bold", color=COLORS["red"])
    ax.text(8.2, 1.6, "物理极值校验\n奇异点保护\n碰撞紧急停止", ha="center", fontsize=7, color=COLORS["slate"])

    save(fig, "02_traditional_vs_e2e.png")


# ═══════════════════════════════════════════════════════════
# 图 3: Diffusion Policy 去噪扩散过程
# ═══════════════════════════════════════════════════════════
def fig3_diffusion_process():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Diffusion Policy：从噪声中生成动作序列", fontproperties=CN_TITLE, y=1.02)

    # (a) 前向扩散 (加噪)
    ax = axes[0]
    t_vals = np.linspace(0, 100, 6)
    noise_levels = np.linspace(0, 1, 6)
    for i, (t, nl) in enumerate(zip(t_vals, noise_levels)):
        x = np.linspace(-3, 3, 200)
        signal = np.sin(x * 3) * np.exp(-x**2/4) * (1 - nl)
        noise = np.random.randn(200) * nl * 0.3
        y = signal + noise + i * 0.3
        alpha = 0.3 + 0.5 * i/5
        ax.plot(x, y, color=COLORS["red"], alpha=alpha, lw=1.5,
                label=f"t={int(t)}" if i in [0, 5] else "")
    ax.set_title("前向扩散 q(xₜ|x₀)：逐步加噪", fontproperties=CN_FONT, fontsize=12, color=COLORS["red"])
    ax.set_xlabel("动作维度"); ax.set_ylabel("噪声水平")
    ax.legend(fontsize=8); ax.set_yticks([])

    # (b) 反向去噪 (推理)
    ax = axes[1]
    np.random.seed(42)
    steps = 6
    x = np.linspace(-3, 3, 200)
    target = np.sin(x * 3) * np.exp(-x**2/4)
    current = np.random.randn(200) * 0.8
    for i in range(steps):
        alpha = 0.2 + 0.6 * i/(steps-1)
        current = current * 0.7 + target * 0.3 + np.random.randn(200) * 0.05 * (1 - i/steps)
        ax.plot(x, current + (steps-1-i)*0.2, color=COLORS["green"], alpha=alpha, lw=1.5)
    ax.plot(x, target, "k--", lw=2, label="专家动作")
    ax.set_title("反向去噪 p(xₜ₋₁|xₜ)：逐步还原", fontproperties=CN_FONT, fontsize=12, color=COLORS["green"])
    ax.set_xlabel("动作维度"); ax.legend(fontsize=8); ax.set_yticks([])

    # (c) 多峰分布示意
    ax = axes[2]
    x = np.linspace(-5, 5, 500)
    # 多峰目标分布
    y_target = (0.3 * np.exp(-(x-2)**2/0.5) + 0.4 * np.exp(-(x+1.5)**2/0.8) +
                0.3 * np.exp(-(x+3.5)**2/0.4))
    y_target /= y_target.max()
    # BC 的 MSE 均值预测
    y_bc = 0.9 * np.exp(-(x-(-0.5))**2/1.5)
    y_bc /= y_bc.max()

    ax.fill(x, y_target, alpha=0.4, color=COLORS["green"], label="真实多峰分布")
    ax.fill(x, y_bc, alpha=0.5, color=COLORS["red"], label="BC (MSE 均值)")
    ax.axvline(x=-0.5, color=COLORS["red"], ls="--", lw=1, alpha=0.7)
    # 标注两个峰
    ax.annotate("模式A\n(左绕)", xy=(-1.5, 0.35), fontsize=9, ha="center", color=COLORS["green"],
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=COLORS["green"]))
    ax.annotate("模式B\n(右绕)", xy=(2, 0.25), fontsize=9, ha="center", color=COLORS["green"],
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=COLORS["green"]))
    ax.annotate("MSE均值\n(不可行!)", xy=(-0.5, 0.65), fontsize=9, ha="center", color=COLORS["red"],
                fontweight="bold")
    ax.set_title("行为克隆 (BC) 的致命缺陷", fontproperties=CN_FONT, fontsize=12)
    ax.set_xlabel("动作空间"); ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper left")

    save(fig, "03_diffusion_process.png")


# ═══════════════════════════════════════════════════════════
# 图 4: VLA (Vision-Language-Action) 模型架构
# ═══════════════════════════════════════════════════════════
def fig4_vla_architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8)
    ax.axis("off")

    title = ax.text(8, 7.5, "VLA (Vision-Language-Action) 大模型架构", ha="center",
                    fontproperties=CN_TITLE, fontsize=18, color=COLORS["dark"])

    # ── 左侧: 输入模态 ──
    inputs = [
        (1.5, 5.5, "🖼️ 视觉编码器\n(ViT / ResNet)", "RGB 图像序列\n128×128×3", COLORS["blue"]),
        (1.5, 3.5, "📝 语言编码器\n(LLM Tokenizer)", "自然语言指令\n\"抓取红色方块\"", COLORS["cyan"]),
        (1.5, 1.5, "📐 状态编码器\n(MLP)", "本体感受\n关节角度 × 9", COLORS["indigo"]),
    ]
    for x, y, title, sub, color in inputs:
        rect = FancyBboxPatch((x-1.1, y-0.6), 3.2, 1.3, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor="white", alpha=0.15)
        ax.add_patch(rect)
        ax.text(x, y+0.15, title, ha="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x, y-0.35, sub, ha="center", fontsize=7, color=COLORS["slate"])

    # ── 中间: Transformer ──
    rect_transformer = FancyBboxPatch((5.8, 1.0), 4.4, 6.2, boxstyle="round,pad=0.4",
                                       facecolor=COLORS["purple"], edgecolor="white", alpha=0.12)
    ax.add_patch(rect_transformer)
    ax.text(8.0, 6.7, "🧠 多模态 Transformer", ha="center", fontsize=12, fontweight="bold", color=COLORS["purple"])

    layers = ["Self-Attention", "Cross-Attention\n(视觉↔文本)", "Feed-Forward\nNetwork", "Layer Norm", "× N 层"]
    for i, layer in enumerate(layers):
        y_pos = 5.5 - i * 0.9
        rect = FancyBboxPatch((6.2, y_pos-0.3), 3.6, 0.7, boxstyle="round,pad=0.1",
                               facecolor="white", edgecolor=COLORS["purple"], alpha=0.6)
        ax.add_patch(rect)
        ax.text(8.0, y_pos+0.05, layer, ha="center", fontsize=8, color=COLORS["slate"])

    # ── 右侧: 输出头 ──
    outputs = [
        (13.5, 5.5, "🎯 动作头\n(Diffusion/MLP)", "关节增量 × 7\n夹爪 × 1", COLORS["green"]),
        (13.5, 3.5, "🎯 状态预测\n(可选)", "未来状态\n世界模型", COLORS["teal"]),
        (13.5, 1.5, "🎯 值函数\n(RL辅助)", "Q-value\nCritic Head", COLORS["amber"]),
    ]
    for x, y, title, sub, color in outputs:
        rect = FancyBboxPatch((x-1.1, y-0.6), 3.0, 1.3, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor="white", alpha=0.15)
        ax.add_patch(rect)
        ax.text(x, y+0.15, title, ha="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x, y-0.35, sub, ha="center", fontsize=7, color=COLORS["slate"])

    # 箭头连接
    for inp_y in [5.5, 3.5, 1.5]:
        ax.annotate("", xy=(5.5, inp_y), xytext=(2.5, inp_y),
                     arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))
    for out_y in [5.5, 3.5, 1.5]:
        ax.annotate("", xy=(12.3, out_y), xytext=(10.3, out_y),
                     arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))

    # 底部注释
    ax.text(8, 0.3, "核心思想：将视觉感知、语言理解和动作生成统一到一个 Transformer 中，输出 Token = 动作指令",
            ha="center", fontsize=9, color=COLORS["slate"], style="italic")

    save(fig, "04_vla_architecture.png")


# ═══════════════════════════════════════════════════════════
# 图 5: Sim-to-Real 迁移策略
# ═══════════════════════════════════════════════════════════
def fig5_sim_to_real():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    ax.text(8, 6.5, "Sim-to-Real Transfer：从仿真到真实世界的三大桥梁",
            ha="center", fontproperties=CN_TITLE, fontsize=17, color=COLORS["dark"])

    bridges = [
        (2.5, "🏗️ 动作空间抽象", COLORS["blue"],
         ["输出 EE Delta Pose (末端相对位姿)",
          "而非底层关节力矩",
          "降低动力学差异",
          "仿真↔真机共用同一控制接口"]),
        (8.0, "🎨 数据视角对齐", COLORS["purple"],
         ["RGB 域随机化 (Domain Rand.)",
          "纹理/光照/颜色增强",
          "深度图作为辅助输入",
          "弱化仿真纹理干扰"]),
        (13.5, "🔧 真机少量微调", COLORS["green"],
         ["遥操作录制 50-100 条真机轨迹",
          "混入仿真数据联合微调",
          "Few-shot Co-fine-tuning",
          "弥合最终 Reality Gap"]),
    ]
    for x, title, color, items in bridges:
        rect = FancyBboxPatch((x-2.3, 1.0), 4.6, 4.8, boxstyle="round,pad=0.3",
                               facecolor=color, edgecolor="white", alpha=0.12)
        ax.add_patch(rect)
        ax.text(x, 5.3, title, ha="center", fontsize=13, fontweight="bold", color=color)
        for i, item in enumerate(items):
            ax.text(x, 4.5 - i*0.55, f"• {item}", ha="center", fontsize=9, color=COLORS["slate"])

    # 底部箭头
    ax.annotate("仿真训练 ➤", xy=(5, 0.5), ha="center", fontsize=11, color=COLORS["blue"], fontweight="bold")
    ax.annotate("➤ 域迁移", xy=(8, 0.5), ha="center", fontsize=11, color=COLORS["purple"], fontweight="bold")
    ax.annotate("➤ 真机部署", xy=(13, 0.5), ha="center", fontsize=11, color=COLORS["green"], fontweight="bold")

    save(fig, "05_sim_to_real.png")


# ═══════════════════════════════════════════════════════════
# 图 6: 模仿学习完整 Pipeline
# ═══════════════════════════════════════════════════════════
def fig6_imitation_learning_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(8, 5.6, "模仿学习 (Imitation Learning) 完整流程",
            ha="center", fontproperties=CN_TITLE, fontsize=17, color=COLORS["dark"])

    stages = [
        ("📹 数据采集", "专家遥操作\n录制演示轨迹", COLORS["blue"]),
        ("📦 预处理", "时间戳对齐\n数据清洗增强", COLORS["cyan"]),
        ("🧠 策略训练", "Diffusion Policy\n/ ACT / VQ-BeT", COLORS["purple"]),
        ("🔄 闭环评测", "仿真环境回测\n成功率统计", COLORS["green"]),
        ("🤖 真机部署", "Sim-to-Real\n安全沙箱保护", COLORS["orange"]),
    ]
    for i, (title, sub, color) in enumerate(stages):
        x = 1.8 + i * 2.8
        rect = FancyBboxPatch((x-1.1, 2.0), 2.2, 2.8, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor="white", alpha=0.15)
        ax.add_patch(rect)
        ax.text(x, 4.3, title, ha="center", fontsize=11, fontweight="bold", color=color)
        ax.text(x, 2.7, sub, ha="center", fontsize=8, color=COLORS["slate"])
        # 步骤编号
        ax.text(x, 4.7, f"STEP {i+1}", ha="center", fontsize=8, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8))

        if i < 4:
            ax.annotate("", xy=(x+1.2, 3.4), xytext=(x+0.9, 3.4),
                         arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=2))

    # 底部关键指标
    ax.text(8, 1.2, "⏱️ 关键指标：轨迹成功率 | 动作平滑度 | 泛化能力 | 推理延迟 (<50ms)",
            ha="center", fontsize=9, color=COLORS["slate"], style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"], edgecolor="none"))

    # 每个阶段的数据标注
    data_flow = ["500+ 条轨迹", "HDF5 → Parquet\n+ MP4 视频", "50K steps\n~2.4 hrs", "闭环成功率\n> 85%", "安全第一\n逐步放权"]
    for i, txt in enumerate(data_flow):
        ax.text(1.8 + i*2.8, 1.6, txt, ha="center", fontsize=7, color=COLORS["gray"])

    save(fig, "06_imitation_learning_pipeline.png")


# ═══════════════════════════════════════════════════════════
# 图 7: Transformer Self-Attention 机制
# ═══════════════════════════════════════════════════════════
def fig7_transformer_attention():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("Transformer 核心：Self-Attention 机制", fontproperties=CN_TITLE, y=1.01)

    # (a) Scaled Dot-Product Attention
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Scaled Dot-Product Attention", fontsize=13, fontweight="bold", color=COLORS["purple"])

    # Q, K, V 矩阵
    matrices = [
        (2, 7.5, "Q (Query)", "我要找什么?", COLORS["blue"]),
        (5, 7.5, "K (Key)", "我有什么?", COLORS["cyan"]),
        (8, 7.5, "V (Value)", "实际内容?", COLORS["green"]),
    ]
    for x, y, title, sub, color in matrices:
        rect = FancyBboxPatch((x-1, y-0.5), 2, 1.2, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="white", alpha=0.2)
        ax.add_patch(rect)
        ax.text(x, y+0.15, title, ha="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x, y-0.3, sub, ha="center", fontsize=7, color=COLORS["slate"])

    # MatMul + Scale + SoftMax
    steps = [
        (5, 5.5, "① Q·Kᵀ", "矩阵乘法 → 注意力分数"),
        (5, 4.0, "② /√dₖ", "缩放 (防梯度消失)"),
        (5, 2.5, "③ SoftMax", "归一化 → 注意力权重"),
        (5, 1.0, "④ × V", "加权求和 → 输出"),
    ]
    for x, y, title, sub in steps:
        rect = FancyBboxPatch((x-1.5, y-0.4), 3, 1.0, boxstyle="round,pad=0.1",
                               facecolor="white", edgecolor=COLORS["purple"], alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, y+0.15, title, ha="center", fontsize=10, fontweight="bold", color=COLORS["purple"])
        ax.text(x, y-0.2, sub, ha="center", fontsize=7, color=COLORS["slate"])

    # 连接箭头
    for s, e in [(7.5, 5.5), (5.5, 4.0), (4.0, 2.5), (2.5, 1.0)]:
        ax.annotate("", xy=(5, e+0.3), xytext=(5, s-0.3),
                     arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))

    ax.text(5, 9.0, "Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V",
            ha="center", fontsize=12, fontweight="bold", color=COLORS["dark"],
            fontfamily="monospace")

    # (b) Multi-Head Attention 示意
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Multi-Head Attention (多头注意力)", fontsize=13, fontweight="bold", color=COLORS["indigo"])

    # 输入
    ax.text(5, 9.2, "输入序列 X = [x₁, x₂, ..., xₙ]", ha="center", fontsize=10, fontweight="bold")
    # 多头
    heads = [
        ("Head 1", "全局依赖", COLORS["blue"]),
        ("Head 2", "局部模式", COLORS["cyan"]),
        ("Head 3", "时序关系", COLORS["green"]),
        ("Head 4", "空间结构", COLORS["orange"]),
    ]
    for i, (name, focus, color) in enumerate(heads):
        x = 1.5 + i * 2.2
        rect = FancyBboxPatch((x-0.8, 5.5), 1.6, 2.5, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="white", alpha=0.2)
        ax.add_patch(rect)
        ax.text(x, 7.5, name, ha="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x, 6.5, focus, ha="center", fontsize=8, color=COLORS["slate"])
        # 注意力热力图示意
        heatmap = np.random.rand(5, 5)
        for r in range(5):
            for c in range(5):
                alpha_val = 0.1 + 0.5 * (1 - abs(r-c)/5)
                ax.add_patch(plt.Rectangle((x-0.5+r*0.2, 6.0-c*0.2), 0.18, 0.18,
                                            facecolor=color, alpha=alpha_val))

    # Concat + Linear
    rect = FancyBboxPatch((3, 2.0), 4, 1.2, boxstyle="round,pad=0.2",
                           facecolor=COLORS["purple"], edgecolor="white", alpha=0.2)
    ax.add_patch(rect)
    ax.text(5, 2.8, "Concat + Linear Projection", ha="center", fontsize=11, fontweight="bold", color=COLORS["purple"])

    # 连线
    for i in range(4):
        ax.annotate("", xy=(5, 2.0), xytext=(1.5+i*2.2, 5.5),
                     arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1, alpha=0.5))

    ax.text(5, 1.2, "核心优势：不同 Head 关注不同方面的依赖关系\n并行计算 → 比 RNN 快 10-100×",
            ha="center", fontsize=9, color=COLORS["slate"])

    save(fig, "07_transformer_attention.png")


# ═══════════════════════════════════════════════════════════
# 图 8: World Model 概念图
# ═══════════════════════════════════════════════════════════
def fig8_world_model():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.6, "World Model（世界模型）：在脑中预演未来",
            ha="center", fontproperties=CN_TITLE, fontsize=17, color=COLORS["dark"])

    # 真实世界
    ax.text(2, 6.5, "🌍 真实物理世界", ha="center", fontsize=12, fontweight="bold", color=COLORS["blue"])
    rect = FancyBboxPatch((0.3, 4.2), 3.4, 2.0, boxstyle="round,pad=0.3",
                           facecolor=COLORS["blue"], edgecolor="white", alpha=0.1)
    ax.add_patch(rect)
    ax.text(2, 5.2, "sₜ (当前状态)\n→ aₜ (执行动作)\n→ sₜ₊₁ (下一状态)", ha="center", fontsize=9, color=COLORS["slate"])

    # 世界模型
    ax.text(7, 6.5, "🧠 世界模型 (内部仿真)", ha="center", fontsize=12, fontweight="bold", color=COLORS["purple"])
    rect_wm = FancyBboxPatch((4.2, 4.2), 5.6, 2.0, boxstyle="round,pad=0.3",
                              facecolor=COLORS["purple"], edgecolor="white", alpha=0.1)
    ax.add_patch(rect_wm)
    ax.text(7, 5.2, "f(sₜ, aₜ) → ŝₜ₊₁ (预测下一帧)\nr(sₜ, aₜ) → r̂ (预测奖励)\n在脑中试错，无需真实交互",
            ha="center", fontsize=9, color=COLORS["slate"])

    # 策略
    ax.text(12, 6.5, "🎯 策略 (Actor)", ha="center", fontsize=12, fontweight="bold", color=COLORS["green"])
    rect_pol = FancyBboxPatch((10.3, 4.2), 3.4, 2.0, boxstyle="round,pad=0.3",
                               facecolor=COLORS["green"], edgecolor="white", alpha=0.1)
    ax.add_patch(rect_pol)
    ax.text(12, 5.2, "π(sₜ) → aₜ\n在世界模型中\n搜索最优动作序列", ha="center", fontsize=9, color=COLORS["slate"])

    # 循环箭头
    ax.annotate("", xy=(3.8, 5.2), xytext=(10.2, 5.2),
                 arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=2, connectionstyle="arc3,rad=0.3"))
    ax.annotate("", xy=(3.8, 4.8), xytext=(10.2, 4.8),
                 arrowprops=dict(arrowstyle="<-", color=COLORS["green"], lw=2, connectionstyle="arc3,rad=-0.3"))

    # 底部：三大能力
    capabilities = [
        (3, "🎬 视觉预测", "预测下一帧 RGB 图像\n捕捉物体运动、碰撞\n隐式学习物理规律", COLORS["cyan"]),
        (7, "🔄 反事实推理", "\"如果我从左边绕呢?\"\n比较不同动作序列的\n未来结果", COLORS["orange"]),
        (11, "⚡ 高效探索", "在想象中尝试 1000 次\n只在实际中执行最优的\n极大提升样本效率", COLORS["green"]),
    ]
    for x, title, sub, color in capabilities:
        rect = FancyBboxPatch((x-1.5, 0.5), 3.0, 3.0, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor="white", alpha=0.1)
        ax.add_patch(rect)
        ax.text(x, 3.0, title, ha="center", fontsize=11, fontweight="bold", color=color)
        ax.text(x, 1.2, sub, ha="center", fontsize=8, color=COLORS["slate"])

    save(fig, "08_world_model.png")


# ═══════════════════════════════════════════════════════════
# 图 9: 具身智能完整技术栈
# ═══════════════════════════════════════════════════════════
def fig9_embodied_stack():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.6, "具身智能完整技术栈 (Embodied AI Full Stack)",
            ha="center", fontproperties=CN_TITLE, fontsize=17, color=COLORS["dark"])

    layers = [
        (7.5, "🧠 高层推理与规划", "LLM/VLM 任务分解 · 常识推理 · 指令理解\nGPT-4V · Gemini · LLaVA · Qwen-VL", COLORS["blue"], 0.2),
        (6.2, "🎯 中层次策略生成", "VLA · Diffusion Policy · ACT · π₀\n端到端 visuomotor policy · 动作序列生成", COLORS["purple"], 0.22),
        (4.9, "🖼️ 感知与表征", "6D 位姿估计 · 目标检测 · 语义分割 · 深度估计\nDINO · SAM · Foundation Models", COLORS["cyan"], 0.18),
        (3.6, "⚙️ 运动规划与控制", "IK/FK · 轨迹优化 · 力控/阻抗控制\nMoveIt · CuRobo · ROS2 Control", COLORS["green"], 0.15),
        (2.3, "🔌 硬件与驱动层", "MCU · 电机驱动 · CAN/EtherCAT 通信\n传感器融合 · 实时操作系统", COLORS["gray"], 0.12),
    ]
    for y, title, sub, color, alpha in layers:
        rect = FancyBboxPatch((0.5, y-0.5), 13, 1.1, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor="white", alpha=alpha)
        ax.add_patch(rect)
        ax.text(7, y+0.2, title, ha="center", fontsize=13, fontweight="bold", color=color)
        ax.text(7, y-0.3, sub, ha="center", fontsize=7.5, color=COLORS["slate"])

    # 右侧：你的定位
    ax.annotate("⭐\n你的\n优势区", xy=(13.2, 3.0), fontsize=9, ha="center", color=COLORS["green"],
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5", edgecolor=COLORS["green"]))
    ax.annotate("🎯\n面试\n目标区", xy=(13.2, 5.8), fontsize=9, ha="center", color=COLORS["purple"],
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E8FF", edgecolor=COLORS["purple"]))

    # 连接箭头
    ax.annotate("", xy=(13.0, 3.8), xytext=(13.0, 5.0),
                 arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=2))
    ax.text(13.5, 4.4, "桥梁", fontsize=8, color=COLORS["purple"], rotation=0)

    save(fig, "09_embodied_stack.png")


# ═══════════════════════════════════════════════════════════
# 图 10: Diffusion Policy 模型详细结构
# ═══════════════════════════════════════════════════════════
def fig10_diffusion_policy_arch():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(8, 7.6, "Diffusion Policy 模型详细架构",
            ha="center", fontproperties=CN_TITLE, fontsize=17, color=COLORS["dark"])

    # ── 输入 ──
    inputs = [
        (1.8, 6.0, "📷 视觉观测 (2帧)", "ResNet-18\n→ 512-d feature", "(2, 3, 96, 96)", COLORS["blue"]),
        (1.8, 3.5, "📐 关节状态 (2帧)", "MLP\n→ 256-d feature", "(2, 9)", COLORS["cyan"]),
        (1.8, 1.0, "⏱️ 扩散时间步 t", "Sinusoidal\nEmbedding → 128-d", "scalar", COLORS["purple"]),
    ]
    for x, y, title, sub, shape, color in inputs:
        rect = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1.3, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor="white", alpha=0.15)
        ax.add_patch(rect)
        ax.text(x, y+0.2, title, ha="center", fontsize=9, fontweight="bold", color=color)
        ax.text(x, y-0.2, sub, ha="center", fontsize=7, color=COLORS["slate"])
        ax.text(x, y-0.45, shape, ha="center", fontsize=6, color=COLORS["gray"])

    # ── FiLM Conditioned UNet ──
    rect = FancyBboxPatch((5.5, 0.5), 5.0, 7.0, boxstyle="round,pad=0.3",
                           facecolor=COLORS["purple"], edgecolor="white", alpha=0.08)
    ax.add_patch(rect)
    ax.text(8, 7.1, "🔄 FiLM-Conditioned 1D UNet", ha="center", fontsize=12, fontweight="bold", color=COLORS["purple"])

    # UNet 内部结构
    unet_layers = [
        ("噪声动作 zₜ\n(64, 8)", "↓"),
        ("Conv1D 512\n+ FiLM", "↓ Down"),
        ("Conv1D 1024\n+ FiLM", "↓ Down"),
        ("Conv1D 2048\n+ FiLM", "— Bottleneck"),
        ("Conv1D 1024\n+ FiLM", "↑ Up"),
        ("Conv1D 512\n+ FiLM", "↑"),
        ("去噪动作 ẑ₀\n(64, 8)", "→"),
    ]
    for i, (label, direction) in enumerate(unet_layers):
        y_pos = 6.3 - i * 0.75
        rect = FancyBboxPatch((6.2, y_pos-0.25), 3.6, 0.6, boxstyle="round,pad=0.08",
                               facecolor="white", edgecolor=COLORS["purple"], alpha=0.5)
        ax.add_patch(rect)
        ax.text(8, y_pos+0.05, label, ha="center", fontsize=7.5, color=COLORS["slate"])
        ax.text(10.3, y_pos+0.05, direction, ha="center", fontsize=7, color=COLORS["gray"])

    # 跳跃连接示意
    for i in range(3):
        ax.annotate("", xy=(6.0, 6.3-i*0.75), xytext=(5.0, 1.8+i*0.75),
                     arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=0.8, alpha=0.4,
                                     connectionstyle="arc3,rad=0.5"))

    # ── 输出 ──
    ax.text(13.5, 4.5, "→ 🎯 动作序列", ha="center", fontsize=11, fontweight="bold", color=COLORS["green"])
    ax.text(13.5, 3.8, "H=64 步\n执行前 N=32 步", ha="center", fontsize=9, color=COLORS["slate"])
    ax.annotate("", xy=(13.5, 2.8), xytext=(10.5, 3.5),
                 arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=1.5))

    # 底部说明
    ax.text(8, 0.2, "FiLM = Feature-wise Linear Modulation：用观测特征调节 UNet 每一层的特征图",
            ha="center", fontsize=8, color=COLORS["gray"], style="italic")

    save(fig, "10_diffusion_policy_arch.png")


# ═══════════════════════════════════════════════════════════
# 图 11: SE(3) 坐标变换 — 手眼标定
# ═══════════════════════════════════════════════════════════
def fig11_se3_transform():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("SE(3) 坐标变换：手眼标定的数学基础", fontproperties=CN_TITLE, y=1.02)

    # (a) 坐标系关系图
    ax = axes[0]
    ax.set_xlim(-2, 10); ax.set_ylim(-2, 8)
    ax.axis("off")
    ax.set_title("坐标系变换链", fontproperties=CN_FONT, fontsize=13, color=COLORS["blue"])

    # 世界坐标系
    ax.arrow(0, 0, 1.5, 0, head_width=0.15, head_length=0.2, fc=COLORS["dark"], lw=2)
    ax.arrow(0, 0, 0, 1.5, head_width=0.15, head_length=0.2, fc=COLORS["dark"], lw=2)
    ax.text(1.8, -0.2, "World {W}", fontsize=10, fontweight="bold", color=COLORS["dark"])

    # 机械臂基座
    ax.text(0, 4, "baseTcam", fontsize=8, color=COLORS["gray"])
    ax.annotate("", xy=(3.5, 4), xytext=(1.3, 0.8),
                 arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=2, connectionstyle="arc3,rad=-0.3"))
    ax.text(4.5, 5.5, "Base {B}", fontsize=10, fontweight="bold", color=COLORS["blue"])
    ax.plot(4, 5, "s", color=COLORS["blue"], markersize=10)

    # 相机坐标系
    ax.text(8, 1.5, "camTobj", fontsize=8, color=COLORS["gray"])
    ax.annotate("", xy=(8.5, 3.8), xytext=(5, 4.5),
                 arrowprops=dict(arrowstyle="->", color=COLORS["cyan"], lw=1.5, connectionstyle="arc3,rad=0.2"))
    ax.text(9, 4.2, "Camera {C}", fontsize=10, fontweight="bold", color=COLORS["cyan"])
    ax.plot(9, 4, "o", color=COLORS["cyan"], markersize=10)

    # 物体
    ax.annotate("", xy=(12, 4), xytext=(9.5, 1.8),
                 arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))
    ax.text(12.2, 3.5, "Object {O}", fontsize=10, fontweight="bold", color=COLORS["green"])
    ax.plot(12, 3, "*", color=COLORS["green"], markersize=15)

    # 手眼标定方程
    ax.text(5, -1, "手眼标定方程:  baseTobj = baseTcam · camTobj", fontsize=11,
            fontweight="bold", color=COLORS["dark"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7", edgecolor=COLORS["amber"]))

    # (b) SE(3) 变换矩阵
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("SE(3) = SO(3) × ℝ³ 刚体变换", fontproperties=CN_FONT, fontsize=13, color=COLORS["purple"])

    # 变换矩阵
    ax.text(5, 7.5, "齐次变换矩阵 T ∈ SE(3)", ha="center", fontsize=12, fontweight="bold", color=COLORS["dark"])

    # 矩阵表示
    matrix_str = (
        "┌                           ┐\n"
        "│  R₃ₓ₃      │  t₃ₓ₁   │\n"
        "│  (旋转)    │  (平移)  │\n"
        "├───────────┼─────────│\n"
        "│  0  0  0   │    1     │\n"
        "└                           ┘"
    )
    ax.text(5, 5.8, matrix_str, ha="center", fontsize=10, fontfamily="monospace", color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=COLORS["purple"], alpha=0.8))

    # SO(3) 旋转表示
    rotations = [
        ("欧拉角 (RPY)", "3 参数\n万向锁问题", COLORS["red"]),
        ("旋转矩阵 R₃ₓ₃", "9 参数\n冗余表示", COLORS["orange"]),
        ("四元数 q", "4 参数\n无万向锁", COLORS["green"]),
        ("轴角 (Axis-Angle)", "4 参数\n几何直观", COLORS["blue"]),
    ]
    for i, (title, sub, color) in enumerate(rotations):
        x = 1.4 + i * 2.2
        rect = FancyBboxPatch((x-0.8, 2.5), 1.7, 2.5, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="white", alpha=0.15)
        ax.add_patch(rect)
        ax.text(x, 4.5, title, ha="center", fontsize=9, fontweight="bold", color=color)
        ax.text(x, 3.2, sub, ha="center", fontsize=8, color=COLORS["slate"])

    # 底部应用
    ax.text(5, 1.5, "在具身智能中：相机内参 (K) + 外参 (T) → 将像素映射到机械臂基座坐标系",
            ha="center", fontsize=9, color=COLORS["slate"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"], edgecolor="none"))

    save(fig, "11_se3_transform.png")


# ═══════════════════════════════════════════════════════════
# 图 12: 多传感器时间同步
# ═══════════════════════════════════════════════════════════
def fig12_sensor_sync():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(8, 6.6, "多模态数据采集：时间同步是最大挑战",
            ha="center", fontproperties=CN_TITLE, fontsize=17, color=COLORS["dark"])

    # 三条时间线
    sensors = [
        ("📷 相机 (30Hz)", 30, 0.033, COLORS["blue"]),
        ("📐 关节编码器 (100Hz)", 100, 0.010, COLORS["cyan"]),
        ("🎮 控制指令 (50Hz)", 50, 0.020, COLORS["green"]),
    ]
    for i, (name, freq, period, color) in enumerate(sensors):
        y = 5.2 - i * 1.5
        ax.text(0.3, y+0.4, name, fontsize=10, fontweight="bold", color=color)
        # 时间线
        ax.axhline(y=y, xmin=0.25, xmax=0.95, color=color, lw=2, alpha=0.5)
        # 采样点
        for t in np.arange(4, 14, period):
            ax.plot(t, y, "o", color=color, markersize=8 - i*2, alpha=0.8)
            ax.axvline(x=t, ymin=(y-0.3)/7, ymax=(y+0.3)/7, color=color, alpha=0.15, lw=0.5)

    # 问题标注
    problems = [
        (2.5, 0.5, "❌ 时间戳未对齐\n\"看未来的图做过去的动作\"\n→ 因果倒置!", COLORS["red"]),
        (8.0, 0.5, "⚠️ 硬件同步方案\nPTP (IEEE 1588)\nGPS 时钟同步", COLORS["amber"]),
        (13.5, 0.5, "✅ 软件插值方案\n线性插值 / 最近邻\n事后时间戳对齐", COLORS["green"]),
    ]
    for x, y, text, color in problems:
        rect = FancyBboxPatch((x-2.2, y-0.2), 4.4, 2.2, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor="white", alpha=0.1)
        ax.add_patch(rect)
        ax.text(x, y+1.2, text, ha="center", fontsize=9, color=COLORS["slate"])

    save(fig, "12_sensor_sync.png")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  AI/ML 概念可视化图表生成器")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    generators = [
        fig1_ai_hierarchy,
        fig2_traditional_vs_e2e,
        fig3_diffusion_process,
        fig4_vla_architecture,
        fig5_sim_to_real,
        fig6_imitation_learning_pipeline,
        fig7_transformer_attention,
        fig8_world_model,
        fig9_embodied_stack,
        fig10_diffusion_policy_arch,
        fig11_se3_transform,
        fig12_sensor_sync,
    ]

    for fn in generators:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  ✅ 完成！共生成了 {len(generators)} 张图表")
    print(f"  📂 输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
