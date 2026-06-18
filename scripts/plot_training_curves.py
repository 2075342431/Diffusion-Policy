#!/usr/bin/env python3
"""
训练曲线可视化工具 — 仅基于训练日志产出
用法:
  python plot_training_curves.py --input logs/training_*.log      # 离线
  python plot_training_curves.py --watch "logs/training_*.log"    # 实时监控
"""

import argparse, os, sys, re, time, glob as _glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

# ── 中文字体（宋体，完整 ASCII+CJK） ──
_FONT_PATH = "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    _name = fm.FontProperties(fname=_FONT_PATH).get_name()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.25,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
})
C = ["#2563EB","#EF4444","#10B981","#F59E0B","#7C3AED","#EC4899","#0891B2","#D97706"]


def parse_log(path: str) -> dict:
    """解析 lerobot-train 日志"""
    pat = re.compile(
        r"step:(?P<step>[\d.]+[KM]?)\s+smpl:(?P<smpl>[\d.]+[KM]?)\s+ep:(?P<ep>[\d.]+[KM]?)\s+"
        r"epch:(?P<epch>[\d.]+)\s+loss:(?P<loss>[\d.]+)\s+grdn:(?P<grdn>[\d.]+)\s+"
        r"lr:(?P<lr>[\de.\-]+)\s+updt_s:(?P<updt>[\d.]+)\s+data_s:(?P<data>[\d.]+)\s+"
        r"smp/s:(?P<smp>[\d.]+)\s+mem_gb:(?P<mem>[\d.]+)"
    )

    def _parse_num(s):
        """解析带 K/M 后缀的数字, e.g. '50K' -> 50000, '1.5M' -> 1500000"""
        s = s.strip()
        if s.endswith("K"): return int(float(s[:-1]) * 1000)
        if s.endswith("M"): return int(float(s[:-1]) * 1_000_000)
        return int(float(s))

    m = {"step":[],"loss":[],"grdn":[],"lr":[],"mem":[],"smp":[],"updt":[],"data":[],"ep":[],"epch":[],"smpl":[]}
    with open(path,"r",errors="ignore") as f:
        for line in f:
            d = pat.search(line)
            if d:
                g = d.groupdict()
                m["step"].append(_parse_num(g["step"]))
                m["loss"].append(float(g["loss"]))
                m["grdn"].append(float(g["grdn"]))
                m["lr"].append(float(g["lr"]))
                m["mem"].append(float(g["mem"]))
                m["smp"].append(float(g["smp"]))
                m["updt"].append(float(g["updt"]))
                m["data"].append(float(g["data"]))
                m["ep"].append(_parse_num(g["ep"]))
                m["epch"].append(float(g["epch"]))
                m["smpl"].append(_parse_num(g["smpl"]))
    return m


def smooth(y, w):
    w = min(w, len(y))
    return np.convolve(y, np.ones(w)/w, mode="valid"), w


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {os.path.basename(path)}")


def gen_all(m: dict, out: str, tag: str = ""):
    if not m["step"]: return
    s = np.array(m["step"])
    n = len(s)
    t = f"_{tag}" if tag else ""
    os.makedirs(out, exist_ok=True)
    w = max(1, n // 40)  # 平滑窗口

    # ━━━ 图1: 6合1 仪表盘 ━━━
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"Diffusion Policy 训练仪表盘 — 已训练 {s[-1]:,} 步", fontsize=15, fontweight="bold", y=0.99)
    # A1 Loss
    ax = axs[0,0]; ax.plot(s, m["loss"], color=C[0], lw=0.7, alpha=0.5)
    if n > w:
        sy, _ = smooth(m["loss"], w); ax.plot(s[w-1:], sy, color=C[3], lw=2, label=f"平滑(w={w})")
    ax.set_title("Loss"); ax.set_ylabel("Loss"); ax.legend(fontsize=7)
    ax.text(0.98,0.95,f"当前:{m['loss'][-1]:.4f}\n最低:{min(m['loss']):.4f}",transform=ax.transAxes,ha="right",va="top",fontsize=9,bbox=dict(boxstyle="round",fc="#F1F5F9"))
    # A2 Grad Norm
    ax = axs[0,1]; ax.plot(s, m["grdn"], color=C[1], lw=1, alpha=0.7)
    ax.set_title("梯度范数"); ax.set_ylabel("||∇||")
    # A3 LR
    ax = axs[1,0]; ax.plot(s, m["lr"], color=C[2], lw=1.5)
    ax.set_title("学习率 (log)"); ax.set_ylabel("LR"); ax.set_yscale("log")
    # A4 GPU
    ax = axs[1,1]; ax.plot(s, m["mem"], color=C[4], lw=1.5)
    ax.fill_between(s, 0, m["mem"], color=C[4], alpha=0.1)
    ax.axhline(8.0, color=C[1], ls="--", lw=1, alpha=0.5, label="8GB上限")
    ax.set_title("GPU 显存"); ax.set_ylabel("GB"); ax.legend(fontsize=7)
    # A5 Throughput
    ax = axs[2,0]; ax.plot(s, m["smp"], color=C[5], lw=1, alpha=0.7)
    ax.set_title("吞吐量"); ax.set_ylabel("Samples/s"); ax.set_xlabel("Step")
    # A6 Epoch
    ax = axs[2,1]; ax.plot(s, m["epch"], color=C[2], lw=1.5)
    ax.set_title("训练进度"); ax.set_ylabel("Epoch"); ax.set_xlabel("Step")
    ax.text(0.98,0.95,f"{m['epch'][-1]:.2f} epochs",transform=ax.transAxes,ha="right",va="top",fontsize=9,bbox=dict(boxstyle="round",fc="#F1F5F9"))
    plt.tight_layout(); save(fig, f"{out}/01_dashboard{t}.png")

    # ━━━ 图2: Loss 多窗口分析 ━━━
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Loss 收敛分析", fontsize=14, fontweight="bold")
    ax = axs[0]
    ax.plot(s, m["loss"], color=C[0], lw=0.4, alpha=0.35, label="原始")
    for wi, ls, lb in [(10,"-","MA-10"),(50,"--","MA-50"),(100,"-.","MA-100")]:
        if n > wi:
            sy, _ = smooth(m["loss"], wi); ax.plot(s[wi-1:], sy, lw=1.8, ls=ls, label=lb)
    ax.set_title("多窗口平滑"); ax.set_ylabel("Loss"); ax.set_xlabel("Step"); ax.legend()
    ax = axs[1]
    tail = max(2, n//5)
    ax.plot(s[-tail:], m["loss"][-tail:], "o-", color=C[0], ms=3, lw=1, alpha=0.8)
    if n > w:
        sy, _ = smooth(m["loss"], w); ax.plot(s[w-1:][-tail:], sy[-tail:], color=C[3], lw=2, label=f"MA-{w}")
    ax.set_title(f"最后 {tail} 步 (放大)"); ax.set_ylabel("Loss"); ax.set_xlabel("Step"); ax.legend()
    plt.tight_layout(); save(fig, f"{out}/02_loss_analysis{t}.png")

    # ━━━ 图3: 梯度流分析 ━━━
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("梯度与优化动态", fontsize=14, fontweight="bold")
    ax = axs[0]; ax.plot(s, m["grdn"], color=C[1], lw=1, alpha=0.7)
    if n > w:
        sy, _ = smooth(m["grdn"], w); ax.plot(s[w-1:], sy, color=C[3], lw=2)
    ax.set_title("梯度范数趋势"); ax.set_xlabel("Step"); ax.set_ylabel("||∇||")
    ax = axs[1]; ax.hist(m["grdn"], bins=40, color=C[1], alpha=0.6, edgecolor="white")
    ax.axvline(np.mean(m["grdn"]), color=C[0], ls="--", lw=2, label=f'均值:{np.mean(m["grdn"]):.2f}')
    ax.set_title("梯度分布"); ax.set_xlabel("||∇||"); ax.set_ylabel("频次"); ax.legend(fontsize=8)
    ax = axs[2]; ax.plot(s, m["lr"], color=C[2], lw=1.5)
    ax.set_title("学习率调度 (Cosine)"); ax.set_xlabel("Step"); ax.set_ylabel("LR"); ax.set_yscale("log")
    plt.tight_layout(); save(fig, f"{out}/03_gradient_dynamics{t}.png")

    # ━━━ 图4: GPU资源 ━━━
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("GPU 资源利用率", fontsize=14, fontweight="bold")
    ax = axs[0]; ax.plot(s, m["mem"], color=C[4], lw=1.5)
    ax.fill_between(s, 0, m["mem"], alpha=0.1, color=C[4])
    ax.axhline(8, color=C[1], ls="--", lw=1); ax.axhline(np.mean(m["mem"]), color=C[5], ls=":", lw=1, label=f'均值:{np.mean(m["mem"]):.2f}GB')
    ax.set_title("显存占用"); ax.set_ylabel("GB"); ax.legend(fontsize=7)
    ax = axs[1]; ax.hist(m["smp"], bins=30, color=C[0], alpha=0.6, edgecolor="white")
    ax.axvline(np.mean(m["smp"]), color=C[1], ls="--", lw=2, label=f'均值:{np.mean(m["smp"]):.0f}smp/s')
    ax.set_title("吞吐量分布"); ax.set_xlabel("Samples/s"); ax.set_ylabel("频次"); ax.legend(fontsize=8)
    ax = axs[2]
    umean = np.mean(m["updt"])*1000; dmean = np.mean(m["data"])*1000
    ax.barh(["数据加载","模型更新"], [dmean, umean], color=[C[2],C[4]], edgecolor="white", height=0.5)
    ax.set_title("每步耗时分解"); ax.set_xlabel("ms")
    for i, v in enumerate([dmean, umean]): ax.text(v+0.5, i, f"{v:.1f} ms", va="center", fontweight="bold")
    plt.tight_layout(); save(fig, f"{out}/04_gpu_resources{t}.png")

    # ━━━ 图5: 训练统计 ━━━
    fig, ax = plt.subplots(figsize=(14, 5.5)); ax.axis("off")
    ax.set_title("训练运行摘要", fontsize=16, fontweight="bold", y=1.02)
    elapsed = (s[-1]-s[0]) * np.mean(m["updt"]) / 60
    eta = max(0, (50000-s[-1]) * np.mean(m["updt"]) / 60)
    rows = [
        ("训练步数",f"{s[-1]:,} / 50,000 ({100*s[-1]/50000:.1f}%)"),
        ("已用时间(估)",f"{elapsed:.0f} 分钟"),
        ("预估剩余",f"{eta:.0f} 分钟" if eta>0 else "已完成!", "🎉" if eta==0 else "⏳"),
        ("Loss 变化",f"{m['loss'][0]:.4f} → {m['loss'][-1]:.4f} (↓{100*(1-m['loss'][-1]/max(m['loss'][0],1e-8)):.1f}%)"),
        ("最低 Loss",f"{min(m['loss']):.4f} @ step {s[m['loss'].index(min(m['loss']))]}"),
        ("完成 Epoch",f"{m['epch'][-1]:.2f}"),
        ("处理样本",f"{m['smpl'][-1]:,.0f}"),
        ("GPU 显存",f"均值 {np.mean(m['mem']):.2f} / 峰值 {max(m['mem']):.2f} GB"),
        ("训练速度",f"均值 {np.mean(m['smp']):.0f} samples/s"),
        ("梯度范数",f"{min(m['grdn']):.2f} ~ {max(m['grdn']):.2f}"),
        ("学习率",f"{m['lr'][0]:.2e} → {m['lr'][-1]:.2e}"),
    ]
    for i,(l,v, *e) in enumerate(rows):
        yp = 4.0-i*0.38
        ax.text(0.3,yp,l,fontsize=11,fontweight="bold",color="#475569")
        ax.text(5.5,yp,v,fontsize=11,color="#1E293B")
        if e: ax.text(4.8,yp,e[0],ha="right",fontsize=14)
    plt.tight_layout(); save(fig, f"{out}/05_summary{t}.png")

    # ━━━ 图6: 训练速度分析 ━━━
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("训练速度与效率", fontsize=14, fontweight="bold")
    ax = axs[0,0]; ax.plot(s, m["updt"], color=C[4], lw=1, alpha=0.6)
    ax.set_title("模型更新耗时"); ax.set_ylabel("秒/步")
    ax = axs[0,1]; ax.plot(s, m["data"], color=C[2], lw=1, alpha=0.6)
    ax.set_title("数据加载耗时"); ax.set_ylabel("秒/步")
    ax = axs[1,0]; ax.plot(s, m["smp"], color=C[5], lw=1, alpha=0.7)
    if n > w:
        sy,_ = smooth(m["smp"], w); ax.plot(s[w-1:], sy, color=C[3], lw=2)
    ax.set_title("吞吐量趋势"); ax.set_ylabel("Samples/s"); ax.set_xlabel("Step")
    ax = axs[1,1]; ax.scatter(m["updt"], m["loss"], c=s, cmap="viridis", s=3, alpha=0.5)
    ax.set_title("Loss vs 更新耗时"); ax.set_xlabel("更新耗时(s)"); ax.set_ylabel("Loss")
    plt.colorbar(ax.collections[0], ax=ax, label="Step")
    plt.tight_layout(); save(fig, f"{out}/06_speed_efficiency{t}.png")

    # ━━━ 图7: Loss分布 ━━━
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Loss 统计分布", fontsize=14, fontweight="bold")
    ax = axs[0]
    bins = max(20, n//20)
    ax.hist(m["loss"], bins=bins, color=C[0], alpha=0.6, edgecolor="white")
    ax.axvline(np.mean(m["loss"]), color=C[1], ls="--", lw=2, label=f'均值:{np.mean(m["loss"]):.4f}')
    ax.axvline(np.median(m["loss"]), color=C[2], ls=":", lw=2, label=f'中位数:{np.median(m["loss"]):.4f}')
    ax.set_title("Loss 分布直方图"); ax.set_xlabel("Loss"); ax.set_ylabel("频次"); ax.legend()
    # 分阶段 Loss
    ax = axs[1]
    stages = min(4, n//100)
    if stages >= 2:
        chunk = n // stages
        for i in range(stages):
            start = i*chunk; end = min((i+1)*chunk, n)
            chunk_loss = m["loss"][start:end]
            ax.boxplot([chunk_loss], positions=[i], widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor=C[i%len(C)], alpha=0.4))
        ax.set_title("Loss 分阶段对比 (早期→晚期)"); ax.set_xlabel("训练阶段"); ax.set_ylabel("Loss")
        ax.set_xticklabels([f"阶段{i+1}\n({s[i*chunk]}-{s[min((i+1)*chunk,n)-1]})" for i in range(stages)])
    plt.tight_layout(); save(fig, f"{out}/07_loss_distribution{t}.png")

    # ━━━ 图8: Epoch 进度 ━━━
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("数据遍历进度", fontsize=14, fontweight="bold")
    ax = axs[0]; ax.plot(s, m["epch"], color=C[2], lw=1.5)
    ax.fill_between(s, 0, m["epch"], alpha=0.1, color=C[2])
    ax.set_title("Epoch 进度"); ax.set_xlabel("Step"); ax.set_ylabel("Epoch")
    episodes = np.array(m["ep"])
    ax = axs[1]; ax.plot(s, episodes, color=C[0], lw=1, alpha=0.7)
    ax.set_title("遍历轨迹数"); ax.set_xlabel("Step"); ax.set_ylabel("Episode")
    ax.text(0.98,0.95,f"已遍历 {episodes[-1]} 条\n共 997 条 ({100*episodes[-1]/997:.1f}%)",
            transform=ax.transAxes,ha="right",va="top",fontsize=9,bbox=dict(boxstyle="round",fc="#F1F5F9"))
    plt.tight_layout(); save(fig, f"{out}/08_epoch_progress{t}.png")

    print(f"\n  📊 共生成 8 张训练图表 → {out}/")


def watch(glob_pat: str, out: str, refresh: int):
    print(f"\n🔍 实时监控 (每 {refresh}s 刷新) — Ctrl+C 退出\n")
    last_step = -1
    try:
        while True:
            files = sorted(_glob.glob(glob_pat))
            if not files: print(f"  [{time.strftime('%H:%M:%S')}] 等待日志..."); time.sleep(refresh); continue
            m = parse_log(files[-1])
            if not m["step"]: print(f"  [{time.strftime('%H:%M:%S')}] 等待数据..."); time.sleep(refresh); continue
            cur = m["step"][-1]
            if cur > last_step:
                last_step = cur
                ts = time.strftime("%H%M%S")
                gen_all(m, out, tag=ts)
                print(f"  [{time.strftime('%H:%M:%S')}] step={cur} loss={m['loss'][-1]:.4f} mem={m['mem'][-1]:.2f}GB")
            time.sleep(refresh)
    except KeyboardInterrupt:
        print(f"\n👋 停止。图表: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="训练曲线可视化")
    p.add_argument("--input","-i",type=str,help="日志文件")
    p.add_argument("--watch","-w",type=str,help="实时监控 glob")
    p.add_argument("--refresh","-r",type=int,default=30)
    p.add_argument("--output-dir","-o",type=str,
                   default=os.path.join(os.path.dirname(__file__),"..","data","figures","training"))
    args = p.parse_args()
    out = os.path.abspath(args.output_dir)
    print(f"\n{'='*55}\n  Diffusion Policy 训练曲线可视化\n  输出: {out}\n{'='*55}\n")
    if args.watch:
        watch(args.watch, out, args.refresh)
    elif args.input:
        m = parse_log(args.input)
        if not m["step"]: print("❌ 未找到训练数据"); sys.exit(1)
        print(f"📊 {len(m['step'])} 条记录 | Step {m['step'][0]}→{m['step'][-1]} | Loss {min(m['loss']):.4f}~{max(m['loss']):.4f}")
        gen_all(m, out)
    else:
        p.print_help()
    print(f"\n✅ 完成: {out}")
