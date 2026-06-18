#!/usr/bin/env python
"""
生成数据可视化图 — 用于机器学习课设 / 项目展示
输出到 data/figures/
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import cv2
from collections import defaultdict

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLAY_H5 = PROJECT_ROOT / "data/replay/trajectory.rgb+depth.pd_joint_delta_pos.physx_cuda.h5"
REPLAY_JSON = PROJECT_ROOT / "data/replay/trajectory.rgb+depth.pd_joint_delta_pos.physx_cuda.json"
OUT_DIR = PROJECT_ROOT / "data/figures"
OUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载重放数据"""
    with open(REPLAY_JSON) as f:
        meta = json.load(f)

    f = h5py.File(REPLAY_H5, "r")
    traj_keys = [k for k in f.keys() if k.startswith("traj_")]

    all_actions = []
    all_states = []
    all_rgb_frames = []  # 每个 episode 采样一帧
    all_depth_frames = []
    episode_lengths = []
    success_flags = []

    for i, key in enumerate(traj_keys):
        traj = f[key]
        actions = traj["actions"][:]
        qpos = traj["obs"]["agent"]["qpos"][:]
        rgb = traj["obs"]["sensor_data"]["base_camera"]["rgb"][:]
        depth = traj["obs"]["sensor_data"]["base_camera"]["depth"][:]
        success = meta["episodes"][i]["success"]

        episode_lengths.append(len(actions))
        success_flags.append(success)

        all_actions.append(actions)
        all_states.append(qpos[:-1])  # 对齐 action 长度（obs 多一帧）

        # 采样中间帧用于展示
        mid = len(rgb) // 2
        all_rgb_frames.append(rgb[mid])
        all_depth_frames.append(depth[mid])

    f.close()

    return {
        "actions": np.concatenate(all_actions, axis=0),
        "states": np.concatenate(all_states, axis=0),
        "rgb_samples": np.stack(all_rgb_frames[:16]),  # 前16条
        "depth_samples": np.stack(all_depth_frames[:16]),
        "episode_lengths": np.array(episode_lengths),
        "success_flags": np.array(success_flags),
        "total_episodes": len(traj_keys),
        "total_frames": sum(episode_lengths),
    }


def fig1_episode_length_distribution(data):
    """图1：轨迹长度分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(data["episode_lengths"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(data["episode_lengths"].mean(), color="red", linestyle="--", linewidth=2,
               label=f'Mean: {data["episode_lengths"].mean():.0f} steps')
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory Length Distribution (997 episodes)")
    ax.legend()

    ax = axes[1]
    ax.hist(data["episode_lengths"][data["success_flags"]], bins=25, color="green", alpha=0.6,
            label=f'Success ({data["success_flags"].sum()})')
    ax.hist(data["episode_lengths"][~data["success_flags"]], bins=25, color="red", alpha=0.6,
            label=f'Failure ({(~data["success_flags"]).sum()})')
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Success vs Failure Length Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_episode_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图1：轨迹长度分布")


def fig2_action_distribution(data):
    """图2：动作空间分布（8维动作的直方图）"""
    actions = data["actions"]
    dim_names = [f"Joint {i}" for i in range(7)] + ["Gripper"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        ax = axes[i]
        ax.hist(actions[:, i], bins=50, color="coral", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(actions[:, i].mean(), color="blue", linestyle="--", linewidth=1.5,
                   label=f'mean={actions[:, i].mean():.3f}')
        ax.set_title(dim_names[i])
        ax.set_xlabel("Action value")
        ax.set_ylabel("Freq")
        ax.legend(fontsize=7)

    fig.suptitle("Action Distribution per Dimension (49539 frames)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_action_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图2：动作分布")


def fig3_joint_state_trajectories(data):
    """图3：单条轨迹的关节状态随时间变化"""
    f = h5py.File(REPLAY_H5, "r")
    traj = f["traj_0"]
    qpos = traj["obs"]["agent"]["qpos"][:]
    f.close()

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        ax.plot(qpos[:, i], linewidth=1.5, color="steelblue")
        if i < 7:
            ax.set_title(f"Joint {i} (rad)")
        else:
            ax.set_title(f"Gripper {i-7} (m)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Joint State Trajectory — Episode 0 (Expert Demonstration)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_joint_state_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图3：关节状态轨迹")


def fig4_state_correlation(data):
    """图4：状态-动作相关性矩阵"""
    states = data["states"]
    actions = data["actions"]
    combined = np.concatenate([states, actions], axis=1)

    labels = [f"S_J{i}" for i in range(9)] + [f"A_J{i}" for i in range(7)] + ["A_G"]

    corr = np.corrcoef(combined.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("State-Action Correlation Matrix", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_state_action_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图4：状态-动作相关性矩阵")


def fig5_rgb_frames(data):
    """图5：专家演示样本帧（4×4 网格 = 16 条轨迹中间帧）"""
    frames = data["rgb_samples"]
    n = len(frames)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(frames[i])
        axes[i].set_title(f"Ep {i}", fontsize=8)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Expert Demonstration Frames — Mid-Episode Samples (16 trajectories)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_expert_rgb_frames.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图5：专家演示 RGB 帧")


def fig6_depth_frames(data):
    """图6：深度图样本"""
    frames = data["depth_samples"]
    n = min(len(frames), 8)
    cols = 4
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(n):
        d = frames[i].squeeze()
        im = axes[i].imshow(d, cmap="viridis")
        axes[i].set_title(f"Ep {i} Depth", fontsize=8)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], shrink=0.7)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Depth Maps — Mid-Episode Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_depth_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图6：深度图样本")


def fig7_action_magnitude(data):
    """图7：动作幅值分析"""
    actions = data["actions"]
    mag = np.linalg.norm(actions, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(mag, bins=80, color="teal", edgecolor="white", alpha=0.8)
    ax.axvline(mag.mean(), color="red", linestyle="--", linewidth=2,
               label=f'Mean: {mag.mean():.4f}')
    ax.axvline(np.percentile(mag, 95), color="orange", linestyle="--", linewidth=2,
               label=f'P95: {np.percentile(mag, 95):.4f}')
    ax.set_xlabel("Action Magnitude (L2 norm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Action Magnitude Distribution")
    ax.legend()

    ax = axes[1]
    dim_mag = np.abs(actions).mean(axis=0)
    colors = ["steelblue"] * 7 + ["coral"]
    dim_names = [f"J{i}" for i in range(7)] + ["Grip"]
    bars = ax.bar(dim_names, dim_mag, color=colors, edgecolor="white")
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Mean Absolute Value")
    ax.set_title("Per-Dimension Action Intensity")
    # 加数值标签
    for bar, val in zip(bars, dim_mag):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_action_magnitude_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图7：动作幅值分析")


def fig8_success_pie(data):
    """图8：成功率饼图 + 数据集摘要"""
    n_success = data["success_flags"].sum()
    n_fail = len(data["success_flags"]) - n_success

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        [n_success, n_fail],
        labels=["Success", "Failure"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%",
        explode=(0, 0.05),
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title(f"Expert Trajectory Success Rate\n({n_success}/{len(data['success_flags'])} episodes)", fontsize=14)

    ax = axes[1]
    ax.axis("off")
    summary_text = f"""
    Dataset Summary
    ═══════════════════

    Total Episodes:      {data['total_episodes']}
    Total Frames:        {data['total_frames']:,}

    Success Episodes:    {n_success}
    Failure Episodes:    {n_fail}
    Success Rate:        {n_success/len(data['success_flags'])*100:.1f}%

    Avg Episode Length:  {data['episode_lengths'].mean():.1f} steps
    Max Episode Length:  {data['episode_lengths'].max()}
    Min Episode Length:  {data['episode_lengths'].min()}

    State Dim:           9 (7 joints + 2 gripper)
    Action Dim:          8 (7 joint delta + 1 gripper cmd)
    Image Resolution:    128×128 RGB + Depth
    Control Mode:        pd_joint_delta_pos
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_dataset_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图8：数据集摘要")


def fig9_pairwise_joint(data):
    """图9：关节状态散点矩阵（采样避免过大）"""
    states = data["states"][::50]  # 采样 1/50
    n_samples = min(len(states), 2000)
    states = states[:n_samples]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            ax = axes[i, j]
            if idx < 9:
                ax.hist(states[:, idx], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
                ax.set_title(f"Joint {idx}" if idx < 7 else f"Gripper {idx-7}")
            else:
                ax.axis("off")

    fig.suptitle("Joint State Distributions (sampled)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_joint_state_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 图9：关节状态分布")


def main():
    print("=" * 60)
    print("Generating Data Visualizations")
    print("=" * 60)
    print(f"\nLoading data from: {REPLAY_H5}")

    data = load_data()
    print(f"  Episodes: {data['total_episodes']}")
    print(f"  Frames: {data['total_frames']:,}")
    print(f"  Actions shape: {data['actions'].shape}")
    print(f"  States shape: {data['states'].shape}")
    print(f"  Success: {data['success_flags'].sum()}/{len(data['success_flags'])}")
    print(f"\nGenerating figures to: {OUT_DIR}/\n")

    fig1_episode_length_distribution(data)
    fig2_action_distribution(data)
    fig3_joint_state_trajectories(data)
    fig4_state_correlation(data)
    fig5_rgb_frames(data)
    fig6_depth_frames(data)
    fig7_action_magnitude(data)
    fig8_success_pie(data)
    fig9_pairwise_joint(data)

    print(f"\n{'=' * 60}")
    print(f"✅ All 9 figures saved to {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
