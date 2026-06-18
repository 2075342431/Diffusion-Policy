#!/usr/bin/env python
"""
Day 1 验证脚本：ManiSkill 3 PickCube-v1 环境验证 + 离屏渲染
运行方式：
    conda activate maniskill
    python scripts/demo_verify.py
输出渲染图到 data/demo_frame_*.png 用于视觉确认
"""

import gymnasium as gym
import mani_skill.envs
import cv2
import numpy as np
from pathlib import Path


def render_to_numpy(env) -> np.ndarray:
    """将 env.render() 的 CUDA tensor 转为 numpy HWC RGB uint8 数组"""
    rgb = env.render()
    if rgb is None:
        return None
    # render_mode="rgb_array" 返回 (1, H, W, 3) CUDA uint8 tensor
    return rgb.squeeze(0).cpu().numpy()


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ManiSkill 3 — PickCube-v1 离屏渲染验证")
    print("=" * 60)

    # 离屏渲染：rgb_array 模式，不弹 GUI，不卡死
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgb+depth",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",      # 离屏渲染高清图（给人看）
        sim_backend="gpu",
    )

    obs, info = env.reset()

    # === 基本信息 ===
    print(f"\n📦 环境信息:")
    print(f"  观测键: {list(obs.keys())}")
    print(f"  关节状态: {obs['agent']['qpos'].shape}  (9 维)")
    print(f"  RGB 相机: {obs['sensor_data']['base_camera']['rgb'].shape}  (128×128×3)")
    print(f"  深度图:   {obs['sensor_data']['base_camera']['depth'].shape}  (128×128×1)")
    print(f"  动作空间: {env.action_space.shape}  ({env.action_space})")
    print(f"  控制模式: pd_joint_delta_pos")

    # === 保存初始帧 ===
    frame = render_to_numpy(env)
    if frame is not None:
        img_path = output_dir / "demo_frame_initial.png"
        cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"\n🖼️  初始帧已保存: {img_path}  ({frame.shape})")

    # === 运行 50 步 ===
    print(f"\n▶ 运行 50 步随机动作...")
    for i in range(50):
        action = env.action_space.sample() * 0.03
        obs, reward, terminated, truncated, info = env.step(action)

        if (i + 1) % 10 == 0:
            reward_val = reward.item() if hasattr(reward, 'item') else reward
            print(f"  Step {i+1:3d}: reward={reward_val:+.4f}")

        if terminated or truncated:
            print(f"  Episode 结束于 step {i+1}")
            obs, info = env.reset()

    # === 保存结束帧 ===
    frame = render_to_numpy(env)
    if frame is not None:
        img_path = output_dir / "demo_frame_final.png"
        cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"\n🖼️  结束帧已保存: {img_path}  ({frame.shape})")

    env.close()

    print()
    print("=" * 60)
    print("✅ PickCube-v1 离屏渲染验证通过！")
    print(f"   查看渲染图: {output_dir}/demo_frame_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
