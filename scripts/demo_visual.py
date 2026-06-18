#!/usr/bin/env python
"""
Day 1 可视化验证：ManiSkill 3 PickCube-v1 GUI 窗口
照着官方 demo_random_action.py 的写法，修复了窗口卡死问题
运行方式：
    conda activate maniskill
    python scripts/demo_visual.py

按 Q 键退出可视化窗口
"""
import gymnasium as gym
import numpy as np
import sapien
import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv


def main():
    print("=" * 60)
    print("ManiSkill 3 — PickCube-v1 可视化 Demo（官方写法）")
    print("=" * 60)

    # 关键：可视化 demo 用 obs_mode="none"，避免传感器渲染和 GUI 渲染冲突
    env: BaseEnv = gym.make(
        "PickCube-v1",
        obs_mode="none",              # 不渲染传感器图，只开 GUI
        control_mode="pd_joint_delta_pos",
        render_mode="human",          # 弹出 GUI 窗口
        sim_backend="gpu",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        enable_shadow=True,
    )

    print(f"动作空间: {env.action_space}")
    print(f"观测模式: none（仅 GUI，不产生传感器图像）")
    print()

    obs, _ = env.reset()
    print("🪟 GUI 窗口应已弹出。按 Q 退出。")

    # === 官方模式：每步调用 env.render() 泵事件循环 ===
    viewer = env.render()
    if isinstance(viewer, sapien.utils.Viewer):
        viewer.paused = False  # 自动播放
        print(f"Viewer 类型: {type(viewer).__name__}, paused={viewer.paused}")

    step = 0
    try:
        while True:
            action = env.action_space.sample() * 0.05
            obs, reward, terminated, truncated, info = env.step(action)

            # ★ 关键：每一步都调用 render()，保持窗口响应
            env.render()

            step += 1
            if step % 100 == 0:
                print(f"  Step {step}")

            if terminated or truncated:
                print(f"  Episode 结束于 step {step}，重置...")
                obs, _ = env.reset()
                env.render()

    except KeyboardInterrupt:
        print("\n👋 用户中断")
    finally:
        env.close()
        print("环境已关闭。")


if __name__ == "__main__":
    main()
