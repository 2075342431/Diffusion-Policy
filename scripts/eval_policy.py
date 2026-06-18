#!/usr/bin/env python3
"""
闭环评测 Diffusion Policy 模型
用法:
  python eval_policy.py                  # 10 episodes, 离屏+视频
  python eval_policy.py -n 50            # 50 episodes 统计成功率
  python eval_policy.py --render         # GUI 可视化（需要显示器）
"""

import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
import cv2

# ── 路径 ──
PROJECT_DIR = Path(__file__).parent.parent
CKPT_DIR = PROJECT_DIR / "outputs/train/pickcube_diffusion/checkpoints/050000/pretrained_model"
VIDEO_DIR = PROJECT_DIR / "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

TASK = "PickCube-v1"
MAX_STEPS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    # ── 加载模型 ──
    print(f"\n{'='*60}")
    print(f"  加载模型: {CKPT_DIR}")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.diffusion import DiffusionPolicy
    from lerobot.policies import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(CKPT_DIR)
    policy = DiffusionPolicy.from_pretrained(CKPT_DIR)
    policy.eval()
    policy.to("cuda")
    print(f"  ✅ {sum(p.numel() for p in policy.parameters()):,} 参数")
    print(f"  n_obs_steps={cfg.n_obs_steps}  horizon={cfg.horizon}  n_action_steps={cfg.n_action_steps}")

    # ── 创建环境 ──
    import gymnasium as gym
    import mani_skill  # noqa

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(
        TASK,
        obs_mode="rgb+depth",
        control_mode="pd_joint_delta_pos",
        render_mode=render_mode,
        sim_backend="gpu",
        max_episode_steps=MAX_STEPS,
    )
    print(f"  ✅ 环境创建: {TASK}")

    # ── 评估 ──
    print(f"\n{'='*60}")
    print(f"  开始评估: {args.episodes} episodes\n")

    successes = []
    all_video_frames = []

    for ep_idx in range(args.episodes):
        obs, info = env.reset()
        policy.reset()  # 清空观测/动作缓存
        frames = []
        total_reward = 0
        step = 0
        success = False

        for step in range(MAX_STEPS):
            # 1. 从 ManiSkill obs 构建 batch
            rgb = obs["sensor_data"]["base_camera"]["rgb"]  # (H, W, 3) uint8 Tensor
            state = obs["agent"]["qpos"]  # (9,) float32 Tensor

            # 转换到 numpy 便于处理
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()

            # 去除 GPU 仿真可能带来的 batch 维度
            if rgb.ndim == 4:
                rgb = rgb[0]
            if state.ndim == 2:
                state = state[0]

            # Resize RGB
            rgb_resized = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA)

            # 转回 torch (C, H, W) float32 [0,1]
            img_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
            state_tensor = torch.from_numpy(state).float()

            # 构建 batch (batch_size=1, 单帧)
            batch = {
                "observation.images.base_camera": img_tensor.unsqueeze(0),  # (1, 3, 96, 96)
                "observation.state": state_tensor.unsqueeze(0),  # (1, 9)
            }

            # 2. 移动到 GPU + 推理
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                action = policy.select_action(batch)  # (action_dim,)

            # 适配 ManiSkill GPU backend: 需要 (1, 8) 的 Tensor
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = torch.clamp(action, -1.0, 1.0)

            # 3. 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 4. 录制
            if not args.no_video and ep_idx < 4:
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    if frame.ndim == 4:
                        frame = frame[0]
                    frames.append(frame)

            # 提取标量（因为在 GPU backend 下这些可能是 batch=1 的 Tensor）
            is_success = info.get("success", False)
            if isinstance(is_success, torch.Tensor):
                is_success = is_success.item()
            
            if is_success:
                success = True
                break

        successes.append(success)
        status = "✅" if success else "❌"
        
        # 提取 total_reward 到标量
        if isinstance(total_reward, torch.Tensor):
            total_reward_val = total_reward.item()
        else:
            total_reward_val = float(total_reward)
            
        print(f"  Episode {ep_idx+1:3d}/{args.episodes}  {status}  "
              f"steps={step+1}  reward={total_reward_val:.2f}")

        if frames:
            all_video_frames.append(frames)

    env.close()

    # ── 统计 ──
    success_rate = sum(successes) / len(successes) * 100
    print(f"\n{'='*60}")
    print(f"  📊 成功率: {success_rate:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"{'='*60}")

    # ── 保存视频 ──
    if all_video_frames:
        for ep_idx, frames in enumerate(all_video_frames):
            if not frames:
                continue
            video_path = VIDEO_DIR / f"eval_ep{ep_idx+1}_{'success' if successes[ep_idx] else 'fail'}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  🎬 {video_path.name}")


if __name__ == "__main__":
    main()
