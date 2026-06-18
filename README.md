# 基于 Diffusion Policy 的端到端视觉机械臂灵巧操作仿真

> 具身智能算法实习生项目 — ManiSkill 3 + LeRobot + Diffusion Policy

## 硬件环境
- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- CUDA: 13.2 Driver (兼容 CUDA 12.x)
- RAM: 14GB
- OS: Ubuntu Linux

## 项目结构
```
embd-ai-project/
├── data/           # 轨迹数据集（ManiSkill 录制）
├── scripts/        # 数据转换、训练、评测脚本
├── checkpoints/    # 模型权重
├── videos/         # Demo 视频
└── logs/           # 训练日志
```

## 技术栈
- 仿真器: ManiSkill 3 (SAPIEN)
- 策略训练: HuggingFace LeRobot
- 算法: Diffusion Policy / ACT
- 任务: PickCube-v1

## 速成路线 (10天)
1. Day 1-2: 环境搭建，跑通 ManiSkill Demo
2. Day 3-4: 专家策略数据采集 (500条轨迹)
3. Day 5-7: LeRobot 数据转换 + Diffusion Policy 训练
4. Day 8-9: 闭环评测与调优
5. Day 10: 包装与复盘
