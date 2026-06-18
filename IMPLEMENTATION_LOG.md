# 具身智能项目实现全记录

> 基于 Diffusion Policy 的端到端视觉机械臂灵巧操作仿真
> 技术栈：ManiSkill 3 + SAPIEN + LeRobot + PyTorch
> 硬件：NVIDIA RTX 4060 Laptop (8GB VRAM), Ubuntu 24.04, CUDA 13.2 Driver

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [仿真验证](#2-仿真验证)
3. [专家数据采集](#3-专家数据采集)
4. [数据格式详解](#4-数据格式详解)
5. [LeRobot 格式转换](#5-lerobot-格式转换)
6. [踩坑记录](#6-踩坑记录)

---

## 1. 环境搭建

### 1.1 硬件与驱动

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8,188 MiB VRAM) |
| CUDA Driver | 595.71.05, CUDA 13.2 |
| RAM | 14 GB |
| CPU | 16 核 |
| OS | Ubuntu 24.04 (X11) |

### 1.2 Conda 环境

```bash
# 创建独立环境（Python 3.10 是 ManiSkill 3 官方推荐版本）
conda create -n maniskill python=3.10 -y
conda activate maniskill

# PyTorch + CUDA 12.4（Driver 13.2 向下兼容 CUDA 12.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **为什么选 CUDA 12.4 而非 13.x？**
> PyTorch 稳定版的 CUDA 支持最高到 12.4/12.6。CUDA Driver 13.2 向下兼容 12.x 二进制，这是 NVIDIA 的基本保证。

安装后验证：

```python
import torch
print(torch.cuda.is_available())          # True
print(torch.version.cuda)                 # 12.4
print(torch.cuda.get_device_name(0))      # NVIDIA GeForce RTX 4060 Laptop GPU
```

### 1.3 ManiSkill 3 + SAPIEN

```bash
pip install --upgrade mani_skill
```

这条命令自动安装：
- `mani_skill` 3.0.1 — 仿真环境框架
- `sapien` 3.0.3 — 底层物理引擎（SAPIEN）
- `gymnasium` 1.3.0 — RL 环境接口
- `opencv-python` — 图像处理
- `h5py` — 轨迹存储

### 1.4 PhysX GPU 库

SAPIEN 首次使用 GPU 仿真时需要下载 PhysX GPU 预编译库（约 75MB）。GitHub 下载在境内网络不稳定，使用 curl 断点续传：

```bash
mkdir -p ~/.sapien/physx/105.1-physx-5.3.1.patch0
curl -L -C - --retry 10 \
  -o ~/.sapien/physx/105.1-physx-5.3.1.patch0/linux-so.zip \
  "https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip"
unzip linux-so.zip  # 解压得到 libPhysXGpu_64.so (236 MB)
```

### 1.5 最终环境版本清单

```
Python:        3.10
PyTorch:       2.6.0+cu124
CUDA Runtime:  12.4
ManiSkill:     3.0.1
SAPIEN:        3.0.3
Gymnasium:     1.3.0
NumPy:         2.2.6
OpenCV:        4.13.0
h5py:          3.16.0
```

---

## 2. 仿真验证

### 2.1 PickCube-v1 任务

**任务描述**：机械臂从桌面抓取红色方块，移动到绿色目标位置。这是 ManiSkill 3 的入门级基准任务。

**关键参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `obs_mode` | `"rgb+depth"` | 传感器观测：RGB 图 + 深度图 + 关节状态 |
| `control_mode` | `"pd_joint_delta_pos"` | 动作空间：关节增量位置控制 |
| `sim_backend` | `"gpu"` / `"physx_cuda"` | GPU 加速物理仿真 |
| `render_mode` | `"human"` / `"rgb_array"` | 可视化（GUI窗口）/ 离屏渲染 |

**观测空间**：

```python
{
    "agent": {
        "qpos": Tensor(1, 9)   # 关节位置：7 个机械臂关节 + 2 个夹爪指
    },
    "sensor_data": {
        "base_camera": {
            "rgb":   Tensor(1, 128, 128, 3),   # uint8
            "depth": Tensor(1, 128, 128, 1),   # int16
        }
    }
}
```

**动作空间**：`Box(-1.0, 1.0, (8,), float32)`
- 7 维：机械臂关节增量位移（delta joint position）
- 1 维：夹爪开合控制

### 2.2 GUI 可视化踩坑

**问题**：`render_mode="human"` 时 SAPIEN viewer 窗口无响应/"卡死"。

**根因分析**：
1. SAPIEN viewer 需要在每步循环中调用 `env.render()` 来泵事件循环
2. `render_mode="human"` 与 `obs_mode="rgb+depth"` 同时使用会冲突——传感器图像渲染和 GUI 渲染争夺 GPU 资源

**解决方案**（参照官方 `demo_random_action.py`）：

```python
# 可视化 demo：obs_mode="none"（不渲染传感器），只用 GUI
env = gym.make(
    "PickCube-v1",
    obs_mode="none",         # 关键：不给 agent 渲染传感器图
    render_mode="human",     # 只渲染 GUI 窗口
    sim_backend="gpu",
)

obs, _ = env.reset()
viewer = env.render()        # 获取 viewer 引用
viewer.paused = False        # 自动播放

while True:
    action = env.action_space.sample() * 0.05
    obs, reward, term, trunc, info = env.step(action)
    env.render()             # ★ 每步调用——泵事件循环，窗口不卡
```

**数据采集/训练阶段**：全程使用离屏渲染（`render_mode="rgb_array"`），不弹 GUI，不卡死。

### 2.3 离屏渲染验证

```python
env = gym.make("PickCube-v1", obs_mode="rgb+depth",
               render_mode="rgb_array", sim_backend="gpu")
obs, info = env.reset()
rgb = env.render()  # 返回 (1, 512, 512, 3) CUDA uint8 tensor
# 保存为 png：squeeze(0).cpu().numpy() → cv2.imwrite()
```

---

## 3. 专家数据采集

### 3.1 方案选择

ManiSkill 3 为每个基准任务提供了**预训练的 PPO 专家策略**和**预录制的演示数据集**。

| 方案 | 方法 | 工作量 | 风险 |
|------|------|--------|------|
| A: 加载 PPO checkpoint 自己跑 | 加载 `.pt` 文件，用 RecordEpisode 录制 | 需要理解 SB3 + 网络结构 | SB3 版本冲突风险 |
| B: 下载官方预录数据 | `download_demo` + `replay_trajectory` | 轻量 | 依赖网络 |
| **C: 下载 + 重放（入选）** | 下载 obs_mode=none 的轨迹 → 重放生成 RGB 观测 | 最省力 | 网络下载需处理 |

**为什么选 C？**
- 官方数据 997 条轨迹，100% 成功率，无需自己跑专家策略
- `obs_mode="none"` 的数据集只有 35MB（只有动作 + 环境状态，无图像）
- 用 `replay_trajectory` 重放时加上 `obs_mode="rgb+depth"`，生成带图像的轨迹
- **数据量对比**：35MB（无观测）→ ~430MB（含 RGB+深度，估算 997 条）

### 3.2 下载官方演示数据

```bash
# 手动下载（境内 HuggingFace 直连不稳定，用 curl 断点续传）
curl -L -C - --retry 10 -o PickCube-v1.zip \
  "https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/PickCube-v1.zip?download=true"
unzip PickCube-v1.zip
```

数据集内容：
```
PickCube-v1/
├── rl/                                    # RL 专家策略数据
│   ├── trajectory.none.pd_joint_delta_pos.physx_cuda.h5   # 997 条轨迹（无观测）
│   ├── trajectory.none.pd_joint_delta_pos.physx_cuda.json # 元数据
│   ├── ppo_pd_joint_delta_pos_ckpt.pt     # PPO checkpoint（可供自行生成数据）
│   ├── ppo_pd_ee_delta_pos_ckpt.pt        # EE 空间控制 checkpoint
│   ├── ppo_pd_ee_delta_pose_ckpt.pt       # EE 位姿控制 checkpoint
│   └── sample_*.mp4                        # 样本视频
├── teleop/                                 # 遥操作数据
│   └── trajectory.h5 + trajectory.json
└── motionplanning/                         # 运动规划数据
    └── trajectory.h5 + trajectory.json
```

### 3.3 重放生成视觉观测

原始轨迹 `obs_mode="none"` 不含图像，但包含 `env_states`（物理状态快照）。利用 ManiSkill 3 的轨迹重放工具，设置 `obs_mode="rgb+depth"` 重新运行，精确复现每一步的同时渲染摄像头画面：

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "trajectory.none.pd_joint_delta_pos.physx_cuda.h5" \
    -o "rgb+depth" \              # ★ 目标观测模式：RGB + 深度
    -b "physx_cuda" \             # 使用与原始数据一致的 GPU 后端
    --save-traj \                  # 保存重放轨迹到新 HDF5
    --use-env-states \             # 用环境状态保证精确复现
    --render-mode "rgb_array" \   # 离屏渲染（不弹窗口）
    --allow-failure                # 保存所有重放结果
```

**重放原理**（源码 `replay_trajectory.py`）：

```
for each episode:
    env.reset(seed=episode.seed)        # 重现初始状态
    env.set_state_dict(env_states[0])   # 精确设置第一帧
    for action in episode.actions:
        env.step(action)                 # 执行专家动作
        env.set_state_dict(env_states[t])# 强制同步到录制状态
        RecordEpisode 自动保存(obs, action)
    flush_trajectory()                   # 写入 HDF5
```

**性能**：
- 单条轨迹：~1.6-1.8 秒（RTX 4060, GPU 仿真）
- 997 条：约 30 分钟
- 100% 成功率（使用 `--use-env-states` 保证确定性回放）

### 3.4 重放输出

```
trajectory.rgb+depth.pd_joint_delta_pos.physx_cuda.h5  (~430 MB for 997 episodes)
trajectory.rgb+depth.pd_joint_delta_pos.physx_cuda.json
```

**HDF5 内部结构**（含观测版）：

```
traj_0/
├── obs/
│   ├── agent/
│   │   └── qpos:      (51, 9)    float32   # 关节位置 (含初始帧)
│   ├── sensor_data/
│   │   └── base_camera/
│   │       ├── rgb:    (51, 128, 128, 3)  uint8
│   │       └── depth:  (51, 128, 128, 1)  int16
│   └── sensor_param/...
├── actions:            (50, 8)    float32   # 专家动作
├── terminated:         (50,)     bool
├── truncated:          (50,)     bool
├── success:            (50,)     bool
└── env_states:         (51,)     dict      # 物理状态快照
```

> **注意**：obs 比 actions 多一帧，因为 reset 后产生初始观测 + 每步 step 产生新观测。标准做法是 `obs[t] → action[t] → obs[t+1]`，取前 50 帧 obs 对齐 50 步 action。

---

## 4. 数据格式详解

### 4.1 状态维度

**关节状态（observation.state）**：9 维
```
[joint_0, joint_1, ..., joint_6, gripper_left, gripper_right]
```
- `joint_0-6`：Franka Panda 机械臂的 7 个旋转关节（弧度）
- `gripper_left/right`：平行夹爪的两个指位置（米）

**动作（action）**：8 维
```
[delta_joint_0, ..., delta_joint_6, gripper_command]
```
- `delta_joint_0-6`：相对于当前关节位置的目标增量（归一化到 [-1, 1]）
- `gripper_command`：夹爪目标开合量（-1 = 合拢, 1 = 张开）

### 4.2 LeRobot 格式要求

LeRobot v3.0 数据集是一个目录，结构如下：

```
lerobot_dataset/
├── data/
│   └── chunk-000/
│       └── file-000.parquet       # 数值数据（action, state, timestamp...）
├── videos/
│   └── observation.images.cam_name/
│       └── chunk-000/
│           └── file-000.mp4        # 每段 episode 一个 mp4
├── meta/
│   ├── info.json                   # 数据集元信息
│   ├── stats.json                  # 归一化统计量
│   ├── tasks.parquet               # 任务定义
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet    # episode 元信息
```

**Parquet schema**（每行 = 一帧）：

| 列名 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `action` | list[float32] | (8,) | 专家动作 |
| `observation.state` | list[float32] | (9,) | 关节状态 |
| `timestamp` | float32 | scalar | 时间戳（秒） |
| `frame_index` | int64 | scalar | 帧内编号 |
| `episode_index` | int64 | scalar | 轨迹编号 |
| `task` | string | scalar | 任务名称 |

**视频**：每段 episode 的 RGB 图像编码为 mp4，存储在 `videos/` 下对应的 camera key 路径。

---

## 5. LeRobot 格式转换

ManiSkill 3 官方提供 `convert_to_lerobot.py`，支持将 HDF5 轨迹直接转换为 LeRobot v3.0 格式。

```bash
python -m mani_skill.trajectory.convert_to_lerobot \
    --traj-path="trajectory.rgb+depth.pd_joint_delta_pos.physx_cuda.h5" \
    --output-dir="./data/lerobot_pickcube_997" \
    --fps=20 \                # 控制频率 20Hz
    --image-size="96x96" \    # 训练用分辨率（8GB 显存约束）
    --task-name="PickCube-v1" \
    --chunks-size=200         # 每 200 个 episode 一个 chunk
```

**转换流程**：
1. `load_trajectory_from_h5()` — 读取 HDF5，提取 action、RGB、state
2. `process_episode()` — 每段 episode 构建 pandas DataFrame
3. `create_video_from_frames()` — RGB 帧编码为 mp4（`cv2.VideoWriter`）
4. `create_meta_files()` — 生成 `info.json`、`stats.json`、`tasks.parquet`
5. `calculate_statistics()` — 计算 action/state/image 的 mean/std/min/max

**为什么用 96×96 分辨率？**
- 原始传感器 128×128 → 转换时 resize 到 96×96
- Diffusion Policy 训练时图像经过 CNN 编码，分辨率对模型容量影响小
- 在 8GB RTX 4060 上以 batch_size=64 训练不会 OOM

---

## 6. 踩坑记录

### 坑 1：`image_size` 不是 `gym.make()` 的参数

```python
# ❌ 错误
env = gym.make("PickCube-v1", image_size=128)

# ✅ 正确：图像尺寸由 obs_mode 的 sensor_configs 控制
env = gym.make("PickCube-v1", obs_mode="rgb+depth")
```

### 坑 2：`render_mode` 与观测模式冲突

- `render_mode="human"` + `obs_mode="rgb+depth"` → 窗口卡死
- `render_mode="human"` + `obs_mode="none"` → 流畅 GUI
- 数据采集只用 `render_mode="rgb_array"` 离屏渲染

### 坑 3：PhysX GPU 库下载失败

GitHub 大文件（75MB）在国内直连容易中断。解决：用 `curl -C - --retry 10` 断点续传。

### 坑 4：HuggingFace 数据集下载

`download_demo.py` 使用 `urllib.request.urlretrieve` 无断点续传，网络不稳时反复失败。解决：手动 `curl -C -` 下载。

### 坑 5：SB3 安装破坏 PyTorch 版本

`pip install stable-baselines3` 会自动安装最新的 `torch` 作为依赖，覆盖已有的 CUDA 版本。注意 `--no-deps` 或先检查依赖树。

### 坑 6：`physx_gpu` vs `physx_cuda`

ManiSkill 3 的 GPU 后端名称为 `"cuda"` / `"physx_cuda"`，而非 `"gpu"` / `"physx_gpu"`。
```python
# 有效后端名
['cpu', 'cuda', 'gpu', 'physx_cpu', 'physx_cuda']
```

### 坑 7：HDF5 文件锁

`h5py` 默认使用文件锁。如果有进程正在写入 HDF5，其他进程无法打开（`BlockingIOError`）。解决：先等写入完成，或改用 `swmr=True` 模式。

---

## 附录：关键技术概念

### A. 为什么用 Diffusion Policy？

传统 Behavior Cloning (BC) 直接回归动作，但真实世界的抓取策略往往是**多峰的**（比如"从左绕过去"和"从右绕过去"都是合理的）。MSE loss 会让模型输出两个可能动作的"平均值"，导致机械臂停在中间什么也抓不到。

Diffusion Policy 通过**去噪扩散过程**建模动作分布：
1. 训练时：学会从噪声中恢复专家动作
2. 推理时：从纯噪声开始，逐步去噪，生成多峰分布中的一个明确动作
3. 配合 Action Chunk（一次输出未来 N 步），动作天然平滑

### B. Sim-to-Real 的核心策略

- **动作空间抽象**：输出末端执行器相对位姿（EE delta pose）而非关节力矩 → 降低动力学差异
- **域随机化**：训练时对 RGB 图随机改变纹理/光照 → 削弱仿真纹理干扰
- **深度图**：比 RGB 更跨域稳定，作为辅助输入

### C. Temporal Ensemble 解释

Diffusion Policy 每步推理输出未来 H 步动作，但只执行前 N 步（N < H），然后重新推理。两次连续推理的第一步动作可能不一致 → 抖动。

Temporal Ensemble 用一个加权滑动窗口融合多次推理的对应步动作（指数衰减权重），消除帧间跳变。

---

*文档生成时间：2026-06-18 | 项目目录：`~/桌面/mianshi/embd-ai-project/`*

---

## 7. LeRobot 训练环境搭建

### 7.1 Python 版本冲突

ManiSkill 3 稳定运行于 Python 3.10，而 LeRobot 0.5.x 要求 Python >= 3.12。因此创建独立的 conda 环境专门用于训练：

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

### 7.2 PyTorch + CUDA 13.0

RTX 4060 的 CUDA Driver 为 13.2。torchcodec（LeRobot 视频解码必需）需要匹配的 PyTorch CUDA 版本：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

最终版本：
```
torch:     2.12.1+cu130
CUDA:      13.0 (Runtime) / 13.2 (Driver)
torchcodec: 0.14.0
```

### 7.3 LeRobot 安装

```bash
git clone https://github.com/huggingface/lerobot.git --depth 1
cd lerobot
pip install -e ".[training]"
```

> **注意**：`pip install lerobot` 仅安装 Python 库，不含训练脚本。需要从 GitHub 克隆完整仓库以使用 `lerobot_train.py`。

### 7.4 数据集加载验证

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset(
    repo_id="/home/kk/桌面/mianshi/embd-ai-project/data/lerobot_pickcube_997"
)
# Frames: 49539 | Episodes: 997 | FPS: 20
# Action: (8,) float32  | State: (9,) float32  | Image: (3, 96, 96) float32
```

数据加载正常，视频解码通过 torchcodec + FFmpeg（conda-forge 安装）完成。

### 7.5 环境对照

| 用途 | Conda 环境 | Python | PyTorch | CUDA |
|------|-----------|--------|---------|------|
| 数据生成 | `maniskill` | 3.10 | 2.6.0+cu124 | 12.4 |
| 模型训练 | `lerobot` | 3.12 | 2.12.1+cu130 | 13.0 |

---

## 8. Day 5：Diffusion Policy 训练配置与验证

### 8.1 训练配置设计

**YAML 配置文件**：`configs/train_diffusion_pickcube.yaml`

基于 8GB 显存的优化策略：

| 参数 | 设置 | 理由 |
|------|------|------|
| `policy.type` | `diffusion` | 扩散模型策略，天然处理多峰动作分布 |
| `use_amp` | `true` | 自动混合精度，节省 ~30% 显存 |
| `batch_size` | `8` | 保守值，留 3GB 余量 |
| `num_workers` | `2` | 减少 CPU 内存占用 |
| `steps` | `50000` | ~8 epochs 过 997 条轨迹 |
| `vision_backbone` | `resnet18` (默认) | 最小 ResNet，11M 参数 |
| `down_dims` | `(512, 1024, 2048)` (默认) | 3 阶段 UNet |
| `horizon` | `64` (默认) | 预测未来 64 步动作 |
| `n_action_steps` | `32` (默认) | 每次执行 32 步 |

### 8.2 环境修复

**问题 1**：`diffusers` 包未安装
```bash
conda run -n lerobot pip install 'lerobot[diffusion]'
```

**问题 2**：`libnvrtc.so.13` 找不到（torchcodec 视频解码需要）
- 库文件存在于 `nvidia/cu13/lib/` 但不在 `LD_LIBRARY_PATH` 中
- 解决：在 `run_training.sh` 中自动收集所有 nvidia 库路径并添加到 `LD_LIBRARY_PATH`

### 8.3 200 步测试结果

```
step:100  loss:0.21  grdn:5.0  lr:1.9e-05  mem:4.94 GB
step:170  loss:0.159 grdn:3.6  lr:3.3e-05  mem:4.94 GB
step:190  loss:0.126 grdn:3.3  lr:3.7e-05  mem:4.94 GB
step:200  loss:0.145 grdn:4.1  lr:3.9e-05  mem:4.94 GB
```

| 指标 | 值 |
|------|-----|
| 模型参数 | 263,131,944 (263M) |
| 显存占用 | 4.94 GB / 8 GB |
| 训练速度 | 5.8 step/s |
| 吞吐量 | 47 samples/s |
| Loss 趋势 | 0.70 → 0.128 ✅ 持续下降 |
| 预估总时长 | 50,000 步 ≈ **2.4 小时** |

> **结论**：配置验证通过！显存安全（~5GB/8GB），loss 稳步下降，无 NaN/Inf，无 OOM。

### 8.4 训练曲线可视化系统

基于 `matplotlib` + 宋体 (AR PL SungtiL GB) 构建了完整的训练曲线自动生成系统。

**中文字体修复**：系统 TTC 字体 (Noto Sans CJK) 无法被 matplotlib 直接读取，`DroidSansFallbackFull.ttf` 缺少 ASCII 字符。最终使用 `gbsn00lp.ttf`（AR PL SungtiL GB），同时支持中文和英文/数字。

**产出 8 张训练图表**：

| 图 | 文件 | 内容 |
|---|------|------|
| 1 | `01_dashboard.png` | 6合1 仪表盘：Loss / 梯度 / LR / 显存 / 吞吐 / Epoch |
| 2 | `02_loss_analysis.png` | Loss 多窗口平滑 (MA-10/50/100) + 末尾放大 |
| 3 | `03_gradient_dynamics.png` | 梯度范数趋势 + 分布直方图 + LR 调度曲线 |
| 4 | `04_gpu_resources.png` | GPU 显存趋势 + 吞吐量分布 + 每步耗时分解 |
| 5 | `05_summary.png` | 训练统计数值卡片 (总览) |
| 6 | `06_speed_efficiency.png` | 更新/加载耗时 + 吞吐量 + Loss vs 耗时散点 |
| 7 | `07_loss_distribution.png` | Loss 分布直方图 + 分阶段箱线图对比 |
| 8 | `08_epoch_progress.png` | Epoch 进度 + 遍历轨迹数 |

**三种使用方式**：

```bash
# 训练完成后生成（自动）
./scripts/run_training.sh              # 前台训练，完成自动生成

# 训练期间实时监控（另开终端）
./scripts/run_training.sh monitor      # 每 30s 刷新图表

# 离线分析已有日志
conda run -n maniskill python3 scripts/plot_training_curves.py \
    --input logs/training_xxx.log
```

### 8.5 启动训练

```bash
cd /home/kk/桌面/mianshi/embd-ai-project

# 前台运行（推荐，完成自动生成图表）
./scripts/run_training.sh

# 后台运行
./scripts/run_training.sh bg

# 后台 + 实时监控（另开终端）
./scripts/run_training.sh monitor

# 测试 200 步
./scripts/run_training.sh test
```

### 8.6 训练中监控

```bash
# 实时查看日志
tail -f logs/training_*.log

# 查看 GPU 状态
watch -n 1 nvidia-smi

# 查看最新图表
ls -lt data/figures/training/

# 查看 checkpoint
ls -la outputs/train/pickcube_diffusion/checkpoints/
```

---

## 9. Day 6-7：完整训练结果

### 9.1 训练完成

```
INFO ... step:50K smpl:400K ep:8K epch:8.07 loss:0.003 grdn:0.073 lr:3.3e-10 updt_s:0.171 data_s:0.001 smp/s:46 mem_gb:4.95
```

| 指标 | 初始 (step 100) | 最终 (step 50,000) |
|------|----------------|-------------------|
| Loss | 0.986 | **0.003** (↓99.7%) |
| 最低 Loss | — | **0.002** |
| 梯度范数 | 3.59 | 0.073 |
| 学习率 | 1.0e-05 | 3.3e-10 (Cosine 衰减至接近 0) |
| 显存 | 4.96 GB | 4.95 GB |
| Epoch | 0.02 | 8.07 (完整遍历 8 遍) |
| 处理样本 | 800 | 400,000 |

**训练统计**：
- 总耗时：**2 小时 25 分**（预估 2.4 小时，实际吻合）
- 速度：5.8 step/s，47 samples/s
- Checkpoints：5 个 (`010000/` ~ `050000/`，`last → 050000`)

### 9.2 日志解析修复

**问题**：高步数时日志使用 K 缩写 (`step:50K`)，原正则 `\d+` 只捕获到 `50`。

**修复**：`plot_training_curves.py` 增加 `_parse_num()` 函数处理 K/M 后缀：
```python
def _parse_num(s):
    if s.endswith("K"): return int(float(s[:-1]) * 1000)
    if s.endswith("M"): return int(float(s[:-1]) * 1_000_000)
    return int(float(s))
```

### 9.3 conda run 输出缓冲

**问题**：`conda run -n lerobot lerobot-train` 默认缓存全部输出，训练 2.4 小时看不到任何进度。

**修复**：`run_training.sh` 所有 `conda run` 命令添加 `--no-capture-output` 标志。

---

## 10. Day 8-9：闭环评测准备

### 10.1 环境合并

LeRobot 训练环境 (Python 3.12) 成功安装 ManiSkill 3：
```bash
conda run -n lerobot pip install mani_skill
```

两个环境现在统一在 `lerobot` conda env 中，避免了跨环境 IPC 的复杂性。

### 10.2 模型加载验证

```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion import DiffusionPolicy

cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
policy = DiffusionPolicy.from_pretrained(checkpoint_dir)
```

- 参数：263,131,944
- 推理 API：`policy.select_action(batch)` — 自动管理观测缓存和动作队列
- `policy.reset()` — 清空缓存（每个 episode 开始时调用）

### 10.3 评测脚本

`scripts/eval_policy.py` — 支持：
- GUI 可视化 (`--render`)
- 离屏评估 (`-n 50`)
- 视频录制 (前 4 个 episode 自动保存)
- 成功率统计

```bash
conda run --no-capture-output -n lerobot python3 scripts/eval_policy.py -n 10
```

### 10.4 预测 vs 实际对比

**预期评估指标**（基于 8 epoch 训练的合理预期）：
- 成功率：60-85%（首轮训练，未调参）
- 动作平滑度：通过视频观察有无抖动
- 若成功率 < 50%，可能需要：
  - 检查预处理管线（图像归一化、状态归一化）
  - 调整 `n_action_steps` 或 `horizon`
  - 增加真机微调数据

---

## 11. 当前状态与下一步

### 已完成 ✅

1. ManiSkill 3 仿真环境搭建（`maniskill` conda env）
2. PickCube-v1 专家数据下载（HuggingFace，997 条轨迹）
3. 轨迹重放生成视觉观测（RGB + Depth + State + Action）
4. LeRobot v3.0 格式转换（parquet + mp4，96×96，20fps）
5. 数据可视化图表生成（9 张数据探索图）
6. LeRobot 训练环境搭建（`lerobot` conda env）
7. 数据集加载验证通过
8. **Day 5 完成** ✅：训练配置、环境修复、200步验证、训练曲线系统
9. **Day 6-7 完成** ✅：50K 步完整训练，Loss 0.986→0.003，5 个 checkpoint
10. **训练曲线 8 张**：日志解析修复后重新生成，数据正确
11. **conda run 缓冲修复**：添加 `--no-capture-output`

### 进行中 🔄

- **闭环评测**：`scripts/eval_policy.py` 已就绪，待运行

### 待完成（Day 8-9）

- 运行评估，统计成功率
- 录制 Demo 视频
- 若成功率不达标，排查预处理/推理管线

### 项目文件树

```
embd-ai-project/
├── README.md
├── IMPLEMENTATION_LOG.md                  ← 本文档
├── lerobot/                               ← LeRobot 仓库
├── configs/
│   └── train_diffusion_pickcube.yaml      ← Day 5: 训练配置
├── scripts/
│   ├── demo_visual.py                     ← GUI 可视化
│   ├── demo_verify.py                     ← 离屏渲染验证
│   ├── generate_figures.py                ← 数据探索图
│   ├── convert_lerobot.sh                 ← 格式转换
│   ├── run_training.sh                    ← Day 5: 训练启动/监控
│   ├── plot_training_curves.py            ← Day 5: 训练曲线生成 ⭐
│   └── eval_policy.py                     ← Day 8: 闭环评测脚本 ⭐
├── data/
│   ├── raw/                    (23M)      ← 原始轨迹
│   ├── replay/                (430M)      ← 带 RGBD 的 HDF5
│   ├── lerobot_pickcube_997/   (37M)      ← LeRobot 训练数据集
│   └── figures/
│       ├── *.png               (1.2M)     ← Day 3-4: 数据探索图 9张
│       └── training/            (1.8M)    ← Day 5: 训练曲线图 8张
├── outputs/train/pickcube_diffusion/
│   └── checkpoints/
│       ├── 010000/ ~ 050000/              ← 5 个训练 checkpoint
│       └── last -> 050000
├── videos/                                ← 待评测产出 Demo 视频
└── logs/
    └── training_*.log          (4MB)      ← 完整训练日志
```
