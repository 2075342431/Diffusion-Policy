#!/bin/bash
# 等重放完成 → 自动转换 LeRobot 格式
# 用法: bash scripts/convert_lerobot.sh

set -e

REPLAY_OUTPUT="/home/kk/.maniskill/demos/PickCube-v1/rl/trajectory.rgb+depth.pd_joint_delta_pos.physx_cuda.h5"
REPLAY_JSON="${REPLAY_OUTPUT%.h5}.json"
LROBOT_DIR="/home/kk/桌面/mianshi/embd-ai-project/data/lerobot_pickcube_997"

echo "============================================"
echo "Day 3-4: 专家数据 → LeRobot 格式转换"
echo "============================================"

if [ ! -f "$REPLAY_OUTPUT" ]; then
    echo "❌ 重放输出文件不存在: $REPLAY_OUTPUT"
    echo "   等待后台重放任务完成..."
    exit 1
fi

SIZE_MB=$(du -m "$REPLAY_OUTPUT" | cut -f1)
echo "✅ 找到重放文件: $REPLAY_OUTPUT (${SIZE_MB}MB)"

# 清理旧输出
rm -rf "$LROBOT_DIR"

echo "▶ 开始转换..."
source /home/kk/miniconda3/bin/activate maniskill

python -m mani_skill.trajectory.convert_to_lerobot \
    --traj-path="$REPLAY_OUTPUT" \
    --output-dir="$LROBOT_DIR" \
    --fps=20 \
    --image-size="96x96" \
    --task-name="PickCube-v1" \
    --chunks-size=200

echo ""
echo "============================================"
echo "✅ 转换完成！"
echo "   LeRobot 数据集: $LROBOT_DIR"
echo "============================================"

# 显示数据集结构
echo ""
echo "文件结构:"
find "$LROBOT_DIR" -type f | head -20
