#!/usr/bin/env bash
# ============================================================================
# Day 5: Diffusion Policy 训练启动脚本
# 硬件：RTX 4060 Laptop (8GB), 使用 lerobot conda 环境
#
# 用法：
#   ./run_training.sh                # 前台运行 + 完成后自动生成训练图表
#   ./run_training.sh bg             # 后台运行
#   ./run_training.sh test           # 200 步测试
#   ./run_training.sh monitor        # 实时监控已有训练（每 30s 刷新图表）
# ============================================================================

set -euo pipefail

# ---- 路径配置 ----
PROJECT_DIR="/home/kk/桌面/mianshi/embd-ai-project"
CONFIG_FILE="${PROJECT_DIR}/configs/train_diffusion_pickcube.yaml"
LOG_DIR="${PROJECT_DIR}/logs"
FIG_DIR="${PROJECT_DIR}/data/figures/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
PLOT_SCRIPT="${PROJECT_DIR}/scripts/plot_training_curves.py"

mkdir -p "${LOG_DIR}" "${FIG_DIR}"
mkdir -p "${PROJECT_DIR}/outputs/train"

# ---- CUDA 库路径（torchcodec 视频解码需要） ----
export LD_LIBRARY_PATH=$(find /home/kk/miniconda3/envs/lerobot/lib/python3.12/site-packages/nvidia \
    -name "*.so*" -type f 2>/dev/null | xargs dirname | sort -u | tr '\n' ':'):${LD_LIBRARY_PATH:-}

# ---- 检查 conda 环境 ----
if ! conda env list | grep -q "lerobot"; then
    echo "❌ 错误: lerobot conda 环境不存在！请先完成 Day 4 的环境搭建。"
    exit 1
fi

echo "=============================================="
echo "  Diffusion Policy 训练 — PickCube-v1"
echo "  数据集: 997 条轨迹, 49,539 帧"
echo "  硬件: RTX 4060 8GB | AMP: ON | Batch: 8"
echo "  训练步数: 50,000 (~8 epochs)"
echo "=============================================="
echo ""
echo "📁 配置文件: ${CONFIG_FILE}"
echo "📝 日志文件: ${LOG_FILE}"
echo "📦 输出目录: ${PROJECT_DIR}/outputs/train/pickcube_diffusion"
echo ""

# ---- 确认 GPU 可见 ----
echo "🔍 检查 GPU 状态..."
conda run -n lerobot python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM Total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
" 2>&1 | grep -v "WARNING\|ERROR" || true
echo ""

# ---- 运行模式 ----
MODE="${1:-fg}"

if [ "$MODE" = "monitor" ]; then
    echo "📊 实时监控模式：每 30 秒更新训练图表"
    echo "   日志匹配: ${LOG_DIR}/training_*.log"
    echo ""
    conda run -n maniskill python3 "${PLOT_SCRIPT}" \
        --watch "${LOG_DIR}/training_*.log" --refresh 30

elif [ "$MODE" = "test" ]; then
    echo "🧪 测试模式：运行 200 步验证配置..."
    echo ""
    conda run --no-capture-output -n lerobot lerobot-train \
        --config_path="${CONFIG_FILE}" \
        --steps=200 \
        --log_freq=10 \
        --save_checkpoint=false \
        2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 测试完成。生成训练图表..."
        conda run -n maniskill python3 "${PLOT_SCRIPT}" --input "${LOG_FILE}" --output-dir "${FIG_DIR}"
        echo "📊 图表已保存至: ${FIG_DIR}/"
        echo "🚀 启动完整训练: ./run_training.sh"
    fi

elif [ "$MODE" = "bg" ]; then
    echo "🚀 后台训练启动中..."
    echo "📊 训练完成后运行: ./run_training.sh plot"
    nohup bash -c "
        export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\"
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate lerobot
        lerobot-train --config_path='${CONFIG_FILE}'
        echo 'DONE' >> /tmp/lerobot_training_done
    " > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "  PID: ${PID}"
    echo "  查看日志: tail -f ${LOG_FILE}"
    echo "  查看进程: ps -p ${PID}"
    echo "  实时出图: ./run_training.sh monitor"
    echo "  终止训练: kill ${PID}"
    echo "${PID}" > "${LOG_DIR}/training_pid.txt"

else
    echo "🚀 前台训练启动（Ctrl+C 终止）..."
    echo ""
    conda run --no-capture-output -n lerobot lerobot-train --config_path="${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "📊 训练完成，生成分析图表..."
        conda run -n maniskill python3 "${PLOT_SCRIPT}" --input "${LOG_FILE}" --output-dir "${FIG_DIR}"
        echo "📊 图表已保存至: ${FIG_DIR}/"
    fi
fi
