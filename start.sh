#!/bin/bash
set -e

echo "============================================"
echo "   GPU SWINIR TRAINING STARTED"
echo "============================================"

echo "[1] Switching into repo folder"
cd /workspace/GpuExe

echo "[2] Installing Python dependencies"
pip install --no-cache-dir -r requirements.txt

echo "[3] Creating output folders"
mkdir -p outputs checkpoints logs

echo "[4] Launching training..."
python main_train_swinir.py --config configs/train_swinir_custom.json

echo "============================================"
echo "   TRAINING COMPLETED SUCCESSFULLY"
echo "============================================"
