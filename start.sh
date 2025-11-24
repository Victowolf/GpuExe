#!/bin/bash
set -e

echo "============================================"
echo "      SWINIR TRAINING JOB STARTED"
echo "============================================"

cd /workspace/SwinIR

echo "[1] Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "[2] Creating output folders..."
mkdir -p outputs checkpoints logs

echo "[3] Launching training..."
python main_train_swinir.py --config configs/train_swinir_custom.json

echo "============================================"
echo "      TRAINING COMPLETED"
echo "============================================"
