#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------
# Script chạy toàn bộ pipeline từ A -> Z bằng 1 lệnh
# -------------------------------------------------
# Gồm 3 bước:
# 1) Chuẩn bị dữ liệu patient-level + split
# 2) Train model theo fold
# 3) Evaluate trên test patient-level
#
# Cách chạy nhanh:
#   cd ThyUS2Path
#   bash run_all.sh
#
# Có thể override bằng biến môi trường:
#   DATA_ROOT=./data
#   PROCESSED_DIR=./data/processed
#   OUTPUT_DIR=./results_refactored
#   FOLD=0
#   RUN_ALL_FOLDS=0
#   K_FOLDS=5
#   TEST_SIZE=0.1
#   SEED=2026
#   BACKBONE=resnet18
#   POOLING=mean
#   IMAGE_SIZE=224
#   BATCH_SIZE=4
#   EPOCHS=20
#   LR=1e-4
#   WEIGHT_DECAY=1e-4
#   THRESHOLD=0.5
#   NUM_WORKERS=4
#   DEVICE=auto
#   PRETRAINED=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Thư mục dữ liệu và output.
DATA_ROOT="${DATA_ROOT:-./data}"
PROCESSED_DIR="${PROCESSED_DIR:-./data/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_refactored}"

# Cấu hình split.
FOLD="${FOLD:-0}"
RUN_ALL_FOLDS="${RUN_ALL_FOLDS:-0}"
K_FOLDS="${K_FOLDS:-5}"
TEST_SIZE="${TEST_SIZE:-0.1}"
SEED="${SEED:-2026}"

# Cấu hình model/train.
BACKBONE="${BACKBONE:-resnet18}"
POOLING="${POOLING:-mean}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
THRESHOLD="${THRESHOLD:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
PRETRAINED="${PRETRAINED:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/3] Chuẩn bị dữ liệu và split"
"$PYTHON_BIN" scripts/prepare_dataset.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$PROCESSED_DIR" \
  --test-size "$TEST_SIZE" \
  --k-folds "$K_FOLDS" \
  --seed "$SEED"

# Tạo danh sách fold cần chạy.
folds=()
if [[ "$RUN_ALL_FOLDS" == "1" ]]; then
  for ((i=0; i<K_FOLDS; i++)); do
    folds+=("$i")
  done
else
  folds+=("$FOLD")
fi

echo "[2/3] Train fold(s): ${folds[*]}"
for fold_idx in "${folds[@]}"; do
  TRAIN_CMD=(
    "$PYTHON_BIN" scripts/train_patient_baseline.py
    --metadata-csv "$PROCESSED_DIR/patient_images.csv"
    --split-dir "$PROCESSED_DIR/splits"
    --fold "$fold_idx"
    --backbone "$BACKBONE"
    --pooling "$POOLING"
    --image-size "$IMAGE_SIZE"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --threshold "$THRESHOLD"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --device "$DEVICE"
    --output-dir "$OUTPUT_DIR"
  )

  if [[ "$PRETRAINED" == "1" ]]; then
    TRAIN_CMD+=(--pretrained)
  fi

  "${TRAIN_CMD[@]}"
done

echo "[3/3] Evaluate trên tập test"
for fold_idx in "${folds[@]}"; do
  RUN_DIR="$OUTPUT_DIR/fold_${fold_idx}_${BACKBONE}_${POOLING}"
  CKPT="$RUN_DIR/best.pt"

  if [[ ! -f "$CKPT" ]]; then
    echo "[WARN] Bỏ qua fold $fold_idx vì chưa có checkpoint: $CKPT"
    continue
  fi

  "$PYTHON_BIN" scripts/evaluate_patient.py \
    --checkpoint "$CKPT" \
    --metadata-csv "$PROCESSED_DIR/patient_images.csv" \
    --patient-ids-csv "$PROCESSED_DIR/splits/test_patients.csv" \
    --threshold "$THRESHOLD" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --output-csv "$RUN_DIR/test_eval_predictions.csv" \
    --output-json "$RUN_DIR/test_eval_metrics.json"
done

echo "[DONE] Hoàn tất pipeline"
echo "Kết quả tại: $OUTPUT_DIR"
