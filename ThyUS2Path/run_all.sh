#!/usr/bin/env bash
set -euo pipefail

# Chạy toàn bộ từ A -> Z theo cấu trúc mới.
# Không cần truyền tham số.
# Muốn tune train: sửa trực tiếp biến DEFAULTS trong scripts/train_patient_baseline.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_WEAK_LABELS="${USE_WEAK_LABELS:-1}" # 1: gộp archive weak, 0: chỉ dùng nhãn mạnh
PREPROCESS_DIR="data/preprocess_data"

echo "[1/4] Xử lý batch pathology"
"$PYTHON_BIN" scripts/step1_data/01_process_batch_pathology.py

echo "[2/4] Xử lý nguồn folder binary"
"$PYTHON_BIN" scripts/step1_data/02_process_folder_binary.py

if [[ "$USE_WEAK_LABELS" == "1" ]]; then
  echo "[3/4] Xử lý archive weak"
  "$PYTHON_BIN" scripts/step1_data/03_process_archive_weak.py

  echo "[4/4] Gộp toàn bộ CSV preprocess -> main_manifest.csv"
  "$PYTHON_BIN" scripts/step1_data/04_merge_to_main.py
else
  echo "[3/4] Bỏ qua archive weak (USE_WEAK_LABELS=0)"

  echo "[4/4] Gộp chỉ nguồn nhãn mạnh"
  "$PYTHON_BIN" scripts/step1_data/04_merge_to_main.py \
    --input-csv "$PREPROCESS_DIR/batch1_image_preprocessed.csv" \
    --input-csv "$PREPROCESS_DIR/batch2_image_preprocessed.csv" \
    --input-csv "$PREPROCESS_DIR/classification_image_preprocessed.csv"
fi

echo "[TRAIN] Bắt đầu train (tham số đọc từ scripts/train_patient_baseline.py)"
"$PYTHON_BIN" scripts/train_patient_baseline.py

echo "[DONE] Hoàn tất pipeline cơ bản."
echo "CSV train chính: data/main/main_manifest.csv"
echo "Kết quả train:   results_refactored/"
