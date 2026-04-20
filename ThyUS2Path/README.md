# ThyUS2Path

Pipeline chẩn đoán tuyến giáp (benign/malignant) theo hướng patient-level (1 bệnh nhân = nhiều ảnh).

## 1) Cấu trúc chính
```text
ThyUS2Path/
  data/
    source_data/           # dữ liệu gốc nhiều nguồn
    preprocess_data/       # CSV đã chuẩn hóa từng nguồn
    main/
      main_manifest.csv    # CSV chính để train
  src/thyus2path/          # mã lõi (core), KHÔNG chạy trực tiếp
  scripts/                 # file chạy (entry points)
    step1_data/            # bước xử lý dữ liệu
    train_patient_baseline.py
    evaluate_patient.py
    infer_patient.py
```

## 2) `src` và `scripts` khác nhau thế nào?
- `src/thyus2path/*`: lớp/hàm lõi (`data`, `dataset`, `model`, `engine`, `metrics`).
- `scripts/*.py`: file chạy thực tế, import hàm từ `src` để dùng.

## 3) Thứ tự chạy chuẩn
```bash
python scripts/step1_data/01_process_batch_pathology.py
python scripts/step1_data/02_process_folder_binary.py
python scripts/step1_data/03_process_archive_weak.py
python scripts/step1_data/04_merge_to_main.py
python scripts/train_patient_baseline.py
```

## 4) Tune tham số train ở đâu?
Mở file `scripts/train_patient_baseline.py`, sửa khối `DEFAULTS`.

Ví dụ các tham số chính:
- `pooling`: `mean` / `max` / `attention`
- `backbone`: `resnet18` / `resnet34`
- `epochs`, `lr`, `batch_size`, `weight_decay`
- `k_folds`, `test_size`, `threshold`

Sau khi sửa, chạy đúng 1 lệnh:
```bash
python scripts/train_patient_baseline.py
```

## 5) Chạy toàn bộ A -> Z bằng 1 lệnh
```bash
bash run_all.sh
```
