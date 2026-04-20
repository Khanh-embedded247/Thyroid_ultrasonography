# DATA_FLOW

## 1) Luồng chạy hiện tại (đã rút gọn)

```text
data/source_data/*
  -> scripts/step1_data/01_process_batch_pathology.py
  -> scripts/step1_data/02_process_folder_binary.py
  -> scripts/step1_data/03_process_archive_weak.py (tùy chọn)
  => data/preprocess_data/*.csv (mỗi nguồn 1 file)

scripts/step1_data/04_merge_to_main.py
  => data/main/main_manifest.csv (file duy nhất để train)

scripts/train_patient_baseline.py
  -> đọc main_manifest.csv
  -> gom theo patient_id (bag nhiều ảnh)
  -> auto split patient-level (train/val/test)
  -> train model (ResNet + pooling)
  => results_refactored/fold_x_backbone_pooling/
       - best.pt
       - report.json
       - test_predictions.csv
```

## 2) Ý nghĩa `src` và `scripts`
- `src/thyus2path/data.py`: xử lý row CSV, group patient, split theo patient.
- `src/thyus2path/dataset.py`: đổi PatientRecord thành tensor bag `[N,C,H,W]`, collate/padding thành batch.
- `src/thyus2path/model.py`: ResNet encoder + pooling (`mean/max/attention`) + classifier.
- `src/thyus2path/engine.py`: train loop, evaluate, save checkpoint/report.
- `src/thyus2path/metrics.py`: tính AUC, AUPRC, accuracy, sensitivity, specificity...

- `scripts/*.py`: điểm chạy thực tế, gọi lại các module lõi trong `src`.

## 3) Input/Output của bước train
Input chính:
- `data/main/main_manifest.csv`
- Cột bắt buộc: `patient_id,batch,local_patient_id,image_name,image_path,label`

Output chính:
- `results_refactored/fold_0_resnet18_mean/best.pt`
- `results_refactored/fold_0_resnet18_mean/report.json`
- `results_refactored/fold_0_resnet18_mean/test_predictions.csv`

## 4) Vì sao split train/val/test ở mức patient?
Vì một bệnh nhân có nhiều ảnh. Nếu split theo ảnh sẽ rò rỉ dữ liệu (cùng bệnh nhân xuất hiện cả train và val/test), metric sẽ ảo.
