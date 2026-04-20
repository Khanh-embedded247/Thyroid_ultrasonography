# Bước 1 - Chuẩn Hóa Dữ Liệu

## Mục tiêu
Đưa nhiều nguồn dữ liệu khác nhau về cùng 1 chuẩn để train patient-level.

## Kết quả đầu ra chuẩn
- `data/ready/strong_labels/image_level_manifest.csv`
- `data/ready/strong_labels/patient_level_table.csv`
- `data/ready/strong_labels/patient_splits/*`

## Lệnh nhanh
```bash
python scripts/scan_data_sources.py
python scripts/build_folder_binary_manifest.py
python scripts/build_archive_weak_manifest.py
python scripts/prepare_dataset.py --data-root data/source_data --output-dir data/ready/strong_labels
```

## Nếu muốn gộp thêm nguồn mở rộng
```bash
python scripts/prepare_dataset.py \
  --data-root data/source_data \
  --extra-metadata-csv data/interim/manifests/classification_image_manifest.csv \
  --extra-metadata-csv data/interim/manifests/archive_weak_manifest.csv \
  --output-dir data/ready/mixed_labels
```

## Ghi chú
- `strong_labels` là bộ dùng chính cho báo cáo luận văn.
- `mixed_labels` là bộ dùng cho thí nghiệm mở rộng.
- Chi tiết schema và quy tắc gộp xem `docs/DATA_SCHEMA_VI.md`.
