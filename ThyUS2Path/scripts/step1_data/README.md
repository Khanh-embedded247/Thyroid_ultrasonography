# STEP 1 - XỬ LÝ DATA (CHẠY TỪNG BƯỚC, KHÔNG CẦN THAM SỐ)

## Bạn chỉ cần chạy lần lượt 4 lệnh này
```bash
python scripts/step1_data/01_process_batch_pathology.py
python scripts/step1_data/02_process_folder_binary.py
python scripts/step1_data/03_process_archive_weak.py
python scripts/step1_data/04_merge_to_main.py
```

## Ý nghĩa
- `01`: xử lý tất cả batch pathology hợp lệ trong `data/source_data/*`.
- `02`: xử lý nguồn `classification_image` (benign/malignant).
- `03`: xử lý nguồn `archive` (nhãn yếu từ TIRADS).
- `04`: gộp tất cả CSV trong `data/preprocess_data/` thành `data/main/main_manifest.csv`.

## Kết quả sau bước 1
- CSV từng nguồn: `data/preprocess_data/*.csv`
- CSV chính để train: `data/main/main_manifest.csv`
- Báo cáo thống kê: `data/main/main_manifest_report.json`

## Nếu muốn bỏ nguồn nào đó khi gộp
```bash
python scripts/step1_data/04_merge_to_main.py \
  --input-csv data/preprocess_data/batch1_image_preprocessed.csv \
  --input-csv data/preprocess_data/batch2_image_preprocessed.csv
```
