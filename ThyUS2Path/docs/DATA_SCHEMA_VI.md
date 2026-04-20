# Chuẩn Hóa Và Gộp Nhiều Nguồn Dữ Liệu

## Mục tiêu
Đưa mọi nguồn dữ liệu khác cấu trúc về cùng một chuẩn để train patient-level.

## Nguyên tắc cốt lõi
- Đơn vị học là **bệnh nhân**.
- Split theo **patient_id** (không split theo ảnh).
- Nhãn chính để báo cáo luận văn: **pathology**.
- Nhãn yếu (ví dụ TIRADS suy diễn) chỉ dùng cho thí nghiệm mở rộng.

## Chuẩn cột CSV khi gộp
### Cột bắt buộc
- `patient_id`
- `batch`
- `local_patient_id`
- `image_name`
- `image_path`
- `label` (`0` là benign, `1` là malignant)

### Cột phụ (nếu có)
- `dataset_name`
- `label_source`
- `age`
- `sex`
- `tirads`
- `study_id`

Cột phụ có thể có hoặc không. Nếu nguồn không có thì để trống.

## Quy tắc chuẩn hóa nhãn
- Mọi nguồn đều quy về nhị phân:
  - `0`: lành tính
  - `1`: ác tính
- Nếu nguồn dùng ký hiệu khác (ví dụ `benign/malignant`, `B/M`, `2/3/4/5`) thì map trước khi gộp.

## Quy tắc ID bệnh nhân
- `patient_id` phải là ID toàn cục, không trùng giữa nguồn.
- Khuyến nghị: `patient_id = "{source_name}:{local_patient_id}"`.

## Quy tắc gộp nhiều nguồn
- Gộp theo danh sách CSV nguồn.
- Loại trùng theo khóa `(patient_id, image_path)`.
- Nếu trùng ảnh mà nhãn khác nhau thì dừng và báo lỗi.
- Thứ tự cột khi ghi file tổng:
  - Cột bắt buộc trước.
  - Cột phụ sau, theo thứ tự ổn định.

## Ý nghĩa file đầu ra
- `image_level_manifest.csv`: bảng chi tiết ảnh (mỗi dòng 1 ảnh).
- `patient_level_table.csv`: bảng tóm tắt bệnh nhân (mỗi dòng 1 bệnh nhân).
- `patient_splits/test_patient_ids.csv`: danh sách bệnh nhân test.
- `patient_splits/fold_k_train_patient_ids.csv`: danh sách train của fold k.
- `patient_splits/fold_k_val_patient_ids.csv`: danh sách val của fold k.

## Quy trình chuẩn theo thứ tự
1. Quét nguồn:
```bash
python scripts/scan_data_sources.py
```

2. Tạo manifest từng nguồn cần chuyển đổi:
```bash
python scripts/build_folder_binary_manifest.py
python scripts/build_archive_weak_manifest.py
```

3. Tạo bộ strong (khuyến nghị dùng chính):
```bash
python scripts/prepare_dataset.py \
  --data-root data/source_data \
  --output-dir data/ready/strong_labels
```

4. Tạo bộ mixed (thí nghiệm mở rộng):
```bash
python scripts/prepare_dataset.py \
  --data-root data/source_data \
  --extra-metadata-csv data/interim/manifests/classification_image_manifest.csv \
  --extra-metadata-csv data/interim/manifests/archive_weak_manifest.csv \
  --output-dir data/ready/mixed_labels
```

## Trường hợp nguồn có tuổi/giới tính/TIRADS
- Giữ các cột đó trong manifest nguồn.
- Khi gộp, file tổng vẫn giữ lại các cột phụ.
- Baseline hiện tại chỉ dùng ảnh + nhãn để train.
- Các cột phụ dùng cho phân tích sau (bias theo tuổi/giới, ablation, mô hình đa modal).
