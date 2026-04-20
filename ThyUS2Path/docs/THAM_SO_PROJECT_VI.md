# THAM_SO_PROJECT_VI

Tài liệu này dành cho người mới học Python/AI, giải thích tham số theo mẫu:
- Khái niệm
- Lý do chọn
- Dẫn chứng

## 1) Nhóm tham số dữ liệu (scripts/train_patient_baseline.py)

### `metadata_csv`
- Khái niệm: file CSV chính chứa metadata ảnh (đường dẫn ảnh, patient_id, nhãn).
- Lý do chọn: đây là điểm vào thống nhất để model đọc dữ liệu từ nhiều nguồn.
- Dẫn chứng: mặc định ở `data/main/main_manifest.csv`.

### `split_dir`
- Khái niệm: thư mục chứa danh sách bệnh nhân train/val/test đã chia sẵn.
- Lý do chọn: nếu có split cố định thì dùng lại để đảm bảo tái lập kết quả.
- Dẫn chứng: nếu `None` thì code tự chia tại lúc train.

### `fold`
- Khái niệm: chỉ số fold đang chạy (0..k-1).
- Lý do chọn: mỗi fold là một lần validation khác nhau để giảm may rủi.
- Dẫn chứng: mặc định `fold=0`, khi chạy đủ 5 fold cần chạy 0,1,2,3,4.

### `k_folds`
- Khái niệm: số phần chia trong cross-validation.
- Lý do chọn: `k=5` là mức cân bằng giữa độ tin cậy và chi phí chạy.
- Dẫn chứng: paper cũng dùng 5-fold cross-validation.

### `test_size`
- Khái niệm: tỉ lệ bệnh nhân giữ ra làm test holdout.
- Lý do chọn: `0.1` (10%) là tỉ lệ phổ biến và khớp paper protocol nội bộ batch1.
- Dẫn chứng: paper ghi rõ 90% train-val, 10% internal test.

### `seed`
- Khái niệm: số ngẫu nhiên cố định để chia dữ liệu và train tái lập được.
- Lý do chọn: có seed thì chạy lại cho kết quả gần nhau hơn.
- Dẫn chứng: script gọi `set_seed(seed)` trước train.

## 2) Nhóm tham số mô hình

### `backbone` (`resnet18` / `resnet34`)
- Khái niệm: mạng trích đặc trưng ảnh.
- Lý do chọn: ResNet là backbone đã được kiểm chứng rộng rãi cho ảnh y khoa.
- Dẫn chứng: kết quả sweep cho thấy `resnet34` cho `val_auc` cao nhất ở fold 0.

### `pooling` (`mean` / `max` / `attention`)
- Khái niệm: cách gộp nhiều ảnh của 1 bệnh nhân thành 1 vector bệnh nhân.
- Lý do chọn:
  - `mean`: ổn định, dễ hội tụ.
  - `max`: nhạy với ảnh nghi ngờ nhất.
  - `attention`: học trọng số ảnh quan trọng.
- Dẫn chứng: script sweep so sánh trực tiếp 3 kiểu pooling.

### `image_size`
- Khái niệm: kích thước ảnh đầu vào model.
- Lý do chọn: `224` là chuẩn phổ biến cho ResNet pretrained, nhanh hơn 256.
- Dẫn chứng: code transform resize về `(224,224)`.

### `pretrained`
- Khái niệm: dùng trọng số học trước (ImageNet) hay không.
- Lý do chọn: dữ liệu y khoa thường không quá lớn, pretrained giúp hội tụ tốt hơn.
- Dẫn chứng: mặc định `True`.

### `dropout` (trong model)
- Khái niệm: kỹ thuật bỏ ngẫu nhiên một phần neuron khi train để giảm overfit.
- Lý do chọn: 0.2 là mức vừa phải, không làm mất quá nhiều thông tin.
- Dẫn chứng: khai báo `dropout=0.2` trong `PatientClassifier`.

## 3) Nhóm tham số huấn luyện

### `batch_size`
- Khái niệm: số bệnh nhân xử lý cùng lúc trong một bước cập nhật.
- Lý do chọn: bài toán patient-level có bag độ dài khác nhau, batch lớn dễ hết VRAM.
- Dẫn chứng: dataset hiện có `max_images_per_patient=31`, nên `batch_size=4` là mức an toàn.

### `lr` (learning rate)
- Khái niệm: tốc độ cập nhật trọng số.
- Lý do chọn: với AdamW + pretrained, `1e-4` thường ổn định.
- Dẫn chứng: sweep của bạn cho thấy `lr=3e-4` cho kết quả kém hơn rõ (`val_auc` thấp hơn).

### `weight_decay`
- Khái niệm: hệ số regularization (phạt trọng số lớn).
- Lý do chọn: giảm overfit khi dữ liệu nhiễu/không đồng đều.
- Dẫn chứng: dùng AdamW với `weight_decay=1e-4` trong engine.

### `epochs`
- Khái niệm: số vòng lặp qua toàn bộ tập train.
- Lý do chọn: 20 là mức thực dụng để cân bằng thời gian và chất lượng.
- Dẫn chứng: run final 20 epoch cải thiện so với sweep 8 epoch.

### `threshold`
- Khái niệm: ngưỡng đổi xác suất thành nhãn 0/1.
- Lý do chọn: mặc định 0.5 dễ hiểu; có thể tune theo mục tiêu sensitivity/specificity.
- Dẫn chứng: metrics trong code được tính theo threshold này.

### `num_workers`
- Khái niệm: số tiến trình đọc dữ liệu song song cho DataLoader.
- Lý do chọn: tăng tốc đọc ảnh; 4 là mức ổn định trên máy hiện tại.
- Dẫn chứng: script đang dùng `num_workers=4`.

### `pos_weight`
- Khái niệm: trọng số lớp dương trong BCE loss để xử lý lệch lớp.
- Lý do chọn: giúp model không nghiêng quá về lớp nhiều mẫu hơn.
- Dẫn chứng: code tự tính theo tỉ lệ âm/dương từ train_records.

## 4) Nhóm tham số tìm kiếm tham số (tự quét)

### `auto_sweep`
- Khái niệm: bật/tắt quét nhanh nhiều cấu hình.
- Lý do chọn: tự động tìm cấu hình tốt mà không phải sửa tay nhiều lần.
- Dẫn chứng: script đã quét 4 cấu hình và lưu summary JSON.

### `sweep_epochs`
- Khái niệm: số epoch ngắn cho mỗi cấu hình trong giai đoạn quét.
- Lý do chọn: 8 để xếp hạng nhanh, giảm thời gian.
- Dẫn chứng: `fold_0_auto_sweep_summary.json` ghi `sweep_epochs=8`.

### `retrain_best`
- Khái niệm: sau khi quét, train lại cấu hình tốt nhất bằng `epochs` đầy đủ.
- Lý do chọn: quét nhanh chỉ để chọn, train full mới là kết quả chốt.
- Dẫn chứng: summary có `final_result` với 20 epoch.

## 5) Tham số evaluate/infer

### Evaluate (`scripts/evaluate_patient.py`)
- `checkpoint`: mô hình đã train.
- `metadata_csv`: dữ liệu ảnh đầu vào để đánh giá.
- `patient_ids_csv`: danh sách bệnh nhân cần đánh giá.
- `threshold`: ngưỡng phân loại.

### Infer (`scripts/infer_patient.py`)
- `checkpoint`: mô hình đã train.
- `images_dir`: thư mục ảnh của 1 bệnh nhân.
- `threshold`: ngưỡng phân loại.
- `device`: chạy CPU hay GPU.

## 6) Trạng thái source `archive` hiện tại: có đang vào train không?

- CÓ, hiện tại đang được gộp vào `main_manifest.csv`.
- Dẫn chứng: `main_manifest_report.json` có file `archive_preprocessed.csv` với `rows=349`.

## 7) Quyết định dùng `archive` cho bài toán benign/malignant

- Kết quả chính luận văn (pathology-based): KHÔNG nên trộn trực tiếp archive vào train chính.
- Lý do: archive suy nhãn từ TIRADS là nhãn gián tiếp, không phải pathology ground truth.
- Cách dùng phù hợp:
  - Thí nghiệm phụ (ablation).
  - Pretrain đặc trưng, rồi fine-tune bằng pathology labels.

## 8) Nên làm gì tiếp: chạy lại sweep hay full train?

Khuyến nghị thực tế ngay bây giờ:
1. KHÔNG cần sweep lại fold 0 (đã có kết quả tốt và rõ ràng).
2. Chốt cấu hình tốt nhất hiện tại.
3. Chạy full train cho fold 1..4 (giữ cùng tham số) để lấy mean ± std trên 5 fold.
4. Sau đó mới quyết định có cần sweep lại sâu hơn hay không.

## 9) Từ điển nhanh thuật ngữ

- `Hyperparameter`: tham số do người đặt trước khi train (lr, batch_size, epochs, ...).
- `Backbone`: mạng trích đặc trưng ảnh.
- `Pooling`: phép gộp nhiều ảnh của 1 bệnh nhân.
- `Cross-validation`: chia nhiều fold để đánh giá ổn định.
- `Holdout test`: tập test giữ riêng, không dùng để chọn model.
- `Pretrain`: học trước trên dữ liệu phụ.
- `Fine-tune`: tinh chỉnh lại trên dữ liệu mục tiêu chính.

