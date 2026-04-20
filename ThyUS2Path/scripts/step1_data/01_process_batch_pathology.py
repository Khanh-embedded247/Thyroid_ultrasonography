#!/usr/bin/env python
from __future__ import annotations

# Script này dùng để xử lý nguồn dữ liệu pathology dạng batch (batch1_image, batch2_image).
# Mục tiêu: mỗi batch -> 1 file CSV riêng trong data/preprocess_data.
# Điểm đặc biệt: chạy KHÔNG cần truyền tham số, script tự quét source_data và xử lý tất cả batch hợp lệ.

import argparse
import csv
from pathlib import Path
from typing import Dict, List

# ROOT là thư mục gốc của project ThyUS2Path.
ROOT = Path(__file__).resolve().parents[2]

# Đuôi ảnh hợp lệ.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".Jpg", ".JPG", ".PNG", ".JPEG"}


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Mặc định:
    - source-root: data/source_data
    - output-dir : data/preprocess_data
    """

    parser = argparse.ArgumentParser(description="Step 1.1 - Process all pathology batches to per-source CSV")

    # Thư mục chứa các nguồn raw.
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "data" / "source_data",
        help="Thư mục chứa dữ liệu nguồn (mặc định: data/source_data)",
    )

    # Thư mục lưu CSV tiền xử lý từng nguồn.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "preprocess_data",
        help="Thư mục lưu CSV tiền xử lý (mặc định: data/preprocess_data)",
    )

    # Nếu muốn chỉ xử lý 1 batch cụ thể thì truyền tên batch vào đây.
    parser.add_argument(
        "--only-batch",
        type=str,
        default="",
        help="Tên batch cần xử lý riêng (ví dụ: batch1_image). Mặc định: xử lý tất cả batch hợp lệ.",
    )

    return parser.parse_args()


def _find_pathology_batches(source_root: Path, only_batch: str) -> List[Path]:
    """Tìm các batch có cấu trúc hợp lệ: có dataset/ và có file csv chứa chữ 'label'."""

    batches: List[Path] = []

    # Duyệt từng thư mục con trong source_root.
    for item in sorted(source_root.iterdir()):
        # Bỏ qua nếu không phải thư mục.
        if not item.is_dir():
            continue

        # Nếu user chọn only-batch thì chỉ nhận đúng tên đó.
        if only_batch and item.name != only_batch:
            continue

        # Chỉ xử lý thư mục có chữ batch trong tên.
        if "batch" not in item.name.lower():
            continue

        # Kiểm tra thư mục ảnh.
        dataset_dir = item / "dataset"
        if not dataset_dir.exists():
            continue

        # Kiểm tra file label csv.
        label_candidates = [p for p in item.glob("*.csv") if "label" in p.name.lower()]
        if not label_candidates:
            continue

        # Nếu pass hết điều kiện thì nhận là 1 nguồn batch hợp lệ.
        batches.append(item)

    return batches


def _pick_label_csv(batch_dir: Path) -> Path:
    """Chọn file label CSV phù hợp nhất trong 1 batch."""

    # Lấy tất cả csv chứa chữ label.
    label_files = [p for p in batch_dir.glob("*.csv") if "label" in p.name.lower()]
    if not label_files:
        raise FileNotFoundError(f"Không tìm thấy label CSV trong {batch_dir}")

    # Ưu tiên file có hậu tố _label để nhất quán.
    preferred = [p for p in label_files if "_label" in p.name.lower()]
    return sorted(preferred or label_files)[0]


def _read_label_map(label_csv: Path) -> Dict[str, int]:
    """Đọc map local_patient_id -> label từ CSV label của batch."""

    mapping: Dict[str, int] = {}

    # Mở CSV bằng utf-8-sig để tránh lỗi BOM.
    with label_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # Kiểm tra cột bắt buộc.
        if "patient_name" not in (reader.fieldnames or []) or "histo_label" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {label_csv} cần có cột patient_name và histo_label")

        # Đọc từng dòng.
        for row in reader:
            local_id = str(row["patient_name"]).strip()
            mapping[local_id] = int(row["histo_label"])

    return mapping


def _iter_images(dataset_dir: Path):
    """Duyệt tất cả ảnh hợp lệ trong thư mục dataset."""

    for p in sorted(dataset_dir.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS:
            yield p


def _extract_local_id(image_name: str) -> str:
    """Tách local patient id từ tên ảnh kiểu 23_001.jpg -> 23."""

    return image_name.split("_")[0]


def _process_one_batch(batch_dir: Path, output_dir: Path) -> Path:
    """Xử lý 1 batch và ghi ra 1 file CSV."""

    source_name = batch_dir.name
    dataset_dir = batch_dir / "dataset"

    # Chọn file label cho batch.
    label_csv = _pick_label_csv(batch_dir)

    # Đọc map label.
    label_map = _read_label_map(label_csv)

    # Chuẩn bị đường dẫn output.
    output_csv = output_dir / f"{source_name}_preprocessed.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Danh sách dòng CSV.
    rows = []

    # Đếm số ảnh bị bỏ qua do không có nhãn.
    skipped_without_label = 0

    # Duyệt từng ảnh.
    for image_path in _iter_images(dataset_dir):
        # Lấy local patient id từ tên ảnh.
        local_id = _extract_local_id(image_path.name)

        # Nếu local id không có trong label map thì bỏ qua.
        if local_id not in label_map:
            skipped_without_label += 1
            continue

        # Lấy nhãn 0/1.
        label = int(label_map[local_id])

        # Tạo global patient_id để tránh trùng giữa các batch.
        patient_id = f"{source_name}:{local_id}"

        # Thêm 1 dòng metadata ảnh.
        rows.append(
            {
                "patient_id": patient_id,
                "batch": source_name,
                "local_patient_id": local_id,
                "image_name": image_path.name,
                "image_path": str(image_path.resolve()),
                "label": str(label),
                "dataset_name": source_name,
                "label_source": "pathology",
            }
        )

    # Khai báo thứ tự cột CSV.
    fieldnames = [
        "patient_id",
        "batch",
        "local_patient_id",
        "image_name",
        "image_path",
        "label",
        "dataset_name",
        "label_source",
    ]

    # Ghi CSV ra file.
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Tính thống kê để in log chi tiết.
    n_patients = len({r["patient_id"] for r in rows})
    n_images = len(rows)
    n_pos = sum(int(r["label"]) for r in rows)
    n_neg = n_images - n_pos

    # In log chi tiết của batch hiện tại.
    print("\n=== BATCH REPORT ===")
    print("[OK] Batch:", source_name)
    print("[OK] Label CSV:", label_csv)
    print("[OK] Output CSV:", output_csv)
    print("[OK] Images used:", n_images)
    print("[OK] Patients:", n_patients)
    print("[OK] Label rows benign(0):", n_neg, "malignant(1):", n_pos)
    print("[INFO] Skipped (không có label):", skipped_without_label)

    return output_csv


def main() -> None:
    """Hàm chính: quét các batch hợp lệ và xử lý lần lượt từng batch."""

    args = parse_args()

    # Chuẩn hóa path tuyệt đối.
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()

    # Kiểm tra source_root tồn tại.
    if not source_root.exists():
        raise FileNotFoundError(f"Không tìm thấy source-root: {source_root}")

    # Tìm danh sách batch pathology cần xử lý.
    batches = _find_pathology_batches(source_root, args.only_batch.strip())

    # Nếu không tìm thấy batch nào thì báo lỗi ngay.
    if not batches:
        raise RuntimeError(
            "Không tìm thấy batch pathology hợp lệ. "
            "Kiểm tra source-root hoặc dùng --only-batch đúng tên thư mục."
        )

    # In thông tin tổng quan trước khi chạy.
    print("=== STEP 1.1 - PROCESS BATCH PATHOLOGY ===")
    print("[INFO] Source root:", source_root)
    print("[INFO] Output dir:", output_dir)
    print("[INFO] Found batches:", [b.name for b in batches])

    # Chạy xử lý từng batch và lưu danh sách file output.
    generated_files: List[Path] = []
    for batch_dir in batches:
        generated_files.append(_process_one_batch(batch_dir, output_dir))

    # In tổng kết cuối cùng.
    print("\n=== STEP 1.1 SUMMARY ===")
    print("[OK] Total batch processed:", len(generated_files))
    for p in generated_files:
        print("[OK] Generated:", p)


if __name__ == "__main__":
    main()
