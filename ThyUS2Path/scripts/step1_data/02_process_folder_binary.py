#!/usr/bin/env python
from __future__ import annotations

# Script này xử lý nguồn dữ liệu dạng thư mục lớp:
# - benign/
# - malignant/
# Đầu ra: 1 file CSV riêng trong data/preprocess_data.
# Chạy mặc định KHÔNG cần truyền tham số.

import argparse
import csv
from pathlib import Path
from typing import Dict, List

# ROOT là thư mục gốc project.
ROOT = Path(__file__).resolve().parents[2]

# Đuôi ảnh hợp lệ.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".Jpg"}


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh (đều có mặc định)."""

    parser = argparse.ArgumentParser(description="Step 1.2 - Process folder-binary source to per-source CSV")

    # Thư mục nguồn mặc định.
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "source_data" / "classification_image",
        help="Thư mục nguồn chứa 2 lớp ảnh (mặc định: data/source_data/classification_image)",
    )

    # Tên nguồn để tạo patient_id và tên file output.
    parser.add_argument(
        "--source-name",
        type=str,
        default="classification_image",
        help="Tên nguồn (mặc định: classification_image)",
    )

    # Tên thư mục class âm tính.
    parser.add_argument("--negative-dir", type=str, default="benign", help="Tên thư mục lớp 0")

    # Tên thư mục class dương tính.
    parser.add_argument("--positive-dir", type=str, default="malignant", help="Tên thư mục lớp 1")

    # Cách suy ra patient_id khi nguồn không có patient_id rõ ràng.
    parser.add_argument(
        "--patient-id-mode",
        choices=["filename_prefix", "filename_stem", "parent_folder"],
        default="filename_stem",
        help="Cách suy ra patient_id",
    )

    # Thư mục output tiền xử lý.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "preprocess_data",
        help="Thư mục lưu CSV tiền xử lý",
    )

    return parser.parse_args()


def _iter_images(root: Path) -> List[Path]:
    """Lấy danh sách ảnh hợp lệ trong thư mục class."""

    files: List[Path] = []

    # Duyệt đệ quy toàn bộ file trong root.
    for p in sorted(root.rglob("*")):
        # Chỉ nhận file có đuôi ảnh hợp lệ.
        if p.is_file() and p.suffix in IMAGE_EXTS:
            files.append(p)

    return files


def _infer_local_patient_id(image_path: Path, mode: str) -> str:
    """Suy ra local patient id từ tên file hoặc thư mục cha."""

    stem = image_path.stem

    # Nếu mode là filename_stem: dùng toàn bộ stem.
    if mode == "filename_stem":
        return stem

    # Nếu mode là parent_folder: dùng tên thư mục cha.
    if mode == "parent_folder":
        return image_path.parent.name

    # Nếu mode là filename_prefix: lấy phần trước dấu gạch dưới.
    parts = stem.split("_")
    return parts[0] if parts else stem


def main() -> None:
    """Hàm chính xử lý nguồn folder binary."""

    args = parse_args()

    # Chuẩn hóa path tuyệt đối.
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Tên nguồn dùng để prefix patient_id.
    source_name = args.source_name.strip() or source_dir.name

    # Xác định 2 thư mục class.
    neg_dir = source_dir / args.negative_dir
    pos_dir = source_dir / args.positive_dir

    # Kiểm tra tồn tại 2 thư mục class.
    if not neg_dir.exists() or not pos_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục lớp. Cần có: {neg_dir} và {pos_dir}. "
            "Nếu tên khác, truyền --negative-dir/--positive-dir"
        )

    # Tạo đường dẫn output CSV.
    output_csv = output_dir / f"{source_name}_preprocessed.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Danh sách rows output.
    rows: List[Dict[str, str]] = []

    # Map patient -> label để phát hiện xung đột nhãn cùng patient.
    patient_label_map: Dict[str, int] = {}

    def add_rows(class_dir: Path, label: int) -> None:
        """Thêm ảnh của 1 class vào rows."""

        # Lấy toàn bộ ảnh trong class_dir.
        images = _iter_images(class_dir)

        # In log số ảnh tìm thấy trong class.
        class_name = class_dir.name
        print(f"[INFO] Class={class_name} label={label} images_found={len(images)}")

        # Duyệt từng ảnh để tạo row.
        for image_path in images:
            # Suy ra local patient id.
            local_id = _infer_local_patient_id(image_path, mode=args.patient_id_mode)

            # Tạo global patient id để tránh trùng giữa nguồn.
            patient_id = f"{source_name}:{local_id}"

            # Nếu đã tồn tại patient mà nhãn khác -> báo lỗi dữ liệu.
            if patient_id in patient_label_map and patient_label_map[patient_id] != label:
                raise ValueError(
                    f"Xung đột nhãn cho patient_id={patient_id}. "
                    "Hãy đổi --patient-id-mode hoặc kiểm tra dữ liệu nguồn."
                )

            # Ghi nhận nhãn patient.
            patient_label_map[patient_id] = label

            # Thêm row metadata.
            rows.append(
                {
                    "patient_id": patient_id,
                    "batch": f"folder_binary:{source_name}",
                    "local_patient_id": local_id,
                    "image_name": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "label": str(label),
                    "dataset_name": source_name,
                    "label_source": "folder_binary",
                }
            )

    # In log mở đầu.
    print("=== STEP 1.2 - PROCESS FOLDER BINARY ===")
    print("[INFO] Source dir:", source_dir)
    print("[INFO] Output dir:", output_dir)
    print("[INFO] source_name:", source_name)
    print("[INFO] patient_id_mode:", args.patient_id_mode)

    # Thêm dữ liệu lớp 0.
    add_rows(neg_dir, 0)

    # Thêm dữ liệu lớp 1.
    add_rows(pos_dir, 1)

    # Nếu không có ảnh hợp lệ thì dừng.
    if not rows:
        raise RuntimeError("Không tìm thấy ảnh hợp lệ trong source-dir")

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

    # Ghi file CSV output.
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Thống kê tổng quan.
    n_patients = len(patient_label_map)
    n_images = len(rows)
    n_pos_patients = sum(1 for v in patient_label_map.values() if v == 1)
    n_neg_patients = n_patients - n_pos_patients

    # In report chi tiết.
    print("\n=== STEP 1.2 SUMMARY ===")
    print("[OK] Output CSV:", output_csv)
    print("[OK] Images:", n_images)
    print("[OK] Patients:", n_patients)
    print("[OK] Patient labels benign(0):", n_neg_patients, "malignant(1):", n_pos_patients)


if __name__ == "__main__":
    main()
