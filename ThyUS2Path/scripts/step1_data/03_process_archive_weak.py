#!/usr/bin/env python
from __future__ import annotations

# Script này xử lý nguồn archive (XML + JPG) và suy nhãn từ TIRADS.
# Đầu ra: 1 file CSV riêng trong data/preprocess_data.
# Chạy mặc định KHÔNG cần truyền tham số.
# Lưu ý: nhãn này là NHÃN YẾU (weak label), không phải pathology chuẩn.

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

# ROOT là thư mục gốc project.
ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh với giá trị mặc định."""

    parser = argparse.ArgumentParser(description="Step 1.3 - Process archive weak source to per-source CSV")

    # Thư mục archive mặc định.
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "data" / "source_data" / "archive",
        help="Thư mục archive chứa XML/JPG",
    )

    # Tên nguồn dùng cho patient_id và tên file output.
    parser.add_argument(
        "--source-name",
        type=str,
        default="archive",
        help="Tên nguồn",
    )

    # Thư mục output tiền xử lý.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "preprocess_data",
        help="Thư mục lưu CSV tiền xử lý",
    )

    # Nếu bật cờ này, ca không suy ra được nhãn sẽ bị bỏ qua ngay.
    parser.add_argument("--skip-unlabeled", action="store_true", help="Bỏ các ca không suy ra được label")

    return parser.parse_args()


def _extract_text(root: ET.Element, tag: str) -> str:
    """Đọc text của 1 tag trong XML, nếu không có thì trả chuỗi rỗng."""

    node = root.find(tag)
    return (node.text or "").strip() if node is not None else ""


def tirads_to_binary_label(tirads: str) -> int | None:
    """Quy đổi TIRADS -> nhãn nhị phân.

    - TIRADS 2/3 -> 0
    - TIRADS 4/5 -> 1
    - Không rõ     -> None
    """

    # Chuẩn hóa text để dễ kiểm tra.
    t = tirads.lower().replace(" ", "")

    # Không có giá trị thì trả None.
    if not t:
        return None

    # Nhóm nguy cơ thấp.
    if t.startswith("2") or t.startswith("3"):
        return 0

    # Nhóm nghi ngờ cao.
    if t.startswith("4") or t.startswith("5"):
        return 1

    # Trường hợp còn lại không map được.
    return None


def main() -> None:
    """Hàm chính xử lý archive weak."""

    args = parse_args()

    # Chuẩn hóa path tuyệt đối.
    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Kiểm tra thư mục archive tồn tại.
    if not archive_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy archive-dir: {archive_dir}")

    # Chuẩn bị đường dẫn output CSV.
    output_csv = output_dir / f"{args.source_name}_preprocessed.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách XML case.
    xml_files = sorted(archive_dir.glob("*.xml"))

    # Danh sách rows output.
    rows = []

    # Đếm các ca bị bỏ qua.
    skipped_cases = 0

    # In log mở đầu.
    print("=== STEP 1.3 - PROCESS ARCHIVE WEAK ===")
    print("[INFO] archive_dir:", archive_dir)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] xml_files_found:", len(xml_files))

    # Duyệt từng XML case.
    for xml_path in xml_files:
        # Parse nội dung XML.
        root = ET.fromstring(xml_path.read_text(encoding="utf-8", errors="ignore"))

        # Lấy case number và tirads.
        case_number = _extract_text(root, "number")
        tirads = _extract_text(root, "tirads")

        # Quy đổi sang nhãn nhị phân.
        label = tirads_to_binary_label(tirads)

        # Nếu bật skip-unlabeled và không có nhãn thì bỏ qua.
        if label is None and args.skip_unlabeled:
            skipped_cases += 1
            continue

        # Nếu thiếu case number thì bỏ qua.
        if not case_number:
            skipped_cases += 1
            continue

        # Tìm ảnh theo pattern <case_number>_*.jpg.
        image_files = sorted(archive_dir.glob(f"{case_number}_*.jpg"))

        # Nếu không có ảnh thì bỏ qua.
        if not image_files:
            skipped_cases += 1
            continue

        # Nếu không map được nhãn thì bỏ qua.
        if label is None:
            skipped_cases += 1
            continue

        # Tạo global patient id.
        patient_id = f"{args.source_name}:{case_number}"

        # Thêm row cho từng ảnh của ca.
        for image_path in image_files:
            rows.append(
                {
                    "patient_id": patient_id,
                    "batch": f"{args.source_name}_tirads_weak",
                    "local_patient_id": case_number,
                    "image_name": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "label": str(label),
                    "dataset_name": args.source_name,
                    "label_source": "tirads_weak",
                    "tirads": tirads,
                }
            )

    # Khai báo cột output.
    fieldnames = [
        "patient_id",
        "batch",
        "local_patient_id",
        "image_name",
        "image_path",
        "label",
        "dataset_name",
        "label_source",
        "tirads",
    ]

    # Ghi CSV output.
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Tính thống kê tổng quan.
    n_patients = len({r["patient_id"] for r in rows})
    n_rows = len(rows)
    n_pos = sum(int(r["label"]) for r in rows)
    n_neg = n_rows - n_pos

    # In report chi tiết.
    print("\n=== STEP 1.3 SUMMARY ===")
    print("[OK] Output CSV:", output_csv)
    print("[OK] Rows:", n_rows)
    print("[OK] Patients:", n_patients)
    print("[OK] Label rows benign(0):", n_neg, "malignant(1):", n_pos)
    print("[INFO] Skipped cases:", skipped_cases)
    print("[WARN] Đây là nhãn yếu từ TIRADS, không phải pathology")


if __name__ == "__main__":
    main()
