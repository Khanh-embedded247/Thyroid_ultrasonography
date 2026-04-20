#!/usr/bin/env python
from __future__ import annotations

# Script này gộp các CSV tiền xử lý trong data/preprocess_data
# thành 1 file CSV chính trong data/main để dùng cho train.
# Chạy mặc định KHÔNG cần tham số.

import argparse
import json
from pathlib import Path
from typing import List

import sys

# ROOT là thư mục gốc project.
ROOT = Path(__file__).resolve().parents[2]

# Đưa src vào PYTHONPATH để import module trong project.
sys.path.insert(0, str(ROOT / "src"))

# Import các hàm xử lý dữ liệu dùng lại từ core module.
from thyus2path.data import (
    load_rows_csv,
    merge_and_deduplicate_rows,
    rows_to_patient_records,
    save_rows_csv,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh (có mặc định đầy đủ)."""

    parser = argparse.ArgumentParser(description="Step 1.4 - Merge per-source preprocessed CSV to one main CSV")

    # Thư mục chứa CSV tiền xử lý từng nguồn.
    parser.add_argument(
        "--preprocess-dir",
        type=Path,
        default=ROOT / "data" / "preprocess_data",
        help="Thư mục chứa CSV tiền xử lý từng nguồn",
    )

    # Có thể truyền danh sách CSV cụ thể muốn gộp.
    parser.add_argument(
        "--input-csv",
        type=Path,
        action="append",
        default=None,
        help="CSV nguồn muốn gộp (truyền nhiều lần). Nếu có thì ưu tiên danh sách này.",
    )

    # File CSV chính đầu ra.
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "main" / "main_manifest.csv",
        help="CSV chính để đưa vào train",
    )

    # File report JSON đầu ra.
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "data" / "main" / "main_manifest_report.json",
        help="Báo cáo thống kê sau gộp",
    )

    return parser.parse_args()


def _collect_csv_files(preprocess_dir: Path, input_csv: List[Path] | None) -> List[Path]:
    """Lấy danh sách file CSV đầu vào để gộp."""

    # Nếu user truyền danh sách input-csv thì dùng danh sách đó.
    if input_csv:
        csv_files = [p.resolve() for p in input_csv]
    else:
        # Nếu không truyền thì lấy toàn bộ *.csv trong preprocess_dir.
        csv_files = sorted(preprocess_dir.glob("*.csv"))

    # Nếu không có file nào thì báo lỗi.
    if not csv_files:
        raise RuntimeError("Không có CSV đầu vào để gộp")

    # Kiểm tra file tồn tại thực tế.
    missing = [p for p in csv_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Một số input CSV không tồn tại: {missing}")

    return csv_files


def main() -> None:
    """Hàm chính: đọc nhiều CSV nguồn -> gộp -> ghi 1 CSV chính + 1 report."""

    args = parse_args()

    # Chuẩn hóa path tuyệt đối.
    preprocess_dir = args.preprocess_dir.resolve()
    output_csv = args.output_csv.resolve()
    report_json = args.report_json.resolve()

    # Kiểm tra thư mục preprocess tồn tại.
    if not preprocess_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy preprocess-dir: {preprocess_dir}")

    # Lấy danh sách file CSV đầu vào.
    csv_files = _collect_csv_files(preprocess_dir, args.input_csv)

    # In log mở đầu.
    print("=== STEP 1.4 - MERGE TO MAIN ===")
    print("[INFO] preprocess_dir:", preprocess_dir)
    print("[INFO] output_csv:", output_csv)
    print("[INFO] report_json:", report_json)
    print("[INFO] input_csv_count:", len(csv_files))
    for i, p in enumerate(csv_files, start=1):
        print(f"[INFO] input_csv[{i}]={p}")

    # Danh sách nhóm rows từ từng file.
    row_groups = []

    # Báo cáo từng file nguồn.
    source_report = []

    # Đọc từng CSV nguồn.
    for csv_path in csv_files:
        rows = load_rows_csv(csv_path)
        row_groups.append(rows)
        source_report.append({"file": str(csv_path), "rows": len(rows)})
        print(f"[INFO] loaded_rows file={csv_path.name} rows={len(rows)}")

    # Gộp và loại trùng theo logic trong core module.
    merged_rows = merge_and_deduplicate_rows(*row_groups)

    # Nếu không còn row hợp lệ sau gộp thì báo lỗi.
    if not merged_rows:
        raise RuntimeError("Không có row hợp lệ sau khi gộp")

    # Tạo thư mục output nếu chưa có.
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Ghi CSV chính đầu ra.
    save_rows_csv(merged_rows, output_csv)

    # Gom theo patient để tính thống kê patient-level.
    records = rows_to_patient_records(merged_rows)

    # Tính thống kê tổng quan dataset.
    summary = summarize_records(records)

    # Tạo report JSON.
    report = {
        "input_files": source_report,
        "summary": {
            "images_total": summary.images_total,
            "patients_total": summary.patients_total,
            "benign_total": summary.benign_total,
            "malignant_total": summary.malignant_total,
            "min_images_per_patient": summary.min_images_per_patient,
            "max_images_per_patient": summary.max_images_per_patient,
            "avg_images_per_patient": summary.avg_images_per_patient,
        },
        "output_csv": str(output_csv),
    }

    # Ghi file report JSON.
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # In report chi tiết.
    print("\n=== STEP 1.4 SUMMARY ===")
    for item in source_report:
        print("[OK] Source file:", item["file"], "rows=", item["rows"])
    print("[OK] Output CSV:", output_csv)
    print("[OK] Images:", summary.images_total)
    print("[OK] Patients:", summary.patients_total)
    print("[OK] benign:", summary.benign_total, "malignant:", summary.malignant_total)
    print("[OK] Report JSON:", report_json)


if __name__ == "__main__":
    main()
