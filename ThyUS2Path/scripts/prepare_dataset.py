#!/usr/bin/env python
from __future__ import annotations

"""Bước 1: Chuẩn hóa dữ liệu về patient-level và tạo split train/val/test.

Chạy file này trước khi train.
"""

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyus2path.data import (
    build_patient_image_rows,
    create_patient_splits,
    rows_to_patient_records,
    save_patient_ids,
    save_patient_table,
    save_rows_csv,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    """Định nghĩa tham số CLI cho bước chuẩn bị dữ liệu."""

    parser = argparse.ArgumentParser(description="Build patient-level metadata and split files")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="Thư mục dữ liệu thô")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed", help="Thư mục output")
    parser.add_argument("--test-size", type=float, default=0.1, help="Tỉ lệ test patient-level")
    parser.add_argument("--k-folds", type=int, default=5, help="Số fold cho train/val")
    parser.add_argument("--seed", type=int, default=2026, help="Seed để tái lập kết quả")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A) Quét dữ liệu thô và tạo metadata mức ảnh.
    rows = build_patient_image_rows(args.data_root)
    metadata_csv = args.output_dir / "patient_images.csv"
    save_rows_csv(rows, metadata_csv)

    # B) Gom lại thành cấu trúc patient-level (mỗi patient có nhiều ảnh).
    records = rows_to_patient_records(rows)
    save_patient_table(records, args.output_dir / "patients.csv")

    # C) Chia tập theo patient_id để tránh leakage.
    summary = summarize_records(records)
    test_ids, folds = create_patient_splits(records, args.test_size, args.k_folds, args.seed)

    # D) Lưu danh sách patient_id cho test và từng fold train/val.
    split_dir = args.output_dir / "splits"
    save_patient_ids(test_ids, split_dir / "test_patients.csv")
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        save_patient_ids(train_ids, split_dir / f"fold_{fold_idx}_train_patients.csv")
        save_patient_ids(val_ids, split_dir / f"fold_{fold_idx}_val_patients.csv")

    # E) Tạo report để kiểm tra nhanh dữ liệu sau xử lý.
    report = {
        "summary": {
            "images_total": summary.images_total,
            "patients_total": summary.patients_total,
            "benign_total": summary.benign_total,
            "malignant_total": summary.malignant_total,
            "min_images_per_patient": summary.min_images_per_patient,
            "max_images_per_patient": summary.max_images_per_patient,
            "avg_images_per_patient": summary.avg_images_per_patient,
        },
        "test_patients": len(test_ids),
        "k_folds": args.k_folds,
        "seed": args.seed,
        "metadata_csv": str(metadata_csv),
        "split_dir": str(split_dir),
    }

    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[OK] Metadata created:", metadata_csv)
    print("[OK] Split files:", split_dir)
    print("[OK] Patients:", summary.patients_total, "Images:", summary.images_total)


if __name__ == "__main__":
    main()
