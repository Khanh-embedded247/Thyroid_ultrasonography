#!/usr/bin/env python
from __future__ import annotations

"""Bước 2: Train baseline patient-level (ResNet + pooling)."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyus2path.data import filter_records, load_patient_ids, load_rows_csv, rows_to_patient_records
from thyus2path.dataset import PatientBagDataset, patient_bag_collate
from thyus2path.engine import evaluate, fit, save_json, set_seed
from thyus2path.model import PatientClassifier


def parse_args() -> argparse.Namespace:
    """Tham số cho quá trình train."""

    parser = argparse.ArgumentParser(description="Train patient-level baseline model")
    parser.add_argument("--metadata-csv", type=Path, default=ROOT / "data" / "processed" / "patient_images.csv")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pooling", choices=["mean", "max", "attention"], default="mean")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4, help="Số patient / batch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_refactored")
    return parser.parse_args()


def _device_from_arg(device_arg: str) -> torch.device:
    """Chọn thiết bị chạy theo tham số."""

    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_pos_weight(records):
    """Tính pos_weight cho BCE nhằm giảm lệch lớp."""

    n_pos = sum(1 for r in records if r.label == 1)
    n_neg = sum(1 for r in records if r.label == 0)
    if n_pos == 0:
        return 1.0
    return max(float(n_neg) / float(n_pos), 1e-6)


def _save_predictions(rows, output_csv: Path):
    """Lưu dự đoán patient-level ra CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label", "prob", "pred"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # 1) Đọc metadata và dựng lại danh sách bệnh nhân.
    rows = load_rows_csv(args.metadata_csv)
    records = rows_to_patient_records(rows)

    # 2) Đọc split patient-level theo fold.
    train_ids = load_patient_ids(args.split_dir / f"fold_{args.fold}_train_patients.csv")
    val_ids = load_patient_ids(args.split_dir / f"fold_{args.fold}_val_patients.csv")
    test_ids = load_patient_ids(args.split_dir / "test_patients.csv")

    train_records = filter_records(records, train_ids)
    val_records = filter_records(records, val_ids)
    test_records = filter_records(records, test_ids)

    # 3) Tạo dataset dạng bag (1 patient = nhiều ảnh).
    train_ds = PatientBagDataset(train_records, image_size=args.image_size, augment=True)
    val_ds = PatientBagDataset(val_records, image_size=args.image_size, augment=False)
    test_ds = PatientBagDataset(test_records, image_size=args.image_size, augment=False)

    # 4) Tạo DataLoader.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=patient_bag_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=patient_bag_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=patient_bag_collate,
    )

    device = _device_from_arg(args.device)

    # 5) Khởi tạo model.
    model = PatientClassifier(
        backbone=args.backbone,
        pooling=args.pooling,
        pretrained=args.pretrained,
        dropout=0.2,
    ).to(device)

    pos_weight = _compute_pos_weight(train_records)
    run_dir = args.output_dir / f"fold_{args.fold}_{args.backbone}_{args.pooling}"
    ckpt_path = run_dir / "best.pt"

    # 6) Train và lưu checkpoint tốt nhất theo val AUC.
    fit_info = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        threshold=args.threshold,
        ckpt_path=ckpt_path,
        meta={
            "backbone": args.backbone,
            "pooling": args.pooling,
            "image_size": args.image_size,
            "fold": args.fold,
            "seed": args.seed,
        },
    )

    # 7) Đánh giá trên test bằng checkpoint tốt nhất.
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    _, test_metrics, test_rows = evaluate(model, test_loader, criterion, device, args.threshold)

    _save_predictions(test_rows, run_dir / "test_predictions.csv")

    report = {
        "config": vars(args),
        "pos_weight": pos_weight,
        "fit": fit_info,
        "best_val_metrics": checkpoint.get("best_val_metrics", {}),
        "test_metrics": test_metrics,
        "train_patients": len(train_records),
        "val_patients": len(val_records),
        "test_patients": len(test_records),
    }
    save_json(report, run_dir / "report.json")

    print("[OK] Run dir:", run_dir)
    print("[OK] Best checkpoint:", ckpt_path)
    print("[OK] Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
