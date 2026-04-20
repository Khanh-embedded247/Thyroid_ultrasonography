#!/usr/bin/env python
from __future__ import annotations

"""Bước 3: Đánh giá checkpoint trên một danh sách bệnh nhân."""

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
from thyus2path.engine import evaluate
from thyus2path.model import PatientClassifier


def parse_args() -> argparse.Namespace:
    """Tham số cho bước evaluate.

    Mặc định metadata đã trỏ vào file chính: data/main/main_manifest.csv
    """

    parser = argparse.ArgumentParser(description="Evaluate checkpoint on patient split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=ROOT / "data" / "main" / "main_manifest.csv")
    parser.add_argument("--patient-ids-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results_refactored" / "eval_predictions.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "results_refactored" / "eval_metrics.json")
    return parser.parse_args()


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = checkpoint.get("meta", {})

    model = PatientClassifier(
        backbone=meta.get("backbone", "resnet18"),
        pooling=meta.get("pooling", "mean"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    rows = load_rows_csv(args.metadata_csv)
    records = rows_to_patient_records(rows)
    patient_ids = load_patient_ids(args.patient_ids_csv)
    records_eval = filter_records(records, patient_ids)

    ds = PatientBagDataset(records_eval, image_size=int(meta.get("image_size", 224)), augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=patient_bag_collate)

    criterion = torch.nn.BCEWithLogitsLoss()
    _, metrics, pred_rows = evaluate(model, loader, criterion, device, args.threshold)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label", "prob", "pred"])
        writer.writeheader()
        writer.writerows(pred_rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
