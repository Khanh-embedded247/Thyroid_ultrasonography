#!/usr/bin/env python
from __future__ import annotations

"""Train chính thức 5-fold cho luận văn (không trộn nhãn yếu archive vào kết quả chính).

Mục tiêu script này:
1) Tự tạo metadata train chính từ `data/main/main_manifest.csv` bằng cách bỏ `tirads_weak`.
2) Chạy full train cho các fold 0..4 với cùng 1 bộ tham số tốt nhất.
3) Tổng hợp báo cáo mean ± std để đưa thẳng vào luận văn.

Lưu ý:
- Script này KHÔNG quét tham số (không sweep). Bạn đã sweep xong ở script baseline.
- Archive vẫn được giữ riêng để pretrain/thử nghiệm phụ, không dùng cho kết quả chính.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyus2path.data import (
    PatientRecord,
    create_patient_splits,
    filter_records,
    load_rows_csv,
    rows_to_patient_records,
    save_patient_ids,
    save_rows_csv,
)
from thyus2path.dataset import PatientBagDataset, patient_bag_collate
from thyus2path.engine import evaluate, fit, save_json, set_seed
from thyus2path.model import PatientClassifier


# =============================== #
# CẤU HÌNH CHÍNH THỨC (TUNE Ở ĐÂY)
# =============================== #
DEFAULTS = {
    # Nguồn metadata đã gộp tất cả source.
    "base_metadata_csv": ROOT / "data" / "main" / "main_manifest.csv",
    # Metadata train chính sẽ tự tạo từ base bằng cách bỏ archive (tirads_weak).
    "train_metadata_csv": ROOT / "data" / "main" / "main_manifest_train_main.csv",
    # Chế độ lọc nguồn train:
    # - train_main: pathology + folder_binary (không archive)
    # - pathology_only: chỉ pathology (batch1 + batch2)
    # - all_sources: giữ cả archive (không khuyến nghị cho kết quả chính)
    "train_profile": "train_main",

    # Split và folds.
    "k_folds": 5,
    "test_size": 0.1,
    "fold_start": 0,
    "fold_end": 4,
    "save_splits_dir": ROOT / "data" / "main" / "splits_main",

    # Bộ tham số tốt nhất đã chốt từ sweep.
    "backbone": "resnet34",
    "pooling": "mean",
    "image_size": 224,
    "pretrained": True,
    "batch_size": 4,
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "threshold": 0.5,
    "num_workers": 4,
    "seed": 2026,
    "device": "auto",

    # Output.
    "output_dir": ROOT / "results_main",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chính thức 5-fold cho luận văn")

    parser.add_argument("--base-metadata-csv", type=Path, default=DEFAULTS["base_metadata_csv"])
    parser.add_argument("--train-metadata-csv", type=Path, default=DEFAULTS["train_metadata_csv"])
    parser.add_argument(
        "--train-profile",
        type=str,
        default=DEFAULTS["train_profile"],
        choices=["train_main", "pathology_only", "all_sources"],
    )

    parser.add_argument("--k-folds", type=int, default=DEFAULTS["k_folds"])
    parser.add_argument("--test-size", type=float, default=DEFAULTS["test_size"])
    parser.add_argument("--fold-start", type=int, default=DEFAULTS["fold_start"])
    parser.add_argument("--fold-end", type=int, default=DEFAULTS["fold_end"])
    parser.add_argument("--save-splits-dir", type=Path, default=DEFAULTS["save_splits_dir"])

    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default=DEFAULTS["backbone"])
    parser.add_argument("--pooling", choices=["mean", "max", "attention"], default=DEFAULTS["pooling"])
    parser.add_argument("--image-size", type=int, default=DEFAULTS["image_size"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULTS["device"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])

    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    args.pretrained = not args.no_pretrained
    return args


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_by_key(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def _build_train_manifest(base_csv: Path, out_csv: Path, profile: str) -> Dict[str, object]:
    """Tạo metadata train chính theo profile đã chọn."""

    all_rows = load_rows_csv(base_csv)
    if not all_rows:
        raise RuntimeError(f"Không có dữ liệu trong: {base_csv}")

    if profile == "train_main":
        # Kết quả chính: dùng pathology + folder_binary, bỏ nhãn yếu archive.
        allow_sources = {"pathology", "folder_binary"}
        rows = [r for r in all_rows if str(r.get("label_source", "")).strip() in allow_sources]
    elif profile == "pathology_only":
        # Chỉ dùng nhãn pathology (batch1 + batch2).
        rows = [r for r in all_rows if str(r.get("label_source", "")).strip() == "pathology"]
    elif profile == "all_sources":
        rows = list(all_rows)
    else:
        raise ValueError(f"Profile không hỗ trợ: {profile}")

    if not rows:
        raise RuntimeError(f"Sau khi lọc profile={profile} thì không còn dòng dữ liệu nào")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_rows_csv(rows, out_csv)

    return {
        "base_csv": str(base_csv),
        "train_csv": str(out_csv),
        "profile": profile,
        "rows_total_before": len(all_rows),
        "rows_total_after": len(rows),
        "label_source_before": _count_by_key(all_rows, "label_source"),
        "label_source_after": _count_by_key(rows, "label_source"),
        "dataset_name_after": _count_by_key(rows, "dataset_name"),
    }


def _compute_pos_weight(records: Sequence[PatientRecord]) -> float:
    n_pos = sum(1 for r in records if r.label == 1)
    n_neg = sum(1 for r in records if r.label == 0)
    if n_pos == 0:
        return 1.0
    return max(float(n_neg) / float(n_pos), 1e-6)


def _build_loaders(
    train_records: Sequence[PatientRecord],
    val_records: Sequence[PatientRecord],
    test_records: Sequence[PatientRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = PatientBagDataset(train_records, image_size=image_size, augment=True)
    val_ds = PatientBagDataset(val_records, image_size=image_size, augment=False)
    test_ds = PatientBagDataset(test_records, image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=patient_bag_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=patient_bag_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=patient_bag_collate,
    )
    return train_loader, val_loader, test_loader


def _save_predictions(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label", "prob", "pred"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_one_fold(
    *,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    train_records: Sequence[PatientRecord],
    val_records: Sequence[PatientRecord],
    test_records: Sequence[PatientRecord],
    manifest_info: Dict[str, object],
) -> Dict[str, object]:
    # Cố định seed theo fold để reproducible.
    set_seed(args.seed + fold)

    print("\n========================================")
    print(f"[FOLD {fold}] Train chính thức")
    print(f"[FOLD {fold}] train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    train_loader, val_loader, test_loader = _build_loaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = PatientClassifier(
        backbone=args.backbone,
        pooling=args.pooling,
        pretrained=args.pretrained,
        dropout=0.2,
    ).to(device)

    pos_weight = _compute_pos_weight(train_records)
    run_dir = args.output_dir / f"fold_{fold}_{args.backbone}_{args.pooling}"
    ckpt_path = run_dir / "best.pt"

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
            "stage": "main_full_train",
            "fold": fold,
            "backbone": args.backbone,
            "pooling": args.pooling,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "manifest_profile": manifest_info["profile"],
        },
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    _, val_metrics, _ = evaluate(model, val_loader, criterion, device, args.threshold)
    _, test_metrics, test_rows = evaluate(model, test_loader, criterion, device, args.threshold)
    _save_predictions(test_rows, run_dir / "test_predictions.csv")

    report = {
        "fold": fold,
        "manifest_info": manifest_info,
        "config": {
            "backbone": args.backbone,
            "pooling": args.pooling,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "threshold": args.threshold,
            "pretrained": args.pretrained,
            "seed": args.seed,
        },
        "pos_weight": pos_weight,
        "fit": fit_info,
        "best_val_metrics": checkpoint.get("best_val_metrics", {}),
        "val_metrics_eval": val_metrics,
        "test_metrics": test_metrics,
        "train_patients": len(train_records),
        "val_patients": len(val_records),
        "test_patients": len(test_records),
    }
    save_json(report, run_dir / "report.json")

    print(
        f"[FOLD {fold}] val_auc={val_metrics.get('auc', float('nan')):.6f} "
        f"test_auc={test_metrics.get('auc', float('nan')):.6f}"
    )

    return {
        "fold": fold,
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(stdev(values))}


def _aggregate_fold_results(fold_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    keys = [
        "auc",
        "auprc",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "loss",
    ]

    agg_val = {}
    agg_test = {}
    for key in keys:
        val_values = [float(r["val_metrics"][key]) for r in fold_results]
        test_values = [float(r["test_metrics"][key]) for r in fold_results]
        agg_val[key] = _mean_std(val_values)
        agg_test[key] = _mean_std(test_values)

    return {
        "val_mean_std": agg_val,
        "test_mean_std": agg_test,
    }


def _print_config(args: argparse.Namespace) -> None:
    print("=== TRAIN MAIN CONFIG ===")
    print("base_metadata_csv:", args.base_metadata_csv)
    print("train_metadata_csv:", args.train_metadata_csv)
    print("train_profile:", args.train_profile)
    print("k_folds:", args.k_folds)
    print("test_size:", args.test_size)
    print("fold_start:", args.fold_start)
    print("fold_end:", args.fold_end)
    print("save_splits_dir:", args.save_splits_dir)
    print("backbone:", args.backbone)
    print("pooling:", args.pooling)
    print("image_size:", args.image_size)
    print("pretrained:", args.pretrained)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("lr:", args.lr)
    print("weight_decay:", args.weight_decay)
    print("threshold:", args.threshold)
    print("num_workers:", args.num_workers)
    print("seed:", args.seed)
    print("device:", args.device)
    print("output_dir:", args.output_dir)


def main() -> None:
    args = parse_args()
    _print_config(args)

    manifest_info = _build_train_manifest(
        base_csv=args.base_metadata_csv.resolve(),
        out_csv=args.train_metadata_csv.resolve(),
        profile=args.train_profile,
    )

    print("\n=== METADATA PROFILE ===")
    print(json.dumps(manifest_info, indent=2, ensure_ascii=False))

    rows = load_rows_csv(args.train_metadata_csv.resolve())
    records = rows_to_patient_records(rows)
    print("[INFO] patients_total:", len(records))

    if args.k_folds < 2:
        raise ValueError("k_folds phải >= 2")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size phải trong khoảng (0,1)")

    # Tạo split một lần để cố định test set và các fold.
    test_ids, folds = create_patient_splits(records, args.test_size, args.k_folds, args.seed)
    print("[INFO] folds_created:", len(folds))
    print("[INFO] test_patients:", len(test_ids))

    if args.fold_start < 0 or args.fold_end >= len(folds) or args.fold_start > args.fold_end:
        raise ValueError(f"Khoảng fold không hợp lệ: {args.fold_start}..{args.fold_end} với k_folds={len(folds)}")

    # Lưu split ra CSV để bạn theo dõi sau.
    args.save_splits_dir.mkdir(parents=True, exist_ok=True)
    save_patient_ids(test_ids, args.save_splits_dir / "test_patient_ids.csv")

    device = _device_from_arg(args.device)
    print("[INFO] device_real:", device)

    all_results: List[Dict[str, object]] = []
    for fold in range(args.fold_start, args.fold_end + 1):
        train_ids, val_ids = folds[fold]
        save_patient_ids(train_ids, args.save_splits_dir / f"fold_{fold}_train_patient_ids.csv")
        save_patient_ids(val_ids, args.save_splits_dir / f"fold_{fold}_val_patient_ids.csv")

        train_records = filter_records(records, train_ids)
        val_records = filter_records(records, val_ids)
        test_records = filter_records(records, test_ids)

        result = _run_one_fold(
            fold=fold,
            args=args,
            device=device,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            manifest_info=manifest_info,
        )
        all_results.append(result)

    summary = {
        "config": {
            "backbone": args.backbone,
            "pooling": args.pooling,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "threshold": args.threshold,
            "k_folds": args.k_folds,
            "test_size": args.test_size,
            "fold_range": [args.fold_start, args.fold_end],
            "seed": args.seed,
        },
        "manifest_info": manifest_info,
        "fold_results": all_results,
        "aggregate": _aggregate_fold_results(all_results),
    }

    summary_path = args.output_dir / "main_5fold_summary.json"
    save_json(summary, summary_path)

    print("\n=== DONE TRAIN MAIN ===")
    print("[OK] Summary:", summary_path)
    print(
        "[OK] test_auc mean±std:",
        f"{summary['aggregate']['test_mean_std']['auc']['mean']:.6f} ± "
        f"{summary['aggregate']['test_mean_std']['auc']['std']:.6f}",
    )


if __name__ == "__main__":
    main()
