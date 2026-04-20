#!/usr/bin/env python
from __future__ import annotations

"""Bước 2: Train baseline patient-level (ResNet + pooling).

Phiên bản này hỗ trợ tự quét 4 cấu hình và tự chọn best config.
Ý tưởng dùng file:
- Bạn tune tham số ngay trong khối DEFAULTS bên dưới.
- Chạy lệnh ngắn: `python scripts/train_patient_baseline.py`
- Không bắt buộc truyền tham số dòng lệnh.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyus2path.data import (
    create_patient_splits,
    filter_records,
    load_patient_ids,
    load_rows_csv,
    rows_to_patient_records,
    save_patient_ids,
)
from thyus2path.dataset import PatientBagDataset, patient_bag_collate
from thyus2path.engine import evaluate, fit, save_json, set_seed
from thyus2path.model import PatientClassifier


# ============================== #
# CẤU HÌNH TRAIN (TUNE TẠI ĐÂY) #
# ============================== #
DEFAULTS = {
    # Data
    "metadata_csv": ROOT / "data" / "main" / "main_manifest.csv",
    "split_dir": None,  # None = tự chia train/val/test khi train
    "fold": 0,
    "k_folds": 5,
    "test_size": 0.1,
    "save_generated_splits_dir": ROOT / "data" / "main" / "splits_auto",

    # Model chung
    "backbone": "resnet18",  # dùng khi không bật auto_sweep
    "pooling": "mean",  # dùng khi không bật auto_sweep
    "image_size": 224,
    "pretrained": True,

    # Train chung
    "batch_size": 4,
    "epochs": 20,  # epochs cho lần train final
    "sweep_epochs": 8,  # epochs cho mỗi cấu hình trong auto sweep
    "lr": 1e-4,  # dùng khi không bật auto_sweep
    "weight_decay": 1e-4,
    "threshold": 0.5,
    "num_workers": 4,
    "seed": 2026,
    "device": "auto",  # auto / cpu / cuda

    # Chế độ tự tìm tham số
    "auto_sweep": True,
    "retrain_best": True,

    # Output
    "output_dir": ROOT / "results_refactored",
}

# 4 cấu hình quét nhanh theo yêu cầu.
SWEEP_CONFIGS = [
    {"name": "A_resnet18_mean_lr1e4", "backbone": "resnet18", "pooling": "mean", "lr": 1e-4},
    {"name": "B_resnet18_mean_lr3e4", "backbone": "resnet18", "pooling": "mean", "lr": 3e-4},
    {"name": "C_resnet18_attention_lr1e4", "backbone": "resnet18", "pooling": "attention", "lr": 1e-4},
    {"name": "D_resnet34_mean_lr1e4", "backbone": "resnet34", "pooling": "mean", "lr": 1e-4},
]


def parse_args() -> argparse.Namespace:
    """Tham số cho quá trình train.

    Lưu ý:
    - Không truyền gì vẫn chạy vì đã có mặc định từ DEFAULTS.
    - Nếu cần, bạn vẫn có thể override tạm bằng CLI.
    """

    parser = argparse.ArgumentParser(description="Train patient-level baseline model")

    parser.add_argument("--metadata-csv", type=Path, default=DEFAULTS["metadata_csv"])
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULTS["split_dir"],
        help="Thư mục split có sẵn. Nếu không truyền hoặc không đủ file thì tự chia tại lúc train.",
    )
    parser.add_argument("--fold", type=int, default=DEFAULTS["fold"])
    parser.add_argument("--k-folds", type=int, default=DEFAULTS["k_folds"], help="Dùng khi auto split tại train")
    parser.add_argument("--test-size", type=float, default=DEFAULTS["test_size"], help="Dùng khi auto split tại train")
    parser.add_argument(
        "--save-generated-splits-dir",
        type=Path,
        default=DEFAULTS["save_generated_splits_dir"],
        help="Nếu auto split, có thể lưu lại split vào thư mục này",
    )

    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default=DEFAULTS["backbone"])
    parser.add_argument("--pooling", choices=["mean", "max", "attention"], default=DEFAULTS["pooling"])
    parser.add_argument("--image-size", type=int, default=DEFAULTS["image_size"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"], help="Số patient / batch")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--sweep-epochs", type=int, default=DEFAULTS["sweep_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Tắt pretrained (mặc định script đang bật pretrained theo DEFAULTS)",
    )

    parser.add_argument(
        "--no-auto-sweep",
        action="store_true",
        help="Tắt quét 4 cấu hình tự động và chỉ train 1 cấu hình backbone/pooling/lr hiện tại",
    )
    parser.add_argument(
        "--no-retrain-best",
        action="store_true",
        help="Khi auto_sweep, không train lại best config full epochs",
    )

    parser.add_argument("--device", type=str, default=DEFAULTS["device"], choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])

    args = parser.parse_args()

    args.pretrained = not args.no_pretrained
    args.auto_sweep = not args.no_auto_sweep
    args.retrain_best = not args.no_retrain_best
    return args


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


def _load_ids_if_exists(path: Path) -> list[str] | None:
    if path.exists():
        return load_patient_ids(path)
    return None


def _load_split_from_dir(split_dir: Path, fold: int):
    """Ưu tiên tên mới, fallback tên cũ."""

    train_ids = _load_ids_if_exists(split_dir / f"fold_{fold}_train_patient_ids.csv")
    val_ids = _load_ids_if_exists(split_dir / f"fold_{fold}_val_patient_ids.csv")
    test_ids = _load_ids_if_exists(split_dir / "test_patient_ids.csv")

    if train_ids is None:
        train_ids = _load_ids_if_exists(split_dir / f"fold_{fold}_train_patients.csv")
    if val_ids is None:
        val_ids = _load_ids_if_exists(split_dir / f"fold_{fold}_val_patients.csv")
    if test_ids is None:
        test_ids = _load_ids_if_exists(split_dir / "test_patients.csv")

    if train_ids is None or val_ids is None or test_ids is None:
        raise FileNotFoundError(f"Split files not found or incomplete in: {split_dir}")

    return train_ids, val_ids, test_ids


def _auto_split(records, fold: int, test_size: float, k_folds: int, seed: int):
    test_ids, folds = create_patient_splits(records, test_size, k_folds, seed)
    if fold < 0 or fold >= len(folds):
        raise ValueError(f"fold={fold} không hợp lệ. k_folds={k_folds}")
    train_ids, val_ids = folds[fold]
    return train_ids, val_ids, test_ids


def _save_generated_splits(out_dir: Path, train_ids, val_ids, test_ids, fold: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_patient_ids(train_ids, out_dir / f"fold_{fold}_train_patient_ids.csv")
    save_patient_ids(val_ids, out_dir / f"fold_{fold}_val_patient_ids.csv")
    save_patient_ids(test_ids, out_dir / "test_patient_ids.csv")


def _print_config(args: argparse.Namespace) -> None:
    """In config train để bạn dễ theo dõi mỗi lần chạy."""

    print("=== TRAIN CONFIG ===")
    print("metadata_csv:", args.metadata_csv)
    print("split_dir:", args.split_dir)
    print("fold:", args.fold)
    print("k_folds:", args.k_folds)
    print("test_size:", args.test_size)
    print("save_generated_splits_dir:", args.save_generated_splits_dir)
    print("backbone:", args.backbone)
    print("pooling:", args.pooling)
    print("image_size:", args.image_size)
    print("pretrained:", args.pretrained)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("sweep_epochs:", args.sweep_epochs)
    print("lr:", args.lr)
    print("weight_decay:", args.weight_decay)
    print("threshold:", args.threshold)
    print("num_workers:", args.num_workers)
    print("seed:", args.seed)
    print("device:", args.device)
    print("auto_sweep:", args.auto_sweep)
    print("retrain_best:", args.retrain_best)
    print("output_dir:", args.output_dir)


def _build_loaders(train_records, val_records, test_records, image_size: int, batch_size: int, num_workers: int):
    """Tạo DataLoader cho train/val/test."""

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


def _run_one_config(
    *,
    cfg_name: str,
    backbone: str,
    pooling: str,
    lr: float,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    split_source: str,
    train_records,
    val_records,
    test_records,
    stage: str,
):
    """Train + evaluate cho một cấu hình, trả về kết quả để so sánh."""

    print("\n----------------------------------------")
    print(f"[RUN] stage={stage} cfg={cfg_name}")
    print(f"[RUN] backbone={backbone} pooling={pooling} lr={lr} epochs={epochs}")

    train_loader, val_loader, test_loader = _build_loaders(
        train_records,
        val_records,
        test_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = PatientClassifier(
        backbone=backbone,
        pooling=pooling,
        pretrained=args.pretrained,
        dropout=0.2,
    ).to(device)

    pos_weight = _compute_pos_weight(train_records)

    run_tag = f"fold_{args.fold}_{stage}_{cfg_name}"
    run_dir = args.output_dir / run_tag
    ckpt_path = run_dir / "best.pt"

    fit_info = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        threshold=args.threshold,
        ckpt_path=ckpt_path,
        meta={
            "stage": stage,
            "cfg_name": cfg_name,
            "backbone": backbone,
            "pooling": pooling,
            "image_size": args.image_size,
            "fold": args.fold,
            "seed": args.seed,
            "split_source": split_source,
            "lr": lr,
            "epochs": epochs,
        },
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    _, val_metrics, _ = evaluate(model, val_loader, criterion, device, args.threshold)
    _, test_metrics, test_rows = evaluate(model, test_loader, criterion, device, args.threshold)

    _save_predictions(test_rows, run_dir / "test_predictions.csv")

    report = {
        "config": {
            "cfg_name": cfg_name,
            "backbone": backbone,
            "pooling": pooling,
            "lr": lr,
            "epochs": epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "threshold": args.threshold,
            "pretrained": args.pretrained,
            "stage": stage,
        },
        "split_source": split_source,
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

    print(f"[OK] cfg={cfg_name} val_auc={val_metrics.get('auc', float('nan')):.6f} test_auc={test_metrics.get('auc', float('nan')):.6f}")

    return {
        "cfg_name": cfg_name,
        "backbone": backbone,
        "pooling": pooling,
        "lr": lr,
        "epochs": epochs,
        "stage": stage,
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_val_auc": float(val_metrics.get("auc", float("nan"))),
        "best_val_auprc": float(val_metrics.get("auprc", float("nan"))),
    }


def _pick_best(sweep_results: list[dict]) -> dict:
    """Chọn best config theo val_auc, nếu hòa thì theo val_auprc."""

    return max(
        sweep_results,
        key=lambda x: (
            float(x.get("best_val_auc", float("-inf"))),
            float(x.get("best_val_auprc", float("-inf"))),
        ),
    )


def main() -> None:
    args = parse_args()
    _print_config(args)
    set_seed(args.seed)

    rows = load_rows_csv(args.metadata_csv)
    records = rows_to_patient_records(rows)
    print("[INFO] records_total:", len(records))

    split_source = "auto_train_time"
    if args.split_dir is not None:
        try:
            train_ids, val_ids, test_ids = _load_split_from_dir(args.split_dir, args.fold)
            split_source = f"from_files:{args.split_dir}"
        except FileNotFoundError:
            train_ids, val_ids, test_ids = _auto_split(records, args.fold, args.test_size, args.k_folds, args.seed)
    else:
        train_ids, val_ids, test_ids = _auto_split(records, args.fold, args.test_size, args.k_folds, args.seed)

    if args.save_generated_splits_dir is not None:
        _save_generated_splits(args.save_generated_splits_dir, train_ids, val_ids, test_ids, args.fold)

    train_records = filter_records(records, train_ids)
    val_records = filter_records(records, val_ids)
    test_records = filter_records(records, test_ids)

    print("[INFO] split_source:", split_source)
    print("[INFO] train_patients:", len(train_records))
    print("[INFO] val_patients:", len(val_records))
    print("[INFO] test_patients:", len(test_records))

    device = _device_from_arg(args.device)
    print("[INFO] device_real:", device)

    if args.auto_sweep:
        print("\n=== AUTO SWEEP: QUÉT 4 CẤU HÌNH ===")
        sweep_results = []

        for idx, cfg in enumerate(SWEEP_CONFIGS, start=1):
            print(f"\n[SWEEP {idx}/{len(SWEEP_CONFIGS)}] {cfg['name']}")
            result = _run_one_config(
                cfg_name=cfg["name"],
                backbone=cfg["backbone"],
                pooling=cfg["pooling"],
                lr=float(cfg["lr"]),
                epochs=args.sweep_epochs,
                args=args,
                device=device,
                split_source=split_source,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage="sweep",
            )
            sweep_results.append(result)

        best_cfg = _pick_best(sweep_results)

        print("\n=== SWEEP RANKING (theo val_auc) ===")
        ranked = sorted(sweep_results, key=lambda x: x["best_val_auc"], reverse=True)
        for i, r in enumerate(ranked, start=1):
            print(
                f"[{i}] {r['cfg_name']} | val_auc={r['best_val_auc']:.6f} | "
                f"val_auprc={r['best_val_auprc']:.6f} | test_auc={r['test_metrics'].get('auc', float('nan')):.6f}"
            )

        print("\n=== BEST CONFIG SAU SWEEP ===")
        print(json.dumps(best_cfg, indent=2, ensure_ascii=False))

        summary = {
            "mode": "auto_sweep",
            "sweep_epochs": args.sweep_epochs,
            "final_epochs": args.epochs,
            "split_source": split_source,
            "sweep_results": sweep_results,
            "best_after_sweep": best_cfg,
        }

        if args.retrain_best:
            print("\n=== RETRAIN BEST CONFIG (FULL EPOCHS) ===")
            final_result = _run_one_config(
                cfg_name=f"best_{best_cfg['cfg_name']}",
                backbone=best_cfg["backbone"],
                pooling=best_cfg["pooling"],
                lr=float(best_cfg["lr"]),
                epochs=args.epochs,
                args=args,
                device=device,
                split_source=split_source,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage="final",
            )
            summary["final_result"] = final_result

            print("\n=== FINAL RESULT ===")
            print(json.dumps(final_result, indent=2, ensure_ascii=False))

        summary_path = args.output_dir / f"fold_{args.fold}_auto_sweep_summary.json"
        save_json(summary, summary_path)
        print("\n[OK] Sweep summary:", summary_path)
        return

    # Chế độ train đơn nếu tắt auto sweep.
    single_name = f"single_{args.backbone}_{args.pooling}_lr{args.lr}"
    single_result = _run_one_config(
        cfg_name=single_name,
        backbone=args.backbone,
        pooling=args.pooling,
        lr=args.lr,
        epochs=args.epochs,
        args=args,
        device=device,
        split_source=split_source,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        stage="single",
    )

    print("\n=== TRAIN DONE (SINGLE MODE) ===")
    print(json.dumps(single_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
