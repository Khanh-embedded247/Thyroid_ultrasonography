from __future__ import annotations

"""Engine train/evaluate cho bài toán patient-level."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import classification_metrics


def set_seed(seed: int) -> None:
    """Cố định seed để kết quả ổn định hơn giữa các lần chạy."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch, device: torch.device):
    """Đưa dữ liệu batch lên thiết bị chạy (CPU/GPU)."""

    bags, mask, labels, patient_ids = batch
    return bags.to(device), mask.to(device), labels.to(device), patient_ids


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train đúng 1 epoch và trả về loss trung bình."""

    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        bags, mask, labels, _ = _to_device(batch, device)

        optimizer.zero_grad()
        logits = model(bags, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * bags.size(0)

    return running_loss / max(len(loader.dataset), 1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    """Evaluate model ở mức bệnh nhân.

    Trả về:
    - loss trung bình
    - metrics tổng hợp
    - danh sách dự đoán theo từng patient
    """

    model.eval()
    running_loss = 0.0
    labels_all: List[int] = []
    probs_all: List[float] = []
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            bags, mask, labels, patient_ids = _to_device(batch, device)
            logits = model(bags, mask)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().tolist()

            running_loss += loss.item() * bags.size(0)

            # Lưu kết quả đúng theo patient-level.
            for pid, y, p in zip(patient_ids, labels_cpu, probs):
                labels_all.append(int(y))
                probs_all.append(float(p))
                rows.append(
                    {
                        "patient_id": pid,
                        "label": int(y),
                        "prob": float(p),
                        "pred": int(p >= threshold),
                    }
                )

    loss_avg = running_loss / max(len(loader.dataset), 1)
    metrics = classification_metrics(labels=labels_all, probs=probs_all, threshold=threshold)
    metrics["loss"] = loss_avg
    return loss_avg, metrics, rows


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    threshold: float,
    ckpt_path: Path,
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Train nhiều epoch và lưu checkpoint tốt nhất theo val AUC."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_auc = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_metrics, _ = evaluate(model, val_loader, criterion, device, threshold)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_auc": float(val_metrics["auc"]),
            "val_auprc": float(val_metrics["auprc"]),
            "val_acc": float(val_metrics["accuracy"]),
        }
        history.append(row)

        if val_metrics["auc"] >= best_auc:
            best_auc = val_metrics["auc"]
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_state": model.state_dict(),
                "best_val_metrics": val_metrics,
                "history": history,
                "meta": meta or {},
            }
            torch.save(payload, ckpt_path)

    return {
        "best_val_auc": best_auc,
        "history": history,
        "checkpoint": str(ckpt_path),
    }


def save_json(data: Dict[str, object], output_path: Path) -> None:
    """Lưu dict ra JSON để phục vụ báo cáo/thống kê."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
