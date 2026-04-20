#!/usr/bin/env python
from __future__ import annotations

"""Bước 4: Suy luận cho 1 bệnh nhân từ thư mục chứa nhiều ảnh."""

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyus2path.model import PatientClassifier


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".Jpg", ".PNG"}


def parse_args() -> argparse.Namespace:
    """Tham số cho bước infer."""

    parser = argparse.ArgumentParser(description="Infer malignancy probability for one patient bag")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True, help="Thư mục chứa toàn bộ ảnh của 1 bệnh nhân")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _device_from_arg(device_arg: str) -> torch.device:
    """Chọn thiết bị chạy theo tham số."""

    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_images(images_dir: Path) -> List[Path]:
    """Lấy danh sách ảnh hợp lệ trong thư mục bệnh nhân."""

    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS])


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)

    # 1) Nạp checkpoint và lấy cấu hình model đã train.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = checkpoint.get("meta", {})

    image_size = int(meta.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 2) Đọc tất cả ảnh của bệnh nhân thành 1 bag.
    image_paths = _collect_images(args.images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.images_dir}")

    bag_tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        bag_tensors.append(transform(image))

    # Tensor đầu vào model có dạng [B, N, C, H, W].
    bag = torch.stack(bag_tensors, dim=0).unsqueeze(0).to(device)
    mask = torch.ones((1, bag.shape[1]), dtype=torch.bool, device=device)

    model = PatientClassifier(
        backbone=meta.get("backbone", "resnet18"),
        pooling=meta.get("pooling", "mean"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 3) Dự đoán xác suất ác tính ở mức bệnh nhân.
    with torch.no_grad():
        logits = model(bag, mask)
        prob = torch.sigmoid(logits).item()

    pred = int(prob >= args.threshold)
    label_text = "malignant" if pred == 1 else "benign"

    print(f"images={len(image_paths)}")
    print(f"prob_malignant={prob:.6f}")
    print(f"pred={pred} ({label_text})")


if __name__ == "__main__":
    main()
