from __future__ import annotations

"""Dataset patient-level và hàm collate cho bag có số ảnh biến thiên.

Mỗi sample = 1 bệnh nhân = nhiều ảnh.
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .data import PatientRecord


class PILToFloatTensorNoNumpy:
    """Chuyển PIL Image -> torch.FloatTensor [C,H,W] mà không dùng numpy.

    Lý do có lớp này:
    - Một số môi trường bị lỗi torch.from_numpy / torchvision.ToTensor.
    - Dùng buffer từ PIL để tránh phụ thuộc numpy trong bước transform.
    """

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")

        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        channels = len(img.getbands())  # RGB -> 3

        # Đọc raw bytes của ảnh và reshape về [H, W, C].
        byte_tensor = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
        tensor = byte_tensor.view(height, width, channels).permute(2, 0, 1).contiguous()

        # Chuẩn hóa về [0, 1] tương đương ToTensor().
        return tensor.float().div(255.0)


class PatientBagDataset(Dataset):
    """Dataset trả về (bag_images, label, patient_id) cho 1 bệnh nhân."""

    def __init__(self, records: Sequence[PatientRecord], image_size: int = 224, augment: bool = False):
        self.records = list(records)
        self.transform = self._build_transform(image_size=image_size, augment=augment)

    @staticmethod
    def _build_transform(image_size: int, augment: bool):
        """Xây transform; augmentation ở mức nhẹ để không phá đặc trưng y khoa."""

        ops = [transforms.Resize((image_size, image_size))]
        if augment:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08),
                ]
            )

        # Không dùng ToTensor() để tránh lỗi from_numpy trong một số env.
        ops.extend(
            [
                PILToFloatTensorNoNumpy(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        images: List[torch.Tensor] = []

        # Đọc toàn bộ ảnh của bệnh nhân và stack thành tensor [N, C, H, W].
        for image_path in record.image_paths:
            path = Path(image_path)
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            images.append(self.transform(image))

        if not images:
            raise RuntimeError(f"Patient {record.patient_id} has no valid images")

        bag = torch.stack(images, dim=0)
        label = torch.tensor(float(record.label), dtype=torch.float32)
        return bag, label, record.patient_id


def patient_bag_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]):
    """Pad các bag có số ảnh khác nhau về cùng chiều trong 1 batch.

    Output:
    - bags: [B, Nmax, C, H, W]
    - mask: [B, Nmax] (True = ảnh hợp lệ)
    - labels: [B]
    - patient_ids: list[str]
    """

    max_instances = max(item[0].shape[0] for item in batch)
    batch_size = len(batch)
    channels, height, width = batch[0][0].shape[1:]

    bags = torch.zeros((batch_size, max_instances, channels, height, width), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_instances), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.float32)
    patient_ids: List[str] = []

    for idx, (bag, label, patient_id) in enumerate(batch):
        n = bag.shape[0]
        bags[idx, :n] = bag
        mask[idx, :n] = True
        labels[idx] = label
        patient_ids.append(patient_id)

    return bags, mask, labels, patient_ids
