from __future__ import annotations

"""Định nghĩa model patient-level cho bài toán benign/malignant.

Luồng chính:
- Mỗi ảnh đi qua CNN backbone để lấy feature.
- Gộp feature các ảnh trong cùng bệnh nhân bằng pooling.
- Classifier dự đoán xác suất ở mức bệnh nhân.
"""

from typing import Literal

import torch
import torch.nn as nn


def _build_resnet(backbone: Literal["resnet18", "resnet34"], pretrained: bool):
    """Khởi tạo backbone và trả về (encoder_bỏ_fc, feature_dim)."""

    if backbone == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(pretrained=pretrained)
    elif backbone == "resnet34":
        from torchvision.models import ResNet34_Weights, resnet34

        try:
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
        except Exception:
            model = resnet34(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    feat_dim = model.fc.in_features
    encoder = nn.Sequential(*list(model.children())[:-1])
    return encoder, feat_dim


class PatientClassifier(nn.Module):
    """Model baseline/MIL-ready với mean, max hoặc attention pooling."""

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34"] = "resnet18",
        pooling: Literal["mean", "max", "attention"] = "mean",
        pretrained: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder, self.feat_dim = _build_resnet(backbone=backbone, pretrained=pretrained)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        if pooling == "attention":
            # Attention 1 tầng để học trọng số từng ảnh trong bag.
            self.attention = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.Tanh(),
                nn.Linear(self.feat_dim // 2, 1),
            )
        else:
            self.attention = None

        self.classifier = nn.Linear(self.feat_dim, 1)

    def _pool(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Gộp feature từ nhiều ảnh thành 1 vector bệnh nhân.

        feats: [B, N, D]
        mask:  [B, N] (True = ảnh hợp lệ)
        """

        if self.pooling == "mean":
            masked = feats * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            return masked.sum(dim=1) / denom

        if self.pooling == "max":
            feats_masked = feats.masked_fill(~mask.unsqueeze(-1), -1e9)
            return feats_masked.max(dim=1).values

        if self.pooling == "attention":
            scores = self.attention(feats).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            return (weights.unsqueeze(-1) * feats).sum(dim=1)

        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward.

        bags: [B, N, C, H, W]
        mask: [B, N]
        return logits: [B]
        """

        bsz, n_ins, channels, height, width = bags.shape

        # Gộp batch và số ảnh để chạy encoder một lần.
        x = bags.view(bsz * n_ins, channels, height, width)
        feats = self.encoder(x).flatten(1)

        # Tách lại theo [B, N, D].
        feats = feats.view(bsz, n_ins, self.feat_dim)

        pooled = self._pool(feats, mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
