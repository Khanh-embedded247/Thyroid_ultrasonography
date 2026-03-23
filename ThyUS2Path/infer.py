#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image

# =========================
# 1️⃣ Định nghĩa mô hình ThyNet
# =========================
class IRAM(nn.Module):
    def __init__(self, C1, C2=128, dropout=True, tau=128):
        super(IRAM, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.tau = tau

        def block():
            layers = [nn.Conv2d(C1, C2, kernel_size=3, padding=1), nn.ReLU()]
            if dropout: layers.append(nn.Dropout(0.25))
            return nn.Sequential(*layers)

        self.psi1 = block()
        self.psi2 = block()
        self.psi3 = block()
        self.psi4 = nn.Sequential(
            nn.Linear(C2, C1),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )

    def forward(self, x):
        x_psi1 = self.psi1(x).view(-1, self.C2)
        x_ds = F.interpolate(x, scale_factor=0.5)
        x_psi2 = self.psi2(x_ds).view(self.C2, -1)
        x_psi3 = self.psi3(x_ds).view(-1, self.C2)
        psi12 = F.softmax(torch.mm(x_psi1, x_psi2) / self.tau, dim=1)
        psi123 = torch.mm(psi12, x_psi3)
        psi1234 = self.psi4(psi123)
        M = psi1234.view(x.shape) + x
        return M, x


class ISAM(nn.Module):
    def __init__(self, batch_size, L, D=256, n_classes=1, dropout=True):
        super(ISAM, self).__init__()
        self.L = L
        self.D = D
        self.batch_size = batch_size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def block(act):
            seq = [nn.Linear(L, D), act()]
            if dropout: seq.append(nn.Dropout(0.25))
            return nn.Sequential(*seq)

        self.attention_a = block(nn.Tanh)
        self.attention_b = block(nn.Sigmoid)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        x_pooled = self.pool(x).view(x.shape[0], self.L)
        a = self.attention_a(x_pooled)
        b = self.attention_b(x_pooled)
        A = self.attention_c(a * b)
        A = F.softmax(A, dim=1)
        return A, x_pooled, x


class ThyNet(nn.Module):
    def __init__(self, C2=128, D=256, batch_size=1, num_cls=2):
        super(ThyNet, self).__init__()
        feature_extractor = resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])
        self.C1 = self.feature_extractor[-1][-1].conv2.out_channels

        self.iram = IRAM(C1=self.C1, C2=C2, dropout=True, tau=C2)
        self.isam = ISAM(batch_size=batch_size, L=self.C1, D=D, dropout=True)
        self.classifier = nn.Linear(self.C1, num_cls)

    def forward(self, x):
        feat = self.feature_extractor(x)
        M, _ = self.iram(feat)
        A, x_pooled, _ = self.isam(M)
        h = torch.bmm(A.view(1, 1, -1), x_pooled.view(1, -1, self.C1)).view(1, self.C1)
        logits = self.classifier(h)
        return logits


# =========================
# 2️⃣ Hàm dự đoán
# =========================
def predict(image_path, model_path, threshold=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model = ThyNet(C2=128, D=256, batch_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        prob_class1 = probs[:, -1].item()

    pred_label = int(prob_class1 > threshold)

    print(f"✅ Prediction logits: {logits.cpu().numpy().squeeze()}")
    print(f"✅ Probability (class=1): {prob_class1:.4f}")
    print(f"✅ Predicted label (> {threshold}): {pred_label}")
    return prob_class1, pred_label


# =========================
# 3️⃣ Chạy ví dụ
# =========================
if __name__ == "__main__":
    image_path = "/home/khanh247/Documents/Thac_si/Paper/deep_learning/ThyUS2Path/data/batch2_image/dataset/5_001_173920.Jpg"
    model_path = "./result/thynet_fold_5.pth"
    predict(image_path, model_path,0.4)
