import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.models import resnet34
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import gradio as gr
import glob

# Load labels for verification
labels_df = pd.read_csv('./data/combined/all_labels.csv')
labels_dict = dict(zip(labels_df['patient_name'].astype(str), labels_df['histo_label']))

# Import các class từ script_x.ipynb (copy paste hoặc import nếu có thể)
# Vì đây là file riêng, tôi sẽ copy các class cần thiết

class Config:
    DATA_ROOT = "./data/"
    SPLIT_DIR = "./split/"
    RESULT_DIR = "./results/"
    LOG_DIR = RESULT_DIR + "logs/"
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.9
    GAMMA = 0.1
    K_FOLDS = 5
    TEST_SIZE = 0.1
    SEED = 202203
    C2 = 128
    D = 256
    USE_TENSORBOARD = True
    RECORD_ITER = 10

class IRAM(nn.Module):
    def __init__(self, C1, C2=128, dropout=True, tau=128):
        super(IRAM, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.tau = tau
        self.psi1 = nn.Sequential(
            nn.Conv2d(C1, C2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.psi2 = nn.Sequential(
            nn.Conv2d(C1, C2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.psi3 = nn.Sequential(
            nn.Conv2d(C1, C2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
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
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.attention_c = nn.Linear(D, n_classes)
        
    def forward(self, x):
        x_pooled = self.pool(x).view(x.shape[0], self.L)
        a = self.attention_a(x_pooled)
        b = self.attention_b(x_pooled)
        A = a.mul(b)
        A = self.attention_c(A)
        if not self.training:
            A = A.view(1, 1, -1)
        else:
            A = A.view(self.batch_size, 1, -1)
        return A, x_pooled, x

class ThyNet(nn.Module):
    def __init__(self, C2, D, batch_size, dropout=True, num_cls=2, n_channels=3):
        super(ThyNet, self).__init__()
        feature_extractor = resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])
        self.C1 = self.feature_extractor[-1][-1].conv2.out_channels
        self.L = self.C1
        self.iram = IRAM(C1=self.C1, C2=C2, dropout=dropout, tau=C2)
        self.isam = ISAM(batch_size=batch_size, L=self.L, D=D, dropout=dropout)
        self.classifier = nn.Linear(self.C1, num_cls)
        self.n_channels = n_channels
        self.batch_size = batch_size
        
    def forward(self, x):
        x = x.view(-1, self.n_channels, x.shape[-2], x.shape[-1])
        feat = self.feature_extractor(x)
        M, _ = self.iram(feat)
        A, x_pooled, _ = self.isam(M)
        A = F.softmax(A, dim=2)
        if not self.training:
            x_pooled = x_pooled.view(1, -1, self.C1)
            h = torch.bmm(A, x_pooled).view(1, self.C1)
        else:
            x_pooled = x_pooled.view(self.batch_size, -1, self.C1)
            h = torch.bmm(A, x_pooled).view(self.batch_size, self.C1)
        logits = self.classifier(h)
        return logits

# Transform cho test
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model - now loads all folds for ensemble
def load_models():
    models = []
    for fold in range(1, 6):
        model_path = f"./results/thynet_fold_{fold}.pth"
        if not os.path.exists(model_path):
            return None, f"Model fold {fold} not found!"
        model = ThyNet(C2=128, D=256, batch_size=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        models.append(model)
    return models, "Ensemble of 5 folds loaded successfully!"

# Dự đoán cho một patient
def predict_patient(patient_folder, model):
    # Tìm tất cả ảnh trong folder patient
    image_paths = glob.glob(os.path.join(patient_folder, "*.jpg")) + glob.glob(os.path.join(patient_folder, "*.Jpg"))
    if not image_paths:
        return "No images found in patient folder!"
    
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img = val_transform(img)
            images.append(img)
        except:
            continue
    
    if not images:
        return "Error loading images!"
    
    # Stack images
    images_tensor = torch.stack(images).unsqueeze(0)  # batch_size=1
    
    with torch.no_grad():
        logits = model(images_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_prob = probs[0, 1].item()  # Probability of class 1 (malignant?)
        pred_class = 1 if pred_prob > 0.5 else 0
    
    result = f"Prediction: {'Malignant' if pred_class == 1 else 'Benign'}\n"
    result += f"Probability of Malignant: {pred_prob:.4f}\n"
    result += f"Number of images: {len(images)}"
    
    return result

# Gradio interface
def predict_from_uploaded(files, models):
    images = []
    patient_names = set()
    for f in files:
        try:
            # f can be a tempfile with .name path or a dict-like object depending on Gradio version
            path = f.name if hasattr(f, 'name') else f
            img = Image.open(path).convert('RGB')
            img = val_transform(img)
            images.append(img)
            # Extract patient name from filename (e.g., 0_001.Jpg -> 0)
            filename = os.path.basename(path)
            patient_name = filename.split('_')[0]
            patient_names.add(patient_name)
        except Exception as e:
            print(f"Error loading uploaded file {f}: {e}")
            continue

    if not images:
        return "No valid images uploaded!"

    # Assume all images are from the same patient
    patient_name = list(patient_names)[0] if patient_names else "Unknown"
    true_label = labels_dict.get(patient_name, "Unknown")

    images_tensor = torch.stack(images).unsqueeze(0)  # batch_size=1

    # Ensemble prediction
    all_logits = []
    for model in models:
        with torch.no_grad():
            logits = model(images_tensor)
            all_logits.append(logits)
    
    # Average logits
    avg_logits = torch.mean(torch.stack(all_logits), dim=0)
    probs = torch.softmax(avg_logits, dim=1)
    pred_prob = probs[0, 1].item()
    pred_class = 1 if pred_prob > 0.5 else 0

    result = f"**Patient Name: {patient_name}**\n\n"
    if true_label != "Unknown":
        result += f"**True Label: {'Malignant' if true_label == 1 else 'Benign'}**\n\n"
    result += f"**Prediction (Ensemble): {'Malignant' if pred_class == 1 else 'Benign'}**\n\n"
    result += f"**Probability of Malignant: {pred_prob:.4f}**\n\n"
    result += f"**Number of images: {len(images)}**"
    return result


def predict_batch_from_files(files, models):
    # Group images by patient
    patient_images = {}
    for f in files:
        try:
            path = f.name if hasattr(f, 'name') else f
            filename = os.path.basename(path)
            patient_name = filename.split('_')[0]
            if patient_name not in patient_images:
                patient_images[patient_name] = []
            img = Image.open(path).convert('RGB')
            img = val_transform(img)
            patient_images[patient_name].append(img)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    if not patient_images:
        return "No valid images uploaded!", None
    
    results = []
    for patient_name, images in patient_images.items():
        if not images:
            continue
        images_tensor = torch.stack(images).unsqueeze(0)  # batch_size=1
        
        # Ensemble
        all_logits = []
        for model in models:
            with torch.no_grad():
                logits = model(images_tensor)
                all_logits.append(logits)
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        probs = torch.softmax(avg_logits, dim=1)
        pred_prob = probs[0, 1].item()
        pred_class = 1 if pred_prob > 0.5 else 0
        
        true_label = labels_dict.get(patient_name, "Unknown")
        results.append({
            "Patient": patient_name,
            "True Label": 'Malignant' if true_label == 1 else ('Benign' if true_label == 0 else 'Unknown'),
            "Prediction": 'Malignant' if pred_class == 1 else 'Benign',
            "Prob_Malignant": f"{pred_prob:.4f}",
            "Num_Images": len(images)
        })
    
    df = pd.DataFrame(results)
    summary = f"Processed {len(results)} patients.\n"
    summary += f"Malignant: {sum(1 for r in results if r['Prediction'] == 'Malignant')}\n"
    summary += f"Benign: {sum(1 for r in results if r['Prediction'] == 'Benign')}"
    
    return summary, df


def gradio_predict_uploaded(uploaded_files):
    models, msg = load_models()
    if models is None:
        return msg
    return predict_from_uploaded(uploaded_files, models)


def gradio_predict_batch(uploaded_files):
    models, msg = load_models()
    if models is None:
        return msg, None
    summary, df = predict_batch_from_files(uploaded_files, models)
    return summary, df


# Create interface with tabs
with gr.Blocks(title="ThyNet Thyroid Ultrasound Classifier") as iface:
    gr.Markdown("# ThyNet Thyroid Ultrasound Classifier")
    gr.Markdown("**Ensemble Prediction**: Uses all 5 trained models (folds) for robust prediction on new images.")
    
    with gr.Tab("Single Patient Test"):
        gr.Markdown("Upload multiple images for one patient. Supported formats: JPG, PNG.")
        uploaded_files = gr.File(file_count="multiple", label="Upload image files (jpg, png)")
        submit_single = gr.Button("Predict")
        output_single = gr.Markdown(label="Prediction Result")
        submit_single.click(gradio_predict_uploaded, inputs=uploaded_files, outputs=output_single)
    
    with gr.Tab("Batch Test (Multiple Patients)"):
        gr.Markdown("Upload multiple images from different patients. Images will be grouped by patient name (from filename). Supported formats: JPG, PNG.")
        batch_files = gr.File(file_count="multiple", label="Upload image files (jpg, png)")
        submit_batch = gr.Button("Predict Batch")
        output_batch_summary = gr.Textbox(label="Summary")
        output_batch_table = gr.Dataframe(label="Detailed Results")
        submit_batch.click(gradio_predict_batch, inputs=batch_files, outputs=[output_batch_summary, output_batch_table])


if __name__ == "__main__":
    iface.launch()