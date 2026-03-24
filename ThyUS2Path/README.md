# ThyUS2Path (Refactored)

Patient-level thyroid ultrasound diagnosis pipeline (benign/malignant) with fast baseline + MIL-ready structure.

## Why this refactor
- Original code mixed experiments and production scripts.
- Patient-level dataset logic needed a strict pipeline to avoid leakage.
- Batch ID collisions were possible (`batch1 patient 23` vs `batch2 patient 23`).

This refactor fixes the structure and introduces a reproducible workflow.

## New structure

```text
ThyUS2Path/
  src/thyus2path/
    data.py                 # build metadata, patient grouping, split by patient
    dataset.py              # patient bag dataset + collate
    model.py                # resnet18/34 + mean/max/attention pooling
    engine.py               # train/eval loop
    metrics.py              # patient-level metrics (AUC, AUPRC, sensitivity...)
  scripts/
    prepare_dataset.py      # build patient metadata + split files
    train_patient_baseline.py
    evaluate_patient.py
    infer_patient.py
  legacy/                   # old notebooks/scripts kept unchanged
  configs/train_baseline.yaml
  train.py                  # wrapper -> scripts/train_patient_baseline.py
  infer.py                  # wrapper -> scripts/infer_patient.py
```

## Critical data fix
Patient ID is now global:

```text
patient_id = "{batch_name}:{local_patient_id}"
```

This prevents cross-batch collisions and label conflicts.

## Quickstart

### 1) Build metadata and patient splits

```bash
cd ThyUS2Path
python scripts/prepare_dataset.py \
  --data-root ./data \
  --output-dir ./data/processed \
  --test-size 0.1 \
  --k-folds 5 \
  --seed 2026
```

Outputs:
- `data/processed/patient_images.csv`
- `data/processed/patients.csv`
- `data/processed/splits/test_patients.csv`
- `data/processed/splits/fold_{i}_train_patients.csv`
- `data/processed/splits/fold_{i}_val_patients.csv`

### 2) Train baseline (patient-level)

```bash
python train.py \
  --metadata-csv ./data/processed/patient_images.csv \
  --split-dir ./data/processed/splits \
  --fold 0 \
  --backbone resnet18 \
  --pooling mean \
  --epochs 20 \
  --batch-size 4 \
  --threshold 0.5
```

### 3) Evaluate checkpoint

```bash
python scripts/evaluate_patient.py \
  --checkpoint ./results_refactored/fold_0_resnet18_mean/best.pt \
  --metadata-csv ./data/processed/patient_images.csv \
  --patient-ids-csv ./data/processed/splits/test_patients.csv
```

### 4) Infer one patient folder

```bash
python infer.py \
  --checkpoint ./results_refactored/fold_0_resnet18_mean/best.pt \
  --images-dir ./data/batch1_image/dataset_patient_x
```

## Metrics (patient-level)
- AUROC
- AUPRC
- Accuracy
- Sensitivity (Recall)
- Specificity
- F1
- Confusion components (TP/TN/FP/FN)

## Legacy
Old files are kept in `legacy/`:
- `train_legacy.py`
- `infer_legacy.py`
- `test_gui_legacy.py`
- `script_legacy.py`
- `script_legacy.ipynb`
- `script_x_legacy.ipynb`
