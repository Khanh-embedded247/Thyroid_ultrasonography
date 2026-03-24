# DATA_FLOW

## 1) End-to-end flow (A -> Z)

```text
Raw data (data/batch*_image/)
  └─ dataset/*.Jpg + *label*.csv
        |
        v
scripts/prepare_dataset.py
  ├─ Build image metadata: data/processed/patient_images.csv
  ├─ Build patient table:  data/processed/patients.csv
  └─ Build split files:    data/processed/splits/*.csv
        |
        v
scripts/train_patient_baseline.py
  ├─ Load metadata + split (patient-level)
  ├─ Build bag dataset/dataloader
  ├─ Train model (mean/max/attention pooling)
  └─ Save checkpoint + report
        |
        v
scripts/evaluate_patient.py
  ├─ Load checkpoint
  ├─ Evaluate on patient list CSV
  └─ Save patient-level metrics + predictions
        |
        v
scripts/infer_patient.py
  └─ Infer one patient folder (many images -> one prediction)
```

## 2) Data format after prepare step

### `data/processed/patient_images.csv`
- `patient_id`: global id (`batch_name:local_patient_id`)
- `batch`
- `local_patient_id`
- `image_name`
- `image_path`
- `label`

### `data/processed/patients.csv`
- one row per patient
- `patient_id`, `label`, `num_images`

### `data/processed/splits/*.csv`
- `test_patients.csv`
- `fold_{k}_train_patients.csv`
- `fold_{k}_val_patients.csv`

All split files are patient-level only, so no leakage between train/val/test.

## 3) Code reading order (recommended)

1. `scripts/prepare_dataset.py`
2. `src/thyus2path/data.py`
3. `src/thyus2path/dataset.py`
4. `src/thyus2path/model.py`
5. `src/thyus2path/engine.py`
6. `scripts/train_patient_baseline.py`
7. `scripts/evaluate_patient.py`
8. `scripts/infer_patient.py`
9. `run_all.sh`

## 4) Why this order

- First understand how metadata and split are built.
- Then understand how one patient becomes one model sample.
- Then see model aggregation (mean/max/attention).
- Finally see training/evaluation/inference entrypoints.
