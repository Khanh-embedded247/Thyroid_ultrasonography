# Refactor Notes

## What changed
- Moved old experiments/scripts/notebooks into `legacy/`.
- Added modular patient-level training pipeline in `src/thyus2path/`.
- Added CLI scripts for:
  - `scripts/prepare_dataset.py`
  - `scripts/train_patient_baseline.py`
  - `scripts/evaluate_patient.py`
  - `scripts/infer_patient.py`

## Critical data fix
- Patient ID now uses `batch_name:local_patient_id` to avoid collisions between batches.
- This prevents label conflicts like `batch1:23` vs `batch2:23`.
