#!/usr/bin/env python
"""Wrapper entrypoint for refactored patient-level training pipeline."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "scripts" / "train_patient_baseline.py"), run_name="__main__")
