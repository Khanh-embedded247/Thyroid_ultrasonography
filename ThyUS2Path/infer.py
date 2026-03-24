#!/usr/bin/env python
"""Wrapper entrypoint for refactored patient-level inference."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "scripts" / "infer_patient.py"), run_name="__main__")
