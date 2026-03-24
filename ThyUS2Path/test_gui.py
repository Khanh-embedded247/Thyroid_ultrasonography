#!/usr/bin/env python
"""Legacy GUI note.

The old Gradio GUI is kept in legacy/test_gui_legacy.py.
Use scripts/infer_patient.py for the new patient-level pipeline.
"""

from pathlib import Path

if __name__ == "__main__":
    legacy_path = Path(__file__).resolve().parent / "legacy" / "test_gui_legacy.py"
    print(f"Legacy GUI moved to: {legacy_path}")
    print("New pipeline: python scripts/infer_patient.py --help")
