from __future__ import annotations

"""Tiện ích xử lý dữ liệu cho bài toán patient-level.

Nguyên tắc cốt lõi của đề tài:
- Đơn vị học là bệnh nhân (1 bag nhiều ảnh), KHÔNG phải 1 ảnh rời.
- Chia train/val/test theo patient_id để tránh leakage.
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Các đuôi ảnh được chấp nhận khi quét dữ liệu.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".Jpg", ".JPG", ".PNG", ".JPEG"}

# Cột ưu tiên cho metadata mở rộng khi gộp nhiều nguồn khác schema.
OPTIONAL_METADATA_KEYS_PRIORITY = (
    "dataset_name",
    "label_source",
    "age",
    "sex",
    "tirads",
    "study_id",
)


@dataclass(frozen=True)
class PatientRecord:
    """Biểu diễn 1 bệnh nhân: id, nhãn và danh sách ảnh."""

    patient_id: str
    label: int
    image_paths: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetSummary:
    """Thông tin thống kê nhanh của dataset."""

    images_total: int
    patients_total: int
    benign_total: int
    malignant_total: int
    min_images_per_patient: int
    max_images_per_patient: int
    avg_images_per_patient: float


def discover_batch_dirs(data_root: Path) -> List[Path]:
    """Tìm các thư mục batch hợp lệ có dataset/ và file label CSV."""

    batches: List[Path] = []
    for item in sorted(data_root.iterdir()):
        if not item.is_dir():
            continue
        if "batch" not in item.name.lower():
            continue
        dataset_dir = item / "dataset"
        if not dataset_dir.exists():
            continue
        if not any("label" in p.name.lower() and p.suffix == ".csv" for p in item.iterdir()):
            continue
        batches.append(item)
    return batches


def _pick_label_csv(batch_dir: Path) -> Path:
    """Chọn file label CSV trong một batch."""

    label_files = [p for p in batch_dir.glob("*.csv") if "label" in p.name.lower()]
    if not label_files:
        raise FileNotFoundError(f"No label CSV found in {batch_dir}")

    # Ưu tiên file có hậu tố _label để nhất quán.
    preferred = [p for p in label_files if "_label" in p.name.lower()]
    return sorted(preferred or label_files)[0]


def _read_label_map(label_csv: Path) -> Dict[str, int]:
    """Đọc map local_patient_id -> histo_label từ CSV."""

    mapping: Dict[str, int] = {}
    with label_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "patient_name" not in reader.fieldnames or "histo_label" not in reader.fieldnames:
            raise ValueError(f"CSV {label_csv} must contain patient_name and histo_label")
        for row in reader:
            patient = str(row["patient_name"]).strip()
            label = int(row["histo_label"])
            mapping[patient] = label
    return mapping


def _iter_image_paths(dataset_dir: Path) -> Iterable[Path]:
    """Duyệt toàn bộ file ảnh trong thư mục dataset của batch."""

    for p in sorted(dataset_dir.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS:
            yield p


def _extract_local_patient_id(image_name: str) -> str:
    """Tách local patient id từ tên ảnh: `23_001.jpg` -> `23`."""

    return image_name.split("_")[0]


def build_patient_image_rows(data_root: Path) -> List[Dict[str, str]]:
    """Tạo metadata mức ảnh nhưng giữ định danh patient-level chuẩn.

    Mỗi dòng output gồm:
    - patient_id (global): batch:local_id
    - batch
    - local_patient_id
    - image_name
    - image_path
    - label
    """

    rows: List[Dict[str, str]] = []
    for batch_dir in discover_batch_dirs(data_root):
        label_csv = _pick_label_csv(batch_dir)
        label_map = _read_label_map(label_csv)
        batch_name = batch_dir.name
        dataset_dir = batch_dir / "dataset"

        for image_path in _iter_image_paths(dataset_dir):
            local_patient_id = _extract_local_patient_id(image_path.name)
            if local_patient_id not in label_map:
                continue

            label = label_map[local_patient_id]

            # Quan trọng: tránh trùng id giữa batch1:23 và batch2:23.
            patient_id = f"{batch_name}:{local_patient_id}"

            rows.append(
                {
                    "patient_id": patient_id,
                    "batch": batch_name,
                    "local_patient_id": local_patient_id,
                    "image_name": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "label": str(label),
                    "dataset_name": batch_name,
                    "label_source": "pathology",
                }
            )
    return rows


def _ordered_metadata_fieldnames(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Sắp thứ tự cột metadata: cột bắt buộc trước, cột phụ sau."""

    required = list(REQUIRED_METADATA_KEYS)
    present_optional = set()
    for row in rows:
        for key in row.keys():
            if key not in REQUIRED_METADATA_KEYS:
                present_optional.add(key)

    ordered_optional = [k for k in OPTIONAL_METADATA_KEYS_PRIORITY if k in present_optional]
    ordered_optional.extend(sorted([k for k in present_optional if k not in OPTIONAL_METADATA_KEYS_PRIORITY]))
    return required + ordered_optional


def save_rows_csv(rows: Sequence[Dict[str, str]], output_csv: Path) -> None:
    """Lưu metadata mức ảnh ra CSV (hỗ trợ cột mở rộng)."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ordered_metadata_fieldnames(rows)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_rows_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Đọc CSV metadata mức ảnh."""

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_to_patient_records(rows: Sequence[Dict[str, str]]) -> List[PatientRecord]:
    """Gom từ mức ảnh sang mức bệnh nhân (bag of images)."""

    grouped: Dict[str, Dict[str, object]] = {}
    for row in rows:
        pid = row["patient_id"]
        label = int(row["label"])
        image_path = row["image_path"]

        if pid not in grouped:
            grouped[pid] = {"label": label, "images": []}

        # Bảo vệ dữ liệu: cùng 1 bệnh nhân không được có 2 nhãn khác nhau.
        if int(grouped[pid]["label"]) != label:
            raise ValueError(f"Conflicting labels for patient {pid}")

        grouped[pid]["images"].append(image_path)

    records: List[PatientRecord] = []
    for pid in sorted(grouped):
        images = tuple(sorted(set(grouped[pid]["images"])))
        records.append(PatientRecord(patient_id=pid, label=int(grouped[pid]["label"]), image_paths=images))
    return records


def summarize_records(records: Sequence[PatientRecord]) -> DatasetSummary:
    """Tính thống kê tổng quan để báo cáo nhanh."""

    if not records:
        return DatasetSummary(0, 0, 0, 0, 0, 0, 0.0)

    num_images = [len(r.image_paths) for r in records]
    benign = sum(1 for r in records if r.label == 0)
    malignant = sum(1 for r in records if r.label == 1)

    return DatasetSummary(
        images_total=sum(num_images),
        patients_total=len(records),
        benign_total=benign,
        malignant_total=malignant,
        min_images_per_patient=min(num_images),
        max_images_per_patient=max(num_images),
        avg_images_per_patient=sum(num_images) / len(num_images),
    )


def save_patient_table(records: Sequence[PatientRecord], output_csv: Path) -> None:
    """Lưu bảng 1 dòng / 1 bệnh nhân để kiểm tra dữ liệu."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label", "num_images"])
        writer.writeheader()
        for r in records:
            writer.writerow({"patient_id": r.patient_id, "label": r.label, "num_images": len(r.image_paths)})


def _stratified_holdout(
    ids_by_label: Dict[int, List[str]],
    test_size: float,
    rng: random.Random,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """Tách tập test có giữ tỷ lệ lớp."""

    test_ids: List[str] = []
    train_val_by_label: Dict[int, List[str]] = {}

    for label, ids in ids_by_label.items():
        ids_copy = ids[:]
        rng.shuffle(ids_copy)

        n_test = max(1, int(round(len(ids_copy) * test_size))) if len(ids_copy) > 1 else 1
        n_test = min(n_test, len(ids_copy))

        test_part = ids_copy[:n_test]
        remain_part = ids_copy[n_test:]

        test_ids.extend(test_part)
        train_val_by_label[label] = remain_part

    return sorted(test_ids), train_val_by_label


def _stratified_kfold_ids(ids_by_label: Dict[int, List[str]], k_folds: int, rng: random.Random) -> List[List[str]]:
    """Tạo k-fold theo cách round-robin trong từng lớp."""

    folds: List[List[str]] = [[] for _ in range(k_folds)]

    for _, ids in ids_by_label.items():
        ids_copy = ids[:]
        rng.shuffle(ids_copy)
        for idx, pid in enumerate(ids_copy):
            folds[idx % k_folds].append(pid)

    for f in folds:
        f.sort()

    return folds


def create_patient_splits(
    records: Sequence[PatientRecord],
    test_size: float,
    k_folds: int,
    seed: int,
) -> Tuple[List[str], List[Tuple[List[str], List[str]]]]:
    """Sinh split patient-level: test và các cặp (train, val) cho từng fold."""

    ids_by_label: Dict[int, List[str]] = defaultdict(list)
    for r in records:
        ids_by_label[int(r.label)].append(r.patient_id)

    rng = random.Random(seed)

    # B1: tách test trước.
    test_ids, train_val_by_label = _stratified_holdout(ids_by_label, test_size, rng)

    # B2: chia phần còn lại thành k fold validation.
    fold_val_ids = _stratified_kfold_ids(train_val_by_label, k_folds, rng)
    all_train_val_ids = sorted([pid for ids in train_val_by_label.values() for pid in ids])

    folds: List[Tuple[List[str], List[str]]] = []
    for val_ids in fold_val_ids:
        val_set = set(val_ids)
        train_ids = [pid for pid in all_train_val_ids if pid not in val_set]
        folds.append((train_ids, val_ids))

    return test_ids, folds


def save_patient_ids(ids: Sequence[str], output_csv: Path) -> None:
    """Lưu danh sách patient_id ra CSV 1 cột."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id"])
        writer.writeheader()
        for pid in ids:
            writer.writerow({"patient_id": pid})


def load_patient_ids(csv_path: Path) -> List[str]:
    """Đọc danh sách patient_id từ CSV 1 cột."""

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [row["patient_id"] for row in csv.DictReader(f)]


def filter_records(records: Sequence[PatientRecord], patient_ids: Sequence[str]) -> List[PatientRecord]:
    """Lọc records theo danh sách patient_id cần dùng."""

    wanted = set(patient_ids)
    return [r for r in records if r.patient_id in wanted]


REQUIRED_METADATA_KEYS = ("patient_id", "batch", "local_patient_id", "image_name", "image_path", "label")


def _validate_row_schema(row: Dict[str, str], source: str = "unknown") -> None:
    """Kiểm tra row có đủ cột bắt buộc cho pipeline train."""

    missing = [k for k in REQUIRED_METADATA_KEYS if k not in row]
    if missing:
        raise ValueError(f"Row from {source} missing keys: {missing}")


def merge_and_deduplicate_rows(*row_groups: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Gộp nhiều nguồn metadata và loại trùng theo (patient_id, image_path).

    Mục tiêu:
    - Cho phép nạp nhiều thư mục dữ liệu.
    - Cho phép nạp thêm manifest CSV ngoài.
    - Không để 1 ảnh bị lặp nhiều lần trong cùng metadata cuối.
    """

    merged: List[Dict[str, str]] = []
    seen = set()
    label_by_key: Dict[Tuple[str, str], str] = {}

    for group_idx, rows in enumerate(row_groups):
        source_name = f"group_{group_idx}"
        for row in rows:
            _validate_row_schema(row, source=source_name)

            normalized = {str(k): "" if v is None else str(v).strip() for k, v in row.items()}
            key = (normalized["patient_id"], normalized["image_path"])
            label = normalized["label"]

            if key in seen:
                if label_by_key.get(key) != label:
                    raise ValueError(
                        f"Conflicting labels for duplicated image row key={key}: "
                        f"old={label_by_key.get(key)} new={label}"
                    )
                continue

            seen.add(key)
            label_by_key[key] = label
            merged.append(normalized)

    return merged
