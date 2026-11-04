"""
Regenerate the Workshop 1 processed dataset (X.npy, metadata.csv).

Downloads MIT-BIH from PhysioNet (if missing) and reuses the exact
feature-extraction code from Workshop 1, so the heartbeat order matches
workshop_1/data/splits.json.

Usage (from workshop_2/):  python src/prepare_data.py
"""
import json
import sys
from pathlib import Path

import wfdb

REPO_ROOT = Path(__file__).resolve().parents[2]
W1_DIR = REPO_ROOT / "workshop_1"
RAW_DIR = W1_DIR / "data" / "raw" / "mitdb"
PROCESSED_DIR = W1_DIR / "data" / "processed"
SPLITS_PATH = W1_DIR / "data" / "splits.json"

# Make Workshop 1's `src` package importable
sys.path.insert(0, str(W1_DIR))
from src.data_loader import get_record_ids          # noqa: E402
from src.features import extract_heartbeat_windows  # noqa: E402
from src.dataset import save_processed_dataset      # noqa: E402


def main():
    # 1. Download the raw dataset if it is not there yet
    if not list(RAW_DIR.glob("*.hea")):
        print("Downloading MIT-BIH from PhysioNet (~90 MB)...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", str(RAW_DIR))

    record_ids = get_record_ids(RAW_DIR)
    print(f"Records found: {len(record_ids)}")

    # 2. Extract heartbeat windows with the Workshop 1 settings
    X, metadata, feature_names = extract_heartbeat_windows(RAW_DIR, record_ids)
    save_processed_dataset(X, metadata, feature_names, PROCESSED_DIR)
    print(f"Saved X with shape {X.shape} to {PROCESSED_DIR}")

    # 3. Check it matches the Workshop 1 splits
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    n_split = sum(len(splits[k]) for k in
                  ["train_indices", "validation_indices", "test_indices"])

    assert len(X) == n_split == 101_418, (
        f"Mismatch: X has {len(X)} rows, splits cover {n_split}"
    )

    test_records = sorted(
        metadata.iloc[splits["test_indices"]]["record_id"].astype(str).unique()
    )
    assert test_records == sorted(map(str, splits["test_records"])), \
        "Test records do not match splits.json"

    print("OK: dataset matches Workshop 1 splits exactly.")
    print(metadata["label"].value_counts().rename({0: "Normal", 1: "Abnormal"}))


if __name__ == "__main__":
    main()