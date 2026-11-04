
from pathlib import Path
import csv
import hashlib
import json

import wfdb


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "mitdb"
REPORTS_DIR = PROJECT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_EXTENSIONS = [".dat", ".hea", ".atr"]


# --------------------------------------------------
# 2. FILE UTILITIES
# --------------------------------------------------

def calculate_sha256(file_path):
    """Calculate the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_directory_size(directory):
    """Calculate the total size of all files in a directory."""
    return sum(
        file.stat().st_size
        for file in directory.rglob("*")
        if file.is_file()
    )


# --------------------------------------------------
# 3. DATASET INVENTORY
# --------------------------------------------------

def main():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    record_ids = sorted(
        file.stem
        for file in DATA_DIR.glob("*.hea")
    )

    inventory = []
    checksum_manifest = []
    missing_files = []

    print("=" * 60)
    print("MIT-BIH ARRHYTHMIA DATABASE - DATASET INFORMATION")
    print("=" * 60)

    print(f"\nDataset directory: {DATA_DIR}")
    print(f"Number of records found: {len(record_ids)}")

    for record_id in record_ids:

        record_path = DATA_DIR / record_id

        missing = [
            ext for ext in REQUIRED_EXTENSIONS
            if not (DATA_DIR / f"{record_id}{ext}").exists()
        ]

        if missing:
            missing_files.append({
                "record": record_id,
                "missing_extensions": missing
            })
            continue

        # Read signal metadata
        record = wfdb.rdrecord(str(record_path))

        # Read reference annotations
        annotations = wfdb.rdann(
            str(record_path),
            "atr"
        )

        # Count all annotation symbols
        symbol_counts = {}

        for symbol in annotations.symbol:
            symbol_counts[symbol] = (
                symbol_counts.get(symbol, 0) + 1
            )

        # Record metadata
        row = {
            "record_id": record_id,
            "channels": record.n_sig,
            "sampling_frequency_hz": record.fs,
            "samples": record.sig_len,
            "duration_seconds": record.sig_len / record.fs,
            "total_annotations": len(annotations.sample),
            "annotation_symbols": json.dumps(
                symbol_counts,
                sort_keys=True
            )
        }

        inventory.append(row)

        # Calculate checksums for the three main files
        for extension in REQUIRED_EXTENSIONS:

            file_path = DATA_DIR / f"{record_id}{extension}"

            checksum_manifest.append({
                "file": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "sha256": calculate_sha256(file_path)
            })

        print(
            f"Record {record_id}: "
            f"{record.n_sig} channels | "
            f"{record.fs} Hz | "
            f"{len(annotations.sample)} annotations"
        )

    # --------------------------------------------------
    # 4. SAVE INVENTORY
    # --------------------------------------------------

    inventory_path = REPORTS_DIR / "dataset_inventory.csv"

    if inventory:
        with open(
            inventory_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=inventory[0].keys()
            )

            writer.writeheader()
            writer.writerows(inventory)

    # --------------------------------------------------
    # 5. SAVE CHECKSUM MANIFEST
    # --------------------------------------------------

    checksum_path = REPORTS_DIR / "checksums_sha256.csv"

    with open(
        checksum_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["file", "size_bytes", "sha256"]
        )

        writer.writeheader()
        writer.writerows(checksum_manifest)

    # --------------------------------------------------
    # 6. SAVE SUMMARY
    # --------------------------------------------------

    total_size = get_directory_size(DATA_DIR)

    summary = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "source": "PhysioNet",
        "version": "1.0.0",
        "records_found": len(record_ids),
        "records_processed": len(inventory),
        "records_with_missing_files": len(missing_files),
        "total_size_bytes": total_size,
        "total_size_megabytes": round(
            total_size / (1024 ** 2), 2
        ),
        "missing_files": missing_files,
        "record_201_and_202_same_subject": True
    }

    summary_path = REPORTS_DIR / "dataset_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # 7. FINAL RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"Records found: {len(record_ids)}")
    print(f"Records processed: {len(inventory)}")
    print(f"Records with missing files: {len(missing_files)}")
    print(f"Dataset size: {summary['total_size_megabytes']} MB")

    print("\nGenerated files:")
    print(inventory_path)
    print(checksum_path)
    print(summary_path)

    if missing_files:
        print("\nWARNING: Some records have missing files.")
    else:
        print("\nAll records have their required files.")

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
