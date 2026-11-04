import numpy as np
import pandas as pd

from src.data_loader import load_record


# AAMI heartbeat classification
AAMI_CLASSES = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q", "?"]
}

SYMBOL_TO_CLASS = {
    symbol: class_name
    for class_name, symbols in AAMI_CLASSES.items()
    for symbol in symbols
}

BINARY_LABELS = {
    "N": 0,
    "S": 1,
    "V": 1,
    "F": 1
}


def extract_heartbeat_windows(
    data_dir,
    record_ids,
    fs=360,
    before_seconds=0.2,
    after_seconds=0.4,
    channel=0
):
    """
    Extract fixed-length ECG windows around annotated heartbeats.

    Returns:
        X: NumPy feature matrix.
        metadata: DataFrame with heartbeat information.
        feature_names: List of feature names.
    """

    samples_before = int(before_seconds * fs)
    samples_after = int(after_seconds * fs)

    window_size = samples_before + samples_after

    X_list = []
    metadata_list = []

    for record_id in record_ids:

        record, annotations = load_record(data_dir, record_id)

        if record.fs != fs:
            raise ValueError(
                f"Unexpected sampling frequency in record {record_id}: "
                f"{record.fs} Hz"
            )

        signal = record.p_signal[:, channel]

        for sample, symbol in zip(
            annotations.sample,
            annotations.symbol
        ):

            if symbol not in SYMBOL_TO_CLASS:
                continue

            aami_class = SYMBOL_TO_CLASS[symbol]

            if aami_class not in BINARY_LABELS:
                continue

            start = sample - samples_before
            end = sample + samples_after

            if start < 0 or end > len(signal):
                continue

            heartbeat = signal[start:end]

            if len(heartbeat) != window_size:
                continue

            if not np.all(np.isfinite(heartbeat)):
                continue

            X_list.append(
                heartbeat.astype(np.float32)
            )

            metadata_list.append({
                "record_id": str(record_id),
                "subject_group": (
                    "201_202"
                    if str(record_id) in ["201", "202"]
                    else str(record_id)
                ),
                "sample": int(sample),
                "symbol": symbol,
                "aami_class": aami_class,
                "label": BINARY_LABELS[aami_class]
            })

    if not X_list:
        raise ValueError("No valid heartbeat windows were extracted.")

    X = np.stack(X_list)

    metadata = pd.DataFrame(metadata_list)

    feature_names = [
        f"ecg_sample_{i:03d}"
        for i in range(window_size)
    ]

    return X, metadata, feature_names