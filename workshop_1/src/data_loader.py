from pathlib import Path

import wfdb


def get_record_ids(data_dir):
    """
    Return the available MIT-BIH recording identifiers.
    """

    data_dir = Path(data_dir)

    record_ids = sorted(
        path.stem
        for path in data_dir.glob("*.hea")
    )

    return record_ids


def load_record(data_dir, record_id):
    """
    Load one ECG recording and its annotations.

    Returns:
        record: WFDB recording.
        annotations: WFDB annotations.
    """

    data_dir = Path(data_dir)
    record_path = data_dir / str(record_id)

    record = wfdb.rdrecord(str(record_path))
    annotations = wfdb.rdann(str(record_path), "atr")

    return record, annotations


def load_ecg_signal(data_dir, record_id, channel=0):
    """
    Load one ECG channel from a recording.
    """

    record, annotations = load_record(data_dir, record_id)

    signal = record.p_signal[:, channel]

    return signal, record.fs, annotations