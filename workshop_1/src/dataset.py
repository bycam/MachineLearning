import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


def save_processed_dataset(
    X,
    metadata,
    feature_names,
    output_dir
):
    """
    Save extracted features, metadata and feature names.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)

    metadata.to_csv(
        output_dir / "metadata.csv",
        index=False
    )

    with open(
        output_dir / "feature_names.json",
        "w"
    ) as file:

        json.dump(feature_names, file, indent=4)


def load_processed_dataset(input_dir):
    """
    Load processed features and metadata.
    """

    input_dir = Path(input_dir)

    X = np.load(input_dir / "X.npy")

    metadata = pd.read_csv(
        input_dir / "metadata.csv"
    )

    with open(
        input_dir / "feature_names.json",
        "r"
    ) as file:

        feature_names = json.load(file)

    return X, metadata, feature_names


def create_group_splits(
    X,
    metadata,
    random_state=42
):
    """
    Create training, validation and test sets
    without sharing patient groups.
    """

    y = metadata["label"].to_numpy()
    groups = metadata["subject_group"].to_numpy()

    test_splitter = StratifiedGroupKFold(
        n_splits=7,
        shuffle=True,
        random_state=random_state
    )

    train_val_idx, test_idx = next(
        test_splitter.split(X, y, groups)
    )

    validation_splitter = StratifiedGroupKFold(
        n_splits=6,
        shuffle=True,
        random_state=random_state
    )

    train_relative_idx, val_relative_idx = next(
        validation_splitter.split(
            X[train_val_idx],
            y[train_val_idx],
            groups[train_val_idx]
        )
    )

    train_idx = train_val_idx[train_relative_idx]
    val_idx = train_val_idx[val_relative_idx]

    splits = {
        "random_state": random_state,
        "method": "StratifiedGroupKFold",
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_records": sorted(
            metadata.iloc[train_idx]["record_id"].unique().tolist()
        ),
        "validation_records": sorted(
            metadata.iloc[val_idx]["record_id"].unique().tolist()
        ),
        "test_records": sorted(
            metadata.iloc[test_idx]["record_id"].unique().tolist()
        )
    }

    return splits


def save_splits(splits, output_path):
    """
    Save dataset split indices and recording identifiers.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(splits, file, indent=4)


def load_splits(input_path):
    """
    Load previously saved dataset splits.
    """

    with open(input_path, "r") as file:
        return json.load(file)