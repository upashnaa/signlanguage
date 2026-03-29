from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(
    data_dir: str = "data",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    LabelEncoder,
]:
    """
    Load .npy sequences from sign subfolders. Each file shape (30, 258).
    Returns stratified 70/15/15 split and the fitted LabelEncoder.
    """
    base = Path(__file__).resolve().parent / data_dir
    if not base.is_dir():
        raise FileNotFoundError(f"Data directory not found: {base}")

    X_list = []
    y_list = []
    for sign_dir in sorted(base.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign_name = sign_dir.name
        for npy_path in sorted(sign_dir.glob("*.npy")):
            seq = np.load(npy_path)
            if seq.shape != (30, 258):
                seq = seq.reshape(30, 258)
            X_list.append(seq)
            y_list.append(sign_name)

    if not X_list:
        raise ValueError(f"No .npy files found under {base}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    label_names = np.array(encoder.classes_)
    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_names,
        encoder,
    )
