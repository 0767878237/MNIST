from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _read_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if labels.shape[0] != size:
        raise ValueError(f"Label count mismatch in {path}: expected {size}, got {labels.shape[0]}")
    return labels


def _read_images(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        images = np.frombuffer(file.read(), dtype=np.uint8)
    expected = size * rows * cols
    if images.shape[0] != expected:
        raise ValueError(f"Image count mismatch in {path}: expected {expected}, got {images.shape[0]}")
    return images.reshape(size, rows, cols)


def load_mnist(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    train_images = _read_images(data_dir / "train-images.idx3-ubyte")
    train_labels = _read_labels(data_dir / "train-labels.idx1-ubyte")
    test_images = _read_images(data_dir / "t10k-images.idx3-ubyte")
    test_labels = _read_labels(data_dir / "t10k-labels.idx1-ubyte")
    return train_images, train_labels, test_images, test_labels
