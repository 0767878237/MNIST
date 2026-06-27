from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def normalize_images(images: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(images).float() / 255.0
    return tensor.unsqueeze(1)


def labels_to_tensor(labels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(labels).long()


def train_validation_split(
    images: torch.Tensor,
    labels: torch.Tensor,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(images.size(0), generator=generator)
    split_index = int(images.size(0) * (1.0 - validation_ratio))

    train_indices = indices[:split_index]
    validation_indices = indices[split_index:]

    return (
        images[train_indices],
        labels[train_indices],
        images[validation_indices],
        labels[validation_indices],
    )
