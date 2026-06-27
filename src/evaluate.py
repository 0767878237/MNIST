from __future__ import annotations

from typing import Dict

import torch


def evaluate_model(model, data_loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def confusion_matrix(model, data_loader, device: torch.device, num_classes: int = 10) -> torch.Tensor:
    model.eval()
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                matrix[true_label, predicted_label] += 1

    return matrix
