from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from src.data_loader import load_mnist
from src.dataset import DataLoaders, MNISTTensorDataset, create_data_loader
from src.evaluate import confusion_matrix, evaluate_model
from src.models.cnn import CNNClassifier
from src.models.mlp import MLPClassifier
from src.preprocess import labels_to_tensor, normalize_images, train_validation_split
from src.visualize import plot_training_curves, save_training_history


def build_model(model_name: str):
    if model_name == "mlp":
        return MLPClassifier()
    if model_name == "cnn":
        return CNNClassifier()
    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(optimizer_name: str, model, learning_rate: float):
    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    if optimizer_name == "sgd":
        return SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def prepare_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    validation_ratio: float,
    seed: int,
) -> DataLoaders:
    train_images_np, train_labels_np, test_images_np, test_labels_np = load_mnist(data_dir)

    train_images = normalize_images(train_images_np)
    train_labels = labels_to_tensor(train_labels_np)
    test_images = normalize_images(test_images_np)
    test_labels = labels_to_tensor(test_labels_np)

    train_images, train_labels, validation_images, validation_labels = train_validation_split(
        train_images,
        train_labels,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    train_dataset = MNISTTensorDataset(train_images, train_labels)
    validation_dataset = MNISTTensorDataset(validation_images, validation_labels)
    test_dataset = MNISTTensorDataset(test_images, test_labels)

    return DataLoaders(
        train=create_data_loader(train_dataset, batch_size=batch_size, shuffle=True),
        validation=create_data_loader(validation_dataset, batch_size=batch_size, shuffle=False),
        test=create_data_loader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def train_one_epoch(model, data_loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = prepare_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, args.learning_rate)

    best_validation_accuracy = 0.0
    best_model_path = Path(args.output_dir) / f"best_{args.model}.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders.train, criterion, optimizer, device)
        validation_metrics = evaluate_model(model, dataloaders.validation, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(validation_metrics["loss"])
        history["val_accuracy"].append(validation_metrics["accuracy"])

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f}"
        )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate_model(model, dataloaders.test, criterion, device)
    matrix = confusion_matrix(model, dataloaders.test, device)

    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(matrix)

    history_path = Path(args.figure_dir) / f"{args.model}_history.json"
    figure_path = Path(args.figure_dir) / f"{args.model}_training_curves.png"
    save_training_history(history, history_path)
    plot_training_curves(history, figure_path)

    print(f"Training history saved to: {history_path}")
    print(f"Training curves saved to: {figure_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST models with PyTorch")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
