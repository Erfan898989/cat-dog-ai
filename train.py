"""Train/evaluate a Cat-vs-Dog image classifier.

Examples:
  python train.py --data data/images --model mobilenet_v3_large --epochs 12
  python train.py --data data/images --model resnet50 --epochs 10
  python train.py --data data/images --model custom_cnn --epochs 20 --no-pretrained
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import CatDogDataset, collect_samples, build_transforms, IDX_TO_CLASS
from src.model import create_model
from src.utils import choose_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Cat vs Dog classifier using CNN / MobileNet / ResNet")
    parser.add_argument("--data", type=str, default="data/images", help="Folder containing extracted labeled images")
    parser.add_argument("--model", choices=["custom_cnn", "mobilenet_v3_small", "mobilenet_v3_large", "resnet50"], default="mobilenet_v3_large")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.10, help="Fraction of full dataset for validation")
    parser.add_argument("--test-split", type=float, default=0.10, help="Fraction of full dataset for final test")
    parser.add_argument("--patience", type=int, default=4, help="Early-stop patience based on validation loss")
    parser.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows; try 2 or 4 for speed")
    parser.add_argument("--max-per-class", type=int, default=0, help="Optional quick run; 0 uses the whole dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use ImageNet pretrained weights")
    return parser.parse_args()


def maybe_limit_samples(samples, max_per_class: int, seed: int):
    if max_per_class <= 0:
        return samples
    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for item in samples:
        groups[item[1]].append(item)
    picked = []
    for label in (0, 1):
        rng.shuffle(groups[label])
        picked.extend(groups[label][:max_per_class])
    rng.shuffle(picked)
    return picked


def get_splits(samples, val_fraction: float, test_fraction: float, seed: int):
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 0.8:
        raise ValueError("Use positive --val-split and --test-split values whose sum is below 0.8.")
    labels = [label for _, label in samples]
    train_val, test = train_test_split(
        samples, test_size=test_fraction, random_state=seed, stratify=labels
    )
    val_relative = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=seed,
        stratify=[label for _, label in train_val],
    )
    return train, val, test


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp_enabled=False, description="train"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_labels, all_predictions = [], []
    progress = tqdm(loader, desc=description, leave=False, unit="batch")

    for images, labels in progress:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            amp_context = torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        count = labels.size(0)
        total_loss += float(loss.detach().item()) * count
        total_correct += int((predictions == labels).sum().item())
        total_seen += count
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / max(total_seen, 1):.4f}", acc=f"{100 * total_correct / max(total_seen, 1):.2f}%")

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "labels": all_labels,
        "predictions": all_predictions,
    }


def plot_training_curves(history: dict, path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Train loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss", marker="o")
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history["train_acc"], label="Train accuracy", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Validation accuracy", marker="o")
    axes[1].set_title("Accuracy by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, path: Path):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.set_title("Test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Cat", "Dog"])
    ax.set_yticks([0, 1], labels=["Cat", "Dog"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        raise SystemExit("epochs/batch-size must be positive and image-size must be at least 64.")

    set_seed(args.seed)
    device = choose_device()
    amp_enabled = device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(" CAT vs DOG AI | Transfer learning + evaluation pipeline")
    print(f"Device: {device} | Model: {args.model} | AMP: {amp_enabled}")
    print("=" * 64)

    samples = collect_samples(args.data)
    samples = maybe_limit_samples(samples, args.max_per_class, args.seed)
    if len(samples) < 20:
        raise SystemExit("Too few valid images. Use more data (at least ~10 images per class).")
    try:
        train_samples, val_samples, test_samples = get_splits(samples, args.val_split, args.test_split, args.seed)
    except ValueError as error:
        raise SystemExit(f"Could not split dataset: {error}. Make sure both classes have enough images.") from error

    print(f"Split sizes -> train: {len(train_samples):,} | val: {len(val_samples):,} | test: {len(test_samples):,}")
    train_tf, eval_tf = build_transforms(args.image_size)
    train_ds = CatDogDataset(train_samples, transform=train_tf)
    val_ds = CatDogDataset(val_samples, transform=eval_tf)
    test_ds = CatDogDataset(test_samples, transform=eval_tf)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": max(0, args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_options)

    model = create_model(args.model, pretrained=(not args.no_pretrained)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_path = output_dir / "best_model.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "learning_rate": []}
    best_val_loss = float("inf")
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp_enabled, "train")
        val_metrics = run_epoch(model, val_loader, criterion, device, amp_enabled=amp_enabled, description="valid")
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        print(
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.2%} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.2%}"
        )

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            stale_epochs = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_name": args.model,
                "image_size": args.image_size,
                "classes": ["cat", "dog"],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_accuracy": val_metrics["accuracy"],
            }
            temp_path = best_path.with_suffix(".tmp")
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, best_path)
            print(f"  ✓ Saved new best model -> {best_path}")
        else:
            stale_epochs += 1
            print(f"  No validation-loss improvement ({stale_epochs}/{args.patience})")

        save_json(output_dir / "history.json", history)
        if stale_epochs >= args.patience:
            print("Early stopping: validation loss stopped improving.")
            break

    if not best_path.exists():
        raise RuntimeError("Training finished without saving a checkpoint.")

    # Evaluate only once on the held-out test set, using the best validation checkpoint.
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, amp_enabled=amp_enabled, description="test")
    report = classification_report(
        test_metrics["labels"], test_metrics["predictions"],
        labels=[0, 1], target_names=["cat", "dog"], output_dict=True, zero_division=0,
    )
    test_accuracy = accuracy_score(test_metrics["labels"], test_metrics["predictions"])
    summary = {
        "model": args.model,
        "device": str(device),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["val_loss"]),
        "best_validation_accuracy": float(checkpoint["val_accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
        "dataset_images_used": len(samples),
        "split_sizes": {"train": len(train_samples), "validation": len(val_samples), "test": len(test_samples)},
        "training_seconds": round(time.time() - start_time, 2),
    }
    save_json(output_dir / "test_metrics.json", summary)
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion(test_metrics["labels"], test_metrics["predictions"], output_dir / "confusion_matrix.png")

    print("\n" + "=" * 64)
    print(f"FINAL TEST ACCURACY: {test_accuracy:.2%}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(classification_report(
        test_metrics["labels"], test_metrics["predictions"],
        labels=[0, 1], target_names=["cat", "dog"], zero_division=0,
    ))
    print(f"Best model: {best_path.resolve()}")
    print(f"Reports/plots: {output_dir.resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
