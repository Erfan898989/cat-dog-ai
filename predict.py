"""Predict whether one image contains a cat or a dog."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data import build_transforms
from src.model import create_model
from src.utils import choose_device


CLASS_NAMES_FA = {"cat": "گربه", "dog": "سگ"}


def load_classifier(checkpoint_path: str | Path, device: torch.device | None = None):
    device = device or choose_device()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    _, eval_transform = build_transforms(int(checkpoint.get("image_size", 224)))
    return model, eval_transform, device


def predict_pil(image: Image.Image, model, transform, device: torch.device) -> dict:
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    label = "cat" if predicted_index == 0 else "dog"
    return {
        "label": label,
        "label_fa": CLASS_NAMES_FA[label],
        "cat_probability": float(probabilities[0]),
        "dog_probability": float(probabilities[1]),
        "confidence": float(probabilities[predicted_index]),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict cat/dog from a photo")
    parser.add_argument("image", help="Path to the image to classify")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth", help="Path to best_model.pth")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    model, transform, device = load_classifier(args.checkpoint)
    with Image.open(image_path) as image:
        result = predict_pil(image, model, transform, device)

    print(f"پیش‌بینی: {result['label_fa']} ({result['label']})")
    print(f"اطمینان مدل: {result['confidence']:.1%}")
    print(f"احتمال گربه: {result['cat_probability']:.1%}")
    print(f"احتمال سگ: {result['dog_probability']:.1%}")
    print("توجه: درصد اطمینان، تضمین درستی پیش‌بینی نیست.")


if __name__ == "__main__":
    main()
