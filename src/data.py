"""Dataset discovery and image transforms."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
CLASS_TO_IDX = {"cat": 0, "dog": 1}
IDX_TO_CLASS = {0: "cat", 1: "dog"}


def label_from_path(path: Path) -> int | None:
    """Infer label from class-folder names or Kaggle-style cat.0.jpg filenames."""
    aliases = {"cat": 0, "cats": 0, "dog": 1, "dogs": 1}

    # Search parent folders from nearest to farthest.
    for part in reversed(path.parts[:-1]):
        normalized = re.sub(r"[ _-]", "", part.lower())
        if normalized in aliases:
            return aliases[normalized]

    match = re.match(r"^(cat|dog)(?:[._ -]|$)", path.name.lower())
    if match:
        return CLASS_TO_IDX[match.group(1)]
    return None


def _is_readable_image(path: Path) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as image:
                image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False
    except Exception:
        # Bad/truncated images occasionally raise other decoder exceptions.
        return False


def collect_samples(root: str | Path) -> list[tuple[Path, int]]:
    """Recursively find labeled, readable images below `root`.

    Supported examples:
      data/PetImages/Cat/0.jpg and data/PetImages/Dog/0.jpg
      data/train/cats/example.jpg and data/train/dogs/example.jpg
      data/cat.0.jpg and data/dog.0.jpg (Kaggle train.zip extracted)
    """
    root = Path(root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    candidates: list[tuple[Path, int]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_from_path(path)
        if label is not None:
            candidates.append((path, label))

    if not candidates:
        raise RuntimeError(
            f"No labeled images found under {root}. Extract Kaggle train.zip first, "
            "or use folders named Cat/Dog (or cats/dogs)."
        )

    good: list[tuple[Path, int]] = []
    bad = 0
    for path, label in tqdm(candidates, desc="Checking image files", unit="img"):
        if _is_readable_image(path):
            good.append((path, label))
        else:
            bad += 1

    counts = {"cat": sum(label == 0 for _, label in good), "dog": sum(label == 1 for _, label in good)}
    print(f"Usable images: {len(good):,} | cats: {counts['cat']:,} | dogs: {counts['dog']:,} | unreadable skipped: {bad:,}")
    if not counts["cat"] or not counts["dog"]:
        raise RuntimeError("Both classes are required. Check that cat and dog images are in the dataset.")
    return good


class CatDogDataset(Dataset):
    def __init__(self, samples: Iterable[tuple[Path, int]], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int = 224):
    """Return augmentation-heavy train transforms and deterministic eval transforms."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = round(image_size * 256 / 224)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.68, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.025),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform
