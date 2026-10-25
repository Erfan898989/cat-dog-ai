"""Safely extract a Kaggle ZIP archive to a dataset folder."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description="Extract Kaggle Cats vs Dogs train.zip")
    parser.add_argument("--zip", dest="zip_path", required=True, help="Path to Kaggle train.zip")
    parser.add_argument("--output", default="data/images", help="Destination folder")
    args = parser.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not zip_path.is_file():
        raise SystemExit(f"ZIP file not found: {zip_path}")

    output.mkdir(parents=True, exist_ok=True)
    destination = str(output)
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (output / member.filename).resolve()
            try:
                target.relative_to(output)
            except ValueError as exc:
                raise SystemExit(f"Unsafe ZIP path detected: {member.filename}") from exc
        archive.extractall(destination)

    print(f"Extracted {zip_path.name} to {output}")
    print("Next: python train.py --data \"" + str(output) + "\"")


if __name__ == "__main__":
    main()
