
from pathlib import Path
import csv
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


ROOT = Path(r"F:\Junshoku\TST2K")
MANIFEST = ROOT / "split_manifest.csv"

SEED = 42
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
IMAGE_SIZE = 256


def find_image(folder: Path, stem: str) -> Path:
    """Find an image by filename, regardless of extension."""
    matches = [
        p for p in folder.iterdir()
        if p.is_file() and p.stem.lower() == stem.lower()
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{stem}' image in {folder}; "
            f"found {len(matches)}"
        )

    return matches[0]


def create_manifest(root=ROOT, manifest_path=MANIFEST):
    """Validate triplets and create a reproducible train/val/test split."""
    root = Path(root)
    folders = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.isdigit()
    )

    valid_ids = []
    rejected = []

    for folder in folders:
        try:
            for name in ("content", "reference", "gt"):
                find_image(folder, name)

            valid_ids.append(folder.name)

        except (FileNotFoundError, OSError) as exc:
            rejected.append((folder.name, str(exc)))

    if not valid_ids:
        raise RuntimeError(f"No valid triplets found in {root}")

    # Fixed shuffle ensures reproducible splits.
    rng = random.Random(SEED)
    rng.shuffle(valid_ids)

    n = len(valid_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    split_ids = {
        "train": valid_ids[:n_train],
        "val": valid_ids[n_train:n_train + n_val],
        "test": valid_ids[n_train + n_val:],
    }

    manifest_path = Path(manifest_path)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "split"]
        )
        writer.writeheader()

        for split, ids in split_ids.items():
            for sample_id in ids:
                writer.writerow({
                    "sample_id": sample_id,
                    "split": split,
                })

    print(f"\nDataset root: {root}")
    print(f"Valid triplets: {n}")
    print(f"Rejected folders: {len(rejected)}")
    print(f"Manifest saved: {manifest_path}")

    for split, ids in split_ids.items():
        print(f"{split}: {len(ids)}")

    if rejected:
        print("\nFirst rejected folders:")
        for folder, reason in rejected[:10]:
            print(f"  {folder}: {reason}")

    return split_ids


class TST2KDataset(Dataset):
    """
    Reference-conditioned tone-transfer dataset.

    Returns:
        source:    [3, H, W], float32 in [0, 1]
        reference: [3, H, W], float32 in [0, 1]
        target:    [3, H, W], float32 in [0, 1]
    """

    def __init__(
        self,
        root=ROOT,
        split="train",
        image_size=IMAGE_SIZE,
        augment=False,
        manifest_path=MANIFEST,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        if not Path(manifest_path).exists():
            create_manifest(self.root, manifest_path)

        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.sample_ids = [
            row["sample_id"]
            for row in rows
            if row["split"] == split
        ]

        if not self.sample_ids:
            raise RuntimeError(
                f"No samples found for split '{split}'"
            )

    def __len__(self):
        return len(self.sample_ids)

    def _load_image(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BICUBIC,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0

        # HWC -> CHW
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        folder = self.root / sample_id

        source_path = find_image(folder, "content")
        reference_path = find_image(folder, "reference")
        target_path = find_image(folder, "gt")

        # Apply identical spatial augmentation to all three images.
        if self.augment and random.random() < 0.5:
            with Image.open(source_path) as im:
                source_img = im.convert("RGB")
            with Image.open(reference_path) as im:
                reference_img = im.convert("RGB")
            with Image.open(target_path) as im:
                target_img = im.convert("RGB")

            source_img = ImageOps.mirror(source_img)
            reference_img = ImageOps.mirror(reference_img)
            target_img = ImageOps.mirror(target_img)

            def to_tensor(image):
                image = image.resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BICUBIC,
                )
                array = np.asarray(image, dtype=np.float32) / 255.0
                return torch.from_numpy(array).permute(2, 0, 1).contiguous()

            source = to_tensor(source_img)
            reference = to_tensor(reference_img)
            target = to_tensor(target_img)

        else:
            source = self._load_image(source_path)
            reference = self._load_image(reference_path)
            target = self._load_image(target_path)

        return {
            "source": source,
            "reference": reference,
            "target": target,
            "sample_id": sample_id,
        }


if __name__ == "__main__":
    create_manifest()

    # Smoke test: load one batch from each split.
    from torch.utils.data import DataLoader

    for split in ("train", "val", "test"):
        dataset = TST2KDataset(split=split)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        print(f"\n{split.upper()} batch:")
        print("Source:", batch["source"].shape)
        print("Reference:", batch["reference"].shape)
        print("Target:", batch["target"].shape)
        print("Pixel range:",
              batch["source"].min().item(),
              batch["source"].max().item())
