
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class PST50Dataset(Dataset):
    def __init__(self, root, indices=None, image_size=256):
        self.root = Path(root)

        self.content_dir = self.root / "content_709"
        self.style_dir = self.root / "paired_style"
        self.target_dir = self.root / "paired_gt"

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),  # RGB, float32, [0, 1]
        ])

        if indices is None:
            indices = range(1, 51)

        self.samples = []

        for i in indices:
            source = self.content_dir / f"in{i}.png"
            reference = self.style_dir / f"tar{i}.png"
            target = self.target_dir / f"gt{i}.png"

            if not all(p.exists() for p in (source, reference, target)):
                print(f"Skipping pair {i}: missing file")
                continue

            self.samples.append((source, reference, target))

        if not self.samples:
            raise RuntimeError(
                f"No valid PST50 samples found under {self.root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_path, reference_path, target_path = self.samples[idx]

        with Image.open(source_path) as im:
            source = self.transform(im.convert("RGB"))

        with Image.open(reference_path) as im:
            reference = self.transform(im.convert("RGB"))

        with Image.open(target_path) as im:
            target = self.transform(im.convert("RGB"))

        return {
            "source": source,
            "reference": reference,
            "target": target,
        }


if __name__ == "__main__":
    root = r"F:\Junshoku\PST50"

    dataset = PST50Dataset(root, image_size=256)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    print("Dataset size:", len(dataset))

    batch = next(iter(loader))
    print("Source:", batch["source"].shape)
    print("Reference:", batch["reference"].shape)
    print("Target:", batch["target"].shape)
