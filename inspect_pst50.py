from operator import index
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

# Change this to your actual PST50 folder.
ROOT = Path(r"F:\Junshoku\PST50")
CONTENT_DIR = ROOT / "content_709"
STYLE_DIR = ROOT / "paired_style"
TARGET_DIR = ROOT / "paired_gt"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SIZE = (256, 256)


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            SIZE, Image.Resampling.LANCZOS
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def main():
    rows = []

    for index in range(1, 51):
        content_path = CONTENT_DIR / f"in{index}.png"
        style_path = STYLE_DIR / f"tar{index}.png"
        target_path = TARGET_DIR / f"gt{index}.png"

        # Some releases may name targets differently.
        # Report missing paths rather than silently pairing incorrectly.
        paths = [content_path, style_path, target_path]
        missing = [str(p) for p in paths if not p.exists()]

        if missing:
            print(f"Pair {index}: missing files")
            for path in missing:
                print(f"  {path}")
            continue

        source = load_image(content_path)
        reference = load_image(style_path)
        target = load_image(target_path)

        # Baseline: do nothing; output equals the source.
        prediction = source.copy()

        psnr = peak_signal_noise_ratio(
            target, prediction, data_range=1.0
        )
        ssim = structural_similarity(
            target, prediction,
            data_range=1.0,
            channel_axis=2,
        )

        rows.append({
            "id": index,
            "source": str(content_path),
            "reference": str(style_path),
            "target": str(target_path),
            "identity_psnr": psnr,
            "identity_ssim": ssim,
        })

        if index <= 3:
            print(
                f"Pair {index}: "
                f"source={source.shape}, "
                f"reference={reference.shape}, "
                f"target={target.shape}"
            )

    if not rows:
        raise RuntimeError(
            "No valid pairs found. Check the folder paths "
            "and ground-truth filename convention."
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "identity_baseline.csv", index=False)

    print(f"\nValid pairs: {len(df)}")
    print(f"Identity PSNR: {df['identity_psnr'].mean():.4f} dB")
    print(f"Identity SSIM: {df['identity_ssim'].mean():.4f}")
    print(f"Saved: {OUTPUT_DIR / 'identity_baseline.csv'}")


if __name__ == "__main__":
    main()