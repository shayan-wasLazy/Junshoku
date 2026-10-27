
from pathlib import Path
from PIL import Image, ImageDraw
import csv

# Dataset paths
ROOT = Path(r"F:\Junshoku\PST50")
CONTENT_DIR = ROOT / "content_709"
STYLE_DIR = ROOT / "paired_style"
TARGET_DIR = ROOT / "paired_gt"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (320, 320)
NUM_PAIRS = 10
LABEL_HEIGHT = 35
GAP = 8
BACKGROUND = "white"


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB").resize(
            IMAGE_SIZE, Image.Resampling.LANCZOS
        )


def main():
    width = IMAGE_SIZE[0] * 3 + GAP * 4
    height = (IMAGE_SIZE[1] + LABEL_HEIGHT) * NUM_PAIRS + GAP * (NUM_PAIRS + 1)

    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    labels = ["SOURCE (Rec.709)", "STYLE REFERENCE", "GROUND TRUTH"]

    for index in range(1, NUM_PAIRS + 1):
        paths = [
            CONTENT_DIR / f"in{index}.png",
            STYLE_DIR / f"tar{index}.png",
            TARGET_DIR / f"gt{index}.png",
        ]

        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"Pair {index}: missing {missing}")
            continue

        y = GAP + (index - 1) * (IMAGE_SIZE[1] + LABEL_HEIGHT + GAP)

        for column, path in enumerate(paths):
            x = GAP + column * (IMAGE_SIZE[0] + GAP)
            draw.text(
                (x + 5, y + 5),
                f"Pair {index} - {labels[column]}",
                fill="black",
            )
            image = load_image(path)
            sheet.paste(image, (x, y + LABEL_HEIGHT))

    output_path = OUTPUT_DIR / "pst50_visual_comparison.jpg"
    sheet.save(output_path, quality=95)

    print(f"Saved comparison sheet: {output_path.resolve()}")


if __name__ == "__main__":
    main()
