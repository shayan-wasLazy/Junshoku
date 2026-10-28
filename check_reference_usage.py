
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Models.VGGBase import PredictModel
from lut_train import apply_lut_bank


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"F:\Junshoku")
DATASET_ROOT = ROOT / "TST2K"

# Keep source fixed while changing the reference.
SOURCE_SAMPLE_ID = "1182"
REFERENCE_A_SAMPLE_ID = "1182"
REFERENCE_B_SAMPLE_ID = "1034"

CHECKPOINT_PATH = ROOT / "checkpoints_tst2k" / "best.pt"
OUTPUT_DIR = ROOT / "Outputs" / "reference_diagnostic"

IMAGE_SIZE = 256
LUT_SIZE = 16
NUM_LUTS = 9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET IMAGE LOCATOR
# ============================================================

def find_dataset_image(sample_id, image_stem):
    sample_dir = DATASET_ROOT / sample_id

    if not sample_dir.is_dir():
        raise FileNotFoundError(
            f"Sample folder not found: {sample_dir}"
        )

    valid_extensions = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }

    matches = [
        p for p in sample_dir.iterdir()
        if p.is_file()
        and p.stem.lower() == image_stem.lower()
        and p.suffix.lower() in valid_extensions
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{image_stem}' image in "
            f"{sample_dir}, found {len(matches)}: {matches}"
        )

    return matches[0]


SOURCE_PATH = find_dataset_image(
    SOURCE_SAMPLE_ID, "content"
)

REFERENCE_A_PATH = find_dataset_image(
    REFERENCE_A_SAMPLE_ID, "reference"
)

REFERENCE_B_PATH = find_dataset_image(
    REFERENCE_B_SAMPLE_ID, "reference"
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(array).permute(
        2, 0, 1
    ).unsqueeze(0).to(DEVICE)


def save_tensor(tensor, path):
    array = (
        tensor[0].detach().float().cpu()
        .permute(1, 2, 0).numpy()
    )

    array = (
        array * 255
    ).round().clip(0, 255).astype(np.uint8)

    Image.fromarray(array).save(path)


def image_difference(a, b):
    """Mean absolute difference, normalized to [0, 1]."""
    return (a.float() - b.float()).abs().mean().item()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():
    model = PredictModel(
        freeze_encoder=True,
        lut_size=LUT_SIZE,
        num_luts=NUM_LUTS,
        pretrained=False,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = None

        for key in (
            "model_state_dict",
            "model",
            "state_dict",
        ):
            if key in checkpoint and isinstance(
                checkpoint[key], dict
            ):
                state = checkpoint[key]
                break

        if state is None:
            state = checkpoint
    else:
        state = checkpoint

    if state and all(
        key.startswith("module.") for key in state
    ):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }

    model.load_state_dict(state)
    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, source, reference):
    with torch.inference_mode():
        luts, weights = model(source, reference)

        transformed = apply_lut_bank(
            source.float(),
            luts.float(),
        )

        output = (
            transformed * weights.unsqueeze(2)
        ).sum(dim=1).clamp(0.0, 1.0)

    return output, weights, luts


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_paths = [
        SOURCE_PATH,
        REFERENCE_A_PATH,
        REFERENCE_B_PATH,
        CHECKPOINT_PATH,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    print("=" * 60)
    print("JUNSHOKU REFERENCE-SENSITIVITY TEST")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Source: {SOURCE_PATH}")
    print(f"Reference A: {REFERENCE_A_PATH}")
    print(f"Reference B: {REFERENCE_B_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    print("\nLoading model...")
    model = load_model()

    source = load_image(SOURCE_PATH)
    ref_a = load_image(REFERENCE_A_PATH)
    ref_b = load_image(REFERENCE_B_PATH)

    print("Running prediction with reference A...")
    output_a, weights_a, luts_a = predict(
        model, source, ref_a
    )

    print("Running prediction with reference B...")
    output_b, weights_b, luts_b = predict(
        model, source, ref_b
    )

    print("Repeating prediction with reference A...")
    output_a_repeat, weights_a_repeat, _ = predict(
        model, source, ref_a
    )

    # --------------------------------------------------------
    # SAVE IMAGES
    # --------------------------------------------------------

    save_tensor(source, OUTPUT_DIR / "source.png")
    save_tensor(ref_a, OUTPUT_DIR / "reference_a.png")
    save_tensor(ref_b, OUTPUT_DIR / "reference_b.png")
    save_tensor(output_a, OUTPUT_DIR / "output_a.png")
    save_tensor(output_b, OUTPUT_DIR / "output_b.png")

    # Amplify differences five times for visualization.
    difference = (
        (output_a - output_b).abs()[0]
        .permute(1, 2, 0).cpu().numpy()
    )

    difference = np.clip(difference * 5.0, 0.0, 1.0)

    Image.fromarray(
        (difference * 255).round().astype(np.uint8)
    ).save(OUTPUT_DIR / "output_difference_5x.png")

    # Save weight maps for visual inspection.
    for i in range(NUM_LUTS):
        for label, weights in (
            ("A", weights_a),
            ("B", weights_b),
        ):
            weight_map = weights[0, i].detach().float().cpu().numpy()

            # Normalize each map for visibility.
            weight_min = weight_map.min()
            weight_max = weight_map.max()

            if weight_max - weight_min > 1e-8:
                display = (
                    (weight_map - weight_min)
                    / (weight_max - weight_min)
                )
            else:
                display = np.zeros_like(weight_map)

            display = (
                display * 255
            ).round().clip(0, 255).astype(np.uint8)

            Image.fromarray(display).save(
                OUTPUT_DIR / f"weight_lut_{i + 1}_{label}.png"
            )

    # --------------------------------------------------------
    # NUMERICAL RESULTS
    # --------------------------------------------------------

    output_diff = image_difference(output_a, output_b)
    weight_diff = image_difference(weights_a, weights_b)

    repeat_diff = image_difference(
        output_a, output_a_repeat
    )

    repeat_weight_diff = image_difference(
        weights_a, weights_a_repeat
    )

    lut_diff = image_difference(luts_a, luts_b)

    print("\n" + "=" * 60)
    print("REFERENCE-SENSITIVITY RESULTS")
    print("=" * 60)

    print(
        f"Output difference (A vs B):        {output_diff:.8f}"
    )
    print(
        f"Weight-map difference (A vs B):    {weight_diff:.8f}"
    )
    print(
        f"Repeated output difference:        {repeat_diff:.8f}"
    )
    print(
        f"Repeated weight difference:        {repeat_weight_diff:.8f}"
    )
    print(
        f"LUT-bank difference (A vs B):      {lut_diff:.8f}"
    )

    # Average weight per LUT across all pixels.
    mean_a = weights_a.mean(dim=(0, 2, 3))
    mean_b = weights_b.mean(dim=(0, 2, 3))

    print("\nAverage LUT usage:")
    print(f"{'LUT':<8}{'Ref A':>12}{'Ref B':>12}{'Abs change':>14}")

    for i in range(NUM_LUTS):
        a = mean_a[i].item()
        b = mean_b[i].item()

        print(
            f"{i + 1:<8}{a:>12.5f}{b:>12.5f}"
            f"{abs(a - b):>14.5f}"
        )

    # Spatial variation of each map.
    print("\nSpatial standard deviation of LUT weights:")

    for i in range(NUM_LUTS):
        std_a = weights_a[0, i].std().item()
        std_b = weights_b[0, i].std().item()

        print(
            f"LUT {i + 1}: Ref A={std_a:.6f}, "
            f"Ref B={std_b:.6f}"
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INITIAL INTERPRETATION")
    print("=" * 60)

    if repeat_diff > 1e-5 or repeat_weight_diff > 1e-5:
        print(
            "WARNING: Repeated inference changed. "
            "Investigate nondeterminism first."
        )
    elif output_diff < 0.003 and weight_diff < 0.003:
        print(
            "Very weak reference sensitivity for this pair. "
            "Test additional references before drawing a conclusion."
        )
    elif output_diff < 0.01:
        print(
            "Some reference sensitivity detected. "
            "Inspect outputs and weight maps visually."
        )
    else:
        print(
            "The output changes measurably with the reference. "
            "This does not prove the style transfer is good."
        )

    print(
        "\nThe LUT-bank difference should be zero because this "
        "architecture learns a global LUT bank. The reference "
        "changes the predicted blending weights, not the LUT values."
    )

    print(f"\nAll diagnostic images saved to:\n{OUTPUT_DIR}")
    print("\nCompare output_a.png and output_b.png.")
    print("Inspect output_difference_5x.png and the weight maps.")


if __name__ == "__main__":
    main()
