
from pathlib import Path

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from tst2k_dataset import TST2KDataset
from Models.VGGBase import PredictModel
from lut_train import apply_lut_bank


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"F:\Junshoku")
DATASET_ROOT = ROOT / "TST2K"

CHECKPOINT_PATH = (
    ROOT / "checkpoints_tst2k" / "best.pt"
)

OUTPUT_DIR = ROOT / "Outputs" / "evaluation_tst2k"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"
WEIGHT_MAP_DIR = OUTPUT_DIR / "weight_maps"

IMAGE_SIZE = 256
LUT_SIZE = 16
NUM_LUTS = 9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def load_checkpoint(model, path):
    """Load model weights from the training checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unexpected checkpoint format.")

    state_dict = None

    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        if checkpoint and all(isinstance(k, str) for k in checkpoint):
            state_dict = checkpoint
        else:
            raise RuntimeError(
                f"Could not locate model weights. "
                f"Checkpoint keys: {list(checkpoint.keys())}"
            )

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)

    return checkpoint


def tensor_to_image(tensor):
    """Convert a (3,H,W) tensor in [0,1] to an RGB image."""

    return (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def calculate_metrics(prediction, target):
    """Calculate image-quality metrics."""

    prediction = np.clip(
        prediction, 0, 1
    ).astype(np.float32)

    target = np.clip(
        target, 0, 1
    ).astype(np.float32)

    difference = prediction - target

    l1 = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference ** 2))

    psnr = float(
        peak_signal_noise_ratio(
            target,
            prediction,
            data_range=1.0,
        )
    )

    ssim = float(
        structural_similarity(
            target,
            prediction,
            data_range=1.0,
            channel_axis=-1,
        )
    )

    return {
        "L1": l1,
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
    }


def save_comparison(
    sample_id,
    source,
    reference,
    target,
    prediction,
    model_metrics,
    identity_metrics,
):
    """Save source, reference, target, and model prediction."""

    images = [
        ("Content / Source", source),
        ("Style Reference", reference),
        ("Ground Truth", target),
        ("Junshoku Prediction", prediction),
    ]

    fig, axes = plt.subplots(
        1, 4, figsize=(16, 5)
    )

    for ax, (title, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"TST2K sample {sample_id} | "
        f"Model L1={model_metrics['L1']:.4f} | "
        f"Identity L1={identity_metrics['L1']:.4f}"
    )

    fig.tight_layout()

    fig.savefig(
        COMPARISON_DIR / f"sample_{sample_id}.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_weight_maps(sample_id, weights):
    """Save spatial LUT weight maps."""

    weights = (
        weights.detach()
        .float()
        .cpu()
        .numpy()
    )  # (K,H,W)

    fig, axes = plt.subplots(
        3, 3, figsize=(12, 12)
    )

    for lut_index, ax in enumerate(axes.ravel()):
        heatmap = ax.imshow(
            weights[lut_index],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        ax.set_title(f"LUT {lut_index + 1}")
        ax.axis("off")

        fig.colorbar(
            heatmap,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    fig.suptitle(
        f"Spatial LUT Weights — TST2K sample {sample_id}"
    )

    fig.tight_layout()

    fig.savefig(
        WEIGHT_MAP_DIR / f"sample_{sample_id}_weights.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    np.save(
        WEIGHT_MAP_DIR / f"sample_{sample_id}_weights.npy",
        weights,
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for directory in (
        OUTPUT_DIR,
        COMPARISON_DIR,
        WEIGHT_MAP_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("=" * 68)
    print("JUNSHOKU — TST2K TEST EVALUATION")
    print("=" * 68)

    print("Device:", DEVICE)
    print("CUDA available:", torch.cuda.is_available())

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    dataset = TST2KDataset(
        root=DATASET_ROOT,
        split="test",
        image_size=IMAGE_SIZE,
        augment=False,
    )

    print("Dataset root:", DATASET_ROOT)
    print("Split: test")
    print("Test samples:", len(dataset))

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = PredictModel(
        freeze_encoder=True,
        lut_size=LUT_SIZE,
        num_luts=NUM_LUTS,
        pretrained=True,
    ).to(DEVICE)

    checkpoint = load_checkpoint(
        model,
        CHECKPOINT_PATH,
    )

    model.eval()

    if isinstance(checkpoint, dict):
        print(
            "Checkpoint epoch:",
            checkpoint.get("epoch", "not recorded"),
        )

        print(
            "Checkpoint validation loss:",
            checkpoint.get("val_loss", "not recorded"),
        )

    print("Loaded checkpoint:", CHECKPOINT_PATH)

    print(
        "Total parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    results = []

    with torch.inference_mode():

        for position in range(len(dataset)):

            sample = dataset[position]

            sample_id = str(sample["sample_id"])

            source = (
                sample["source"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            reference = (
                sample["reference"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            target_tensor = sample["target"]

            # Predict LUT bank and spatial weights.
            luts, weights = model(
                source,
                reference,
            )

            # Apply each LUT to the source image.
            transformed = apply_lut_bank(
                source.float(),
                luts.float(),
            )

            # Spatially blend the transformed images.
            prediction_tensor = (
                transformed
                * weights.unsqueeze(2).float()
            ).sum(dim=1).clamp(0, 1)

            prediction = tensor_to_image(
                prediction_tensor[0]
            )

            source_image = tensor_to_image(
                sample["source"]
            )

            reference_image = tensor_to_image(
                sample["reference"]
            )

            target_image = tensor_to_image(
                target_tensor
            )

            # Model metrics.
            model_metrics = calculate_metrics(
                prediction,
                target_image,
            )

            # Identity baseline: unchanged source.
            identity_metrics = calculate_metrics(
                source_image,
                target_image,
            )

            # Save visual comparisons.
            save_comparison(
                sample_id,
                source_image,
                reference_image,
                target_image,
                prediction,
                model_metrics,
                identity_metrics,
            )

            # Save LUT weight maps.
            save_weight_maps(
                sample_id,
                weights[0],
            )

            row = {
                "sample_id": sample_id,
            }

            row.update({
                f"model_{key}": value
                for key, value in model_metrics.items()
            })

            row.update({
                f"identity_{key}": value
                for key, value in identity_metrics.items()
            })

            results.append(row)

            print(
                f"Sample {position + 1:03d}/{len(dataset):03d} "
                f"(ID {sample_id}) | "
                f"Model L1={model_metrics['L1']:.5f} | "
                f"Identity L1={identity_metrics['L1']:.5f} | "
                f"PSNR={model_metrics['PSNR']:.3f} dB | "
                f"SSIM={model_metrics['SSIM']:.4f}"
            )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    per_image_path = (
        OUTPUT_DIR / "metrics_per_image.csv"
    )

    df.to_csv(
        per_image_path,
        index=False,
    )

    metric_names = [
        "L1",
        "MSE",
        "PSNR",
        "SSIM",
    ]

    summary_rows = []

    for method in ("model", "identity"):

        row = {
            "method": method,
        }

        for metric in metric_names:

            column = f"{method}_{metric}"

            row[f"mean_{metric}"] = df[column].mean()

            row[f"std_{metric}"] = df[column].std(
                ddof=1
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        OUTPUT_DIR / "metrics_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    improvement = {
        "L1": (
            df["identity_L1"] - df["model_L1"]
        ).mean(),

        "MSE": (
            df["identity_MSE"] - df["model_MSE"]
        ).mean(),

        "PSNR": (
            df["model_PSNR"] - df["identity_PSNR"]
        ).mean(),

        "SSIM": (
            df["model_SSIM"] - df["identity_SSIM"]
        ).mean(),
    }

    print("\n" + "=" * 68)
    print("TST2K TEST SUMMARY")
    print("=" * 68)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\nMean model improvement over identity "
        "(positive is better):"
    )

    for metric, value in improvement.items():
        print(f"  {metric}: {value:+.6f}")

    wins = int(
        (
            df["model_L1"]
            < df["identity_L1"]
        ).sum()
    )

    print(
        f"\nModel beats identity on L1 for "
        f"{wins}/{len(df)} samples."
    )

    print("\nSaved files:")
    print("  Per-image metrics:", per_image_path)
    print("  Summary metrics:  ", summary_path)
    print("  Comparisons:      ", COMPARISON_DIR)
    print("  Weight maps:      ", WEIGHT_MAP_DIR)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
