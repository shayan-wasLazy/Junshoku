
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from dataset import PST50Dataset
from Models.VGGBase import PredictModel
from lut_train import apply_lut_bank


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"F:\Junshoku")
DATASET_ROOT = ROOT / "PST50"
CHECKPOINT_PATH = ROOT / "checkpoints_spatial_lut" / "best.pt"
OUTPUT_DIR = ROOT / "Outputs" / "evaluation"

IMAGE_SIZE = 256
LUT_SIZE = 16
NUM_LUTS = 8

# Must match the validation indices used in train.py
VAL_INDICES = list(range(41, 51))

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "comparisons").mkdir(exist_ok=True)
(OUTPUT_DIR / "weight_maps").mkdir(exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_checkpoint(model, path):
    """Load a checkpoint from common PyTorch checkpoint formats."""
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = None

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            if key in checkpoint and isinstance(
                checkpoint[key], dict
            ):
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            # Also supports checkpoints containing only model weights.
            if checkpoint and all(
                isinstance(k, str) for k in checkpoint
            ):
                state_dict = checkpoint
            else:
                raise RuntimeError(
                    "Could not locate model weights in checkpoint. "
                    f"Available keys: {list(checkpoint.keys())}"
                )
    else:
        raise RuntimeError("Unexpected checkpoint format.")

    # Support checkpoints saved using DataParallel.
    state_dict = {
        k.removeprefix("module."): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def tensor_to_image(tensor):
    """Convert a (3,H,W) tensor in [0,1] into an RGB NumPy image."""
    return (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def calculate_metrics(prediction, target):
    """Calculate image-quality metrics for RGB images in [0,1]."""
    prediction = np.clip(prediction, 0, 1).astype(np.float32)
    target = np.clip(target, 0, 1).astype(np.float32)

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
    metrics,
):
    """Save the four-image comparison for one validation example."""
    images = [
        ("Content / Source", source),
        ("Style Reference", reference),
        ("Ground Truth", target),
        ("Junshoku Prediction", prediction),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (title, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Validation sample {sample_id} | "
        f"L1={metrics['L1']:.4f} | "
        f"PSNR={metrics['PSNR']:.2f} dB | "
        f"SSIM={metrics['SSIM']:.4f}"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "comparisons" /
        f"sample_{sample_id:02d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_weight_maps(sample_id, weights):
    """
    Save eight spatial LUT weight maps.

    weights shape: (K,H,W)
    """
    weights = weights.detach().float().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for lut_index, ax in enumerate(axes):
        heatmap = ax.imshow(
            weights[lut_index],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"LUT {lut_index + 1}")
        ax.axis("off")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Spatial LUT Weights — Validation Sample {sample_id}"
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "weight_maps" /
        f"sample_{sample_id:02d}_weights.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save raw maps so they can be analyzed later.
    np.save(
        OUTPUT_DIR / "weight_maps" /
        f"sample_{sample_id:02d}_weights.npy",
        weights,
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 65)
    print("JUNSHOKU — MODEL EVALUATION")
    print("=" * 65)

    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load validation data.
    dataset = PST50Dataset(
        root=DATASET_ROOT,
        indices=VAL_INDICES,
        image_size=IMAGE_SIZE,
    )

    print(f"Validation samples: {len(dataset)}")

    # Recreate the architecture used for training.
    model = PredictModel(
        freeze_encoder=True,
        lut_size=LUT_SIZE,
        num_luts=NUM_LUTS,
        pretrained=True,
    ).to(DEVICE)

    checkpoint = load_checkpoint(model, CHECKPOINT_PATH)
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

    print(f"Loaded: {CHECKPOINT_PATH}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    results = []

    # No gradients needed during evaluation.
    with torch.inference_mode():
        for position in range(len(dataset)):
            sample = dataset[position]

            source = sample["source"].unsqueeze(0).to(DEVICE)
            reference = sample["reference"].unsqueeze(0).to(DEVICE)
            target_tensor = sample["target"]

            # Predict LUT bank and spatial mixture weights.
            luts, weights = model(source, reference)

            # Apply each LUT to the source image.
            transformed = apply_lut_bank(
                source.float(),
                luts.float(),
            )

            # Blend the transformed outputs using spatial weights.
            # transformed: (B,K,3,H,W)
            # weights:     (B,K,H,W)
            prediction_tensor = (
                transformed * weights.unsqueeze(2).float()
            ).sum(dim=1).clamp(0, 1)

            prediction = tensor_to_image(prediction_tensor[0])
            source_image = tensor_to_image(source[0])
            reference_image = tensor_to_image(reference[0])
            target_image = tensor_to_image(target_tensor)

            # Evaluate the model against the ground truth.
            model_metrics = calculate_metrics(
                prediction,
                target_image,
            )

            # Identity baseline: use the original source unchanged.
            identity_metrics = calculate_metrics(
                source_image,
                target_image,
            )

            sample_id = VAL_INDICES[position]

            save_comparison(
                sample_id,
                source_image,
                reference_image,
                target_image,
                prediction,
                model_metrics,
            )

            save_weight_maps(
                sample_id,
                weights[0],
            )

            results.append({
                "sample_id": sample_id,
                **{
                    f"model_{key}": value
                    for key, value in model_metrics.items()
                },
                **{
                    f"identity_{key}": value
                    for key, value in identity_metrics.items()
                },
            })

            print(
                f"Sample {sample_id:02d} | "
                f"Model L1={model_metrics['L1']:.5f} | "
                f"PSNR={model_metrics['PSNR']:.3f} dB | "
                f"SSIM={model_metrics['SSIM']:.4f}"
            )

    # Save per-image results.
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "metrics_per_image.csv"
    df.to_csv(csv_path, index=False)

    # Summarize model versus identity baseline.
    metric_names = ["L1", "MSE", "PSNR", "SSIM"]
    summary_rows = []

    for method in ("model", "identity"):
        row = {"method": method}

        for metric in metric_names:
            column = f"{method}_{metric}"
            row[f"mean_{metric}"] = df[column].mean()
            row[f"std_{metric}"] = df[column].std(ddof=1)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved files:")
    print(f"  Per-image metrics: {csv_path}")
    print(f"  Summary metrics:   {summary_path}")
    print(f"  Comparisons:       {OUTPUT_DIR / 'comparisons'}")
    print(f"  Weight maps:       {OUTPUT_DIR / 'weight_maps'}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
