
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PST50Dataset
from tst2k_dataset import TST2KDataset
from Models.VGGBase import PredictModel
from lut_train import apply_lut_bank, grading_loss


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"F:\Junshoku\TST2K")
CHECKPOINT_DIR = Path(
    r"F:\Junshoku\checkpoints_tst2k"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
LUT_SIZE = 16
NUM_LUTS = 9
BATCH_SIZE = 2

EPOCHS = 20
PATIENCE = 5
MIN_DELTA = 1e-4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

SEED = 42
PRETRAINED_VGG = True
USE_AMP = True

LUT_SMOOTH_WEIGHT = 0.01
MAP_SMOOTH_WEIGHT = 0.01


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAINING
# ============================================================

def main():
    seed_everything(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = USE_AMP and device.type == "cuda"

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    # Small initial experiment only.
    # Train: 1-40; validation: 41-50.
    train_dataset = TST2KDataset(
        root=ROOT,
        split="train",
        image_size=IMAGE_SIZE,
    )

    val_dataset = TST2KDataset(
        root=ROOT,
        split="val",
        image_size=IMAGE_SIZE,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = PredictModel(
        freeze_encoder=True,
        lut_size=LUT_SIZE,
        num_luts=NUM_LUTS,
        pretrained=PRETRAINED_VGG,
    ).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_path = CHECKPOINT_DIR / "best.pt"
    latest_path = CHECKPOINT_DIR / "latest.pt"

    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        print(f"\nEpoch {epoch}/{EPOCHS}")

        for step, batch in enumerate(train_loader, start=1):

            source = batch["source"].to(
                device, non_blocking=True
            )
            reference = batch["reference"].to(
                device, non_blocking=True
            )
            target = batch["target"].to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            # VGG encoder and predictor.
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                luts, weights = model(source, reference)

            # LUT interpolation and loss in FP32 for stability.
            with torch.autocast(
                device_type=device.type,
                enabled=False,
            ):
                luts = luts.float()
                weights = weights.float()

                # Shape: (B, K, 3, H, W)
                transformed = apply_lut_bank(
                    source.float(), luts
                )

                # Broadcast weights over RGB channels.
                # Shape: (B, K, 1, H, W)
                prediction = (
                    transformed * weights.unsqueeze(2)
                ).sum(dim=1)

                prediction = prediction.clamp(0.0, 1.0)

                loss, metrics = grading_loss(
                    prediction=prediction,
                    target=target.float(),
                    luts=luts,
                    weights=weights,
                    lut_smooth_weight=LUT_SMOOTH_WEIGHT,
                    map_smooth_weight=MAP_SMOOTH_WEIGHT,
                )

            # Backpropagation.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            n = source.size(0)
            train_loss_sum += loss.item() * n
            train_samples += n

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"  Train {step}/{len(train_loader)}"
                    f" | Loss: {loss.item():.5f}"
                    f" | L1: {metrics['l1']:.5f}"
                )

        train_loss = train_loss_sum / max(train_samples, 1)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss_sum = 0.0
        val_l1_sum = 0.0
        val_samples = 0

        with torch.no_grad():

            for batch in val_loader:

                source = batch["source"].to(
                    device, non_blocking=True
                )
                reference = batch["reference"].to(
                    device, non_blocking=True
                )
                target = batch["target"].to(
                    device, non_blocking=True
                )

                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    luts, weights = model(source, reference)

                with torch.autocast(
                    device_type=device.type,
                    enabled=False,
                ):
                    luts = luts.float()
                    weights = weights.float()

                    transformed = apply_lut_bank(
                        source.float(), luts
                    )

                    prediction = (
                        transformed * weights.unsqueeze(2)
                    ).sum(dim=1).clamp(0.0, 1.0)

                    loss, metrics = grading_loss(
                        prediction,
                        target.float(),
                        luts,
                        weights,
                        LUT_SMOOTH_WEIGHT,
                        MAP_SMOOTH_WEIGHT,
                    )

                n = source.size(0)
                val_loss_sum += loss.item() * n
                val_l1_sum += metrics["l1"] * n
                val_samples += n

        val_loss = val_loss_sum / max(val_samples, 1)
        val_l1 = val_l1_sum / max(val_samples, 1)

        print(f"\nTrain loss: {train_loss:.6f}")
        print(f"Val loss:   {val_loss:.6f}")
        print(f"Val L1:     {val_l1:.6f}")

        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_l1": val_l1,
            "config": {
                "image_size": IMAGE_SIZE,
                "lut_size": LUT_SIZE,
                "num_luts": NUM_LUTS,
            },
        }

        torch.save(checkpoint, latest_path)

        if val_loss < best_val_loss - MIN_DELTA:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(checkpoint, best_path)

            print("Saved best checkpoint:", best_path)

        else:
            epochs_without_improvement += 1

            print(
                "No improvement:",
                f"{epochs_without_improvement}/{PATIENCE}",
            )

        if device.type == "cuda":
            allocated = (
                torch.cuda.memory_allocated() / 1024**3
            )
            reserved = (
                torch.cuda.memory_reserved() / 1024**3
            )

            print(
                f"GPU memory: {allocated:.2f} GB allocated"
                f" | {reserved:.2f} GB reserved"
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("Best checkpoint:", best_path)
    print("Latest checkpoint:", latest_path)
    print(
        "Note: this 40/10 split is a pipeline experiment, "
        "not a reliable generalization benchmark."
    )
    print(
        f"  Train {step}/{len(train_loader)}"
        f" | Loss: {loss.item():.5f}"
        f" | L1: {metrics['l1']:.5f}"
        f" | LUT curvature: {metrics['lut_curvature']:.5f}"
        f" | Usage: {metrics['lut_usage']:.5f}"
    )


if __name__ == "__main__":
    main()
