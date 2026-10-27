from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PST50Dataset
from Models.VGG_base import PredictModel
from lut_train import apply_lut


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("F:/Cog")

TRAIN_CSV = ROOT / "data" / "pairs" / "train.csv"
VAL_CSV = ROOT / "data" / "pairs" / "val.csv"

# Separate folder to preserve previous experiments
CHECKPOINT_DIR = ROOT / "checkpoints_VGG_2epochs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
LUT_SIZE = 16
BATCH_SIZE = 2

EPOCHS = 2
PATIENCE = 3
MIN_DELTA = 0.0001

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


# ============================================================
# DEVICE
# ============================================================

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)


# ============================================================
# DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = PST50Dataset(
    root=ROOT / "data" / "pairs" / "train",
    image_size=IMAGE_SIZE
)

val_dataset = PST50Dataset(
    root=ROOT / "data" / "pairs" / "val",
    image_size=IMAGE_SIZE
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

if len(train_dataset) == 0:
    raise RuntimeError(f"No training samples found: {TRAIN_CSV}")

if len(val_dataset) == 0:
    raise RuntimeError(f"No validation samples found: {VAL_CSV}")


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP
)


# ============================================================
# MODEL
# ============================================================

model = PredictModel(
    freeze_encoder=True,
    lut_size=LUT_SIZE
).to(DEVICE)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("\nModel created.")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(train_loader, start=1):

        source = batch["source"].to(
            DEVICE,
            non_blocking=True
        )

        reference = batch["reference"].to(
            DEVICE,
            non_blocking=True
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            predicted_lut = model(source, reference)

            predicted_image = apply_lut(
                source,
                predicted_lut
            )

            loss = criterion(
                predicted_image,
                target
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_index % 100 == 0:
            print(
                f"  Batch {batch_index}/{len(train_loader)}"
                f" | Running L1: "
                f"{total_loss / total_samples:.6f}"
            )

    return total_loss / total_samples


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:

        source = batch["source"].to(
            DEVICE,
            non_blocking=True
        )

        reference = batch["reference"].to(
            DEVICE,
            non_blocking=True
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            predicted_lut = model(source, reference)

            predicted_image = apply_lut(
                source,
                predicted_lut
            )

            loss = criterion(
                predicted_image,
                target
            )

        batch_size = source.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# TRAINING + EARLY STOPPING
# ============================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

best_checkpoint = CHECKPOINT_DIR / "best.pt"
latest_checkpoint = CHECKPOINT_DIR / "latest.pt"

print("\nStarting training...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 60)

    # Train
    train_loss = train_one_epoch()

    # Validate
    val_loss = validate()

    print(f"\nTrain L1: {train_loss:.6f}")
    print(f"Val L1:   {val_loss:.6f}")

    # Save latest model and optimizer state
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        latest_checkpoint
    )

    # Check whether validation loss improved
    if val_loss < best_val_loss - MIN_DELTA:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        # Save plain state_dict for compatibility
        # with evaluators using model.load_state_dict(...)
        torch.save(
            model.state_dict(),
            best_checkpoint
        )

        print("\nNew best model saved:")
        print(best_checkpoint)

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    # GPU memory usage
    if DEVICE == "cuda":

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

    # Early stopping check
    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered after "
            f"{PATIENCE} consecutive epochs without "
            f"validation improvement."
        )

        break


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("Training complete.")

print(f"Best validation L1: {best_val_loss:.6f}")
print(f"Best checkpoint: {best_checkpoint}")
print(f"Latest checkpoint: {latest_checkpoint}")