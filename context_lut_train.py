
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Models.ContextLUT import ContextLUTModel
from tst2k_dataset import TST2KDataset


# ---------------- CONFIG ----------------

ROOT = Path(r"F:\Junshoku\TST2K")
SAVE_DIR = Path(r"F:\Junshoku\checkpoints_context_lut")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
LUT_SIZE = 16
BATCH_SIZE = 2
EPOCHS = 20
PATIENCE = 5

LR = 1e-4
WEIGHT_DECAY = 1e-5
LUT_SMOOTH_WEIGHT = 0.01
MAP_SMOOTH_WEIGHT = 0.01
NUM_WORKERS = 0
USE_AMP = True
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- 3D LUT INTERPOLATION ----------------

def apply_3d_lut(image, lut):
    """
    image: (B, 3, H, W), RGB in [0, 1]
    lut:   (N, N, N, 3), indexed by R, G, B
    """
    batch, _, height, width = image.shape
    size = lut.shape[0]

    pixels = image.permute(0, 2, 3, 1).reshape(
        batch, -1, 3
    ).clamp(0, 1)

    coords = pixels * (size - 1)
    lower = coords.floor().long()
    upper = (lower + 1).clamp(max=size - 1)
    frac = coords - lower.to(coords.dtype)

    r0, g0, b0 = lower.unbind(-1)
    r1, g1, b1 = upper.unbind(-1)
    fr, fg, fb = frac.unbind(-1)

    flat = lut.reshape(-1, 3)

    def lookup(r, g, b):
        return flat[(r * size + g) * size + b]

    def blend(a, b, w):
        return a * (1 - w.unsqueeze(-1)) + b * w.unsqueeze(-1)

    c000 = lookup(r0, g0, b0)
    c001 = lookup(r0, g0, b1)
    c010 = lookup(r0, g1, b0)
    c011 = lookup(r0, g1, b1)
    c100 = lookup(r1, g0, b0)
    c101 = lookup(r1, g0, b1)
    c110 = lookup(r1, g1, b0)
    c111 = lookup(r1, g1, b1)

    c00 = blend(c000, c001, fb)
    c01 = blend(c010, c011, fb)
    c10 = blend(c100, c101, fb)
    c11 = blend(c110, c111, fb)

    c0 = blend(c00, c01, fg)
    c1 = blend(c10, c11, fg)
    output = blend(c0, c1, fr)

    return output.reshape(
        batch, height, width, 3
    ).permute(0, 3, 1, 2).contiguous()


def apply_context_lut(source, luts, context):
    """
    luts:    (2, N, N, N, 3)
    context: (B, 1, H, W), values in [0, 1]
    """
    output_0 = apply_3d_lut(source, luts[0])
    output_1 = apply_3d_lut(source, luts[1])

    # Interpolate between the two LUT slices per pixel.
    prediction = (
        output_0 * (1.0 - context)
        + output_1 * context
    )

    return prediction.clamp(0, 1)


# ---------------- REGULARIZATION ----------------

def lut_smoothness(luts):
    terms = []
    for dim in (1, 2, 3):
        terms.append(
            torch.diff(luts, dim=dim).abs().mean()
        )
    return sum(terms) / len(terms)


def context_smoothness(context):
    horizontal = (
        context[:, :, :, 1:] - context[:, :, :, :-1]
    ).abs().mean()
    vertical = (
        context[:, :, 1:, :] - context[:, :, :-1, :]
    ).abs().mean()
    return (horizontal + vertical) / 2


def calculate_loss(prediction, target, luts, context):
    reconstruction = F.l1_loss(prediction, target)
    lut_reg = lut_smoothness(luts)
    map_reg = context_smoothness(context)

    total = (
        reconstruction
        + LUT_SMOOTH_WEIGHT * lut_reg
        + MAP_SMOOTH_WEIGHT * map_reg
    )

    return total, reconstruction.detach().item()


# ---------------- TRAINING ----------------

def main():
    seed_everything(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = USE_AMP and device.type == "cuda"

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = TST2KDataset(
        root=ROOT, split="train", image_size=IMAGE_SIZE
    )
    val_dataset = TST2KDataset(
        root=ROOT, split="val", image_size=IMAGE_SIZE
    )

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

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    model = ContextLUTModel(
        lut_size=LUT_SIZE,
        pretrained=True,
        freeze_encoder=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = float("inf")
    no_improvement = 0

    best_path = SAVE_DIR / "best_context_lut.pt"
    latest_path = SAVE_DIR / "latest_context_lut.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_l1 = 0.0
        train_count = 0

        for batch in train_loader:
            source = batch["source"].to(device)
            reference = batch["reference"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, enabled=use_amp
            ):
                luts, context = model(source, reference)

            with torch.autocast(
                device_type=device.type, enabled=False
            ):
                luts = luts.float()
                context = context.float()

                prediction = apply_context_lut(
                    source.float(), luts, context
                )
                loss, l1 = calculate_loss(
                    prediction, target.float(), luts, context
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            n = source.size(0)
            train_total += loss.item() * n
            train_l1 += l1 * n
            train_count += n

        train_loss = train_total / max(train_count, 1)
        train_l1 /= max(train_count, 1)

        model.eval()
        val_total = 0.0
        val_l1 = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                source = batch["source"].to(device)
                reference = batch["reference"].to(device)
                target = batch["target"].to(device)

                with torch.autocast(
                    device_type=device.type, enabled=use_amp
                ):
                    luts, context = model(source, reference)

                with torch.autocast(
                    device_type=device.type, enabled=False
                ):
                    luts = luts.float()
                    context = context.float()
                    prediction = apply_context_lut(
                        source.float(), luts, context
                    )
                    loss, l1 = calculate_loss(
                        prediction, target.float(), luts, context
                    )

                n = source.size(0)
                val_total += loss.item() * n
                val_l1 += l1 * n
                val_count += n

        val_loss = val_total / max(val_count, 1)
        val_l1 /= max(val_count, 1)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss {train_loss:.5f} | "
            f"Train L1 {train_l1:.5f} | "
            f"Val loss {val_loss:.5f} | "
            f"Val L1 {val_l1:.5f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_l1": val_l1,
            "config": {
                "image_size": IMAGE_SIZE,
                "lut_size": LUT_SIZE,
                "context_slices": 2,
            },
        }
        torch.save(checkpoint, latest_path)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improvement = 0
            torch.save(checkpoint, best_path)
            print("Saved best checkpoint:", best_path)
        else:
            no_improvement += 1
            print(f"No improvement: {no_improvement}/{PATIENCE}")

        if device.type == "cuda":
            print(
                "GPU memory allocated:",
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            )

        if no_improvement >= PATIENCE:
            print("Early stopping.")
            break

    print("Training complete.")
    print("Best validation loss:", best_val)
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
