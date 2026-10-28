
import torch
from torch.utils.data import DataLoader
from Models.VGGBase import PredictModel
from dataset import PST50Dataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = r"F:\Junshoku\PST50"
CHECKPOINT_PATH = r"F:\Junshoku\checkpoints_usage_0005\best.pt"

IMAGE_SIZE = 256
LUT_SIZE = 16
NUM_LUTS = 9
VAL_INDICES = list(range(41, 51))
BATCH_SIZE = 1

# --------------------------------------------------
# Load validation dataset
# --------------------------------------------------
dataset = PST50Dataset(
    root=DATASET_ROOT,
    indices=VAL_INDICES,
    image_size=IMAGE_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# --------------------------------------------------
# Load model and checkpoint
# --------------------------------------------------
model = PredictModel(
    num_luts=NUM_LUTS,
    lut_size=LUT_SIZE,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

state = (
    checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
    else checkpoint
)

model.load_state_dict(state)
model.eval()

print(f"Device: {DEVICE}")
print(f"Validation samples: {len(dataset)}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

# --------------------------------------------------
# Aggregate LUT usage across validation samples
# --------------------------------------------------
sum_weights = torch.zeros(NUM_LUTS, dtype=torch.float64)
sum_max_weights = torch.zeros(NUM_LUTS, dtype=torch.float64)
sum_dominant_pixels = torch.zeros(NUM_LUTS, dtype=torch.float64)

total_pixels = 0
total_samples = 0

with torch.no_grad():
    for sample in loader:
        source = sample["source"].to(DEVICE)
        reference = sample["reference"].to(DEVICE)

        output = model(source, reference)

        # PredictModel returns (LUT bank, spatial weights)
        _, weights = output

        # Expected shape: (B, K, H, W)
        if weights.ndim != 4 or weights.shape[1] != NUM_LUTS:
            raise ValueError(
                f"Unexpected weight shape: {tuple(weights.shape)}. "
                "Expected (B, 9, H, W)."
            )

        weights = weights.detach().cpu().double()
        batch_size, num_luts, height, width = weights.shape

        # Mean softmax weight per LUT for each image
        sum_weights += weights.mean(dim=(0, 2, 3))

        # Per-image maximum weight for each LUT
        sum_max_weights += weights.amax(dim=(2, 3)).sum(dim=0)

        # Percentage of pixels where each LUT has the highest weight
        dominant = weights.argmax(dim=1)

        for k in range(NUM_LUTS):
            sum_dominant_pixels[k] += (dominant == k).sum()

        total_pixels += batch_size * height * width
        total_samples += batch_size

mean_weights = sum_weights / total_samples
mean_max_weights = sum_max_weights / total_samples
dominant_percentages = (
    sum_dominant_pixels / total_pixels * 100.0
)

# --------------------------------------------------
# Report
# --------------------------------------------------
print("\n" + "=" * 72)
print("SEMANTIC LUT USAGE — VALIDATION SET")
print("=" * 72)

print(
    f"{'LUT':<8}"
    f"{'Mean weight':>15}"
    f"{'Mean max weight':>19}"
    f"{'Dominant pixels':>20}"
)

for k in range(NUM_LUTS):
    print(
        f"{k + 1:<8}"
        f"{mean_weights[k].item():>15.4f}"
        f"{mean_max_weights[k].item():>19.4f}"
        f"{dominant_percentages[k].item():>19.2f}%"
    )

print("\nChecks:")
print(f"Sum of mean weights: {mean_weights.sum().item():.4f}")
print(
    "Sum of dominant-pixel percentages: "
    f"{dominant_percentages.sum().item():.2f}%"
)

print("\nInterpretation:")
print("- Mean weight measures average contribution across all pixels.")
print("- Mean max weight measures each LUT's strongest activation per image,")
print("  averaged over validation images.")
print("- Dominant pixels measures how often a LUT wins the per-pixel argmax.")
print("- A LUT can contribute without being the dominant LUT.")
