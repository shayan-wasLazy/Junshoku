
from pathlib import Path
import torch

from Models.VGGBase import PredictModel

ROOT = Path(r"F:\Junshoku")
CHECKPOINT = ROOT / "checkpoints_spatial_lut" / "best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PredictModel(
    freeze_encoder=True,
    lut_size=16,
    num_luts=8,
    pretrained=True,
).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

if "model_state_dict" in checkpoint:
    state = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
    state = checkpoint["model"]
else:
    state = checkpoint

state = {
    key.removeprefix("module."): value
    for key, value in state.items()
}

model.load_state_dict(state)
model.eval()

luts = model.lut_bank().detach().float().cpu()

# Shape expected: (K, N, N, N, 3)
print("LUT bank shape:", tuple(luts.shape))

k, n, _, _, _ = luts.shape

axis = torch.linspace(0, 1, n)
r, g, b = torch.meshgrid(axis, axis, axis, indexing="ij")
identity = torch.stack([r, g, b], dim=-1)

print("\n--- Difference from identity LUT ---")

for i in range(k):
    difference = luts[i] - identity

    print(
        f"LUT {i + 1}: "
        f"mean_abs_deviation={difference.abs().mean().item():.6f}, "
        f"max_abs_deviation={difference.abs().max().item():.6f}, "
        f"std={luts[i].std().item():.6f}"
    )

print("\n--- Pairwise differences between LUTs ---")

for i in range(k):
    for j in range(i + 1, k):
        difference = (luts[i] - luts[j]).abs().mean().item()
        print(f"LUT {i + 1} vs LUT {j + 1}: {difference:.6f}")
