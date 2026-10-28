
from pathlib import Path
import torch

from Models.VGGBase import PredictModel
from lut_train import lut_smoothness
from Models.VGGBase import make_initial_luts

ROOT = Path(r"F:\Junshoku")
CHECKPOINT = ROOT / "checkpoints_tst2k" / "best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PredictModel(
    freeze_encoder=True,
    lut_size=16,
    num_luts=9,
    pretrained=False,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

if isinstance(checkpoint, dict):
    state = next(
        (
            checkpoint[k]
            for k in ("model_state_dict", "model", "state_dict")
            if k in checkpoint and isinstance(checkpoint[k], dict)
        ),
        checkpoint,
    )
else:
    state = checkpoint

if state and all(k.startswith("module.") for k in state):
    state = {k.removeprefix("module."): v for k, v in state.items()}

model.load_state_dict(state)
model.eval()

learned = model.lut_bank().detach().float()
initial = make_initial_luts(size=16).to(DEVICE).float()

identity = initial[0]

print("\nLEARNED LUT DIAGNOSTICS")
print("-" * 75)
print(
    f"{'LUT':<8}{'Change from init':>19}"
    f"{'Distance from identity':>25}{'Output range':>18}"
)

for i in range(learned.shape[0]):
    lut = learned[i]

    change = (lut - initial[i]).abs().mean().item()
    identity_distance = (lut - identity).abs().mean().item()
    output_range = (
        lut.min().item(),
        lut.max().item(),
    )

    print(
        f"{i + 1:<8}{change:>19.6f}"
        f"{identity_distance:>25.6f}"
        f"{output_range[0]:.3f} to {output_range[1]:.3f}"
    )

print("\nLUT bank curvature:", lut_smoothness(learned).item())
print("Learned LUTs saved in checkpoint:", learned.shape)



# Add this block to the end of inspect_lut_bank.py,
# after the existing diagnostics.

print("\nPAIRWISE LUT DISTANCES")
print("Mean absolute difference between each pair:")
print("      " + " ".join(f"{i+1:>8}" for i in range(learned.shape[0])))

for i in range(learned.shape[0]):
    row = []

    for j in range(learned.shape[0]):
        distance = (learned[i] - learned[j]).abs().mean().item()
        row.append(f"{distance:8.4f}")

    print(f"{i+1:>4}  " + " ".join(row))
