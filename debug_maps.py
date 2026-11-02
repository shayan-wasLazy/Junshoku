
import torch
import yaml
from pathlib import Path

from node_graph.model import NodeGraph
from node_graph.data import TST2KDataset

CONFIG = "node_graph/configs/default.yaml"

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TST2KDataset(
    root=cfg["data"]["root"],
    split="test",
    image_size=cfg["data"]["image_size"],
    augment=False,
    manifest_path=cfg["data"]["manifest"],
    seed=cfg["seed"],
)

model = NodeGraph(cfg).to(device)

checkpoint_path = (
    Path(cfg["training"]["checkpoint_dir"]) / "best.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

state = checkpoint.get(
    "model",
    checkpoint.get("state_dict", checkpoint)
)

model.load_state_dict(state, strict=True)
model.eval()

print("Checkpoint:", checkpoint_path)
print("Map limits:", cfg["model"]["map_min_gate"],
      cfg["model"]["map_max_gate"])

with torch.inference_mode():
    for sample_idx in [0, 1, 2]:
        sample = dataset[sample_idx]

        source = sample["source"].unsqueeze(0).to(device)
        reference = sample["reference"].unsqueeze(0).to(device)

        result = model(source, reference)

        print("\nSample:", sample["sample_id"])

        for i, context in enumerate(result["maps"]):
            print(
                f"Node {i+1}: "
                f"min={context.min().item():.6f}, "
                f"max={context.max().item():.6f}, "
                f"mean={context.mean().item():.6f}, "
                f"std={context.std(unbiased=False).item():.8f}"
            )

print("\nDiagnostic completed.")
