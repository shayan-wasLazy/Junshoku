import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import load_config
from .data import TST2KDataset
from .model import NodeGraph
from .graph import apply_graph


def to_image(tensor):
    return tensor.detach().float().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()


def metrics(prediction, target):
    prediction = np.clip(prediction, 0, 1).astype(np.float32)
    target = np.clip(target, 0, 1).astype(np.float32)
    diff = prediction - target
    return {
        "L1": float(np.abs(diff).mean()),
        "MSE": float(np.square(diff).mean()),
        "PSNR": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
        "SSIM": float(structural_similarity(target, prediction, data_range=1.0, channel_axis=-1)),
    }


def save_comparison(sample_id, source, reference, target, prediction, model_m, identity_m, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (title, image) in zip(axes, [
        ("Content / Source", source), ("Style Reference", reference),
        ("Ground Truth", target), ("Node Graph Prediction", prediction)
    ]):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"TST2K {sample_id} | Prediction L1={model_m['L1']:.4f} | "
        f"Identity L1={identity_m['L1']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_{sample_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_context_maps(sample_id, maps, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Node 1: Exposure / WB", "Node 2: Contrast", "Node 3: Creative Colour"]
    for i, (ax, title, context) in enumerate(zip(axes, titles, maps)):
        heat = context.detach().float().cpu().squeeze().numpy()
        im = ax.imshow(heat, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{title}\nmean={heat.mean():.3f}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        np.save(out_dir / f"sample_{sample_id}_node{i+1}_map.npy", heat)
    fig.suptitle(f"Learned node context maps — TST2K {sample_id}")
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_{sample_id}_context_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Junshoku node graph on the TST2K test split.")
    parser.add_argument("--config", default="node_graph/configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="Outputs/node_graph_evaluation")
    parser.add_argument("--sample-id", default=None, help="Evaluate one test sample, e.g. 0429.")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or str(Path(cfg["training"]["checkpoint_dir"]) / "best.pt")
    output_dir = Path(args.output_dir)
    comparisons_dir = output_dir / "comparisons"
    maps_dir = output_dir / "context_maps"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("JUNSHOKU — TST2K NODE GRAPH EVALUATION")
    print("=" * 68)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", checkpoint_path)

    dataset = TST2KDataset(
        root=cfg["data"]["root"],
        split="test",
        image_size=cfg["data"]["image_size"],
        augment=False,
        manifest_path=cfg["data"]["manifest"],
        seed=cfg["seed"],
    )
    if args.sample_id is not None:
        indices = [i for i, sid in enumerate(dataset.ids) if str(sid) == str(args.sample_id)]
        if not indices:
            raise ValueError(f"Sample {args.sample_id} is not in the TST2K test split.")
    else:
        indices = list(range(len(dataset)))
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    print("Test samples to evaluate:", len(indices))

    model = NodeGraph(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown")

    rows = []
    with torch.inference_mode():
        for position, index in enumerate(indices, start=1):
            sample = dataset[index]
            sample_id = str(sample["sample_id"])
            source = sample["source"].unsqueeze(0).to(device)
            reference = sample["reference"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)

            result = model(source, reference)
            prediction_tensor = apply_graph(source, result, cfg)

            source_img = to_image(source)
            reference_img = to_image(reference)
            target_img = to_image(target)
            prediction_img = to_image(prediction_tensor)

            model_metrics = metrics(prediction_img, target_img)
            identity_metrics = metrics(source_img, target_img)
            save_comparison(sample_id, source_img, reference_img, target_img,
                            prediction_img, model_metrics, identity_metrics, comparisons_dir)
            save_context_maps(sample_id, result["maps"], maps_dir)

            row = {"sample_id": sample_id}
            row.update({f"model_{k}": v for k, v in model_metrics.items()})
            row.update({f"identity_{k}": v for k, v in identity_metrics.items()})
            for node_name, params in zip(
                ("exposure_wb", "contrast", "creative_colour"), result["params"]
            ):
                for j, value in enumerate(params[0].detach().float().cpu().tolist()):
                    row[f"{node_name}_param_{j+1}"] = value
            for j, context in enumerate(result["maps"]):
                row[f"node{j+1}_map_mean"] = float(context.mean().item())
                row[f"node{j+1}_map_std"] = float(context.std(unbiased=False).item())
                row[f"node{j+1}_map_min"] = float(context.min().item())
                row[f"node{j+1}_map_max"] = float(context.max().item())
            rows.append(row)

            print(
                f"[{position:03d}/{len(indices):03d}] ID {sample_id} | "
                f"L1={model_metrics['L1']:.5f} (identity {identity_metrics['L1']:.5f}) | "
                f"PSNR={model_metrics['PSNR']:.3f} dB | SSIM={model_metrics['SSIM']:.4f}"
            )

    if not rows:
        raise RuntimeError("No test samples were evaluated.")
    csv_path = output_dir / "metrics_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = ("L1", "MSE", "PSNR", "SSIM")
    print("\nSUMMARY (mean over evaluated samples)")
    for method in ("model", "identity"):
        print(method.upper())
        for name in metric_names:
            vals = [r[f"{method}_{name}"] for r in rows]
            print(f"  {name}: {np.mean(vals):.6f} ± {np.std(vals, ddof=1) if len(vals)>1 else 0:.6f}")
    wins = sum(r["model_L1"] < r["identity_L1"] for r in rows)
    print(f"\nModel beats unchanged-source baseline on L1: {wins}/{len(rows)}")
    print("Per-image CSV:", csv_path)
    print("Comparison images:", comparisons_dir)
    print("Context maps:", maps_dir)


if __name__ == "__main__":
    main()
