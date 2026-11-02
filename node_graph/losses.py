import torch
import torch.nn.functional as F


def total_loss(prediction, target, maps, cfg):
    reconstruction = F.l1_loss(prediction, target)
    smoothness = prediction.new_tensor(0.0)
    variance_floor = prediction.new_tensor(0.0)

    min_std = float(cfg["loss"].get("map_min_std", 0.03))
    for context in maps:
        dx = context[:, :, :, 1:] - context[:, :, :, :-1]
        dy = context[:, :, 1:, :] - context[:, :, :-1, :]
        smoothness = smoothness + dx.abs().mean() + dy.abs().mean()

        # A small variance floor discourages a perfectly constant gate.
        # It does not prescribe where a node should act; that must be learned
        # from the paired reconstruction signal.
        spatial_std = context.flatten(2).std(dim=2, unbiased=False).mean()
        variance_floor = variance_floor + F.relu(min_std - spatial_std).square()
    smoothness = smoothness / len(maps)
    variance_floor = variance_floor / len(maps)

    weights = cfg["loss"]
    total = (
        float(weights["reconstruction"]) * reconstruction
        + float(weights["map_smoothness"]) * smoothness
        + float(weights.get("map_variance", 0.01)) * variance_floor
    )
    return total, {
        "reconstruction": reconstruction.detach(),
        "map_smoothness": smoothness.detach(),
        "map_variance": variance_floor.detach(),
    }
