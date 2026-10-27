
import torch
import torch.nn.functional as F


def apply_lut(image, lut):
    """
    image: (B, 3, H, W), RGB in [0, 1]
    lut:   (N, N, N, 3), indexed by R, G, B
    returns: (B, 3, H, W)
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Image must have shape (B, 3, H, W)")

    if lut.ndim != 4 or lut.shape[-1] != 3:
        raise ValueError("LUT must have shape (N, N, N, 3)")

    batch, _, height, width = image.shape
    size = lut.shape[0]

    pixels = image.permute(0, 2, 3, 1).reshape(
        batch, -1, 3
    ).clamp(0.0, 1.0)

    coords = pixels * (size - 1)

    lower = coords.floor().long()
    upper = (lower + 1).clamp(max=size - 1)
    fraction = coords - lower.to(coords.dtype)

    r0, g0, b0 = lower.unbind(-1)
    r1, g1, b1 = upper.unbind(-1)

    fr, fg, fb = fraction.unbind(-1)
    flat_lut = lut.reshape(-1, 3)

    def lookup(r, g, b):
        index = (r * size + g) * size + b
        return flat_lut[index]

    def blend(a, b, weight):
        weight = weight.unsqueeze(-1)
        return a * (1.0 - weight) + b * weight

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


def apply_lut_bank(image, luts):
    """
    image: (B, 3, H, W)
    luts:  (K, N, N, N, 3)
    returns: (B, K, 3, H, W)
    """
    if luts.ndim != 5:
        raise ValueError("LUT bank must have shape (K, N, N, N, 3)")

    return torch.stack(
        [apply_lut(image, lut) for lut in luts],
        dim=1,
    )


def lut_smoothness(luts):
    """Penalize abrupt changes between neighboring LUT entries."""
    loss_r = (
        luts[:, 1:, :, :, :] - luts[:, :-1, :, :, :]
    ).abs().mean()

    loss_g = (
        luts[:, :, 1:, :, :] - luts[:, :, :-1, :, :]
    ).abs().mean()

    loss_b = (
        luts[:, :, :, 1:, :] - luts[:, :, :, :-1, :]
    ).abs().mean()

    return (loss_r + loss_g + loss_b) / 3.0


def spatial_smoothness(weights):
    """Encourage neighboring pixels to use similar LUT mixtures."""
    horizontal = (
        weights[:, :, :, 1:] - weights[:, :, :, :-1]
    ).abs().mean()

    vertical = (
        weights[:, :, 1:, :] - weights[:, :, :-1, :]
    ).abs().mean()

    return (horizontal + vertical) / 2.0


def grading_loss(
    prediction,
    target,
    luts,
    weights,
    lut_smooth_weight=0.01,
    map_smooth_weight=0.01,
):
    reconstruction = F.l1_loss(prediction, target)
    lut_reg = lut_smoothness(luts)
    map_reg = spatial_smoothness(weights)

    total = (
        reconstruction
        + lut_smooth_weight * lut_reg
        + map_smooth_weight * map_reg
    )

    metrics = {
        "l1": reconstruction.detach().item(),
        "lut_smoothness": lut_reg.detach().item(),
        "map_smoothness": map_reg.detach().item(),
    }

    return total, metrics
