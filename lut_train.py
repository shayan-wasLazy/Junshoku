
import torch
import torch.nn as nn
import torch.nn.functional as F



# apply lut
def apply_lut(image, lut):
    """
    image: (B, 3, H, W), RGB values normalized to [0, 1]
    lut:   (N, N, N, 3), indexed by [R, G, B]

    returns: (B, 3, H, W)
    """
    batch, channels, height, width = image.shape
    assert channels == 3

    size = lut.shape[0]
    assert lut.shape == (size, size, size, 3)

    # Convert BCHW image into (B, number_of_pixels, RGB).
    pixels = image.permute(0, 2, 3, 1).reshape(batch, -1, 3)
    pixels = pixels.clamp(0, 1) * (size - 1)

    # Lower and upper grid coordinates.
    lower = pixels.floor().long()
    upper = (lower + 1).clamp(max=size - 1)
    fraction = pixels - lower.float()

    r0, g0, b0 = lower.unbind(dim=-1)
    r1, g1, b1 = upper.unbind(dim=-1)

    fr, fg, fb = fraction.unbind(dim=-1)

    # Flatten LUT grid so each RGB coordinate maps to one index.
    flat_lut = lut.reshape(-1, 3)

    def lookup(r, g, b):
        index = (r * size + g) * size + b
        return flat_lut[index]

    # Retrieve the eight surrounding corners.
    c000 = lookup(r0, g0, b0)
    c001 = lookup(r0, g0, b1)
    c010 = lookup(r0, g1, b0)
    c011 = lookup(r0, g1, b1)
    c100 = lookup(r1, g0, b0)
    c101 = lookup(r1, g0, b1)
    c110 = lookup(r1, g1, b0)
    c111 = lookup(r1, g1, b1)

    # Trilinear interpolation.
    def blend(a, b, weight):
        return a * (1 - weight.unsqueeze(-1)) + \
               b * weight.unsqueeze(-1)

    c00 = blend(c000, c001, fb)
    c01 = blend(c010, c011, fb)
    c10 = blend(c100, c101, fb)
    c11 = blend(c110, c111, fb)

    c0 = blend(c00, c01, fg)
    c1 = blend(c10, c11, fg)

    output = blend(c0, c1, fr)

    # Return to BCHW format.
    output = output.reshape(batch, height, width, 3)
    return output.permute(0, 3, 1, 2).contiguous()


class SingleLUT(nn.Module):
    def __init__(self, size=16):
        super().__init__()

        axis = torch.linspace(0, 1, size)
        r, g, b = torch.meshgrid(
            axis, axis, axis, indexing="ij"
        )

        identity = torch.stack([r, g, b], dim=-1)
        self.lut = nn.Parameter(identity)

    def forward(self, image):
        return apply_lut(image, self.lut)
    



def lut_smoothness(lut):
    """
    Penalize abrupt changes between neighboring LUT entries.
    """
    loss_r = (lut[1:, :, :, :] - lut[:-1, :, :, :]).abs().mean()
    loss_g = (lut[:, 1:, :, :] - lut[:, :-1, :, :]).abs().mean()
    loss_b = (lut[:, :, 1:, :] - lut[:, :, :-1, :]).abs().mean()

    return (loss_r + loss_g + loss_b) / 3


def grading_loss(prediction, target, lut, smooth_weight=0.01):
    reconstruction = F.l1_loss(prediction, target)
    smoothness = lut_smoothness(lut)

    total = reconstruction + smooth_weight * smoothness

    return total, {
        "l1": reconstruction.detach().item(),
        "smoothness": smoothness.detach().item(),
    }

