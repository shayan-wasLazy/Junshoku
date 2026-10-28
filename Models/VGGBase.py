
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


def make_initial_luts(size=16):
    """Create eight distinct, gently initialized grading LUTs."""
    axis = torch.linspace(0.0, 1.0, size)
    r, g, b = torch.meshgrid(axis, axis, axis, indexing="ij")
    rgb = torch.stack([r, g, b], dim=-1)

    luminance = (
        0.2126 * r + 0.7152 * g + 0.0722 * b
    )

    def clamp(x):
        return x.clamp(0.0, 1.0)

    # 1. Brightness / exposure
    brightness = clamp(rgb + 0.07)

    # 2. Contrast
    contrast = clamp(0.5 + 1.15 * (rgb - 0.5))

    # 3. Red gain / warm shift
    red = clamp(torch.stack([
        r * 1.12,
        g * 0.99,
        b * 0.97,
    ], dim=-1))

    # 4. Green gain
    green = clamp(torch.stack([
        r * 0.98,
        g * 1.10,
        b * 0.98,
    ], dim=-1))

    # 5. Blue gain / cool shift
    blue = clamp(torch.stack([
        r * 0.97,
        g * 0.99,
        b * 1.12,
    ], dim=-1))

    # 6. Lift shadows gently, with a stronger effect in dark regions.
    shadow_strength = (1.0 - luminance).square() * 0.075
    shadows = clamp(rgb + shadow_strength.unsqueeze(-1))

    # 7. Gently lift highlights while preserving headroom.
    highlight_strength = luminance.square() * 0.07
    highlights = clamp(
        rgb + highlight_strength.unsqueeze(-1) * (1.0 - rgb)
    )

    # 8. Subtle teal shadows + warm highlights.
    shadow_mask = (1.0 - luminance).square()
    highlight_mask = luminance.square()

    creative = torch.stack([
        r - 0.025 * shadow_mask + 0.045 * highlight_mask,
        g + 0.015 * shadow_mask + 0.005 * highlight_mask,
        b + 0.045 * shadow_mask - 0.020 * highlight_mask,
    ], dim=-1)
    creative = clamp(creative)

    # The identity LUT is retained as a neutral reference.
    identity = rgb.clone()

    return torch.stack([
        identity,
        brightness,
        contrast,
        red,
        green,
        blue,
        shadows,
        highlights,
        creative,
    ], dim=0)


class VGG16Encoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()

        weights = VGG16_Weights.DEFAULT if pretrained else None
        backbone = vgg16(weights=weights)
        self.features = backbone.features

        if freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.features(image)


class LUTBank(nn.Module):
    def __init__(self, num_luts=9, size=16):
        super().__init__()

        if num_luts != 9:
            raise ValueError(
                "This initialization defines 9 LUTs: identity plus 8 grading LUTs."
            )

        initial_luts = make_initial_luts(size=size)
        self.luts = nn.Parameter(initial_luts)

    def forward(self):
        return self.luts


class SpatialPredictor(nn.Module):
    def __init__(self, feature_channels=512, num_luts=9):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(feature_channels * 2, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_luts, 1),
        )

    def forward(self, source_features, reference_features, output_size):
        features = torch.cat(
            [source_features, reference_features], dim=1
        )

        logits = self.net(features)
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return torch.softmax(logits, dim=1)


class PredictModel(nn.Module):
    def __init__(
        self,
        freeze_encoder=True,
        lut_size=16,
        num_luts=9,
        pretrained=True,
    ):
        super().__init__()

        self.encoder = VGG16Encoder(
            pretrained=pretrained,
            freeze=freeze_encoder,
        )

        self.lut_bank = LUTBank(
            num_luts=num_luts,
            size=lut_size,
        )

        self.spatial_predictor = SpatialPredictor(
            feature_channels=512,
            num_luts=num_luts,
        )

    def forward(self, source, reference):
        source_features = self.encoder(source)
        reference_features = self.encoder(reference)

        weights = self.spatial_predictor(
            source_features,
            reference_features,
            source.shape[-2:],
        )

        return self.lut_bank(), weights
