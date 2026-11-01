
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


def make_identity_lut(size=16):
    axis = torch.linspace(0.0, 1.0, size)
    r, g, b = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([r, g, b], dim=-1)


class VGG16Encoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()

        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.features = vgg16(weights=weights).features

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

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


class ContextPredictor(nn.Module):
    """Predicts one continuous context value per pixel."""

    def __init__(self, feature_channels=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(feature_channels * 2, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, source_features, reference_features, output_size):
        features = torch.cat(
            [source_features, reference_features], dim=1
        )
        context = self.net(features)

        return F.interpolate(
            context,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class ContextLUTModel(nn.Module):
    """
    Two-slice 4D LUT:
      LUT dimensions: (context, R, G, B, RGB_output)
      Context map:    (B, 1, H, W), values in [0, 1]
    """

    def __init__(
        self,
        lut_size=16,
        pretrained=True,
        freeze_encoder=True,
    ):
        super().__init__()

        self.encoder = VGG16Encoder(
            pretrained=pretrained,
            freeze=freeze_encoder,
        )

        identity = make_identity_lut(lut_size)

        # Both context slices begin as identity transformations.
        # Training learns how the transformations should differ.
        self.lut_bank = nn.Parameter(
            torch.stack([identity, identity.clone()], dim=0)
        )

        self.context_predictor = ContextPredictor(feature_channels=512)

    def forward(self, source, reference):
        source_features = self.encoder(source)
        reference_features = self.encoder(reference)

        context = self.context_predictor(
            source_features,
            reference_features,
            source.shape[-2:],
        )

        return self.lut_bank, context
