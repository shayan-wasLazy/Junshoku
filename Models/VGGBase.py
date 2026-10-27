
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


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
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.features(image)


class LUTBank(nn.Module):
    def __init__(self, num_luts=8, size=16, noise=0.001):
        super().__init__()

        axis = torch.linspace(0.0, 1.0, size)
        r, g, b = torch.meshgrid(
            axis, axis, axis, indexing="ij"
        )

        identity = torch.stack([r, g, b], dim=-1)
        luts = identity.unsqueeze(0).repeat(
            num_luts, 1, 1, 1, 1
        )

        if noise > 0:
            luts = (
                luts + noise * torch.randn_like(luts)
            ).clamp(0.0, 1.0)

        self.luts = nn.Parameter(luts)

    def forward(self):
        return self.luts


class SpatialPredictor(nn.Module):
    def __init__(self, feature_channels=512, num_luts=8):
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

        # Weights across the LUT bank sum to 1 at each pixel.
        return torch.softmax(logits, dim=1)


class PredictModel(nn.Module):
    def __init__(
        self,
        freeze_encoder=True,
        lut_size=16,
        num_luts=8,
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

        # LUTs: (K, N, N, N, 3)
        # Weights: (B, K, H, W)
        return self.lut_bank(), weights
