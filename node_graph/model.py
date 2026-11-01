import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
    )


class VGGFeatureEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16(weights=weights).features
        self.blocks = nn.ModuleList([
            features[:4], features[4:9], features[9:16], features[16:23]
        ])
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer("std", torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self, x):
        x = (x - self.mean) / self.std
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs


class ParamHead(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(True),
            nn.Linear(dim, dim // 2), nn.ReLU(True),
            nn.Linear(dim // 2, out),
        )

    def forward(self, x):
        return self.net(x)


class ContextHead(nn.Module):
    """Spatial gate initialized near 0.5 and bounded away from exact 0/1."""
    def __init__(self, cin, map_size, initial_gate=0.5, min_gate=0.05, max_gate=0.95):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(cin, 128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 1),
        )
        self.map_size = int(map_size)
        self.min_gate = float(min_gate)
        self.max_gate = float(max_gate)
        # Start with a neutral, non-saturated gate. Small nonzero weights allow
        # spatial features to influence the gate from the beginning.
        final = self.net[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(final.bias, 0.0)
        self.initial_gate = float(initial_gate)

    def forward(self, x):
        logits = self.net(x)
        gate = self.min_gate + (self.max_gate - self.min_gate) * torch.sigmoid(logits)
        return F.interpolate(
            gate, size=(self.map_size, self.map_size),
            mode="bilinear", align_corners=False
        )


class NodeGraph(nn.Module):
    """Three serial nodes: exposure/WB -> tonal contrast -> creative colour."""
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.encoder = VGGFeatureEncoder(m["pretrained"], m["freeze_encoder"])
        channels = [64, 128, 256, 512]
        self.spatial_fuse = nn.Sequential(
            conv_block(sum(channels) * 2, 256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(.1),
            nn.Linear(512, m["hidden_dim"]), nn.ReLU(True),
        )
        hidden = int(m["hidden_dim"])
        self.exposure_head = ParamHead(hidden, 4)
        self.contrast_head = ParamHead(hidden, 4)
        self.creative_head = ParamHead(hidden, 5)
        self.maps = nn.ModuleList([
            ContextHead(128, m["map_size"], m.get("map_initial_gate", .5),
                        m.get("map_min_gate", .05), m.get("map_max_gate", .95))
            for _ in range(3)
        ])

    def forward(self, source, reference):
        source_features = self.encoder(source)
        reference_features = self.encoder(reference)
        global_features = torch.cat([
            source_features[-1].mean((2, 3)),
            reference_features[-1].mean((2, 3)),
        ], dim=1)
        global_features = self.global_mlp(global_features)

        spatial_size = source_features[0].shape[-2:]
        fused_features = []
        for src, ref in zip(source_features, reference_features):
            pair = torch.cat([src, ref], dim=1)
            fused_features.append(F.interpolate(
                pair, size=spatial_size, mode="bilinear", align_corners=False
            ))
        spatial_features = self.spatial_fuse(torch.cat(fused_features, dim=1))

        params = [
            self.exposure_head(global_features),
            self.contrast_head(global_features),
            self.creative_head(global_features),
        ]
        maps = [head(spatial_features) for head in self.maps]
        return {"params": params, "maps": maps}
