import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout,cout,3,padding=1), nn.ReLU(inplace=True))

class VGGFeatureEncoder(nn.Module):
    """Shared VGG16 features; returns shallow-to-deep maps at 1/2..1/16."""
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        net = vgg16(weights=weights).features
        self.blocks = nn.ModuleList([
            net[:4], net[4:9], net[9:16], net[16:23]
        ])
        self.freeze = freeze
        if freeze:
            for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("mean",torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer("std",torch.tensor([.229,.224,.225])[None,:,None,None])
    def forward(self,x):
        x=(x-self.mean)/self.std
        outs=[]
        for block in self.blocks:
            x=block(x); outs.append(x)
        return outs

class ParamHead(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(True),
                               nn.Linear(dim,dim//2),nn.ReLU(True),nn.Linear(dim//2,out))
    def forward(self,x): return self.net(x)

class ContextHead(nn.Module):
    def __init__(self, cin, map_size):
        super().__init__()
        self.net=nn.Sequential(conv_block(cin,128),nn.Conv2d(128,64,3,padding=1),
                               nn.ReLU(True),nn.Conv2d(64,1,1),nn.Sigmoid())
        self.map_size=map_size
    def forward(self,x):
        return torch.nn.functional.interpolate(self.net(x),size=(self.map_size,self.map_size),
                                               mode="bilinear",align_corners=False)

class NodeGraph(nn.Module):
    """Three serial nodes: exposure/WB -> tonal contrast -> creative colour."""
    def __init__(self,cfg):
        super().__init__()
        m=cfg["model"]; self.map_size=int(m["map_size"])
        self.encoder=VGGFeatureEncoder(m["pretrained"],m["freeze_encoder"])
        channels=[64,128,256,512]
        self.spatial_fuse=nn.Sequential(conv_block(sum(channels)*2,256),
                                        nn.Conv2d(256,128,3,padding=1),nn.ReLU(True))
        self.global_mlp=nn.Sequential(nn.Linear(1024,512),nn.ReLU(True),
                                      nn.Dropout(.1),nn.Linear(512,m["hidden_dim"]),nn.ReLU(True))
        h=m["hidden_dim"]
        self.exposure_head=ParamHead(h,4) # EV, temperature, tint, WB mix
        self.contrast_head=ParamHead(h,4) # contrast, black, white, pivot
        self.creative_head=ParamHead(h,5) # saturation, hue, shadow tint, highlight tint, strength
        self.maps=nn.ModuleList([ContextHead(128,self.map_size) for _ in range(3)])
        self.limits={k:float(m[k]) for k in ("exposure_limit_ev","temperature_limit",
            "tint_limit","contrast_limit","saturation_limit","hue_limit")}
    def forward(self,source,reference):
        sf=self.encoder(source); rf=self.encoder(reference)
        deep=torch.cat([sf[-1].mean((2,3)),rf[-1].mean((2,3))],1)
        global_feat=self.global_mlp(deep)
        size=sf[0].shape[-2:]
        spatial=[]
        for a,b in zip(sf,rf):
            spatial.append(torch.nn.functional.interpolate(torch.cat([a,b],1),size=size,
                            mode="bilinear",align_corners=False))
        fused=self.spatial_fuse(torch.cat(spatial,1))
        params=[self.exposure_head(global_feat),self.contrast_head(global_feat),
                self.creative_head(global_feat)]
        maps=[head(fused) for head in self.maps]
        return {"params":params,"maps":maps}
