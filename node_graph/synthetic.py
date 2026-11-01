import math
import random
import torch
from torch.utils.data import Dataset
from .data import TST2KDataset
from .nodes import exposure_wb, tonal_contrast, creative_colour, masked_node

def make_smooth_mask(height, width, rng):
    yy, xx = torch.meshgrid(torch.linspace(-1,1,height), torch.linspace(-1,1,width), indexing="ij")
    mask = torch.zeros((1,height,width))
    for _ in range(rng.randint(1,3)):
        cx,cy=rng.uniform(-.65,.65),rng.uniform(-.65,.65)
        rx,ry=rng.uniform(.18,.65),rng.uniform(.18,.65)
        angle=rng.uniform(-math.pi,math.pi)
        xr=(xx-cx)*math.cos(angle)+(yy-cy)*math.sin(angle)
        yr=-(xx-cx)*math.sin(angle)+(yy-cy)*math.cos(angle)
        blob=torch.exp(-.5*((xr/rx)**2+(yr/ry)**2))
        mask=torch.maximum(mask,blob.unsqueeze(0)*rng.uniform(.35,.95))
    return mask.clamp(0,1)

class SyntheticNodeDataset(Dataset):
    """Synthetic paired data with exact spatial masks and node parameter labels."""
    def __init__(self, base_dataset, cfg, length=4000, seed=42):
        self.base=base_dataset; self.cfg=cfg; self.length=int(length); self.seed=int(seed)
        self.map_size=int(cfg["model"]["map_size"])
    def __len__(self): return self.length
    def __getitem__(self,index):
        rng=random.Random(self.seed+index)
        base=self.base[index % len(self.base)]
        source=base["source"]
        masks=[make_smooth_mask(self.map_size,self.map_size,rng) for _ in range(3)]
        params=[
            torch.tensor([rng.uniform(-.65,.65),rng.uniform(-.6,.6),rng.uniform(-.6,.6),rng.uniform(-.5,.5)]),
            torch.tensor([rng.uniform(-.7,.7),rng.uniform(-.5,.5),rng.uniform(-.5,.5),rng.uniform(-.5,.5)]),
            torch.tensor([rng.uniform(-.7,.7),rng.uniform(-.6,.6),rng.uniform(-.6,.6),rng.uniform(-.6,.6),rng.uniform(-.5,.5)])
        ]
        x=source.unsqueeze(0)
        x=masked_node(x,exposure_wb(x,params[0].unsqueeze(0),self.cfg),masks[0].unsqueeze(0))
        x=masked_node(x,tonal_contrast(x,params[1].unsqueeze(0),self.cfg),masks[1].unsqueeze(0))
        x=masked_node(x,creative_colour(x,params[2].unsqueeze(0),self.cfg),masks[2].unsqueeze(0))
        target=x.squeeze(0).detach()
        # The graded image is the synthetic reference cue for this supervised pretraining task.
        return {"source":source,"reference":target.clone(),"target":target,
                "params":params,"maps":masks}
