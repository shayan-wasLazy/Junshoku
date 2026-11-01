import torch
import torch.nn.functional as F

def masked_node(x, transformed, mask):
    mask=F.interpolate(mask,size=x.shape[-2:],mode="bilinear",align_corners=False)
    return (x + mask*(transformed-x)).clamp(0,1)

def exposure_wb(x,p,cfg):
    lim=cfg["model"]
    ev=torch.tanh(p[:,0:1])*float(lim["exposure_limit_ev"])
    temp=torch.tanh(p[:,1:2])*float(lim["temperature_limit"])
    tint=torch.tanh(p[:,2:3])*float(lim["tint_limit"])
    strength=torch.sigmoid(p[:,3:4])
    gain=torch.pow(2.0,ev).view(-1,1,1,1)
    # Simple differentiable RGB-axis white-balance approximation.
    wb=torch.cat([1+temp-tint*.25, 1+tint*.25, 1-temp-tint*.25],1).view(-1,3,1,1)
    return (x*gain*(1-strength+strength*wb)).clamp(0,1)

def tonal_contrast(x,p,cfg):
    c=float(cfg["model"]["contrast_limit"])
    contrast=1+torch.tanh(p[:,0:1])*c
    black=.08*torch.tanh(p[:,1:2])
    white=.08*torch.tanh(p[:,2:3])
    pivot=.35+.3*torch.sigmoid(p[:,3:4])
    y=((x-black.view(-1,1,1,1)-pivot.view(-1,1,1,1))*contrast.view(-1,1,1,1)
       +pivot.view(-1,1,1,1)+white.view(-1,1,1,1))
    return y.clamp(0,1)

def creative_colour(x,p,cfg):
    lim=cfg["model"]
    sat=1+torch.tanh(p[:,0:1])*float(lim["saturation_limit"])
    hue=torch.tanh(p[:,1:2])*float(lim["hue_limit"])
    sh=torch.tanh(p[:,2:3])*.12
    hi=torch.tanh(p[:,3:4])*.12
    strength=torch.sigmoid(p[:,4:5])
    luma=(.299*x[:,0:1]+.587*x[:,1:2]+.114*x[:,2:3])
    gray=luma.expand_as(x)
    y=gray+(x-gray)*sat.view(-1,1,1,1)
    # Small rotation in the R-B plane; green is left unchanged.
    angle=hue.view(-1,1,1,1)
    r=y[:,0:1]*torch.cos(angle)-y[:,2:3]*torch.sin(angle)
    b=y[:,0:1]*torch.sin(angle)+y[:,2:3]*torch.cos(angle)
    y=torch.cat([r,y[:,1:2],b],1)
    shadows=(1-luma).clamp(0,1); highlights=luma.clamp(0,1)
    tint=torch.cat([sh,torch.zeros_like(sh),-sh],1).view(-1,3,1,1)*shadows
    tint=tint+torch.cat([-hi,torch.zeros_like(hi),hi],1).view(-1,3,1,1)*highlights
    transformed=(y+tint).clamp(0,1)
    s=strength.view(-1,1,1,1)
    return (x+s*(transformed-x)).clamp(0,1)

def run_nodes(source,params,maps,cfg):
    from . import _unused
