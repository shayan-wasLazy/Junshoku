import torch
import torch.nn.functional as F

def total_loss(pred,target,maps,cfg):
    rec=F.l1_loss(pred,target)
    smooth=pred.new_tensor(0.)
    for m in maps:
        smooth=smooth+(m[:,:,:,1:]-m[:,:,:,:-1]).abs().mean()
        smooth=smooth+(m[:,:,1:,:]-m[:,:,:-1,:]).abs().mean()
    smooth=smooth/len(maps)
    w=cfg["loss"]
    return float(w["reconstruction"])*rec+float(w["map_smoothness"])*smooth, {
        "l1":rec.detach(),"map_smoothness":smooth.detach()}
