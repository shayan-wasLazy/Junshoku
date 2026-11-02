import argparse, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import load_config
from .data import TST2KDataset
from .synthetic import SyntheticNodeDataset
from .model import NodeGraph
from .graph import apply_graph

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="node_graph/configs/default.yaml")
    ap.add_argument("--epochs",type=int,default=None)
    ap.add_argument("--batch-size",type=int,default=None)
    args=ap.parse_args(); cfg=load_config(args.config)
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    base=TST2KDataset(cfg["data"]["root"],"train",cfg["data"]["image_size"],False,cfg["data"]["manifest"],cfg["seed"])
    ds=SyntheticNodeDataset(base,cfg,cfg["synthetic"]["samples"],cfg["seed"])
    bs=args.batch_size or cfg["data"]["batch_size"]
    loader=DataLoader(ds,batch_size=bs,shuffle=True,num_workers=0,pin_memory=device.type=="cuda")
    model=NodeGraph(cfg).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    amp=bool(cfg["training"]["amp"] and device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    out=Path(cfg["training"]["synthetic_checkpoint_dir"]); out.mkdir(parents=True,exist_ok=True)
    epochs=args.epochs or cfg["synthetic"]["epochs"]
    for epoch in range(1,epochs+1):
        model.train(); sums=[0.,0.,0.,0.]
        for batch in tqdm(loader,desc=f"Synthetic {epoch}/{epochs}"):
            s=batch["source"].to(device); r=batch["reference"].to(device); t=batch["target"].to(device)
            gt_params=[p.to(device) for p in batch["params"]]
            gt_maps=[m.to(device) for m in batch["maps"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                outp=model(s,r)
                map_loss=sum(F.l1_loss(a,b) for a,b in zip(outp["maps"],gt_maps))/3
                param_loss=sum(F.smooth_l1_loss(a,b) for a,b in zip(outp["params"],gt_params))/3
                pred=apply_graph(s,outp,cfg); rec=F.l1_loss(pred,t)
                loss=(cfg["synthetic"]["map_weight"]*map_loss+
                      cfg["synthetic"]["param_weight"]*param_loss+
                      cfg["synthetic"]["reconstruction_weight"]*rec)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer); scaler.update()
            for i,v in enumerate((loss,map_loss,param_loss,rec)): sums[i]+=v.item()
        n=max(1,len(loader))
        print(f"Epoch {epoch}: total={sums[0]/n:.5f} maps={sums[1]/n:.5f} params={sums[2]/n:.5f} reconstruction={sums[3]/n:.5f}")
        torch.save({"epoch":epoch,"model":model.state_dict(),"config":cfg},out/"latest.pt")
    print("Synthetic pretraining checkpoint:",out/"latest.pt")
if __name__=="__main__": main()
