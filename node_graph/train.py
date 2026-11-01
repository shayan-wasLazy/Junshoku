import argparse, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import load_config
from .data import TST2KDataset
from .model import NodeGraph
from .graph import apply_graph
from .losses import total_loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="node_graph/configs/default.yaml")
    ap.add_argument("--data-root",default=None)
    ap.add_argument("--epochs",type=int,default=None)
    ap.add_argument("--batch-size",type=int,default=None)
    ap.add_argument("--lr",type=float,default=None)
    args=ap.parse_args()
    cfg=load_config(args.config)
    if args.data_root: cfg["data"]["root"]=args.data_root
    if args.epochs: cfg["training"]["epochs"]=args.epochs
    if args.batch_size: cfg["data"]["batch_size"]=args.batch_size
    if args.lr: cfg["training"]["learning_rate"]=args.lr
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print(torch.cuda.get_device_name(0))
    train_ds=TST2KDataset(cfg["data"]["root"],"train",cfg["data"]["image_size"],cfg["data"]["augment"],cfg["data"]["manifest"],cfg["seed"])
    val_ds=TST2KDataset(cfg["data"]["root"],"val",cfg["data"]["image_size"],False,cfg["data"]["manifest"],cfg["seed"])
    print(f"TST2K samples: train={len(train_ds)}, validation={len(val_ds)}")
    train_dl=DataLoader(train_ds,batch_size=cfg["data"]["batch_size"],shuffle=True,
                        num_workers=cfg["data"]["num_workers"],pin_memory=device.type=="cuda")
    val_dl=DataLoader(val_ds,batch_size=cfg["data"]["batch_size"],shuffle=False,
                      num_workers=cfg["data"]["num_workers"],pin_memory=device.type=="cuda")
    model=NodeGraph(cfg).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg["training"]["learning_rate"],
                          weight_decay=cfg["training"]["weight_decay"])
    amp=bool(cfg["training"]["amp"] and device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    out=Path(cfg["training"]["checkpoint_dir"]); out.mkdir(parents=True,exist_ok=True)
    best=float("inf")
    for epoch in range(1,int(cfg["training"]["epochs"])+1):
        model.train(); tr=0.; tr_parts={"reconstruction":0.,"map_smoothness":0.,"map_variance":0.}
        for batch in tqdm(train_dl,desc=f"Train {epoch}"):
            s=batch["source"].to(device,non_blocking=True)
            r=batch["reference"].to(device,non_blocking=True)
            t=batch["target"].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                result=model(s,r); pred=apply_graph(s,result,cfg)
                loss,parts=total_loss(pred,t,result["maps"],cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params,float(cfg["training"]["grad_clip_norm"]))
            scaler.step(opt); scaler.update()
            tr+=loss.item()
            for key,value in parts.items(): tr_parts[key]+=float(value.item())
        model.eval(); va=0.
        with torch.no_grad():
            for batch in val_dl:
                s=batch["source"].to(device); r=batch["reference"].to(device); t=batch["target"].to(device)
                result=model(s,r); pred=apply_graph(s,result,cfg)
                loss,_=total_loss(pred,t,result["maps"],cfg); va+=loss.item()
        tr/=max(1,len(train_dl)); va/=max(1,len(val_dl))
        print(f"Epoch {epoch}: train={tr:.5f} val={va:.5f} | " + " ".join(f"{k}={v/max(1,len(train_dl)):.5f}" for k,v in tr_parts.items()))
        state={"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"config":cfg}
        torch.save(state,out/"latest.pt")
        if va<best:
            best=va; torch.save(state,out/"best.pt"); print("Saved best checkpoint.")
    print("Finished. Best validation loss:",best)

if __name__=="__main__": main()
