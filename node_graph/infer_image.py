import argparse, torch
from .utils import load_config,load_rgb,save_rgb,get_device
from .model import NodeGraph
from .graph import apply_graph

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="node_graph/configs/default.yaml")
    ap.add_argument("--source",default=None); ap.add_argument("--reference",default=None)
    ap.add_argument("--output",default=None); ap.add_argument("--checkpoint",default=None)
    a=ap.parse_args(); cfg=load_config(a.config)
    source=a.source or cfg["inference"]["input"]
    ref=a.reference or cfg["inference"]["reference"]
    output=a.output or cfg["inference"]["output"]
    ckpt=a.checkpoint or cfg["inference"]["checkpoint"]
    device=get_device(); model=NodeGraph(cfg).to(device)
    state=torch.load(ckpt,map_location=device,weights_only=False)
    model.load_state_dict(state["model"]); model.eval()
    s=load_rgb(source,cfg["data"]["image_size"],device)
    r=load_rgb(ref,cfg["data"]["image_size"],device)
    with torch.no_grad():
        result=model(s,r); graded=apply_graph(s,result,cfg)
    save_rgb(graded,output)
    print("Saved:",output)
    print("Node parameter tensors:",[x.detach().cpu().tolist() for x in result["params"]])
    print("Context map means:",[round(x.mean().item(),4) for x in result["maps"]])
if __name__=="__main__": main()
