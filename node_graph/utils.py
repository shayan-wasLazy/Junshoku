from pathlib import Path
import yaml, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

def load_config(path):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def load_rgb(path,size,device):
    with Image.open(path) as im:
        x=to_tensor(im.convert("RGB").resize((size,size),Image.Resampling.LANCZOS))
    return x.unsqueeze(0).to(device)

def save_rgb(x,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    to_pil_image(x.detach().cpu().squeeze(0).clamp(0,1)).save(path)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
