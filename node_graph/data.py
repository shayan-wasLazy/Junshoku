from pathlib import Path
import csv
import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

def find_image(folder: Path, stem: str) -> Path:
    matches=[p for p in folder.iterdir() if p.is_file() and p.stem.lower()==stem.lower()]
    if len(matches)!=1:
        raise FileNotFoundError(f"Expected exactly one '{stem}' image in {folder}; found {len(matches)}")
    return matches[0]

def create_manifest(root, manifest_path, seed=42, train_ratio=.90, val_ratio=.05):
    root=Path(root); manifest_path=Path(manifest_path)
    folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    valid=[]; rejected=[]
    for folder in folders:
        try:
            for name in ("content","reference","gt"): find_image(folder,name)
            valid.append(folder.name)
        except (FileNotFoundError,OSError) as exc: rejected.append((folder.name,str(exc)))
    if not valid: raise RuntimeError(f"No valid TST2K triplets found under {root}")
    rng=random.Random(seed); rng.shuffle(valid)
    n=len(valid); n_train=int(n*train_ratio); n_val=int(n*val_ratio)
    splits={"train":valid[:n_train],"val":valid[n_train:n_train+n_val],"test":valid[n_train+n_val:]}
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    with manifest_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["sample_id","split"]); writer.writeheader()
        for split,ids in splits.items():
            for sid in ids: writer.writerow({"sample_id":sid,"split":split})
    print(f"TST2K valid={n}, rejected={len(rejected)}; split sizes: "+", ".join(f"{k}={len(v)}" for k,v in splits.items()))
    return splits

class TST2KDataset(Dataset):
    """TST2K triplets: numbered folder/content, reference, gt."""
    def __init__(self,root,split="train",image_size=256,augment=False,manifest_path=None,seed=42):
        self.root=Path(root); self.split=split; self.image_size=int(image_size); self.augment=augment
        if split not in {"train","val","test"}: raise ValueError("split must be train, val, or test")
        self.manifest=Path(manifest_path) if manifest_path else self.root/"split_manifest.csv"
        if not self.manifest.exists(): create_manifest(self.root,self.manifest,seed=seed)
        with self.manifest.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        self.ids=[r["sample_id"] for r in rows if r["split"]==split]
        if not self.ids: raise RuntimeError(f"No TST2K samples for split={split}. Check {self.manifest}")
    def __len__(self): return len(self.ids)
    def _tensor(self,path):
        with Image.open(path) as im: im=im.convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BICUBIC)
        arr=np.asarray(im,dtype=np.float32)/255.0
        return torch.from_numpy(arr).permute(2,0,1).contiguous()
    def __getitem__(self,index):
        sid=self.ids[index]; folder=self.root/sid
        paths=[find_image(folder,n) for n in ("content","reference","gt")]
        if self.augment and random.random()<.5:
            imgs=[]
            for path in paths:
                with Image.open(path) as im: imgs.append(ImageOps.mirror(im.convert("RGB")))
            tensors=[]
            for im in imgs:
                im=im.resize((self.image_size,self.image_size),Image.Resampling.BICUBIC)
                arr=np.asarray(im,dtype=np.float32)/255.0
                tensors.append(torch.from_numpy(arr).permute(2,0,1).contiguous())
            source,reference,target=tensors
        else: source,reference,target=[self._tensor(p) for p in paths]
        return {"source":source,"reference":reference,"target":target,"sample_id":sid}
