from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PST50Pairs(Dataset):
    """PST50 convention used by the existing Junshoku dataset.py:
    content_709/inN.png, paired_style/tarN.png, paired_gt/gtN.png.
    The target is the ground-truth graded content, not the reference image.
    """
    def __init__(self, root, indices, image_size=256):
        root = Path(root)
        self.samples = []
        for i in indices:
            paths = (root/"content_709"/f"in{i}.png",
                     root/"paired_style"/f"tar{i}.png",
                     root/"paired_gt"/f"gt{i}.png")
            if all(p.is_file() for p in paths):
                self.samples.append(paths)
            else:
                print(f"[data] skipping {i}: missing one or more files")
        if not self.samples:
            raise RuntimeError(f"No valid PST50 triplets found under {root}")
        self.tf = transforms.Compose([
            transforms.Resize((image_size,image_size),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        src, ref, tgt = self.samples[idx]
        def load(path):
            with Image.open(path) as im: return self.tf(im.convert("RGB"))
        return {"source":load(src), "reference":load(ref), "target":load(tgt)}
