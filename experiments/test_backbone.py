import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # project root (this file lives in experiments/)
sys.path.insert(0, str(ROOT))                   # so `import reid` works when run from anywhere

from pathlib import Path
import torch
import timm
from PIL import Image

MODEL = "convnext_small.dinov3_lvd1689m"

# num_classes=0 -> remove the classification head, return the pooled feature vector
model = timm.create_model(MODEL, pretrained=True, num_classes=0).cuda().eval()
print("parameters (millions):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))

# timm knows which image size / normalization this model expects
cfg = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**cfg)
print("expected input:", cfg["input_size"], " mean:", cfg["mean"], " std:", cfg["std"])

# take any one crop and turn it into an embedding
crop_path = next((ROOT / "data" / "crops").iterdir())
img = transform(Image.open(crop_path).convert("RGB")).unsqueeze(0).cuda()

with torch.no_grad():
    emb = model(img)

print("crop:", crop_path.name)
print("embedding shape:", tuple(emb.shape))
print("first 5 numbers:", emb[0, :5].tolist())