from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms as T

DATA = Path("C:/falcon/data")
SPLITS = DATA / "splits"
OUT = Path("C:/falcon/runs/frozen")
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "convnext_small.dinov3_lvd1689m"
SIZE = 256

model = timm.create_model(MODEL, pretrained=True, num_classes=0).cuda().eval()
cfg = timm.data.resolve_data_config({}, model=model)

# our own transform: squeeze the WHOLE crop to 256x256 (no center-cut), model's normalization
transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize(cfg["mean"], cfg["std"]),
])

@torch.no_grad()
def embed(image_ids, batch_size=64):
    feats = []
    for i in range(0, len(image_ids), batch_size):
        batch = torch.stack([
            transform(Image.open(DATA / "crops" / f"{iid}.jpg").convert("RGB"))
            for iid in image_ids[i:i + batch_size]
        ]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(batch) + model(torch.flip(batch, dims=[3]))   # flip averaging
        feats.append(f.float().cpu())
        print(f"\r embedded {min(i + batch_size, len(image_ids))}/{len(image_ids)}", end="")
    print()
    feats = torch.cat(feats)
    return torch.nn.functional.normalize(feats, dim=1).numpy()      # length 1 -> dot = cosine

q_ids = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
g_ids = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
q_emb, g_emb = embed(q_ids), embed(g_ids)

sims = q_emb @ g_emb.T                                # (num_queries, num_gallery) similarities
order = np.argsort(-sims, axis=1)[:, :10]             # top 10 gallery indices per query

# submission.csv: no header, query_id + 10 gallery ids
with open(OUT / "submission.csv", "w") as f:
    for qi, qid in enumerate(q_ids):
        f.write(",".join([qid] + [g_ids[j] for j in order[qi]]) + "\n")

# candidates.csv: top-1 match with its similarity as confidence (no refusal yet)
pd.DataFrame({
    "query_id": q_ids,
    "gallery_id": [g_ids[order[qi, 0]] for qi in range(len(q_ids))],
    "confidence": [sims[qi, order[qi, 0]] for qi in range(len(q_ids))],
}).to_csv(OUT / "candidates.csv", index=False)

np.save(OUT / "embeddings.npy", np.concatenate([q_emb, g_emb]).astype(np.float32))
print("saved to", OUT)