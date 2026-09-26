"""
Teacher vectors for distillation (docs/PLAN.md A11): the Base+ViT ensemble (flip, glued 0.5/0.5 —
exactly the "accurate" mode) embeds every TRAINING photo once. train_final.py --distill then teaches the
student ViT to reproduce the ensemble's similarity structure.
The teacher must never have seen the validation cars of the split being tested:
  seed 42 : --base runs/convnext_b_v6_ema/best_ema.pth --vit runs/val_vit/best_ema.pth
  seed 7  : --base runs/split7/base.pth               --vit runs/split7_vit/vit.pth
  final   : --base weights/base.pth                   --vit runs/final_vit_e20/vit.pth

  python tools/teacher_embed.py --train-csv data/splits/train_split.csv \
      --base runs/convnext_b_v6_ema/best_ema.pth --vit runs/val_vit/best_ema.pth --out runs/teacher_s42.npz
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from reid.data import CROPS, make_transforms  # noqa: E402
from reid.model import load_reid  # noqa: E402

BACKBONES = {"base": "convnext_base.dinov3_lvd1689m", "vit": "vit_base_patch16_dinov3.lvd1689m"}


@torch.no_grad()
def embed(model, size, ids, device, bs=32):
    _, tf = make_transforms((size, size))
    model.backbone.set_grad_checkpointing(False)
    out = []
    for i in range(0, len(ids), bs):
        batch = torch.stack([tf(Image.open(CROPS / f"{x}.jpg").convert("RGB")) for x in ids[i:i + bs]]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model(batch) + model(torch.flip(batch, dims=[3]))          # flip, as in accurate mode
        out.append(F.normalize(f.float(), dim=1).cpu())
        print(f"\r  {min(i + bs, len(ids))}/{len(ids)}", end="", flush=True)
    print()
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--vit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = pd.read_csv(args.train_csv, dtype={"image_id": str}).image_id.tolist()
    parts = []
    for name in ("base", "vit"):
        t0 = time.time()
        model, size = load_reid(getattr(args, name), BACKBONES[name], device)
        parts.append(np.sqrt(0.5) * embed(model, size, ids, device))
        print(f"{name}: {len(ids)} photos in {time.time() - t0:.0f}s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    vecs = np.concatenate(parts, axis=1)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, image_ids=np.array(ids), vectors=vecs.astype(np.float32),
             base=str(args.base), vit=str(args.vit))
    print(f"saved -> {args.out}  {vecs.shape}")


if __name__ == "__main__":
    main()
