"""
Experiment G3: does a third model (DINOv3 ViT) make accurate mode better?
Accurate mode = flip averaging + glued ensemble (sqrt(w) scaling) + streaming re-ranking,
scored with the organizers' ranking_metrics on top-10, exactly like submission.csv.

  seed 42:
  python experiments/ensemble_test.py --base runs/convnext_b_v6_ema/best_ema.pth ^
      --small runs/convnext_s_v2_ema/best_ema.pth --vit runs/val_vit/best_ema.pth
  seed 7 (same weights grid, no re-tuning):
  python experiments/ensemble_test.py --splits data/splits_seed7 --base runs/split7/base.pth ^
      --small runs/split7/small.pth --vit runs/split7_vit/vit.pth

Keep rule: the best combo WITH the ViT must beat Base+Small (0.6/0.4) by >= 0.7 on BOTH splits,
using the SAME weights on both splits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root

import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from reid import evaluate as official
from reid.data import CROPS, make_transforms
from reid.model import load_reid
from reid.rerank import re_ranking_streaming

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = {"base": "convnext_base.dinov3_lvd1689m", "small": "convnext_small.dinov3_lvd1689m",
             "vit": "vit_base_patch16_dinov3.lvd1689m"}
COMBOS = [                                   # fixed in advance: no tuning on the validation set
    {"base": 1.0},
    {"vit": 1.0},
    {"base": 0.6, "small": 0.4},             # current accurate mode (reference)
    {"base": 0.5, "vit": 0.5},
    {"base": 0.6, "vit": 0.4},
    {"base": 0.4, "small": 0.3, "vit": 0.3},
    {"base": 0.5, "small": 0.25, "vit": 0.25},
]


@torch.no_grad()
def embed(model, ids, size, device, bs=32):
    _, tf = make_transforms((size, size))
    out = []
    for i in range(0, len(ids), bs):
        batch = torch.stack([tf(Image.open(CROPS / f"{x}.jpg").convert("RGB")) for x in ids[i:i + bs]]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model(batch) + model(torch.flip(batch, dims=[3]))          # flip averaging
        out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def score(q, g, q_ids, g_ids, query, gallery):
    sims = re_ranking_streaming(q, g, 6, 2, topk=100)
    order = np.argsort(-sims, axis=1)[:, :10]
    ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
    return 100 * official.ranking_metrics(query, gallery, ranked)["mAP@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    for name in BACKBONES:
        ap.add_argument(f"--{name}", required=True, help=f"{name} weights")
    args = ap.parse_args()

    splits = ROOT / args.splits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query, gallery = official.load_gt(splits / "val_gt.csv")
    q_ids = pd.read_csv(splits / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
    g_ids = pd.read_csv(splits / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()

    emb = {}
    for name, backbone in BACKBONES.items():
        t0 = time.time()
        model, size = load_reid(ROOT / getattr(args, name), backbone, device)
        emb[name] = (embed(model, q_ids, size, device), embed(model, g_ids, size, device))
        del model
        torch.cuda.empty_cache()
        print(f"embedded with {name} ({size}x{size}) in {time.time() - t0:.0f}s")

    rows = []
    for combo in COMBOS:
        q = np.concatenate([np.sqrt(w) * emb[n][0] for n, w in combo.items()], axis=1)   # glued
        g = np.concatenate([np.sqrt(w) * emb[n][1] for n, w in combo.items()], axis=1)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        label = " + ".join(f"{n} {w}" for n, w in combo.items())
        rows.append({"combo": label, "mAP@10 (flip + re-rank)": score(q, g, q_ids, g_ids, query, gallery)})
        print(rows[-1])

    df = pd.DataFrame(rows)
    ref = df.loc[df.combo == "base 0.6 + small 0.4", "mAP@10 (flip + re-rank)"].item()
    df["gain vs Base+Small"] = df["mAP@10 (flip + re-rank)"] - ref
    print("\n" + df.round(2).to_string(index=False))
    out = ROOT / "runs" / f"ensemble_{Path(args.splits).name}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
