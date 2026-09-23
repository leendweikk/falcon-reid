"""
Experiment: test-time multi-size (no training).
Each car photo is embedded at several input sizes; the normalized vectors are averaged.
Scored with the organizers' own ranking_metrics, exactly like submission.csv (top-10).

  python experiments/multisize_test.py --weights runs/convnext_b_v6_ema/best_ema.pth
  python experiments/multisize_test.py --weights runs/split7/base.pth --splits data/splits_seed7

Keep rule: a combo is adopted only if it beats 256-alone by >= 0.7 mAP@10 on BOTH splits
(with re-ranking, fast mode = Base, no flip).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from reid import evaluate as official
from reid.data import CROPS, MEAN, STD
from reid.model import ReIDModel
from reid.rerank import re_ranking_streaming

ROOT = Path(__file__).resolve().parents[1]


def load_model(backbone, path, device):
    state = torch.load(path, map_location="cpu")
    m = ReIDModel(num_classes=state["classifier.weight"].shape[0], backbone=backbone, pretrained=False)
    m.load_state_dict(state)
    return m.to(device).eval()


@torch.no_grad()
def embed(model, ids, size, device, bs=32):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    out = []
    for i in range(0, len(ids), bs):
        batch = torch.stack([tf(Image.open(CROPS / f"{x}.jpg").convert("RGB")) for x in ids[i:i + bs]]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model(batch)
        out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def score(q, g, q_ids, g_ids, query, gallery, rerank):
    sims = re_ranking_streaming(q, g, 6, 2, topk=100) if rerank else q @ g.T
    order = np.argsort(-sims, axis=1)[:, :10]
    ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
    return 100 * official.ranking_metrics(query, gallery, ranked)["mAP@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--backbone", default="convnext_base.dinov3_lvd1689m")
    ap.add_argument("--sizes", default="224,256,288,320")
    args = ap.parse_args()

    splits = ROOT / args.splits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query, gallery = official.load_gt(splits / "val_gt.csv")
    q_ids = pd.read_csv(splits / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
    g_ids = pd.read_csv(splits / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
    model = load_model(args.backbone, ROOT / args.weights, device)

    sizes = [int(s) for s in args.sizes.split(",")]
    emb = {}
    for s in sizes:
        t0 = time.time()
        emb[s] = (embed(model, q_ids, s, device), embed(model, g_ids, s, device))
        print(f"embedded at {s}x{s} in {time.time() - t0:.0f}s")

    combos = [(s,) for s in sizes]
    combos += [c for n in (2, 3) for c in itertools.combinations(sizes, n) if 256 in c]
    rows = []
    for c in combos:
        q = sum(emb[s][0] for s in c)
        g = sum(emb[s][1] for s in c)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        rows.append({"sizes": "+".join(map(str, c)),
                     "mAP@10 (cosine)": score(q, g, q_ids, g_ids, query, gallery, rerank=False),
                     "mAP@10 (re-rank)": score(q, g, q_ids, g_ids, query, gallery, rerank=True)})
        print(rows[-1])

    df = pd.DataFrame(rows)
    base = df.loc[df.sizes == "256", "mAP@10 (re-rank)"].item()
    df["gain vs 256"] = df["mAP@10 (re-rank)"] - base
    print("\n" + df.round(2).to_string(index=False))
    out = ROOT / "runs" / f"multisize_{Path(args.splits).name}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
