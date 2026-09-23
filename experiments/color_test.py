import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # project root (this file lives in experiments/)
sys.path.insert(0, str(ROOT))                   # so `import reid` works when run from anywhere

"""
Does an explicit COLOR score help? (Leen's question, 23.09)
Blends a soft color similarity (HSV histogram of the car's central region) into the ranking.
It can only nudge, never veto. Measured with the official scorer, with and without re-ranking.

  python experiments/color_test.py runs/restructure_check
"""
import numpy as np
import pandas as pd
from PIL import Image

from reid import evaluate as official
from reid.data import CROPS
from reid.rerank import re_ranking_streaming

SPLITS = ROOT / "data" / "splits"
PRED = Path(sys.argv[1])

q_ids = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
g_ids = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
gt_query, gt_gallery = official.load_gt(SPLITS / "val_gt.csv")
emb = np.load(PRED / "embeddings.npy")
q_emb, g_emb = emb[:len(q_ids)], emb[len(q_ids):]


def color_hist(image_id):
    """8 hue x 4 saturation x 4 value histogram of the central 60% of the crop (less road/background)."""
    img = Image.open(CROPS / f"{image_id}.jpg").convert("HSV")
    w, h = img.size
    img = img.crop((int(0.2 * w), int(0.2 * h), int(0.8 * w), int(0.8 * h)))
    a = np.asarray(img).reshape(-1, 3).astype(np.int32)
    idx = (a[:, 0] * 8 // 256) * 16 + (a[:, 1] * 4 // 256) * 4 + (a[:, 2] * 4 // 256)
    hist = np.bincount(idx, minlength=128).astype(np.float32)
    return hist / hist.sum()


print("computing color histograms...")
qc = np.stack([color_hist(i) for i in q_ids])
gc = np.stack([color_hist(i) for i in g_ids])
color_sim = np.minimum(qc[:, None, :], gc[None, :, :]).sum(axis=2)        # histogram intersection, 0..1


def score(sims):
    order = np.argsort(-sims, axis=1)[:, :10]
    ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
    m = official.ranking_metrics(gt_query, gt_gallery, ranked)
    return m["mAP@10"], m["Rank-1"], m["Rank-5"]


cos = q_emb @ g_emb.T
print("re-ranking (top-100)...")
rr = re_ranking_streaming(q_emb, g_emb, 6, 2, topk=100)
rows = []
for a in [0.0, 0.05, 0.1, 0.2, 0.3]:
    rows.append((f"cosine + {a:.2f} x color", *score(cos + a * color_sim)))
for a in [0.0, 0.05, 0.1, 0.2]:
    rows.append((f"re-rank top100 + {a:.2f} x color", *score(rr + a * color_sim)))
df = pd.DataFrame(rows, columns=["method", "mAP@10", "Rank-1", "Rank-5"])
print(df.round(4).to_string(index=False))
