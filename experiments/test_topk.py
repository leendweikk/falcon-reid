import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # project root (this file lives in experiments/)
sys.path.insert(0, str(ROOT))                   # so `import reid` works when run from anywhere

import time
from pathlib import Path

import numpy as np
import pandas as pd

from reid import evaluate as official
from reid.rerank import re_ranking_streaming

SPLITS = (ROOT / "data" / "splits")
PRED = (ROOT / "runs" / "pipeline_val_ema")

q_ids = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
g_ids = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
gt_query, gt_gallery = official.load_gt(SPLITS / "val_gt.csv")
emb = np.load(PRED / "embeddings.npy")
q, g = emb[:len(q_ids)], emb[len(q_ids):]


def score(sims):
    order = np.argsort(-sims, axis=1)[:, :10]
    ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
    return official.ranking_metrics(gt_query, gt_gallery, ranked)["mAP@10"]


print(f"no re-ranking      : mAP@10 {score(q @ g.T):.4f}")
for topk in [None, 200, 100, 50, 30]:
    t0 = time.time()
    sims = re_ranking_streaming(q, g, 6, 2, topk=topk)
    ms = 1000 * (time.time() - t0) / len(q_ids)
    print(f"re-rank topk={str(topk):4s}: mAP@10 {score(sims):.4f} | {ms:.1f} ms/query")