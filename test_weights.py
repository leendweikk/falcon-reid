from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as official
from rerank import re_ranking_streaming

SPLITS = Path("C:/falcon/data/splits")
RUNS = Path("C:/falcon/runs")

q_ids = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
g_ids = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
gt_query, gt_gallery = official.load_gt(SPLITS / "val_gt.csv")
nq = len(q_ids)


def load(run):
    emb = np.load(RUNS / run / "val_pred" / "embeddings.npy")
    return emb[:nq], emb[nq:]


def score(sims):
    order = np.argsort(-sims, axis=1)[:, :10]
    ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
    m = official.ranking_metrics(gt_query, gt_gallery, ranked)
    return m["mAP@10"], m["Rank-1"], m["Rank-5"]


qB, gB = load("convnext_b_v6_ema")
qS, gS = load("convnext_s_v2_ema")

rows = [("Base EMA alone", *score(qB @ gB.T)),
        ("Small EMA alone", *score(qS @ gS.T))]
for w in [0.7, 0.6, 0.5]:
    qE = np.concatenate([np.sqrt(w) * qB, np.sqrt(1 - w) * qS], axis=1)
    gE = np.concatenate([np.sqrt(w) * gB, np.sqrt(1 - w) * gS], axis=1)
    rows.append((f"Ensemble {w:.1f}/{1 - w:.1f}", *score(qE @ gE.T)))
    rows.append((f"Ensemble {w:.1f}/{1 - w:.1f} + re-rank top100",
                 *score(re_ranking_streaming(qE, gE, 6, 2, topk=100))))

df = pd.DataFrame(rows, columns=["method", "mAP@10", "Rank-1", "Rank-5"])
print(df.round(4).to_string(index=False))
df.to_csv(RUNS / "ensemble_weights_ema.csv", index=False)