import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # project root (this file lives in experiments/)
sys.path.insert(0, str(ROOT))                   # so `import reid` works when run from anywhere

from pathlib import Path

import numpy as np
import pandas as pd

from reid import evaluate as official

SPLITS = (ROOT / "data" / "splits")
PRED = (ROOT / "runs" / "convnext_b_v2_softmargin" / "val_pred")

q_ids = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
g_ids = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
emb = np.load(PRED / "embeddings.npy")
q_emb, g_emb = emb[:len(q_ids)], emb[len(q_ids):]
sims = q_emb @ g_emb.T

query, gallery = official.load_gt(SPLITS / "val_gt.csv")

# Scenario B: hide each query's same-car + same-camera "twin" from the gallery
qv, qc = query.loc[q_ids, "vehicle_id"].values, query.loc[q_ids, "camera_id"].values
gv, gc = gallery.loc[g_ids, "vehicle_id"].values, gallery.loc[g_ids, "camera_id"].values
twins = (qv[:, None] == gv[None, :]) & (qc[:, None] == gc[None, :])
sims_no_twins = np.where(twins, -2.0, sims)


def sweep(S):
    top = S.argmax(axis=1)
    conf = S[np.arange(len(q_ids)), top]
    rows = []
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        cands = {q: [(g_ids[top[i]], float(conf[i]))] for i, q in enumerate(q_ids) if conf[i] >= t}
        m = official.candidate_metrics(query, gallery, cands)       # official scoring code
        rows.append((t, m["F1"], m["TNR"], m["Precision"], m["Recall"]))
    return pd.DataFrame(rows, columns=["threshold", "F1", "TNR", "Precision", "Recall"])


A = sweep(sims)
B = sweep(sims_no_twins)
both = A[["threshold"]].copy()
both["F1_A"], both["TNR_A"] = A.F1, A.TNR
both["F1_B"], both["TNR_B"] = B.F1, B.TNR
both["F1_mean"] = (A.F1 + B.F1) / 2

print(both.iloc[::5].round(3).to_string(index=False))          # every 0.05
for name, col in [("A (as-is)", "F1_A"), ("B (no twins)", "F1_B"), ("robust (mean)", "F1_mean")]:
    best = both.loc[both[col].idxmax()]
    print(f"best for {name:14s}: threshold {best.threshold:.2f} | "
          f"F1_A {best.F1_A:.3f} TNR_A {best.TNR_A:.3f} | F1_B {best.F1_B:.3f} TNR_B {best.TNR_B:.3f}")

both.to_csv(PRED / "threshold_sweep.csv", index=False)           # for the slides / README