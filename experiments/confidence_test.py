import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # project root (this file lives in experiments/)
sys.path.insert(0, str(ROOT))                   # so `import reid` works when run from anywhere

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reid import evaluate as official

SPLITS = (ROOT / "data" / "splits")
RUNS = (ROOT / "runs")

q_ids = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
g_ids = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
query, gallery = official.load_gt(SPLITS / "val_gt.csv")
nq = len(q_ids)


def load(run):
    emb = np.load(RUNS / run / "val_pred" / "embeddings.npy")
    return emb[:nq], emb[nq:]


qB, gB = load("convnext_b_v2_softmargin")
qS, gS = load("convnext_s_v1")
qE = np.concatenate([np.sqrt(0.7) * qB, np.sqrt(0.3) * qS], axis=1)     # our ensemble
gE = np.concatenate([np.sqrt(0.7) * gB, np.sqrt(0.3) * gS], axis=1)
sims = qE @ gE.T

qv, qc = query.loc[q_ids, "vehicle_id"].values, query.loc[q_ids, "camera_id"].values
gv, gc = gallery.loc[g_ids, "vehicle_id"].values, gallery.loc[g_ids, "camera_id"].values
twins = (qv[:, None] == gv[None, :]) & (qc[:, None] == gc[None, :])
SCEN = {"A": sims, "B": np.where(twins, -2.0, sims)}                   # B = no same-camera twin


def features(S):
    valid = S > -1.5                                                    # ignore hidden twins
    srt = -np.sort(-S, axis=1)
    top1, top2 = srt[:, 0], srt[:, 1]
    mean = np.array([row[v].mean() for row, v in zip(S, valid)])
    std = np.array([row[v].std() for row, v in zip(S, valid)])
    z = (top1 - mean) / std
    return np.stack([top1, top2, top1 - top2, z], axis=1)


def evaluate_conf(conf, S, threshold=None):
    """Returns PR-AUC, best F1, TNR at best F1, best threshold (or F1 at a given threshold)."""
    top = S.argmax(axis=1)

    def metrics_at(t):
        cands = {q: [(g_ids[top[i]], float(conf[i]))] for i, q in enumerate(q_ids) if conf[i] >= t}
        return official.candidate_metrics(query, gallery, cands)       # official scoring code

    if threshold is not None:
        return metrics_at(threshold)["F1"]
    pr_auc = metrics_at(-np.inf)["PR-AUC"]
    best = max(((metrics_at(t), t) for t in np.quantile(conf, np.linspace(0, 1, 101))),
               key=lambda mt: mt[0]["F1"])
    return pr_auc, best[0]["F1"], best[0]["TNR"], best[1]


XA, XB = features(SCEN["A"]), features(SCEN["B"])
top_correct_A = (gv[SCEN["A"].argmax(axis=1)] == qv).astype(int)       # is the top-1 the right car?

logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pA_log = cross_val_predict(logistic, XA, top_correct_A, groups=qv,
                           cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
pB_log = logistic.fit(XA, top_correct_A).predict_proba(XB)[:, 1]        # trained on A only

methods = {
    "raw cosine (current)": (XA[:, 0], XB[:, 0]),
    "Z-score": (XA[:, 3], XB[:, 3]),
    "logistic (top1, top2, gap, z)": (pA_log, pB_log),
}

rows = []
for name, (cA, cB) in methods.items():
    print("evaluating:", name)
    prA, f1A, tnrA, tA = evaluate_conf(cA, SCEN["A"])
    prB, f1B, tnrB, _ = evaluate_conf(cB, SCEN["B"])
    f1B_transfer = evaluate_conf(cB, SCEN["B"], threshold=tA)
    rows.append((name, prA, f1A, tnrA, tA, prB, f1B, f1B_transfer))

df = pd.DataFrame(rows, columns=["method", "PR-AUC_A", "bestF1_A", "TNR_A", "thr_A",
                                 "PR-AUC_B", "bestF1_B", "F1_B_with_A_thr"])
print(df.round(3).to_string(index=False))
df.to_csv(RUNS / "confidence_results.csv", index=False)