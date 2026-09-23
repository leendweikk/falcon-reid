from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as official

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
    m = official.ranking_metrics(gt_query, gt_gallery, ranked)     # official scoring code
    return m["mAP@10"], m["Rank-1"], m["Rank-5"]


def dba(g, k):
    """Gallery-only smoothing: average each gallery vector with its k nearest OTHER gallery vectors."""
    gg = g @ g.T
    np.fill_diagonal(gg, -np.inf)
    nn = np.argsort(-gg, axis=1)[:, :k]
    g_new = g + g[nn].sum(axis=1)
    return g_new / np.linalg.norm(g_new, axis=1, keepdims=True)


def re_ranking(q, g, k1=20, k2=6, lam=0.3):
    """k-reciprocal re-ranking (Zhong et al. 2017). Returns a similarity (higher = better)."""
    feats = np.concatenate([q, g])
    dist = np.clip(2 - 2 * feats @ feats.T, 0, None)          # squared euclidean for unit vectors
    dist = (dist / dist.max(axis=0)).T.astype(np.float32)
    n = len(feats)
    rank = np.argsort(dist, axis=1)
    V = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        fwd = rank[i, :k1 + 1]
        recip = fwd[np.where(rank[fwd, :k1 + 1] == i)[0]]
        expansion = recip
        for c in recip:
            half = int(np.around(k1 / 2)) + 1
            c_fwd = rank[c, :half]
            c_recip = c_fwd[np.where(rank[c_fwd, :half] == c)[0]]
            if len(np.intersect1d(c_recip, recip)) > 2 / 3 * len(c_recip):
                expansion = np.append(expansion, c_recip)
        expansion = np.unique(expansion)
        w = np.exp(-dist[i, expansion])
        V[i, expansion] = w / w.sum()
    if k2 > 1:
        V = np.stack([V[rank[i, :k2]].mean(axis=0) for i in range(n)])
    jaccard = np.zeros((len(q), n), dtype=np.float32)
    for i in range(len(q)):
        jaccard[i] = 1 - np.minimum(V[i], V).sum(axis=1) / np.maximum(V[i], V).sum(axis=1)
    final = jaccard * (1 - lam) + dist[:len(q)] * lam
    return -final[:, len(q):]                                  # minus distance = similarity


qB, gB = load("convnext_b_v2_softmargin")
qS, gS = load("convnext_s_v1")
simB, simS = qB @ gB.T, qS @ gS.T

results = [("Base alone (current best)", *score(simB)),
           ("Small alone", *score(simS))]

for w in [0.8, 0.7, 0.6]:
    results.append((f"Ensemble {w:.1f} Base + {1 - w:.1f} Small", *score(w * simB + (1 - w) * simS)))

for k in [1, 2]:
    results.append((f"DBA gallery-only k={k} (Base)", *score(qB @ dba(gB, k).T)))

for k1, k2 in [(20, 6), (10, 3), (6, 2)]:
    results.append((f"Re-ranking k1={k1} k2={k2} (Base)", *score(re_ranking(qB, gB, k1, k2))))

# ---- combinations: glue the two fingerprints into one (0.7 Base + 0.3 Small) ----
qE = np.concatenate([np.sqrt(0.7) * qB, np.sqrt(0.3) * qS], axis=1)
gE = np.concatenate([np.sqrt(0.7) * gB, np.sqrt(0.3) * gS], axis=1)

results.append(("Ensemble glued (should match 0.7/0.3)", *score(qE @ gE.T)))
results.append(("Ensemble + DBA k=1   [SAFE]", *score(qE @ dba(gE, 1).T)))
results.append(("Base + DBA k=1 + Re-ranking 10/3", *score(re_ranking(qB, dba(gB, 1), 10, 3))))
results.append(("Ensemble + Re-ranking 10/3", *score(re_ranking(qE, gE, 10, 3))))
results.append(("Ensemble + DBA k=1 + Re-ranking 10/3", *score(re_ranking(qE, dba(gE, 1), 10, 3))))

df = pd.DataFrame(results, columns=["method", "mAP@10", "Rank-1", "Rank-5"])
print(df.round(4).to_string(index=False))
df.to_csv(RUNS / "posthoc_results.csv", index=False)


