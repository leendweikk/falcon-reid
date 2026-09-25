"""
Refusal study (Track B): which confidence signal and which threshold maximize the OFFICIAL
candidate score  0.7 * F1 + 0.3 * TNR  (official answer #15), with the open-set share set to
the real test's 20% (official answer #17).

Signals (all computed for ONE query against the gallery; no other queries are used):
  cosine   cosine similarity of the re-ranked top-1 (current method)
  gap      top-1 cosine minus the best cosine among the OTHER gallery items (is the model torn?)
  cos+gap  the sum of both

  python experiments/threshold_study.py runs/speed_base_noflip
  python experiments/threshold_study.py <seed-7 run dir> --splits data/splits_seed7
Needs in the run dir: embeddings.npy + all_candidates.csv (written by run_inference.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root

import argparse

import numpy as np
import pandas as pd

from reid import evaluate as official

ROOT = Path(__file__).resolve().parents[1]
OPENSET_SHARE = 0.20                       # official answer #17: 20% of test queries have no pair


def score_at(ok, has, sig, t, w):
    """ok: top-1 correct; has: query has a valid pair; w: weight of open-set queries."""
    ans = sig >= t
    tp = np.sum(ans & has & ok)
    fp_m = np.sum(ans & has & ~ok)
    fn = np.sum(~ans & has)
    fp_o = np.sum(ans & ~has) * w
    tn = np.sum(~ans & ~has) * w
    f1 = 2 * tp / (2 * tp + fp_m + fp_o + fn) if tp else 0.0
    tnr = tn / (tn + fp_o) if (tn + fp_o) else float("nan")
    return 0.7 * f1 + 0.3 * tnr, f1, tnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="run dir with embeddings.npy + all_candidates.csv")
    ap.add_argument("--splits", default="data/splits")
    args = ap.parse_args()
    pred, splits = Path(args.pred), ROOT / args.splits

    query, gallery = official.load_gt(splits / "val_gt.csv")
    q_ids = pd.read_csv(splits / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
    g_ids = pd.read_csv(splits / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
    emb = np.load(pred / "embeddings.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q, g = emb[:len(q_ids)], emb[len(q_ids):]
    assert len(g) == len(g_ids), "embeddings.npy does not match the split CSVs"

    cand = pd.read_csv(pred / "all_candidates.csv", dtype={"query_id": str, "gallery_id": str}).set_index("query_id")
    g_index = {gid: j for j, gid in enumerate(g_ids)}
    gal_vid = gallery.vehicle_id.to_dict()
    cos = q @ g.T

    has, ok, conf, gap = [], [], [], []
    for i, qid in enumerate(q_ids):
        row = query.loc[qid]
        top = cand.loc[qid, "gallery_id"]
        j = g_index[top]
        has.append(official.valid_positives(row, gallery) > 0)
        ok.append(gal_vid[top] == row.vehicle_id)
        c = cos[i, j]
        rest = np.delete(cos[i], j)
        conf.append(c)
        gap.append(c - rest.max())
    has, ok, conf, gap = map(np.array, (has, ok, conf, gap))

    n_open, n_match = (~has).sum(), has.sum()
    w20 = (OPENSET_SHARE / (1 - OPENSET_SHARE)) * n_match / n_open
    print(f"queries: {len(has)} | with a pair: {n_match} | open-set: {n_open} "
          f"({n_open / len(has):.1%}) -> open-set weight for 20%: {w20:.2f}")

    # sanity check against the official code: cosine signal, threshold 0.72, real share
    t = 0.72
    cands = {qid: [(cand.loc[qid, "gallery_id"], float(conf[i]))] for i, qid in enumerate(q_ids) if conf[i] >= t}
    m = official.candidate_metrics(query, gallery, cands)
    _, f1, tnr = score_at(ok, has, conf, t, 1.0)
    print(f"sanity @0.72: official F1 {m['F1']:.4f} TNR {m['TNR']:.4f} | ours F1 {f1:.4f} TNR {tnr:.4f}")

    rows = []
    for name, sig in [("cosine", conf), ("gap", gap), ("cos+gap", conf + gap)]:
        grid = np.unique(np.quantile(sig, np.linspace(0.01, 0.99, 400)))
        for share, w in [("real", 1.0), ("20%", w20)]:
            res = np.array([score_at(ok, has, sig, t, w) for t in grid])
            best = res[:, 0].argmax()
            plateau = grid[res[:, 0] >= res[best, 0] - 0.005]
            center = float(np.median(plateau))
            s_c, f1_c, tnr_c = score_at(ok, has, sig, center, w)
            rows.append({"signal": name, "open-set share": share,
                         "best score": res[best, 0], "at threshold": grid[best],
                         "plateau": f"{plateau.min():.3f}-{plateau.max():.3f}",
                         "chosen (plateau center)": center, "score there": s_c,
                         "F1": f1_c, "TNR": tnr_c,
                         "PR-AUC": official.pr_auc(sig, has.astype(int))})
    df = pd.DataFrame(rows)
    print("\n" + df.round(4).to_string(index=False))
    df.to_csv(pred / "threshold_study.csv", index=False)
    print(f"\ncandidate points (10% of the total) = 10 x score.  saved -> {pred / 'threshold_study.csv'}")


if __name__ == "__main__":
    main()
