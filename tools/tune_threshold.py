"""
Refusal threshold for ANY pipeline (item B4), tuned on our validation splits only.

Input: run folders written by `python -m falcon.predict ... --gt ...` (all_candidates.csv = top-1 +
confidence for every query) and the matching ground truth. Objective = the official candidate score
0.7*F1 + 0.3*TNR (answer #15), with the open-set queries re-weighted to the real test's 20% (answer #17).
Choice rule: the plateau (within 0.005 of the best score) on EACH split; the threshold is the middle of
the range where the plateaus overlap, so it is justified on both splits, not tuned to one.

  python tools/tune_threshold.py --run runs/ts_val42 --gt data/splits/val_gt.csv \
                                 --run runs/ts_val7 --gt data/splits_seed7/val_gt.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reid import evaluate as official  # noqa: E402

OPENSET_SHARE = 0.20


def score_at(ok, has, sig, t, w):
    ans = sig >= t
    tp = np.sum(ans & has & ok)
    fp_m = np.sum(ans & has & ~ok)
    fn = np.sum(~ans & has)
    fp_o = np.sum(ans & ~has) * w
    tn = np.sum(~ans & ~has) * w
    f1 = 2 * tp / (2 * tp + fp_m + fp_o + fn) if tp else 0.0
    tnr = tn / (tn + fp_o) if (tn + fp_o) else float("nan")
    return 0.7 * f1 + 0.3 * tnr, f1, tnr


def load(run, gt):
    query, gallery = official.load_gt(gt)
    cand = pd.read_csv(Path(run) / "all_candidates.csv", dtype={"query_id": str, "gallery_id": str})
    gal_vid = gallery.vehicle_id.to_dict()
    has = np.array([official.valid_positives(query.loc[q], gallery) > 0 for q in cand.query_id])
    ok = np.array([gal_vid[g] == query.loc[q].vehicle_id for q, g in zip(cand.query_id, cand.gallery_id)])
    sig = cand.confidence.to_numpy(float)
    if has.all() or not has.any():
        sys.exit(f"{run}: needs queries WITH and WITHOUT a pair in the gallery (use our validation splits)")
    w = (OPENSET_SHARE / (1 - OPENSET_SHARE)) * has.sum() / (~has).sum()
    return ok, has, sig, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--gt", action="append", required=True)
    args = ap.parse_args()
    assert len(args.run) == len(args.gt)

    splits = [load(r, g) for r, g in zip(args.run, args.gt)]
    lo_all, hi_all, rows = -np.inf, np.inf, []
    for (ok, has, sig, w), run in zip(splits, args.run):
        grid = np.unique(np.quantile(sig, np.linspace(0.01, 0.99, 400)))
        res = np.array([score_at(ok, has, sig, t, w)[0] for t in grid])
        best = res.argmax()
        plateau = grid[res >= res[best] - 0.005]
        lo_all, hi_all = max(lo_all, plateau.min()), min(hi_all, plateau.max())
        rows.append({"run": run, "queries": len(sig), "open-set share (real)": round(float((~has).mean()), 3),
                     "best score": round(float(res[best]), 4), "at": round(float(grid[best]), 4),
                     "plateau": f"{plateau.min():.3f}-{plateau.max():.3f}",
                     "PR-AUC": round(official.pr_auc(sig, has.astype(int)), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    if lo_all <= hi_all:
        t = float(np.round((lo_all + hi_all) / 2, 3))
        print(f"\nplateaus overlap on all splits: {lo_all:.3f}-{hi_all:.3f}  ->  threshold {t}")
    else:
        t = float(np.round((lo_all + hi_all) / 2, 3))
        print(f"\n!! plateaus do NOT overlap ({hi_all:.3f} < {lo_all:.3f}); compromise threshold {t} (check by hand)")
    for (ok, has, sig, w), run in zip(splits, args.run):
        s, f1, tnr = score_at(ok, has, sig, t, w)
        print(f"  {run}: at {t}: score {s:.4f} (F1 {f1:.4f}, TNR {tnr:.4f}, 20% open-set weighting)")


if __name__ == "__main__":
    main()
