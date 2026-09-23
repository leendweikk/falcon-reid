import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reid import evaluate as official

ROOT = Path(__file__).resolve().parent

SPLITS = (ROOT / "data" / "splits")
pred = Path(sys.argv[1])                       # folder with all_candidates.csv

query, gallery = official.load_gt(SPLITS / "val_gt.csv")
c = pd.read_csv(pred / "all_candidates.csv", dtype={"query_id": str, "gallery_id": str})

rows = []
for t in np.round(np.arange(0.30, 0.901, 0.01), 2):
    sub = c[c.confidence >= t]
    cands = {r.query_id: [(r.gallery_id, float(r.confidence))] for r in sub.itertuples()}
    m = official.candidate_metrics(query, gallery, cands)          # official scoring code
    rows.append((t, m["F1"], m["TNR"], m["Precision"], m["Recall"]))

df = pd.DataFrame(rows, columns=["threshold", "F1", "TNR", "Precision", "Recall"])
print(df.iloc[::5].round(3).to_string(index=False))

best = df.loc[df.F1.idxmax()]
plateau = df[df.F1 >= best.F1 - 0.005]                             # thresholds within 0.005 of best F1
center = plateau.threshold.median()
print(f"\nbest F1 {best.F1:.3f} at threshold {best.threshold:.2f} (TNR {best.TNR:.3f})")
print(f"plateau (F1 within 0.005 of best): {plateau.threshold.min():.2f} - {plateau.threshold.max():.2f}")
print(f"recommended threshold = plateau center = {center:.2f}")
df.to_csv(pred / "threshold_sweep.csv", index=False)