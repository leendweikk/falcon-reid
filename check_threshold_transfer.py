"""
Compares top-1 confidence distributions: validation (matched queries only) vs real test.
  python check_threshold_transfer.py <val_pred_dir> <test_pred_dir> <val_threshold>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = Path("C:/falcon/data/splits")
val_dir, test_dir, val_thr = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])

gt = pd.read_csv(SPLITS / "val_gt.csv", dtype={"image_id": str})
cars_in_gallery = set(gt[gt.split == "gallery"].vehicle_id)
car_of = dict(zip(gt.image_id, gt.vehicle_id))

val = pd.read_csv(val_dir / "all_candidates.csv", dtype={"query_id": str})
matched = val.query_id.map(car_of).isin(cars_in_gallery)       # drop the 50 "stranger" cars
v = val.confidence[matched].values
t = pd.read_csv(test_dir / "all_candidates.csv").confidence.values

print(f"{'percentile':>10} | {'val (matched)':>13} | {'test':>6}")
for p in [5, 10, 25, 50, 75, 90]:
    print(f"{p:>10} | {np.percentile(v, p):>13.3f} | {np.percentile(t, p):>6.3f}")

shift = np.median(t) - np.median(v)
new_thr = val_thr + shift
print(f"\nmedian shift (test - val): {shift:+.3f}")
print(f"threshold: validation {val_thr:.2f} -> suggested for test {new_thr:.2f}")
print(f"test queries answered at {val_thr:.2f}: {(t >= val_thr).mean():.1%} | "
      f"at {new_thr:.2f}: {(t >= new_thr).mean():.1%}")
print(f"(for reference: matched val queries answered at {val_thr:.2f}: {(v >= val_thr).mean():.1%})")