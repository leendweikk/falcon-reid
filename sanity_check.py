"""
Compares two pipeline outputs on the same test: how often do they pick the same top-1 / overlap in top-10?
  python sanity_check.py <pred_dir_A> <pred_dir_B>
"""
import sys
from pathlib import Path

import pandas as pd


def load(d):
    rows = [line.strip().split(",") for line in open(Path(d) / "submission.csv")]
    return {r[0]: r[1:] for r in rows}


a, b = load(sys.argv[1]), load(sys.argv[2])
common = a.keys() & b.keys()
top1 = sum(a[q][0] == b[q][0] for q in common) / len(common)
overlap = sum(len(set(a[q]) & set(b[q])) for q in common) / (10 * len(common))
print(f"queries compared: {len(common)}")
print(f"same top-1 match : {top1:.1%}")
print(f"top-10 overlap   : {overlap:.1%}")
print("healthy if top-1 agreement is roughly 80%+; very low values mean something is wrong")