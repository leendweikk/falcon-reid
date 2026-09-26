"""
Model soup: average the weights of several models trained from the same starting point.
  python soup.py <output.pth> <model1.pth> <model2.pth> ...
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # project root (file moved to experiments/legacy/)
import sys

import torch

out_path, in_paths = sys.argv[1], sys.argv[2:]
states = [torch.load(p, map_location="cpu") for p in in_paths]

soup = {}
for key, first in states[0].items():
    if first.is_floating_point():
        soup[key] = sum(s[key] for s in states) / len(states)     # average weights and BN statistics
    else:
        soup[key] = first                                        # counters (e.g. num_batches_tracked)

torch.save(soup, out_path)
print(f"soup of {len(states)} models -> {out_path}")