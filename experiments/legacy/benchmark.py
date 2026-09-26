"""
Speed benchmark, measured the way the organizers describe it:
  1) time to build the feature of ONE vehicle (batch = 1)
  2) throughput (images per second) with batch processing
  3) re-ranking time per query (top-100 candidates)
for several pipeline options, on GPU (if present) and CPU.

  python benchmark.py                 # GPU + CPU
  python benchmark.py --no-cpu        # GPU only (CPU is slow to measure)
Results -> docs/benchmark.md
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # project root (file moved to experiments/legacy/)
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from reid.data import test_transform
from reid.model import ReIDModel
from reid.rerank import re_ranking_streaming

ROOT = Path(__file__).resolve().parents[2]
LONG_SIDE = 384


def load(backbone, path, device):
    state = torch.load(path, map_location="cpu")
    m = ReIDModel(num_classes=state["classifier.weight"].shape[0], backbone=backbone, pretrained=False)
    m.load_state_dict(state)
    return m.to(device).eval()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def features(models, batch, device, flip):
    fp16 = device.type == "cuda"
    parts = []
    for m, w in models:
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            f = m(batch)
            if flip:
                f = f + m(torch.flip(batch, dims=[3]))
        parts.append(np.sqrt(w) * torch.nn.functional.normalize(f.float(), dim=1))
    return torch.cat(parts, dim=1)


def prepare(path, box):
    img = Image.open(path).convert("RGB")
    x, y, w, h = box
    car = img.crop((x, y, x + w, y + h))
    s = LONG_SIDE / max(car.size)
    if s < 1:
        car = car.resize((round(car.width * s), round(car.height * s)), Image.BICUBIC)
    return test_transform(car)


def timeit(fn, n, device, warmup=3):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    ap.add_argument("--query", default=str(ROOT / "data" / "test_query.csv"))
    ap.add_argument("--gallery", default=str(ROOT / "data" / "test_gallery.csv"))
    args = ap.parse_args()

    q = pd.read_csv(args.query, dtype={"image_id": str})
    files = {p.stem: p for p in Path(args.images).iterdir()}
    rows = q.head(64)
    samples = [(files[r.image_id], (r.x, r.y, r.w, r.h)) for r in rows.itertuples()]

    devices = [torch.device("cuda")] if torch.cuda.is_available() else []
    if not args.no_cpu:
        devices.append(torch.device("cpu"))
    options = [("Base only, no flip", ["base"], False), ("Base only + flip", ["base"], True),
               ("Ensemble, no flip", ["base", "small"], False), ("Ensemble + flip (submitted)", ["base", "small"], True)]
    weights = {"base": ("convnext_base.dinov3_lvd1689m", ROOT / "weights" / "base.pth", 0.6),
               "small": ("convnext_small.dinov3_lvd1689m", ROOT / "weights" / "small.pth", 0.4)}

    results = []
    prep_ms = 1000 * timeit(lambda: prepare(*samples[0]), 20, torch.device("cpu"))
    print(f"image decode + crop + resize (CPU, per vehicle): {prep_ms:.1f} ms")

    for device in devices:
        loaded = {k: load(bb, p, device) for k, (bb, p, _) in weights.items()}
        one = prepare(*samples[0]).unsqueeze(0).to(device)
        batch32 = torch.stack([prepare(*s) for s in samples[:32]]).to(device)
        n1, nb = (50, 10) if device.type == "cuda" else (5, 1)
        for name, keys, flip in options:
            models = [(loaded[k], weights[k][2]) for k in keys]
            t1 = timeit(lambda: features(models, one, device, flip), n1, device)
            tb = timeit(lambda: features(models, batch32, device, flip), nb, device, warmup=1)
            results.append(dict(device=device.type.upper(), option=name,
                                model_ms_batch1=round(1000 * t1, 1),
                                total_ms_batch1=round(1000 * t1 + prep_ms, 1),
                                model_fps_batch32=round(32 / tb, 1)))
            print(results[-1])
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # re-ranking cost per query (CPU numpy), realistic gallery size, 1792-d ensemble features
    g = pd.read_csv(args.gallery)
    rng = np.random.default_rng(0)
    qe = rng.normal(size=(20, 1792)).astype(np.float32); qe /= np.linalg.norm(qe, axis=1, keepdims=True)
    ge = rng.normal(size=(len(g), 1792)).astype(np.float32); ge /= np.linalg.norm(ge, axis=1, keepdims=True)
    t0 = time.perf_counter(); re_ranking_streaming(qe, ge, 6, 2, topk=100)
    rerank_ms = 1000 * (time.perf_counter() - t0) / len(qe)
    print(f"re-ranking (top-100, gallery {len(g)}): {rerank_ms:.1f} ms per query")

    df = pd.DataFrame(results)
    out = ROOT / "experiments" / "legacy" / "benchmark_old.md"
    out.parent.mkdir(exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    out.write_text(
        "# Speed benchmark\n\n"
        f"Hardware: GPU = {gpu}; CPU = {torch.get_num_threads()} threads; torch {torch.__version__}\n\n"
        f"- Image decode + crop + resize (CPU, per vehicle): **{prep_ms:.1f} ms**\n"
        f"- Re-ranking (streaming, top-100, gallery {len(g)}): **{rerank_ms:.1f} ms per query**\n\n"
        "`model_ms_batch1` = feature for ONE vehicle; `total_ms_batch1` = + decode/crop/resize; "
        "`model_fps_batch32` = images/second in batches of 32.\n\n" + df.to_markdown(index=False) + "\n",
        encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
