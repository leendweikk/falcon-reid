"""
Speed test of EXACTLY what we ship (falcon.Extractor), following the organizers' protocol
(answers #31, #32, #34, #37):
  latency_b1 : full cycle for one vehicle (read file, decode, crop, preprocess, forward, L2),
               median of 300 runs after 50 warm-up runs, CUDA-synchronized
  throughput : sustained images/s at batch 1 / 8 / 16 / 32, >= 10 s each; the best counts
  also       : peak VRAM, total weight size of the mode, determinism (two runs, same input)
Points: latency <= 40 ms -> 10, 40-80 linear, > 80 -> 0; FPS >= 100 -> 10, 50-100 linear, < 50 -> 0.

  python tools/speed_bench.py --mode fast --images data/images --csv data/test_query.csv data/test_gallery.csv
  python tools/speed_bench.py --mode accurate --decode pil-draft
  python tools/speed_bench.py --config falcon/config_val42.json --mode fast_vit
Run it only when the GPU is otherwise idle. Results are appended to docs/speed_log_falcon.csv.
"""
import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from falcon.extractor import Extractor, load_config  # noqa: E402

WEIGHT_EXT = {".pt", ".pth", ".bin", ".onnx", ".engine", ".plan", ".safetensors", ".ckpt", ".trt", ".pb",
              ".tflite", ".npz"}                                          # the list from answer #37


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def points_latency(ms):
    return 10.0 if ms <= 40 else max(0.0, 10.0 * (80 - ms) / 40)


def points_fps(fps):
    return 10.0 if fps >= 100 else max(0.0, 10.0 * (fps - 50) / 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="default: falcon/config.json")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--decode", default=None)
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    ap.add_argument("--csv", nargs="+", default=[str(ROOT / "data" / "test_query.csv"),
                                                   str(ROOT / "data" / "test_gallery.csv")])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg["mode"]
    t0 = time.perf_counter()
    ex = Extractor(cfg, mode=mode, decode=args.decode)
    load_s = time.perf_counter() - t0
    dev = ex.device
    files = {p.stem: p for p in Path(args.images).iterdir()}
    df = pd.concat([pd.read_csv(c, dtype={"image_id": str}) for c in args.csv], ignore_index=True)
    items = [(files[r.image_id], (r.x, r.y, r.w, r.h)) for r in df.itertuples()]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for i in range(args.warmup + args.runs):
        sync(dev)
        t = time.perf_counter()
        ex.extract(*items[i % len(items)])
        sync(dev)
        times.append(time.perf_counter() - t)
    lat_ms = 1000 * float(np.median(times[args.warmup:]))

    fps = {}
    for bs in (1, 8, 16, 32):
        ex.extract_batch(items[:bs])
        sync(dev)
        n, i, t = 0, 0, time.perf_counter()
        while time.perf_counter() - t < args.seconds:
            ex.extract_batch([items[(i + k) % len(items)] for k in range(bs)])
            n += bs
            i += bs
        sync(dev)
        fps[bs] = n / (time.perf_counter() - t)

    a, b = ex.extract_batch(items[:32]), ex.extract_batch(items[:32])
    weights_mb = sum(p.stat().st_size for p in (ROOT / "weights").rglob("*")
                     if p.suffix in WEIGHT_EXT and any(p.name == Path(m["file"]).name
                                                        for m in cfg["modes"][mode]["models"])) / 2**20
    best = max(fps, key=fps.get)
    row = {"mode": mode, "decode": ex.decode, "decode_workers": ex.decode_workers, "cpu_threads": __import__("os").cpu_count(),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "torch": torch.__version__, "latency_b1_ms": round(lat_ms, 2),
           **{f"fps_b{k}": round(v, 1) for k, v in fps.items()}, "best_fps": round(fps[best], 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else 0,
           "weights_mb": round(weights_mb), "load_s": round(load_s, 2),
           "determinism_max_diff": float(np.abs(a - b).max()),
           "points_latency": round(points_latency(lat_ms), 2), "points_fps": round(points_fps(fps[best]), 2)}
    for k, v in row.items():
        print(f"{k:>22}: {v}")
    print(f"\n  => speed points on THIS machine: {row['points_latency'] + row['points_fps']:.1f} / 20")
    log = ROOT / "docs" / "speed_log_falcon.csv"
    new = pd.DataFrame([{"machine": platform.node(), **row}])
    old = pd.read_csv(log) if log.exists() else None
    (new if old is None else pd.concat([old, new], ignore_index=True)).to_csv(log, index=False)   # union of columns
    print(f"appended -> {log}")


if __name__ == "__main__":
    main()
