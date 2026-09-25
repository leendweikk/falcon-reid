"""
Speed benchmark that copies the organizers' protocol (official answers #31, #34, #37):
  - full cycle per vehicle: read file from disk -> decode JPEG -> crop by bbox -> preprocess
    -> forward -> L2-normalize  (gallery search and re-ranking are NOT timed)
  - latency_b1: median of 300 runs after 50 warm-up runs, CUDA-synchronized around each run
  - throughput: sustained images/second at batch 1 / 8 / 16 / 32, >= 10 s each; the best counts
  - also: peak VRAM, weights size, determinism (two runs on the same data)
Points (answer #34): latency <=40 ms = 10, 40-80 ms linear, >80 ms = 0;
                     throughput >=100 FPS = 10, 50-100 linear, <50 = 0.

Decode variants (accuracy must be re-checked on validation before one is adopted):
  pil        PIL full decode, crop, long side -> 384, resize 256 (= current pipeline, = training crops)
  pil-draft  PIL decodes the JPEG directly at 1/2, 1/4 or 1/8 size when the car stays >= 384 px
  gpu        nvJPEG decode on the GPU (torchvision.io), crop + resize on the GPU

  python experiments/official_bench.py --model base --weights weights/base.pth --decode pil
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import decode_jpeg, read_file

from reid.data import MEAN, STD
from reid.model import load_reid

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = {"base": "convnext_base.dinov3_lvd1689m", "small": "convnext_small.dinov3_lvd1689m",
             "vit": "vit_base_patch16_dinov3.lvd1689m"}
LONG_SIDE = 384


class Extractor:
    """extract(list of (path, (x, y, w, h))) -> L2-normalized float32 embeddings (N, D)."""

    def __init__(self, model, size, device, decode, workers, channels_last):
        self.model, self.size, self.device, self.decode = model, size, device, decode
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.channels_last = channels_last

    # ---- CPU decode paths: return uint8 HxWx3 numpy of the car, long side <= 384 ----
    def _pil(self, path, box):
        img = Image.open(path)
        x, y, w, h = box
        if self.decode == "pil-draft":
            scale = 1
            for s in (8, 4, 2):                                   # biggest reduction keeping the car >= 384 px
                if max(w, h) / s >= LONG_SIDE:
                    scale = s
                    break
            if scale > 1:
                orig_w = img.width
                img.draft("RGB", (img.width // scale, img.height // scale))
                f = orig_w / img.width                            # reduction the decoder actually applied
                x, y, w, h = x / f, y / f, w / f, h / f
        img = img.convert("RGB")
        car = img.crop((round(x), round(y), round(x + w), round(y + h)))
        s = LONG_SIDE / max(car.size)
        if s < 1:
            car = car.resize((round(car.width * s), round(car.height * s)), Image.BICUBIC)
        car = car.resize((self.size, self.size), Image.BILINEAR)   # = torchvision Resize default
        return np.asarray(car)

    def _gpu(self, path, box):
        img = decode_jpeg(read_file(str(path)), device=self.device)      # 3xHxW uint8 on GPU
        x, y, w, h = (int(round(v)) for v in box)
        car = img[:, y:y + h, x:x + w].unsqueeze(0).float()
        return F.interpolate(car, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)

    @torch.no_grad()
    def extract(self, items):
        if self.decode == "gpu":
            batch = torch.cat([self._gpu(p, b) for p, b in items]) / 255.0
        else:
            arrays = list(self.pool.map(lambda it: self._pil(*it), items)) if self.pool \
                else [self._pil(p, b) for p, b in items]
            batch = torch.from_numpy(np.stack(arrays)).to(self.device, non_blocking=True)
            batch = batch.permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - self.mean) / self.std
        if self.channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            f = self.model(batch)
        return F.normalize(f.float(), dim=1).cpu().numpy()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def points_latency(ms):
    return 10.0 if ms <= 40 else max(0.0, 10.0 * (80 - ms) / 40)


def points_fps(fps):
    return 10.0 if fps >= 100 else max(0.0, 10.0 * (fps - 50) / 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=BACKBONES, default="base")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "base.pth"))
    ap.add_argument("--decode", choices=["pil", "pil-draft", "gpu"], default="pil")
    ap.add_argument("--workers", type=int, default=8, help="decode threads for batches")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    ap.add_argument("--csv", nargs="+", default=[str(ROOT / "data" / "test_query.csv"),
                                                   str(ROOT / "data" / "test_gallery.csv")])
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    model, size = load_reid(args.weights, BACKBONES[args.model], device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    load_s = time.perf_counter() - t0
    files = {p.stem: p for p in Path(args.images).iterdir()}
    df = pd.concat([pd.read_csv(c, dtype={"image_id": str}) for c in args.csv], ignore_index=True)
    items = [(files[r.image_id], (r.x, r.y, r.w, r.h)) for r in df.itertuples()]
    ex = Extractor(model, size, device, args.decode, args.workers, args.channels_last)
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- latency, batch = 1: 50 warm-up + 300 timed, median ----
    times = []
    for i in range(350):
        sync(device)
        t = time.perf_counter()
        ex.extract([items[i % len(items)]])
        sync(device)
        times.append(time.perf_counter() - t)
    lat_ms = 1000 * float(np.median(times[50:]))

    # ---- throughput at batch 1/8/16/32, >= N seconds each ----
    fps = {}
    for bs in (1, 8, 16, 32):
        ex.extract(items[:bs])                                     # warm-up for this shape
        sync(device)
        n, i, t = 0, 0, time.perf_counter()
        while time.perf_counter() - t < args.seconds:
            batch = [items[(i + k) % len(items)] for k in range(bs)]
            ex.extract(batch)
            n += bs
            i += bs
        sync(device)
        fps[bs] = n / (time.perf_counter() - t)

    # ---- determinism: same 32 vehicles twice ----
    a, b = ex.extract(items[:32]), ex.extract(items[:32])
    max_diff = float(np.abs(a - b).max())

    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
    size_mb = Path(args.weights).stat().st_size / 2**20
    best_bs = max(fps, key=fps.get)
    row = {"model": args.model, "decode": args.decode, "workers": args.workers,
           "channels_last": args.channels_last, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "latency_b1_ms": round(lat_ms, 2), **{f"fps_b{k}": round(v, 1) for k, v in fps.items()},
           "best_fps": round(fps[best_bs], 1), "peak_vram_mb": round(peak_vram),
           "weights_mb": round(size_mb), "load_s": round(load_s, 2), "determinism_max_diff": max_diff,
           "points_latency": round(points_latency(lat_ms), 2), "points_fps": round(points_fps(fps[best_bs]), 2)}
    for k, v in row.items():
        print(f"{k:>22}: {v}")
    print(f"\n  => speed points on THIS machine: {row['points_latency'] + row['points_fps']:.1f} / 20 "
          f"(their RTX A5000 is faster)")

    log = ROOT / "docs" / "speed_log.csv"
    log.parent.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"appended -> {log}")


if __name__ == "__main__":
    main()
