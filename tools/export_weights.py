"""
Turn a training checkpoint into a slim inference file for falcon/:
  - drops the ID classifier head (only needed for training)
  - stores the weights in float16 (half the size; the model still runs with fp16 autocast as before)
  - records backbone name, pooling and input size, so inference needs no guessing
  - checks that the slim model gives the same vectors as the original (cosine >= 0.999)
  - prints the sha256 (goes into the README and the Dockerfile check, answer #39)

  python tools/export_weights.py --model base --src weights/base.pth --dst weights/base_infer.pth
  python tools/export_weights.py --model vit  --src weights/vit.pth  --dst weights/vit_infer.pth
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from falcon.extractor import load_infer_weights  # noqa: E402
from reid.model import load_reid  # noqa: E402

BACKBONES = {"base": "convnext_base.dinov3_lvd1689m", "small": "convnext_small.dinov3_lvd1689m",
             "vit": "vit_base_patch16_dinov3.lvd1689m"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=BACKBONES, required=True)
    ap.add_argument("--src", required=True, help="training checkpoint (train.py / train_final.py)")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    full, size = load_reid(args.src, BACKBONES[args.model], "cpu")
    state = {k: v.half() for k, v in full.state_dict().items()
             if k != "classifier.weight" and v.is_floating_point()}
    state.update({k: v for k, v in full.state_dict().items() if not v.is_floating_point()})  # e.g. BN counters
    pool = "gem" if "gem_p" in state else "avg"
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "falcon-infer-v1", "backbone": BACKBONES[args.model], "pool": pool,
                "input_size": size, "state_dict": state}, dst)

    # same vectors? (random images, CPU, float32 both sides)
    slim, size2 = load_infer_weights(dst, "cpu")
    assert size2 == size
    full.backbone.set_grad_checkpointing(False)
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        a = torch.nn.functional.normalize(full(x), dim=1)
        b = torch.nn.functional.normalize(slim(x), dim=1)
    cos = (a * b).sum(1).min().item()
    print(f"{args.src} -> {dst}")
    print(f"  backbone {BACKBONES[args.model]} | pool {pool} | input {size} | "
          f"size {Path(args.src).stat().st_size / 2**20:.0f} MB -> {dst.stat().st_size / 2**20:.0f} MB")
    print(f"  min cosine(original, slim) on random input: {cos:.6f}")
    assert cos >= 0.999, "slim model differs from the original!"
    print(f"  sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
