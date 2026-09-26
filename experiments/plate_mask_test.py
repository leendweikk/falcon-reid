"""
A15: plate-masking self-test (docs/PLAN.md). Answer #48: finalists are re-tested with the licence plates
PAINTED OVER. This test does the same on OUR validation photos and measures how much each model loses.
It is also the second check for VeRi (A9): the VeRi model is kept only if, with plates painted over,
it is STILL better than the current ViT on both splits (so it did not just learn to read VeRi plates).

Step 1 — prepare (CPU is fine, so it can run while the GPU trains):
  python experiments/plate_mask_test.py prepare
    detects plates (same pinned public detector as prepare_veri.py) in the val query+gallery crops of
    BOTH splits, paints each plate box (+10% margin) with solid gray, saves data/crops_masked/<id>.jpg
    (photos without a detected plate are copied unchanged) and runs/plate_mask_check.jpg to check by eye.

Step 2 — evaluate (GPU, after training):
  python experiments/plate_mask_test.py eval --splits data/splits \
      --weights now=runs/val_vit/best_ema.pth veri=runs/veri_s42/vit.pth
  python experiments/plate_mask_test.py eval --splits data/splits_seed7 \
      --weights now=runs/split7_vit/vit.pth veri=runs/veri_s7/vit.pth
    fast mode (ViT 256, no flip, re-ranking k1=6 k2=2 λ=0.3 top-100), scored with the official
    ranking_metrics, on the original crops and on the masked crops.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from reid.data import CROPS  # noqa: E402

MASKED = ROOT / "data" / "crops_masked"
VIT = "vit_base_patch16_dinov3.lvd1689m"
FILL = (128, 128, 128)


def val_ids(split_dirs):
    ids = []
    for d in split_dirs:
        for f in ("val_query.csv", "val_gallery.csv"):
            ids += pd.read_csv(ROOT / d / f, dtype={"image_id": str}).image_id.tolist()
    return sorted(set(ids))


def paint(img, boxes, margin=0.10):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        draw.rectangle((max(0, x0 - margin * w), max(0, y0 - margin * h),
                        min(img.width - 1, x1 + margin * w), min(img.height - 1, y1 + margin * h)), fill=FILL)
    return out


def prepare(args):
    from experiments.prepare_veri import contact_sheet, load_detector
    ids = val_ids(args.split_dirs)
    if args.limit:
        ids = ids[:args.limit]
    assert ids, "no validation photos found"
    detector, det_path = load_detector(args.detector)
    MASKED.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    check = set(rng.choice(len(ids), size=min(32, len(ids)), replace=False).tolist())
    pairs, n_with, t0 = [], 0, time.time()
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        imgs = [Image.open(CROPS / f"{x}.jpg").convert("RGB") for x in chunk]
        results = detector.predict([np.asarray(im)[:, :, ::-1] for im in imgs], conf=args.conf,
                                   device=args.device, verbose=False)
        for k, (x, im, res) in enumerate(zip(chunk, imgs, results)):
            boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
            n_with += bool(boxes)
            if boxes:
                out = paint(im, boxes)
                out.save(MASKED / f"{x}.jpg", quality=95)
            else:
                out = im
                shutil.copyfile(CROPS / f"{x}.jpg", MASKED / f"{x}.jpg")
            if i + k in check:
                pairs.append((im, out, len(boxes)))
        print(f"\r  {min(i + args.batch, len(ids))}/{len(ids)}  with a plate: {n_with}", end="", flush=True)
    print()
    (ROOT / "runs").mkdir(exist_ok=True)
    contact_sheet(pairs, ROOT / "runs" / "plate_mask_check.jpg")
    print(f"detector: {det_path}")
    print(f"val photos: {len(ids)} | plate found and painted: {n_with} ({n_with / len(ids):.1%}) "
          f"| {time.time() - t0:.0f}s")
    print(f"saved -> {MASKED}  and  runs/plate_mask_check.jpg (CHECK IT BY EYE)")


def evaluate(args):
    import torch
    from torchvision import transforms as T

    from reid import evaluate as official
    from reid.data import MEAN, STD
    from reid.model import load_reid
    from reid.rerank import re_ranking_streaming

    splits = ROOT / args.splits
    query, gallery = official.load_gt(splits / "val_gt.csv")
    q_ids = pd.read_csv(splits / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
    g_ids = pd.read_csv(splits / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
    missing = [x for x in q_ids + g_ids if not (MASKED / f"{x}.jpg").exists()]
    assert not missing, f"{len(missing)} masked photos missing — run the 'prepare' step first"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def embed(model, size, ids, folder, bs=32):
        tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
        out = []
        for i in range(0, len(ids), bs):
            batch = torch.stack([tf(Image.open(folder / f"{x}.jpg").convert("RGB"))
                                 for x in ids[i:i + bs]]).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                f = model(batch)
            out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
        return torch.cat(out).numpy()

    def score(q, g):
        sims = re_ranking_streaming(q, g, 6, 2, 0.3, topk=100)
        order = np.argsort(-sims, axis=1, kind="stable")[:, :10]
        ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
        return 100 * official.ranking_metrics(query, gallery, ranked)["mAP@10"]

    rows = []
    for spec in args.weights:
        label, path = spec.split("=", 1)
        model, size = load_reid(ROOT / path, args.backbone, device)
        res = {}
        for name, folder in (("original", CROPS), ("plates painted", MASKED)):
            res[name] = score(embed(model, size, q_ids, folder), embed(model, size, g_ids, folder))
        rows.append({"model": label, "weights": path, "original": round(res["original"], 2),
                     "plates painted": round(res["plates painted"], 2),
                     "drop": round(res["original"] - res["plates painted"], 2)})
        print(rows[-1])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    name = Path(args.splits).name
    print(f"\n=== {name}: {len(q_ids)} queries, {len(g_ids)} gallery (fast mode, re-rank) ===")
    print(df.to_string(index=False))
    if len(df) == 2:
        a, b = df.iloc[0], df.iloc[1]
        print(f"\nwith plates painted: {b.model} - {a.model} = {b['plates painted'] - a['plates painted']:+.2f} "
              f"({'still better' if b['plates painted'] > a['plates painted'] else 'NOT better'})")
    out = ROOT / "runs" / f"plate_mask_{name}.csv"
    df.to_csv(out, index=False)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--split-dirs", nargs="+", default=["data/splits", "data/splits_seed7"])
    p.add_argument("--detector", default=None, help="local .pt (default: the pinned HF file)")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu", help="cpu by default: the GPU may be training")
    p.add_argument("--limit", type=int, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--splits", required=True)
    e.add_argument("--weights", nargs="+", required=True, help="label=path, reference model first")
    e.add_argument("--backbone", default=VIT)
    args = ap.parse_args()
    prepare(args) if args.step == "prepare" else evaluate(args)


if __name__ == "__main__":
    main()
