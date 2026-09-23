"""
Batch inference: produces the three official files.

  python run_inference.py --images <dir> --query test_query.csv --gallery test_gallery.csv --out <dir>

Works offline (backbones are built empty, then our trained weights are loaded),
on GPU if available, otherwise on CPU.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from data import test_transform
from model import ReIDModel
from rerank import re_ranking_streaming

LONG_SIDE = 384        # identical to make_crops.py: crop by box, longer side -> 384, then 256x256


def load_model(backbone, weights_path, device):
    state = torch.load(weights_path, map_location="cpu")
    model = ReIDModel(num_classes=state["classifier.weight"].shape[0],
                      backbone=backbone, pretrained=False)          # no internet needed
    model.load_state_dict(state)
    return model.to(device).eval()


def crop_car(files, row):
    img = Image.open(files[row.image_id]).convert("RGB")
    car = img.crop((row.x, row.y, row.x + row.w, row.y + row.h))
    scale = LONG_SIDE / max(car.size)
    if scale < 1:
        car = car.resize((round(car.width * scale), round(car.height * scale)), Image.BICUBIC)
    return car


@torch.no_grad()
def embed_all(models, df, files, device, batch_size=32):
    use_fp16 = device.type == "cuda"
    out = []
    for i in range(0, len(df), batch_size):
        rows = df.iloc[i:i + batch_size]
        batch = torch.stack([test_transform(crop_car(files, r)) for r in rows.itertuples()]).to(device)
        parts = []
        for model, weight in models:
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
                f = model(batch) + model(torch.flip(batch, dims=[3]))       # flip averaging
            f = torch.nn.functional.normalize(f.float(), dim=1)
            parts.append(np.sqrt(weight) * f)                                # glued ensemble
        out.append(torch.cat(parts, dim=1).cpu())
        print(f"\r  embedded {min(i + batch_size, len(df))}/{len(df)}", end="")
    print()
    return torch.cat(out).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with the full camera frames")
    ap.add_argument("--query", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-weights", default="weights/base.pth")
    ap.add_argument("--small-weights", default="weights/small.pth")
    ap.add_argument("--threshold", type=float, default=0.72)    # plateau center, validation
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--k1", type=int, default=6)
    ap.add_argument("--k2", type=int, default=2)
    ap.add_argument("--topk", type=int, default=100)            # re-rank only top-100 candidates
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    q_df = pd.read_csv(args.query, dtype={"image_id": str})
    g_df = pd.read_csv(args.gallery, dtype={"image_id": str})
    files = {p.stem: p for p in Path(args.images).iterdir()}

    models = [(load_model("convnext_base.dinov3_lvd1689m", args.base_weights, device), 0.6),
              (load_model("convnext_small.dinov3_lvd1689m", args.small_weights, device), 0.4)]

    t0 = time.time()
    q_emb = embed_all(models, q_df, files, device)
    g_emb = embed_all(models, g_df, files, device)
    t_embed = time.time() - t0

    # 1) embeddings.npy: queries first, then gallery, in CSV order
    np.save(out / "embeddings.npy", np.concatenate([q_emb, g_emb]))

    cos = q_emb @ g_emb.T
    t0 = time.time()
    sims = cos if args.no_rerank else re_ranking_streaming(q_emb, g_emb, args.k1, args.k2, topk=args.topk)
    t_rerank = time.time() - t0
    order = np.argsort(-sims, axis=1)[:, :10]
    q_ids, g_ids = q_df.image_id.tolist(), g_df.image_id.tolist()

    # 2) submission.csv: no header, query + top-10
    with open(out / "submission.csv", "w") as f:
        for i, qid in enumerate(q_ids):
            f.write(",".join([qid] + [g_ids[j] for j in order[i]]) + "\n")

    # 3) candidates.csv: top-1 (after re-ranking) with its cosine confidence; below threshold = refusal
    top1 = order[:, 0]
    conf = cos[np.arange(len(q_ids)), top1]
    all_c = pd.DataFrame({"query_id": q_ids, "gallery_id": [g_ids[j] for j in top1],
                          "confidence": np.round(conf, 4)})
    all_c.to_csv(out / "all_candidates.csv", index=False)                  # for threshold tuning
    all_c[all_c.confidence >= args.threshold].to_csv(out / "candidates.csv", index=False)

    n = len(q_ids) + len(g_ids)
    print(f"done -> {out}")
    print(f"embedding: {t_embed:.1f}s for {n} images ({1000 * t_embed / n:.1f} ms/image, batch 32)")
    print(f"re-ranking: {t_rerank:.1f}s for {len(q_ids)} queries ({1000 * t_rerank / len(q_ids):.1f} ms/query)")
    print(f"answered {(conf >= args.threshold).sum()}/{len(q_ids)} (threshold {args.threshold})")


if __name__ == "__main__":
    main()