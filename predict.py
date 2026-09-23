import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from data import test_transform, CROPS
from model import ReIDModel


def load_model(ckpt_path, backbone="convnext_base.dinov3_lvd1689m"):
    state = torch.load(ckpt_path, map_location="cpu")
    num_classes = state["classifier.weight"].shape[0]      # 1241 (val model) or 1541 (final)
    model = ReIDModel(num_classes=num_classes, backbone=backbone)
    model.load_state_dict(state)
    return model.cuda().eval()


@torch.no_grad()
def embed(model, image_ids, batch_size=64):
    feats = []
    for i in range(0, len(image_ids), batch_size):
        batch = torch.stack([test_transform(Image.open(CROPS / f"{iid}.jpg").convert("RGB"))
                             for iid in image_ids[i:i + batch_size]]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(batch) + model(torch.flip(batch, dims=[3]))     # flip averaging
        feats.append(f.float().cpu())
        print(f"\r embedded {min(i + batch_size, len(image_ids))}/{len(image_ids)}", end="")
    print()
    return torch.nn.functional.normalize(torch.cat(feats), dim=1).numpy()


def write_outputs(q_ids, g_ids, q_emb, g_emb, out_dir, threshold):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sims = q_emb @ g_emb.T
    order = np.argsort(-sims, axis=1)[:, :10]

    # 1) embeddings.npy : all queries (file order), then all gallery (file order)
    np.save(out_dir / "embeddings.npy", np.concatenate([q_emb, g_emb]).astype(np.float32))

    # 2) submission.csv : no header, query + top-10 gallery ids
    with open(out_dir / "submission.csv", "w") as f:
        for qi, qid in enumerate(q_ids):
            f.write(",".join([qid] + [g_ids[j] for j in order[qi]]) + "\n")

    # 3) candidates.csv : top-1 with confidence, only if confidence >= threshold (else refusal)
    rows = []
    for qi, qid in enumerate(q_ids):
        conf = float(sims[qi, order[qi, 0]])
        if conf >= threshold:
            rows.append((qid, g_ids[order[qi, 0]], round(conf, 4)))
    pd.DataFrame(rows, columns=["query_id", "gallery_id", "confidence"]).to_csv(
        out_dir / "candidates.csv", index=False)

    print(f"saved to {out_dir} | answered {len(rows)}/{len(q_ids)} queries, "
          f"refused {len(q_ids) - len(rows)} (threshold {threshold})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=-1.0)   # -1 = never refuse
    ap.add_argument("--backbone", default="convnext_base.dinov3_lvd1689m")
    args = ap.parse_args()

    q_ids = pd.read_csv(args.query, dtype={"image_id": str}).image_id.tolist()
    g_ids = pd.read_csv(args.gallery, dtype={"image_id": str}).image_id.tolist()

    model = load_model(args.ckpt, args.backbone)
    q_emb, g_emb = embed(model, q_ids), embed(model, g_ids)
    write_outputs(q_ids, g_ids, q_emb, g_emb, args.out, args.threshold)