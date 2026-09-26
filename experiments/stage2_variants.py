"""
Two-stage (A10) — can stage 2 be cheaper without losing accuracy? (docs/PLAN.md D5: total run time)

Stage 1 always = fast ViT, no flip (the timed extract()). Stage 2 variants (Base 0.5 + ViT 0.5, glued):
  a) Base flip   + ViT flip      (current two_stage; needs a second ViT pass)
  b) Base flip   + ViT reused    (the stage-1 ViT vectors are reused: only Base is computed in stage 2)
  c) Base noflip + ViT reused    (cheapest)
Each embedding pass is timed on ALL images of the split (batched, like the real run).
Run with the laptop PLUGGED IN (on battery the GPU is throttled).

  python experiments/stage2_variants.py --config falcon/config_val42.json --splits data/splits
  python experiments/stage2_variants.py --config falcon/config_val7.json  --splits data/splits_seed7
Keep rule: the cheapest variant whose mAP@10 is within 0.7 of variant (a) on BOTH splits.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import copy  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.stage_tests import embed, search_all, unit  # noqa: E402
from falcon.extractor import Extractor, load_config  # noqa: E402
from reid import evaluate as official  # noqa: E402


def glue(base, vit):
    return unit(np.concatenate([np.sqrt(0.5) * unit(base), np.sqrt(0.5) * unit(vit)], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    args = ap.parse_args()

    splits = ROOT / args.splits
    name = Path(args.splits).name
    query, gallery = official.load_gt(splits / "val_gt.csv")
    q_df = pd.read_csv(splits / "val_query.csv", dtype={"image_id": str})
    g_df = pd.read_csv(splits / "val_gallery.csv", dtype={"image_id": str})
    g_ids = g_df.image_id.tolist()
    files = {p.stem: p for p in Path(args.images).iterdir()}

    cfg = load_config(args.config)
    base = [m for m in cfg["modes"]["accurate"]["models"] if m["name"] == "base"][0]
    vit = [m for m in cfg["modes"]["accurate"]["models"] if m["name"] == "vit"][0]
    passes = {                                              # name -> (model entry, flip)
        "vit_noflip": (vit, False), "base_flip": (base, True), "base_noflip": (base, False), "vit_flip": (vit, True)}

    vecs, secs = {}, {}
    for pname, (entry, flip) in passes.items():
        c = copy.deepcopy(cfg)
        c["modes"]["_pass"] = {"models": [dict(entry, weight=1.0)], "flip": flip, "refusal_threshold": None}
        ex = Extractor(c, mode="_pass")
        embed(ex, g_df.iloc[:32], files)                   # warm-up (not timed)
        if ex.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        vecs[pname] = (embed(ex, q_df, files), embed(ex, g_df, files))
        if ex.device.type == "cuda":
            torch.cuda.synchronize()
        secs[pname] = time.time() - t0
        print(f"{pname:<12} {secs[pname]:6.1f}s for {len(q_df) + len(g_df)} images")
        del ex

    qf, gf = map(unit, vecs["vit_noflip"])
    variants = {
        "a) Base flip + ViT flip (current)": (glue(vecs["base_flip"][0], vecs["vit_flip"][0]),
                                              glue(vecs["base_flip"][1], vecs["vit_flip"][1]),
                                              secs["vit_noflip"] + secs["base_flip"] + secs["vit_flip"]),
        "b) Base flip + ViT reused": (glue(vecs["base_flip"][0], vecs["vit_noflip"][0]),
                                      glue(vecs["base_flip"][1], vecs["vit_noflip"][1]),
                                      secs["vit_noflip"] + secs["base_flip"]),
        "c) Base noflip + ViT reused": (glue(vecs["base_noflip"][0], vecs["vit_noflip"][0]),
                                        glue(vecs["base_noflip"][1], vecs["vit_noflip"][1]),
                                        secs["vit_noflip"] + secs["base_noflip"]),
    }
    rows = []
    for label, (qh, gh, embed_s) in variants.items():
        t0 = time.time()
        ranked, _ = search_all(qf, gf, qh, gh, 100, 6, 2, 0.3)
        search_s = time.time() - t0
        r = {qid: [g_ids[j] for j in ranked[i]] for i, qid in enumerate(q_df.image_id)}
        m = 100 * official.ranking_metrics(query, gallery, r)["mAP@10"]
        rows.append({"variant": label, "mAP@10": round(m, 2), "embed s (all passes)": round(embed_s, 1),
                     "search s": round(search_s, 1), "total s": round(embed_s + search_s, 1)})
    df = pd.DataFrame(rows)
    df["vs a)"] = (df["mAP@10"] - df.loc[0, "mAP@10"]).round(2)
    n = len(q_df) + len(g_df)
    print(f"\n=== {name}: {len(q_df)} queries + {len(g_df)} gallery = {n} images ===")
    print(df.to_string(index=False))
    print(f"(organizers' rough limit: latency_b1 x n x 3; with the laptop's 33.8 ms: {0.0338 * n * 3:.0f} s)")
    out = ROOT / "runs" / f"stage2_variants_{name}.csv"
    df.to_csv(out, index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
