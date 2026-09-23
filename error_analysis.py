"""
Error analysis on validation: which queries fail, and why?
  python error_analysis.py runs/pipeline_val_final
Writes: <pred>/error_analysis/summary.txt, per_query.csv, worst_XX.jpg grids
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageStat

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
CROPS = ROOT / "data" / "crops"
PRED = Path(sys.argv[1])
OUT = PRED / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)
N_GRIDS = 20

qdf = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str})
gdf = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str})
gt = pd.read_csv(SPLITS / "val_gt.csv", dtype={"image_id": str}).set_index("image_id")
q_ids, g_ids = qdf.image_id.tolist(), gdf.image_id.tolist()
emb = np.load(PRED / "embeddings.npy")
q_emb, g_emb = emb[:len(q_ids)], emb[len(q_ids):]
cos = q_emb @ g_emb.T
g_index = {g: j for j, g in enumerate(g_ids)}
ranked = {r[0]: r[1:] for r in (line.strip().split(",") for line in open(PRED / "submission.csv"))}

gv, gc = gt.loc[g_ids, "vehicle_id"].values, gt.loc[g_ids, "camera_id"].values


def brightness(image_id):
    return ImageStat.Stat(Image.open(CROPS / f"{image_id}.jpg").convert("L")).mean[0]


rows = []
for qi, row in qdf.iterrows():
    qid = row.image_id
    v, c = gt.loc[qid, "vehicle_id"], gt.loc[qid, "camera_id"]
    positives = [g_ids[j] for j in np.where((gv == v) & (gc != c))[0]]
    if not positives:
        continue                                            # open-set query: not in mAP
    junk = {g_ids[j] for j in np.where((gv == v) & (gc == c))[0]}
    clean = [g for g in ranked[qid] if g not in junk][:10]  # same as official protocol
    rel = np.array([gt.loc[g, "vehicle_id"] == v for g in clean])
    hits = np.cumsum(rel)
    ap = float((hits / np.arange(1, len(rel) + 1) * rel).sum() / min(len(positives), 10)) if rel.any() else 0.0
    first = int(np.argmax(rel)) + 1 if rel.any() else 99
    best_true = max(positives, key=lambda g: cos[qi, g_index[g]])
    wrong_top = next((g for g in clean if gt.loc[g, "vehicle_id"] != v), None)
    rows.append(dict(
        query=qid, vehicle=v, AP=ap, first_correct_rank=first,
        box_area=row.w * row.h, brightness=brightness(qid),
        n_positives=len(positives), has_twin=bool(junk),
        sim_best_true=cos[qi, g_index[best_true]],
        sim_best_wrong=cos[qi, g_index[wrong_top]] if wrong_top else np.nan,
        best_true=best_true, clean_top5="|".join(clean[:5]),
    ))

df = pd.DataFrame(rows)
df.to_csv(OUT / "per_query.csv", index=False)
fail = df[df.first_correct_rank > 5]
lines = [f"queries scored: {len(df)} | mAP@10 {df.AP.mean():.4f} | "
         f"Rank-1 {(df.first_correct_rank == 1).mean():.4f} | Rank-5 {(df.first_correct_rank <= 5).mean():.4f}",
         f"failures (correct car not in top-5): {len(fail)} ({len(fail) / len(df):.1%})", ""]


def by_quartile(col, label):
    qs = pd.qcut(df[col].rank(method="first"), 4,                        # rank first: robust to repeated values
                 labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    t = df.groupby(qs, observed=True).agg(mAP=("AP", "mean"),
                                          fail_rate=("first_correct_rank", lambda r: (r > 5).mean()),
                                          range_min=(col, "min"), range_max=(col, "max"))
    lines.extend([f"--- by {label} ---", t.round(3).to_string(), ""])


by_quartile("box_area", "box size (pixels)")
by_quartile("brightness", "brightness (0 dark - 255 bright)")
lines.extend(["--- by number of correct gallery photos ---",
              df.groupby("n_positives").agg(queries=("AP", "size"), mAP=("AP", "mean")).round(3).to_string(), ""])
lookalike = (fail.sim_best_wrong > fail.sim_best_true).mean() if len(fail) else float("nan")
lines.extend([
    "--- why failures happen ---",
    f"in {lookalike:.0%} of failures, a WRONG car looks more similar than the best TRUE match (lookalike problem)",
    f"median similarity in failures: best true {fail.sim_best_true.median():.3f} vs best wrong {fail.sim_best_wrong.median():.3f}",
    f"median similarity in successes: best true {df[df.first_correct_rank == 1].sim_best_true.median():.3f}",
])
summary = "\n".join(lines)
print(summary)
(OUT / "summary.txt").write_text(summary, encoding="utf-8")

# ---- picture grids for the worst queries: query | top-5 (red=wrong, green=right) | true match ----
T = (192, 144)


def tile(image_id, color, label):
    im = Image.open(CROPS / f"{image_id}.jpg").convert("RGB").resize(T)
    canvas = Image.new("RGB", (T[0] + 8, T[1] + 26), color)
    canvas.paste(im, (4, 4))
    ImageDraw.Draw(canvas).text((6, T[1] + 8), label, fill="white")
    return canvas


worst = df.sort_values(["AP", "sim_best_true"]).head(N_GRIDS)
for n, r in enumerate(worst.itertuples(), 1):
    qi = q_ids.index(r.query)
    tiles = [tile(r.query, (60, 60, 200), f"QUERY  bright {r.brightness:.0f}")]
    for k, g in enumerate(r.clean_top5.split("|"), 1):
        ok = gt.loc[g, "vehicle_id"] == r.vehicle
        tiles.append(tile(g, (0, 150, 0) if ok else (190, 0, 0), f"#{k} sim {cos[qi, g_index[g]]:.2f}"))
    tiles.append(tile(r.best_true, (0, 150, 0), f"TRUE sim {r.sim_best_true:.2f} rank {r.first_correct_rank if r.first_correct_rank < 99 else '>10'}"))
    grid = Image.new("RGB", (len(tiles) * tiles[0].width, tiles[0].height), "black")
    for i, t in enumerate(tiles):
        grid.paste(t, (i * t.width, 0))
    grid.save(OUT / f"worst_{n:02d}.jpg", quality=90)
print(f"\nsaved {len(worst)} picture grids + per_query.csv + summary.txt to {OUT}")