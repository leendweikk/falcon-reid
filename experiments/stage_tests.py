"""
Saturday step 3 (docs/PLAN.md A10, A12): short tests that need NO training, on one validation split.
Everything goes through the shipped code (falcon.Extractor on full frames + falcon.search.rank_candidates),
scored with the organizers' ranking_metrics on top-10, exactly like submission.csv.

  1. fast ViT, default re-ranking             (reference; ~79.1 on seed 42 / ~78.0 on seed 7 expected)
  2. fast ViT, small fixed re-ranking grid     (A12: k1, k2, lambda; the grid is fixed in advance)
  3. full ensemble Base+ViT with flip         (reference for the upper bound, ~82.9 / ~81.3)
  4. TWO-STAGE (A10): the fast ViT finds the top-K candidates; the ensemble re-orders ONLY those.
     Allowed: answer #28 (re-ranking with a heavier model inside one query's top-K), not timed (#31).
Each query is handled alone against the gallery (answer #38). Vectors are cached, so a re-run is fast.

  python experiments/stage_tests.py --config falcon/config_val42.json --splits data/splits
  python experiments/stage_tests.py --config falcon/config_val7.json  --splits data/splits_seed7
Keep rules: a change is adopted only if it wins by >= 0.7 on seed 42 AND still wins on seed 7.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import itertools  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from falcon.extractor import Extractor, load_config  # noqa: E402
from falcon.search import rank_candidates  # noqa: E402
from reid import evaluate as official  # noqa: E402

GRID = {"k1": [4, 6, 8], "k2": [1, 2], "lam": [0.2, 0.3, 0.5]}     # fixed in advance (A12)


def embed(ex, df, files, bs=32):
    out = []
    for i in range(0, len(df), bs):
        rows = df.iloc[i:i + bs]
        out.append(ex.extract_batch([(files[r.image_id], (r.x, r.y, r.w, r.h)) for r in rows.itertuples()]))
    return np.concatenate(out)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def search_all(q_cand, g_cand, q_rank, g_rank, topk, k1, k2, lam):
    """For each query ALONE: candidates = cosine top-K with the (q_cand, g_cand) vectors;
    ordering = re-ranking with the (q_rank, g_rank) vectors, restricted to those candidates."""
    ranked, confs = [], []
    for i in range(len(q_cand)):
        cand = np.argsort(-(g_cand @ q_cand[i]), kind="stable")[:topk]
        cos = g_rank[cand] @ q_rank[i]
        pre = np.argsort(-cos, kind="stable")                   # rank_candidates expects cosine order
        cand, cos = cand[pre], cos[pre]
        order, conf = rank_candidates(q_rank[i], g_rank[cand], cos, k1, k2, lam, True)
        ranked.append(cand[order][:10])
        confs.append(conf)
    return ranked, np.array(confs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    ap.add_argument("--fast-mode", default="fast_vit")
    ap.add_argument("--heavy-mode", default="accurate")
    args = ap.parse_args()

    splits = ROOT / args.splits
    name = Path(args.splits).name
    cache = ROOT / "runs" / f"stage_cache_{name}.npz"
    query, gallery = official.load_gt(splits / "val_gt.csv")
    q_df = pd.read_csv(splits / "val_query.csv", dtype={"image_id": str})
    g_df = pd.read_csv(splits / "val_gallery.csv", dtype={"image_id": str})
    g_ids = g_df.image_id.tolist()

    if cache.exists():
        z = np.load(cache)
        qf, gf, qh, gh = z["qf"], z["gf"], z["qh"], z["gh"]
        print(f"vectors from cache {cache}")
    else:
        cfg = load_config(args.config)
        files = {p.stem: p for p in Path(args.images).iterdir()}
        vecs = {}
        for mode in (args.fast_mode, args.heavy_mode):
            t0 = time.time()
            ex = Extractor(cfg, mode=mode)
            vecs[mode] = (embed(ex, q_df, files), embed(ex, g_df, files))
            print(f"embedded {mode} in {time.time() - t0:.0f}s")
            del ex
        qf, gf = vecs[args.fast_mode]
        qh, gh = vecs[args.heavy_mode]
        np.savez(cache, qf=qf, gf=gf, qh=qh, gh=gh)
    qf, gf, qh, gh = map(unit, (qf, gf, qh, gh))

    def score(ranked):
        r = {qid: [g_ids[j] for j in ranked[i]] for i, qid in enumerate(q_df.image_id)}
        return 100 * official.ranking_metrics(query, gallery, r)["mAP@10"]

    rows = []

    def run(label, **kw):
        t0 = time.time()
        ranked, _ = search_all(**kw)
        rows.append({"test": label, "mAP@10": score(ranked), "seconds": round(time.time() - t0, 1)})
        print(f"{label:<52} {rows[-1]['mAP@10']:.2f}  ({rows[-1]['seconds']}s)")

    print(f"\n=== {name}: {len(q_df)} queries, {len(g_df)} gallery ===")
    run("1  fast ViT, default re-rank (k1=6 k2=2 lam=0.3)", q_cand=qf, g_cand=gf, q_rank=qf, g_rank=gf,
        topk=100, k1=6, k2=2, lam=0.3)
    for k1, k2, lam in itertools.product(GRID["k1"], GRID["k2"], GRID["lam"]):
        if (k1, k2, lam) == (6, 2, 0.3):
            continue
        run(f"2  fast ViT, re-rank k1={k1} k2={k2} lam={lam}", q_cand=qf, g_cand=gf, q_rank=qf, g_rank=gf,
            topk=100, k1=k1, k2=k2, lam=lam)
    run("3  full ensemble Base+ViT flip, default re-rank", q_cand=qh, g_cand=gh, q_rank=qh, g_rank=gh,
        topk=100, k1=6, k2=2, lam=0.3)
    for topk in (20, 50, 100):
        run(f"4  TWO-STAGE: ViT top-{topk} -> ensemble re-orders", q_cand=qf, g_cand=gf, q_rank=qh, g_rank=gh,
            topk=topk, k1=6, k2=2, lam=0.3)

    df = pd.DataFrame(rows)
    ref = df.loc[0, "mAP@10"]
    df["gain vs fast ViT"] = df["mAP@10"] - ref
    out = ROOT / "runs" / f"stage_tests_{name}.csv"
    df.to_csv(out, index=False)
    print("\n" + df.round(2).to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
