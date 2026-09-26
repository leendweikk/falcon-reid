"""
Batch run: images folder + query/gallery CSVs -> the three official files (answers #40, #41).

    python -m falcon.predict --images data/images --query test_query.csv --gallery test_gallery.csv --out out/

Writes to --out:
  embeddings.npy     float32 (n_query + n_gallery, D): queries first, then gallery, CSV order (#27)
  submission.csv     no header: query_id + 10 gallery_ids for EVERY query (#21)
  candidates.csv     header query_id,gallery_id,confidence; a query is refused = no row (#18, #20)
  all_candidates.csv top-1 + confidence for every query (not submitted; used to tune the threshold)
  run_info.json      settings, versions and timings of this run

Validation (our own splits, where the answers are known):
    python -m falcon.predict --images data/images --query data/splits/val_query.csv \
        --gallery data/splits/val_gallery.csv --out runs/check_fast --gt data/splits/val_gt.csv
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from falcon.extractor import ROOT, Extractor, load_config
from falcon.search import Gallery


def find_images(folder):
    files = {}
    for p in Path(folder).iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            files[p.stem] = p
    return files


def embed_csv(ex, df, files, batch_size, label):
    out = []
    t0 = time.time()
    for i in range(0, len(df), batch_size):
        rows = df.iloc[i:i + batch_size]
        items = [(files[r.image_id], (r.x, r.y, r.w, r.h)) for r in rows.itertuples()]
        out.append(ex.extract_batch(items))
        print(f"\r  {label}: {min(i + batch_size, len(df))}/{len(df)}", end="", flush=True)
    print(f"  ({time.time() - t0:.1f}s)")
    return np.concatenate(out) if out else np.zeros((0, ex.dim), np.float32)


def main():
    ap = argparse.ArgumentParser(description="Falcon ReID: produce submission.csv, embeddings.npy, candidates.csv")
    ap.add_argument("--images", required=True, help="folder with the full camera frames")
    ap.add_argument("--query", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="default: falcon/config.json")
    ap.add_argument("--mode", default=None, help="fast | accurate (default: from config)")
    ap.add_argument("--threshold", type=float, default=None, help="override the refusal threshold")
    ap.add_argument("--decode", default=None, help="pil | pil-draft (default: from config)")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gt", default=None, help="ground truth CSV (our validation splits only) -> print official metrics")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg["mode"]
    threshold = args.threshold if args.threshold is not None else cfg["modes"][mode]["refusal_threshold"]
    if threshold is None:
        raise SystemExit(f"mode '{mode}' has no tuned refusal threshold yet: tune it or pass --threshold")
    rr = cfg["rerank"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    spec = cfg["modes"][mode]
    ex = Extractor(cfg, mode=mode, device=args.device, decode=args.decode)
    # two-stage (answer #28): a heavier model re-orders the fast model's top-K; it is not timed (#31)
    rescore_mode = spec.get("rescore_mode")
    ex2 = Extractor(cfg, mode=rescore_mode, device=args.device, decode=args.decode) if rescore_mode else None
    t_load = time.time() - t_start
    print(f"mode {mode} | device {ex.device} | decode {ex.decode} | flip {ex.flip} | "
          f"models {[m['name'] for m in spec['models']]} | dim {ex.dim}"
          + (f" | stage 2: {rescore_mode} re-orders top-{spec['rescore_topk']}" if ex2 else "")
          + f" | loaded in {t_load:.1f}s")

    q_df = pd.read_csv(args.query, dtype={"image_id": str})
    g_df = pd.read_csv(args.gallery, dtype={"image_id": str})
    files = find_images(args.images)
    missing = [i for i in pd.concat([q_df.image_id, g_df.image_id]) if i not in files]
    if missing:
        raise SystemExit(f"{len(missing)} images listed in the CSVs are missing from {args.images}, e.g. {missing[:3]}")

    # 1) vectors: every image on its own (the gallery is the static base, answer #38)
    t0 = time.time()
    g_emb = embed_csv(ex, g_df, files, cfg["batch_size"], "gallery")
    q_emb = embed_csv(ex, q_df, files, cfg["batch_size"], "query")
    t_embed = time.time() - t0
    # embeddings.npy = the vectors of the timed feature extractor (stage 1), queries first (answer #27)
    np.save(out / "embeddings.npy", np.concatenate([q_emb, g_emb]).astype(np.float32))
    t_embed2, g_emb2, q_emb2 = 0.0, None, None
    if ex2:
        t0 = time.time()
        g_emb2 = embed_csv(ex2, g_df, files, cfg["batch_size"], "gallery (stage 2)")
        q_emb2 = embed_csv(ex2, q_df, files, cfg["batch_size"], "query (stage 2)")
        t_embed2 = time.time() - t0

    # 2) search: queries are handled ONE BY ONE against the gallery (stream protocol, answers #38/#40)
    gallery = Gallery(g_emb, g_df.image_id, rr["k1"], rr["k2"], rr["lambda"],
                      spec.get("rescore_topk", rr["topk"]), rerank=rr["enabled"] and not args.no_rerank,
                      rescore_vectors=g_emb2)
    t0 = time.time()
    sub_lines, cands = [], []
    for i, qid in enumerate(q_df.image_id):
        ranked, top, conf = gallery.search(q_emb[i], q_rescore=None if q_emb2 is None else q_emb2[i])
        sub_lines.append(",".join([qid] + [gallery.ids[j] for j in ranked]))
        cands.append((qid, gallery.ids[top], round(conf, 6)))
    t_search = time.time() - t0

    with open(out / "submission.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sub_lines) + "\n")
    all_c = pd.DataFrame(cands, columns=["query_id", "gallery_id", "confidence"])
    all_c.to_csv(out / "all_candidates.csv", index=False)
    answered = all_c[all_c.confidence >= threshold]
    answered.to_csv(out / "candidates.csv", index=False)

    n = len(q_df) + len(g_df)
    info = {
        "mode": mode, "decode": ex.decode, "flip": ex.flip, "rerank": gallery.rerank, "rerank_params": rr,
        "refusal_signal": "cos+gap", "refusal_threshold": threshold,
        "n_query": len(q_df), "n_gallery": len(g_df), "embedding_dim": int(ex.dim),
        "answered": int(len(answered)), "refused": int(len(q_df) - len(answered)),
        "two_stage": {"rescore_mode": rescore_mode, "rescore_topk": spec.get("rescore_topk")} if ex2 else None,
        "seconds": {"load": round(t_load, 2), "embed": round(t_embed, 2), "embed_stage2": round(t_embed2, 2),
                    "search": round(t_search, 2),
                    "total": round(time.time() - t_start, 2)},
        "ms_per_image_batched": round(1000 * t_embed / max(n, 1), 2),
        "device": str(ex.device), "gpu": torch.cuda.get_device_name(0) if ex.device.type == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"done -> {out}  | answered {info['answered']}/{len(q_df)} (threshold {threshold}) | "
          f"embed {t_embed:.1f}s ({info['ms_per_image_batched']} ms/image batched)"
          + (f", stage-2 embed {t_embed2:.1f}s" if ex2 else "")
          + f", search {t_search:.1f}s, TOTAL {info['seconds']['total']}s")

    if args.gt:
        import sys
        sys.path.insert(0, str(ROOT))
        from reid import evaluate as official
        query, gal = official.load_gt(args.gt)
        ranked = official.load_submission(out / "submission.csv", set(gal.index))
        rm = official.ranking_metrics(query, gal, ranked)
        cm = official.candidate_metrics(query, gal, official.load_candidates(out / "candidates.csv"))
        score = 0.7 * cm["F1"] + 0.3 * cm["TNR"] if cm["TNR"] == cm["TNR"] else float("nan")
        print(f"[official metrics] mAP@10 {100 * rm['mAP@10']:.2f} | Rank-1 {100 * rm['Rank-1']:.2f} | "
              f"Rank-5 {100 * rm['Rank-5']:.2f} | F1 {cm['F1']:.4f} | TNR {cm['TNR']:.4f} | "
              f"0.7*F1+0.3*TNR {score:.4f}")
        info["validation"] = {"ranking": rm, "candidates": cm, "candidate_score": score}
        (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
