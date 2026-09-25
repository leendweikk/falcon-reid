# Organizers' official answers — summary

Source: the official Q&A table «Вопросы и ответы Бизнес 7. Фалькон Тех» (sheet «Объединённые вопросы участников»),
plus the ТЗ (technical brief). Numbers (#) are the question numbers in that table. We quote them in
`RULES_CHECKLIST.md`, `EXPERIMENTS.md` and the README so every design choice points to an official answer.

## Scoring (ТЗ §9 + #10, #15, #34)
| Part | Weight | How it is measured |
|---|---|---|
| Accuracy | 45% | **mAP@10 from `submission.csv`** (top-10, after junk filtering); AP normalized by min(n_pos, 10); macro over queries |
| Performance | 20% | 10% latency (batch 1, full cycle: ≤40 ms full, 40–80 ms linear, >80 ms zero) + 10% throughput (best of batch 1/8/16/32: ≥100 FPS full, 50–100 linear, <50 zero). Weights > 2 GB → no performance score |
| Engineering quality | 15% | reproducibility from code, clean architecture, Docker quality, stack compliance (ТЗ §6), documentation completeness |
| Refusal ("candidates") | 10% | **0.7 × F1 + 0.3 × TNR** (micro, per query, top-1 only) |
| Defense | 10% | pitch, architecture reasoning, error analysis, threshold defense |
| Tie-breakers (ТЗ §10) | 0% | web UI (upload, ranking view, confidence, **export**), Grad-CAM/attention maps, ANN search (FAISS/HNSW) at ~10^6 gallery, deep error analysis |

Reference only (no points): full-ranking mAP and mINP from `embeddings.npy`; PR-AUC from `candidates.csv` (#16).

## Evaluation protocol
- Junk: gallery items with the **same vehicle_id AND same camera_id** as the query are removed **before** cutting to top-10; same camera + other vehicle stays (hard negatives) (#11).
- Queries with no valid positive are **excluded** from mAP/Rank (not AP = 0) and scored only in refusal mode (#11, #13, #22).
- Closed test: **20% of queries have no pair** in the gallery, unmarked; the other 80% have ≥1 cross-camera positive. The public test has pairs for every query → calibrate the threshold on our own split with ~20% open-set (#17).
- Closed test has the same structure/pipeline as the public test (#6). Public test: 1110 queries + 750 gallery (#27).

## Output files (#18, #20, #21, #27, #29, О-2)
- `submission.csv`: always 10 candidates for every query, no header; refusal is NOT expressed here.
- `candidates.csv`: header `query_id,gallery_id,confidence`; refusal = **no row** for that query; any number of rows allowed, only the top-confidence one counts; confidence = any monotone score.
- `embeddings.npy`: float32, shape (n_query + n_gallery, D), queries then gallery in CSV order; L2 normalization not required. Used for reference metrics and to check the submission comes from a real model (#12).
- The threshold must be stated and justified in the README (#26).

## What is allowed
- `camera_id` in training: **only** for PK-sampling and cross-camera validation; never as model input (#2).
- Re-ranking inside the top-K of **one** query against the gallery (k-reciprocal, a heavier pairwise/local model on the top-K, anything not using other queries) (#28, #38). The gallery is a static base that can be pre-built (#38). Re-ranking must be described in the solution (#12).
- Single-image TTA (flips, crops) (#38).
- Any **public** pretrained weights/datasets with a clear source (URL, version/tag, author), any licence incl. CC BY-NC, gated access, DINOv2/v3 (#46). External re-ID datasets with readable plates are allowed, no masking needed (#47).
- FP16/BF16, ONNX Runtime, TensorRT, TorchScript (#36).
- Any stack for the demo service/UI (#43); the demo takes image **+ bbox**, no detection (#45).

## What is forbidden
- Query expansion, clustering over test queries, anything using other queries (#38) — also checked in the source code (#40).
- Using closed-test labels in any form, tuning on organizers' answers (#38).
- Plate features: finalists are re-tested with plates painted over; a clear drop is questioned (#48, ТЗ §9).
- Network access at run time; everything must be inside the image (#39).

## Hardware and running (#30, #31, #32, #39, #40, #41)
- Test machine: NVIDIA **RTX A5000 24 GB** (Ampere SM86), **CUDA driver 12.2**, 2× Xeon Gold 6338 (128 logical CPUs); GPU in Docker via `--gpus all`; measured on GPU.
- Timed cycle = everything inside `extract()`: read file, decode, crop, preprocess, forward, postprocess, L2 norm. Latency: median of 300 runs after 50 warm-up, CUDA-synced. Throughput: ≥10 s per batch size. Also recorded: peak VRAM, weight load time, total weight size, determinism (two runs).
- Weight size = all weight files used at inference (.pt .pth .bin .onnx .engine .safetensors …) (#37).
- Batch run: they pass the images folder + CSVs and expect the 3 files; time limit ≈ latency_b1 × n_test × 3 (~4 min now) (#40).
- `docker build` may use the network; pinned versions (`==`); weights in the repo or downloaded in the Dockerfile **with sha256 check**; pretrained weights pinned by version with checksum (#39).
- Mandatory minimum: one command producing the 3 files. The full service (DB + web UI) must also start with one command (`docker-compose up`) offline (#41) and is part of ТЗ §6 (engineering quality).

## Still unknown
- The `extractor.py` contract mentioned in #43 (not in the dataset README, ТЗ or example_submission.zip) — asked.
- How long the demo prototype must stay online (Telegram, unanswered).
