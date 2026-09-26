# PLAN — Falcon ReID master checklist (single source of truth)

Deadline: **Tue 29 Sep 2026, 23:59 MSK**. Model freeze: **Sun 27 Sep, 20:00**. Links uploaded: **Mon 28 Sep evening**.
Tuesday = buffer only.
Rule: every patch updates this file. Status: ✅ done · 🔄 in progress · ⬜ todo · ❌ dropped (with reason).
Sources merged here: HANDOFF.md §11, the two outside reviews (26 Sep 01:17), the ТЗ, the official answers (#N), Leen's requests.

## Scoring reminder (decides every choice)
45% mAP@10 · 20% speed (≤40 ms incl. decode, ≥100 FPS, on RTX A5000) · 15% engineering (reproducibility, Docker, architecture, docs — the web service is optional, #41) · 10% refusal (0.7·F1+0.3·TNR) · 10% defense.
The model is chosen by **total points**, from measured numbers only.

---

## A. Accuracy (45%)
| # | Item | Status | Notes / evidence |
|---|---|---|---|
| A1 | Base+ViT ensemble | ✅ | 82.94 (s42) / 81.29 (s7), gate passed |
| A2 | ViT recipe: 30 epochs vs 20 vs LLRD | ✅ | **20 epochs kept.** s42 fast: e30 80.27 vs e20 79.14, but s7: e30 77.06 vs e20 **78.00** → e30 fails the gate. LLRD 78.97 (rejected) |
| A2b | Final ViT on all 1,541 cars | ✅ | `runs/final_vit_e20/vit.pth` (trained 26 Sep 02:27–03:07) |
| A3 | Camera-aware batches | ❌ | 77.23 vs 78.98 (worse) |
| A4 | GeM pooling | ❌ | 78.45 vs 78.98 (not better) |
| A5 | P=32 batches | ❌ | doesn't fit in 6 GB (shared-memory spill, 5–12× slower) |
| A6 | ConvNeXt-Large (as a teacher only) | ⬜ low | 1-epoch fit test first; only if distillation works and time remains |
| A7 | Letterbox input | ❌ | rectangular input already lost; low expected gain; no GPU time |
| A8 | **Label cleaning** (train only) | ⬜ | ~45% of the worst failures look like label errors. Manual review (Leen, no GPU). Also good for the defense |
| A9 | **VeRi + plate blurring** | ❌ | 26 Sep, real pipeline, seed 42: two-stage top-100 **82.02 vs 82.25 (−0.23)**; fast ViT in the same test 79.07 vs 79.36 (−0.29); fast mode via multisize_test s42 79.39 vs 79.14 (+0.25), s7 78.08 vs 78.00 (+0.08). Fails the +0.7 bar on seed 42 → seed-7 two-stage run not needed. **Rejected: VeRi does not help our cars.** Final ViT stays (defense line: tried 49k extra photos, measured, rejected) |
| A10 | **Two-stage** (fast ViT top-100 → ensemble re-orders) | ✅ | **passes the gate**: s42 82.25 vs 79.36 (+2.89), s7 80.29 vs 78.15 (+2.15). Full ensemble 82.85/80.86. In falcon/ as mode `two_stage` (patch21). Allowed #28, not timed #31 |
| A11 | Distillation ensemble → ViT (relational KD) | ❌ | fast mode s42 79.53 vs 79.14 (+0.39), s7 78.03 vs 78.00 (+0.03): below the +0.7 bar |
| A12 | Re-tune re-ranking k1/k2/λ/top-K | ❌ | best grid point (k1=4,k2=2,λ=0.5): s42 +0.90 but s7 −0.12 → fails the gate; defaults k1=6,k2=2,λ=0.3 stay |
| A13 | Gallery-side smoothing | ⬜ low | allowed #38; only if time |
| A14 | Synthetic data (VehicleX) | ❌ | no time |
| A15 | Plate-masking self-test (paint the plate area, check the mAP drop) | ✅ | experiments/plate_mask_test.py (patch26). Plates found and painted in 2,124 / 3,471 val photos (61.2%); check sheet: boxes only on plates. **Current fast ViT: seed 42 79.14 → 79.64 painted (+0.51), seed 7 78.00 → 77.71 (−0.29)** — both inside the ~0.7 seed noise → no measurable use of the plate zone. The 'original' numbers equal the known references (79.14 / 78.00), so the test matches the real pipeline. Evidence for #48 in README + slides |

**MODEL DECISION (26 Sep 16:05): FINAL = the current models (fast ViT 20 epochs = stage 1; Base + ViT with flip = stage 2). No retraining needed; only an optional rented-GPU bigger-batch test could still change this before the freeze.**

## B. Refusal (10%)
| # | Item | Status | Notes |
|---|---|---|---|
| B1–B3 | objective, 20% strangers, cos+gap signal | ✅ | |
| B3b | cos+gap in the real code | ✅ | falcon/search.py; validation 0.9569 (patch16) |
| B4 | **Re-tune the threshold for the FINAL pipeline** | ✅ | two_stage (Base flip+ViT flip): **0.809** from plateau overlap 0.779–0.839; score 0.9669 on both splits. **In falcon/config.json since patch28.** Re-run if any model or stage 2 changes |
| B5 | README justification (plateau on 2 splits) | ⬜ | |

## C. Speed (20%)
| # | Item | Status | Notes |
|---|---|---|---|
| C1 | Official-protocol benchmark on what we ship | ✅ | tools/speed_bench.py. Laptop: ViT 33.8 ms / 107 FPS (20/20), Base 50.8 ms (17.3), ensemble 85 ms (0) |
| C2 | Fast decoding (pil-draft / nvJPEG) | ⬜ low | ViT already under 40 ms; only if A5000 margin is thin. Accuracy check required |
| C3 | fp16 weights / channels_last / torch.compile | ⬜ low | same condition as C2 |
| C4 | Slim fp16 weights without classifier | ✅ | tools/export_weights.py (cosine ≥ 0.99999) |
| C5 | Choose the fast model by speed × accuracy | ✅ | **ViT (20 ep)**: s42 79.14 vs Base 78.98, s7 78.00 vs Base 76.04; 33.8 ms vs 50.8 ms |
| C6 | **Rent an RTX A5000** (real speed + real Docker GPU test) | ⬜ ⚠️ | **Moved BEFORE the Sunday freeze**: it is the only measurement that could still change the model choice (40 ms bar, whole-run limit ≈ latency×n×3, #40). Speed test = any A5000 pod; Docker GPU test needs a full VM. Pick driver 12.2 (R535) like the judges' (#32) |

## D. Docker / reproducibility (mandatory)
| # | Item | Status | Notes |
|---|---|---|---|
| D1 | extractor.py / extract() contract | ✅ | moderator 26 Sep: "a mistake, don't pay attention" → our `falcon.extract(image, bbox)` stays |
| D2 | CUDA-12 torch (not CUDA 13) | ✅ | Dockerfile: torch 2.14.0 from cu126 index + build-time assert |
| D3 | Pinned versions + sha256 weights | 🔄 | manifest + fetch_weights done; **GitHub Release with the final weights** todo |
| D4 | Offline run | ✅ | real Docker run with `--network none` on 26 Sep (I6) |
| D5 | Total run time within ~latency×n×3 | ✅ | PLUGGED IN, stage-2 variants (26 Sep): a) Base flip+ViT flip 96 s / 89 s vs limit ~192 / 186 s (≈50%) → **keep a)**. b) ViT reused −0.46/−0.35 (not needed), c) Base no-flip −0.19/**−1.51** (fails s7). Re-measure on the A5000 (C6) |
| D6 | Output format = example_submission.zip + passes evaluate.py | ✅ | checked 26 Sep |
| D7 | Clean-machine test | ⬜ | teammate, Monday |
| D8 | Install WSL + Docker Desktop | ✅ | 26 Sep: WSL + Ubuntu + Docker Desktop 4.92; GPU visible in a container (RTX 4050, driver 610 / CUDA 13.3) |
| D9 | Regenerate `submission/` from the final pipeline | ✅ | 26 Sep, commit 848c9fa (I2) |

## E. Docs
| # | Item | Status | Notes |
|---|---|---|---|
| E1 | **README.md (Russian + English summary)** | ⬜ ⚠️ | architecture, methods, one-command run, val metrics on both splits, threshold reasoning, all libraries/datasets with versions (ТЗ §12) |
| E2 | EXPERIMENTS.md (all experiments incl. rejected) | ⬜ | |
| E3 | Error analysis write-up with real examples | ⬜ | docs/error_analysis exists |
| E4 | Grad-CAM images | ✅ | docs/gradcam (10 images) |
| E5 | Limitations section | ⬜ | |
| E6 | Research sources list (Leen's docx + ours) | ⬜ | goes into the README references |

## F. Service — OPTIONAL (answers #41, #43, #44: tie-breaker only, NOT in the main 90%; engineering 15% = inference pipeline, Docker, docs)
| # | Item | Status | Notes |
|---|---|---|---|
| F1 | Backend (FastAPI) + pgvector + web client + Swagger + compose | ✅ written, 🔄 testing | patch18; service == batch 14/14 |
| F2 | Leen's Windows test (dev_server, CPU) | ⬜ | |
| F3 | Docker test of the full compose | ⬜ | after D8 |
| F4 | **Hosted prototype link** (ТЗ §13) | ⬜ ⚠️ | decide where to host; how long it must stay online is unanswered |
| F5 | 10^6-gallery ANN demo (HNSW) | ⬜ | tie-breaker |
| F6 | Grad-CAM in the UI | ⬜ low | tie-breaker |
| F7 | API_CONTRACT without plate_number | ✅ | patch18 |
| F8 | Service uses the SAME pipeline as the submission (two-stage) | ✅ | patch28: shared `falcon.search.order_candidates`; pgvector stores the stage-2 vector per row; tested in the sandbox with random weights: service == batch 10/10, mode-change guard works. Old gallery DBs must be reset once (`docker compose down -v`, or delete runs/pgdev) |
| D10 | Test the judges' driver 12.2 before Monday (Leen's own laptop if it has an NVIDIA GPU: driver 536.xx) | ⬜ | waiting for Leen's GPU check |

## G. Presentation (defense 10%)
| # | Item | Status | Notes |
|---|---|---|---|
| G1 | Read the template, plan slides 7–11 (strict) | ⬜ | template received 25 Sep, not opened yet |
| G2 | Slides: method + backbone choice, viewpoint/light, user journey, scaling, honest errors, model-choice-by-points table | ⬜ | |
| G3 | Demo video | ⬜ low | |

## H. Team, organizers, housekeeping
| # | Item | Status | Notes |
|---|---|---|---|
| H1 | Teammate: presentation in her own Claude chat (English content; template slides 7–11 keep their Russian labels), diagrams, UI screenshots, Monday clean-machine test | 🔄 | brief + prompt sent 26 Sep; all technical claims come from Leen/this chat; numbers = XX until the freeze |
| H2 | Organizer questions: upload contents, prototype online duration | 🔄 | sent 26 Sep; moderator passed them to the mentor |
| H3 | Repo cleanup: loose scripts into folders, old .patch files, stray data.py, queue scripts | ⬜ | |
| H4 | Back up the new weights (Drive) | ⬜ | after the final models exist |
| H5a | Deep research (competition methods, VeRi, decoding, recipe) | ✅ | 26 Sep: 2026 DINOv3 vehicle re-ID paper (256 px, one strong backbone + re-rank, big batch 512), AI City 2021 winners; only untested lever = bigger batch (C7) |

---

## I. Full audit 26 Sep 16:15 — every rule and every review point mapped to an item
Sources walked line by line: ТЗ §3–§13, dataset README, evaluate.py, all 54 official answers (original
sheet, read 26 Sep), Telegram (slides 7–11 rule, deadline), both outside reviews, Claude's review.
| # | Item | Source | Status | Notes |
|---|---|---|---|---|
| I1 | **Final ViT exported to weights/vit_infer.pth** | D3, #39 | ✅ | was MISSING until 26 Sep 16:12 (config pointed at a file that did not exist). sha256 ca781eacac1f6c39ae358378d3ab118d36b156c451f178c60f7e1038aa6506ed |
| I2 | Full final pipeline run on the public test (two-stage default) → time vs limit, answered count | D5, D9, #40 | ✅ | laptop 26 Sep: TOTAL 99.6 s vs limit ≈ 33.8 ms × 1860 × 3 ≈ 189 s (53%); answered 991/1110 at 0.809. The 3 files are in submission/ (commit 848c9fa) |
| I3 | .dockerignore whitelists ONLY manifest + the 2 inference files | #37 (all weight files in the solution dir count, cap 2 GB) | ✅ | patch30; before, a local build copied ~1.8 GB of training/validation weights into the image |
| I4 | GitHub Release weights-v1 (base_infer + vit_infer + LICENSE_DINOv3.md) + real sha256/bytes in manifest | #39, ТЗ §9 reproducibility | ✅ | release published 26 Sep; verified from a fresh clone on a clean machine: both downloads pass sha256, both stages load offline (768-d / 1792-d) |
| I5 | torch 2.14.0+cu126 wheel exists for cp311 linux | D2 | ✅ | checked on download.pytorch.org 26 Sep |
| I6 | Real Docker build from a fresh clone + offline run (`--network none`) + no OOM | ТЗ §6–§8, #39–#41 | ✅ | 26 Sep laptop (Docker Desktop, RTX 4050): build 657 s, all checks pass; offline GPU run TOTAL 98.1 s, answered 991/1110. vs laptop run: top-1 identical 100%, full top-10 rows identical 98.6% (the rest = near-ties swapped by fp16 math of a different CUDA build), embeddings min cosine 0.999999, same answered set → reproduced. Say this in the README |
| I7 | Speed on a weaker-than-judges machine (free Colab T4, 2 CPU threads, driver 580) + determinism | #30–#34, #31 | 🔄 | 26 Sep Colab, torch 2.14.0+cu126, released weights: **latency 36.6 ms → 10/10** even there; **determinism 0.0**; FPS 47 = CPU-bound (2 threads decode), not predictive for the judges' 128 threads. pil-draft: 32.3 ms / 48.8 FPS (draft only triggers for boxes ≥ 768 px, so little gain). → patch33: decode_workers = auto (one per CPU thread, max 16). Laptop re-check pending. Paid A5000 rental dropped (no budget) |
| I8 | Live build demo on request | ТЗ §6 «продемонстрировать процесс сборки … в реальном времени» | ⬜ | rehearse once; note build time |
| I9 | DINOv3 pretrained weights: exact source (timm/HF id + revision) and licence notice for our released fine-tuned weights | #39, #46 | ⬜ | README + release notes |
| I10 | README states: embeddings.npy = stage-1 ViT vectors, submission.csv = two-stage order (why they differ) + re-ranking described | #12, #28 | ⬜ | part of E1 |
| I11 | README lists ALL libraries/frameworks/datasets WITH versions; datasets = organizers' only (VeRi tried, rejected, not used); plate detector listed as analysis-only tool | ТЗ §7, §12; #46 | ⬜ | part of E1 |
| I12 | README language: English + a short Russian summary at the top (recommended) | judges are Russian | ⬜ | Leen decides |
| I13 | Threshold justification in README (plateau on 2 splits, 20% open-set re-weighting, 0.7·F1+0.3·TNR) | README dataset, #26, ТЗ §12 | ⬜ | = B5 |
| I14 | Error analysis on the held-out val split with real examples; back the "label errors" claim with example pairs (or drop the %) | ТЗ §9–§11, #50 | ⬜ | = E3 + A8 (time-boxed) |
| I15 | Remove HANDOFF.md (personal/informal) from the public repo before submission; PLAN.md → keep as dev log or move out | public repo | ⬜ | before links go up |
| I16 | Quarantine/remove forbidden reference code: experiments/posthoc.py "ALL QUERIES [FORBIDDEN]" re-rank | #40 (source code is checked) | ⬜ | cleanup (H3) |
| I17 | Commit docs/speed_log_falcon.csv (speed evidence); delete stray root data.py (check unused) + old .patch files | H3 | ⬜ | |
| I18 | Hosted prototype: password-protected (do not publish organizers' images openly); online through the expert review (30 Sep–14 Oct) and the finals; CPU host is enough (~1.3 s/query two-stage) | ТЗ §13 | ⬜ | = F4 |
| I19 | Presentation content check: no "~90% ceiling" framing; include VeRi rejection, plate test, model-choice-by-points table, honest errors | ТЗ §11 | ⬜ | teammate + Leen |
| I20 | Submission links (repo, presentation PDF, prototype, docs) uploaded Monday evening; each opened logged-out to check | ТЗ §13, Telegram | ⬜ | |
| I21 | Refresh RULES_CHECKLIST.md statuses and walk it line by line before upload | own rule | ⬜ | last step before upload |
| I22 | Rejected ideas documented with reasons (distillation, torch.compile, TensorRT, INT8, letterbox, ConvNeXt-L, VeRi, 320, re-rank tuning, GeM, camera-aware, soup, CosFace) | defense, reviews | ⬜ | = E2 |
| C7 | Decision: rent an A5000 for speed test (+ optional bigger-batch ViT P=32 gate, stop by Sun 14:00) | research 26 Sep | ⬜ | Leen decides; ~$1 |

## Order from 26 Sep 16:15 (one step at a time; tick items above as they finish)
**Saturday:** I2 final pipeline on the public test → C7 rental decision → (rental: I7 speed/driver/determinism, optional bigger-batch gate) → I4 weights release → I6 Docker build from a fresh clone, offline.
**Sunday:** (only if the bigger batch passed: final ViT retrain + threshold re-tune, then redo I2/I4) · E1 README (I9–I13) · E2/I22 experiments · E3/I14 error analysis · H3/I15–I17 cleanup · model freeze 20:00 at the latest.
**Monday:** D7 clean-machine test (teammate) · presentation PDF (G, I19) · hosted prototype (I18) · I21 rules walk-through · I20 upload links in the evening.
**Tuesday:** buffer only; fix what the clean-machine test found; nothing new after ~20:00.
