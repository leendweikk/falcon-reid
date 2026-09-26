# PLAN — Falcon ReID master checklist (single source of truth)

Deadline: **Tue 29 Sep 2026, 23:59 MSK**. Model freeze: **Sun 27 Sep, 20:00**. Links uploaded: **Mon 28 Sep evening**.
Tuesday = buffer only.
Rule: every patch updates this file. Status: ✅ done · 🔄 in progress · ⬜ todo · ❌ dropped (with reason).
Sources merged here: HANDOFF.md §11, the two outside reviews (26 Sep 01:17), the ТЗ, the official answers (#N), Leen's requests.

## Scoring reminder (decides every choice)
45% mAP@10 · 20% speed (≤40 ms incl. decode, ≥100 FPS, on RTX A5000) · 15% engineering · 10% refusal (0.7·F1+0.3·TNR) · 10% defense.
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
| A9 | **VeRi + plate blurring** | ⬜ | allowed #46/#47; blur for #48 safety. GPU: only if distillation is done by Sat evening. Must pass the gate |
| A10 | **Two-stage** (fast ViT top-K → ensemble re-scores) | 🔄 | experiments/stage_tests.py ready (patch20) | allowed #28, not timed #31; measure the gain + total run ≤ ~4 min (#40); declare in README |
| A11 | **Distillation** ensemble → ViT (feature/similarity level, not logits) | ⬜ | first GPU job Saturday after the ViT decision |
| A12 | Re-tune re-ranking k1/k2/λ/top-K for the final model | 🔄 | experiments/stage_tests.py (fixed 18-point grid) | cheap, no training; fixed grid, check both splits |
| A13 | Gallery-side smoothing | ⬜ low | allowed #38; only if time |
| A14 | Synthetic data (VehicleX) | ❌ | no time |
| A15 | Plate-masking self-test (paint the plate area, check the mAP drop) | ⬜ | evidence for #48; needed if VeRi is used, nice for defense anyway |

## B. Refusal (10%)
| # | Item | Status | Notes |
|---|---|---|---|
| B1–B3 | objective, 20% strangers, cos+gap signal | ✅ | |
| B3b | cos+gap in the real code | ✅ | falcon/search.py; validation 0.9569 (patch16) |
| B4 | **Re-tune the threshold for the FINAL fast model (ViT)** | ⬜ | threshold_study on both splits; update config |
| B5 | README justification (plateau on 2 splits) | ⬜ | |

## C. Speed (20%)
| # | Item | Status | Notes |
|---|---|---|---|
| C1 | Official-protocol benchmark on what we ship | ✅ | tools/speed_bench.py. Laptop: ViT 33.8 ms / 107 FPS (20/20), Base 50.8 ms (17.3), ensemble 85 ms (0) |
| C2 | Fast decoding (pil-draft / nvJPEG) | ⬜ low | ViT already under 40 ms; only if A5000 margin is thin. Accuracy check required |
| C3 | fp16 weights / channels_last / torch.compile | ⬜ low | same condition as C2 |
| C4 | Slim fp16 weights without classifier | ✅ | tools/export_weights.py (cosine ≥ 0.99999) |
| C5 | Choose the fast model by speed × accuracy | ✅ | **ViT (20 ep)**: s42 79.14 vs Base 78.98, s7 78.00 vs Base 76.04; 33.8 ms vs 50.8 ms |
| C6 | **Rent an RTX A5000** (real speed + real Docker GPU test) | ⬜ | Monday, after the Docker image works |

## D. Docker / reproducibility (mandatory)
| # | Item | Status | Notes |
|---|---|---|---|
| D1 | **extractor.py / extract() contract** | ⬜ ⚠️ | #43 mentions it; we have `falcon.extract(image, bbox)`. Check Telegram: was the question sent/answered? |
| D2 | CUDA-12 torch (not CUDA 13) | ✅ | Dockerfile: torch 2.14.0 from cu126 index + build-time assert |
| D3 | Pinned versions + sha256 weights | 🔄 | manifest + fetch_weights done; **GitHub Release with the final weights** todo |
| D4 | Offline run | 🔄 | tested without network in the sandbox; real test in Docker todo |
| D5 | Total run time within ~latency×n×3 | ⬜ | measure the full batch run in Docker |
| D6 | Output format = example_submission.zip + passes evaluate.py | ✅ | checked 26 Sep |
| D7 | Clean-machine test | ⬜ | teammate, Monday |
| D8 | **Install WSL + Docker Desktop** on the laptop | 🔄 | WSL + Ubuntu installed 26 Sep 09:50; Docker Desktop next |
| D9 | Regenerate `submission/` from the final pipeline | ⬜ | current files are from the old Base+Small |

## E. Docs
| # | Item | Status | Notes |
|---|---|---|---|
| E1 | **README.md (Russian + English summary)** | ⬜ ⚠️ | architecture, methods, one-command run, val metrics on both splits, threshold reasoning, all libraries/datasets with versions (ТЗ §12) |
| E2 | EXPERIMENTS.md (all experiments incl. rejected) | ⬜ | |
| E3 | Error analysis write-up with real examples | ⬜ | docs/error_analysis exists |
| E4 | Grad-CAM images | ✅ | docs/gradcam (10 images) |
| E5 | Limitations section | ⬜ | |
| E6 | Research sources list (Leen's docx + ours) | ⬜ | goes into the README references |

## F. Service (engineering 15% + tie-breakers)
| # | Item | Status | Notes |
|---|---|---|---|
| F1 | Backend (FastAPI) + pgvector + web client + Swagger + compose | ✅ written, 🔄 testing | patch18; service == batch 14/14 |
| F2 | Leen's Windows test (dev_server, CPU) | ⬜ | |
| F3 | Docker test of the full compose | ⬜ | after D8 |
| F4 | **Hosted prototype link** (ТЗ §13) | ⬜ ⚠️ | decide where to host; how long it must stay online is unanswered |
| F5 | 10^6-gallery ANN demo (HNSW) | ⬜ | tie-breaker |
| F6 | Grad-CAM in the UI | ⬜ low | tie-breaker |
| F7 | API_CONTRACT without plate_number | ✅ | patch18 |

## G. Presentation (defense 10%)
| # | Item | Status | Notes |
|---|---|---|---|
| G1 | Read the template, plan slides 7–11 (strict) | ⬜ | template received 25 Sep, not opened yet |
| G2 | Slides: method + backbone choice, viewpoint/light, user journey, scaling, honest errors, model-choice-by-points table | ⬜ | |
| G3 | Demo video | ⬜ low | |

## H. Team, organizers, housekeeping
| # | Item | Status | Notes |
|---|---|---|---|
| H1 | Teammate tasks: slides, UI polish, clean-machine test | ⬜ | agree on Saturday |
| H2 | Telegram: upload-button contents, prototype online duration, D1 | ⬜ | |
| H3 | Repo cleanup: loose scripts into folders, old .patch files, stray data.py, queue scripts | ⬜ | |
| H4 | Back up the new weights (Drive) | ⬜ | after the final models exist |
| H5 | Deep research promised on 25 Sep (competition methods, VeRi use, decoding) | 🔄 | CUDA part done; Leen's docx read 26 Sep; rest folded into A9–A11 decisions |

---

## Saturday 26 Sep: order
1. Night-queue results → ViT recipe decision (A2) → final ViT chosen.
2. Install WSL + Docker (D8), restart.
3. Threshold for the ViT (B4) · two-stage test (A10) · re-rank tuning (A12). All short, no training.
4. GPU: distillation (A11); then VeRi (A9) only if there's time before Sunday 20:00.
5. In parallel, no GPU: README (E1), label review (A8, Leen), Telegram questions (H2, D1), teammate plan (H1), template (G1).
