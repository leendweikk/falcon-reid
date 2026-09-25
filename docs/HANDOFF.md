# HANDOFF — Falcon ReID (ЛЦТ 2026, task 7 «ФАЛЬКОН.Tech»)

Written 25 Sep 2026, ~17:00 Amman time (UTC+3 = Moscow time), to continue the work in a fresh
conversation. Everything here is from files, logs and messages; items marked **(unverified)**
were not re-checked when this was written.

---

## 0. How to use this file (for the new assistant)

1. Read this whole file, then `docs/ORGANIZER_ANSWERS.md` and `docs/RULES_CHECKLIST.md` (both in the repo).
2. Ask Leen for anything marked **(unverified)** before relying on it.
3. Keep to "How Leen likes to work" (section 2). It matters as much as the technical content.
4. Never state a number that hasn't been measured. Test every script against the real conditions it
   will meet (old log formats, Windows paths, UTF-16 files written by PowerShell, and so on). Two recent bugs came
   from skipping this (see section 12).

---

## 1. The project in one paragraph

A vehicle **re-identification** service: given a photo of a car (full camera frame + bounding
box), rank the gallery photos of **the same physical car** taken by **other cameras**, without
using licence plates (plates are already blurred). It also has to **refuse** ("no match") when
the car isn't in the gallery. It is NOT classification and there is NO detector (YOLO etc.): the
organizers give the boxes. Deliverables: 3 files (`submission.csv`, `embeddings.npy`,
`candidates.csv`) produced by **one offline Docker command**, plus a microservice demo (backend,
vector DB, browser UI, OpenAPI, docker-compose), README/docs and a presentation.

**Deadline: Tue 29 Sep 2026, 23:59 Moscow time** (= 23:59 in Amman, same UTC+3). The upload button closes
at exactly midnight. The organizers advise uploading links early and updating the materials behind them.
Timeline after that: 30 Sep–14 Oct experts review ONLY the uploaded materials (no pitches) and pick the
top 10 → 15 Oct finalists announced → 16–20 Oct finalists improve their solutions → 23 Oct online pitch
→ 30 Oct awards in Moscow.

---

## 2. People and how Leen likes to work

- **Leen**: team lead, a student at 42 (C curriculum), limited ML background, very motivated
  ("aim for the BEST, fix every negative, don't just say sorry"). Works on her brother's laptop.
- A **teammate** in another country: frontend + slides (template slides 7–11), later the clean-machine test.
- Her working style:
  - **step by step, one small step per reply**; wait for her screenshot or result before the next step
  - she often asks to "regive" or "resend" the **full code of a file** rather than edits
  - **simple language**, analogies, narrative explanations, recaps before moving on
  - she wants honest critique; no endless apologizing; own mistakes briefly and fix them
  - check every idea against the rules (`RULES_CHECKLIST.md`) BEFORE trying it; she was upset when rules
    were discovered mid-way before
  - she is fine with long GPU runs and doesn't want ideas dropped because of time ("try everything worth
    trying; I'll handle the deadline"). Still order the work so the mandatory and risky items are
    handled early.
- Code delivery method: the assistant's sandbox **cannot push** to GitHub (403). The assistant commits
  in its sandbox, exports `git format-patch -1 --stdout > patchN.patch`, sends the file; Leen runs
  `Move-Item $HOME\Downloads\patchN.patch C:\falcon\` → `git am patchN.patch` → `git push`.
  Patches must be applied **in order**. If a download gets renamed ("patch13 (1).patch"), adjust the name.
  Commits by the assistant use the identity Claude <noreply@anthropic.com> plus the session trailer lines.
- Old `.patch` files lie untracked in `C:\falcon` and should be deleted at some point.

---

## 3. Machine and environment

- Laptop (her brother's): **RTX 4050 Laptop 6 GB**, 16 GB RAM, Windows 11, 12 CPU threads.
- **Smart App Control** switched itself ON on 24 Sep and blocked pandas DLLs ("An Application Control policy has
  blocked this file"). On 25 Sep pandas/torch imported fine again (`3.0.6 2.14.0+cu132 True`), so it
  appears solved **(unverified: whether it was turned off)**.
- Repo: `C:\falcon`, GitHub `https://github.com/leendweikk/falcon-reid` (public or private: **unverified**;
  must be open by submission, ТЗ §13).
- venv: `C:\falcon\.venv`, Python 3.11. Activate in a new terminal:
  `(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& C:\falcon\.venv\Scripts\Activate.ps1)`
  The prompt must show `(.venv) PS C:\falcon>`.
- Key versions (`requirements.txt`, pinned, pushed, ASCII-encoded, `+cu132` suffix removed):
  torch 2.14.0 (+cu132 locally), torchvision 0.29.0, timm 1.0.30, numpy 2.4.6, pandas 3.0.6, pillow 12.3.0,
  scikit-learn 1.9.1, scipy 1.17.1, tabulate 0.10.0, huggingface_hub 1.32.0, safetensors 0.8.0.
- Opening a terminal: VS Code menu Terminal → New Terminal (or Ctrl+Shift+`), or Windows key → "PowerShell".
- Long jobs run in their **own PowerShell window** so they can't be closed by accident:
  `Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd C:\falcon; .\.venv\Scripts\Activate.ps1; powershell -ExecutionPolicy Bypass -File queue.ps1'`
- PowerShell 5 `Tee-Object` writes **UTF-16** files (the queue logs). Read them accordingly.
- Backups on Google Drive: folder `falcon backup/final` (weights/base.pth, small.pth) and `falcon backup/validation`
  (validation models). **Unverified** whether these were uploaded: `runs/split7/*.pth`, `runs/val_vit/best_ema.pth`,
  the crops zip, the logs.
- Leen also has a laptop with a GTX 1650 (4 GB), and is open to renting a cloud GPU (for example an RTX A5000 on
  RunPod to measure real speed on the organizers' GPU model).

---

## 4. Data

- Organizers' dataset: `train.csv` (9,556 rows; columns image_id,x,y,w,h,vehicle_id,camera_id; 1,541 cars),
  `test_query.csv` (1,110 rows) + `test_gallery.csv` (750 rows) = the **public** test (every query has a pair).
  Full frames in `C:\falcon\data\images`. The closed test has the same structure, with **20% of queries having
  no pair** (answer #17). One frame = one vehicle = one image_id (#1, #27).
- `make_crops.py`: crops each box, shrinks the longer side to 384 (bicubic, only if larger) → `data/crops/<id>.jpg`.
- Validation splits (`make_val_split.py <seed>`): 300 cars held out, 50 of them "strangers" (queries only).
  - seed 42 → `data/splits/` (train_split 1,241 cars / 7,661 photos; val 1,240 queries of which 308 = 24.8%
    have no valid pair). All comparisons use it.
  - seed 7 → `data/splits_seed7/` (7,718 train photos; 1,217 queries, 313 = 25.7% no pair). The honest second check.
  - Files: `val_query.csv`, `val_gallery.csv`, `val_gt.csv` (image_id, vehicle_id, camera_id, split), `train_split.csv`.
- Seed-to-seed noise ≈ **0.7 mAP**. A **gate** means: a change is kept only if it wins by ≥ 0.7 on seed 42 AND
  still wins on seed 7, in the **real pipeline** (re-ranking included). The quick training metric can mislead
  (the 320 story, section 8).
- Error analysis (earlier): about 45% of the worst failures are **label errors in the dataset**; darkest quarter of
  photos 72.5–74% vs the average; front vs rear views; two cars in one box. Top-5 wrong guesses are sensible lookalikes.
- VeRi: Leen **has it downloaded**; `experiments/prepare_veri.py` exists. Earlier rejected out of caution (visible
  plates), but **answers #46/#47 now allow it**.

---

## 5. Current model recipe (all in the repo)

- `reid/model.py` — `ReIDModel(num_classes, backbone, head="linear", pretrained=True, pool="avg"|"gem")`:
  timm backbone (`num_classes=0`, gradient checkpointing ON) → BNNeck (BatchNorm1d, frozen bias) → linear
  classifier. Eval mode returns `feat_bn`. GeM = learnable p (init 3) on `forward_features` + timm's head LayerNorm
  (ConvNeXt only). `load_reid(path, backbone, device)` → (model, input_size): reads an optional `input_size` key,
  detects GeM via `gem_p`, builds with `pretrained=False` (offline).
- Backbones (timm): `convnext_base.dinov3_lvd1689m` ("base"), `convnext_small.dinov3_lvd1689m` ("small"),
  `vit_base_patch16_dinov3.lvd1689m` ("vit", 86M params, 768-d).
- `reid/losses.py` — ID cross-entropy (label smoothing 0.1) + batch-hard **soft-margin** triplet (`margin=None`).
- `reid/data.py` — `INPUT_HW=(256,256)`, `make_transforms(hw)`; train: Resize, HFlip, Pad 10, RandomCrop,
  ColorJitter(brightness 0.2, contrast 0.15; no hue), ToTensor, Normalize(ImageNet), RandomErasing 0.5.
  `STRONG_LIGHT_AUG=False` (rejected). `TrainSet(csvs, transform)` stores the transform ON the dataset (Windows
  worker processes) and `cams` (camera_id). `PKSampler(P=16, K=4, camera_aware=False)`: camera-aware = round-robin
  over cameras (allowed by answer #2: camera_id only for sampling/validation, never a model input).
- `reid/optim.py` — `param_groups(model, lr_backbone, lr_head, llrd)`: 2 groups, or layer-wise LR decay for ViT.
- Training: AdamW, lr 1e-4 backbone / 1e-3 head (ViT: 5e-5 backbone), wd 1e-4, fp16 autocast + GradScaler,
  30-epoch schedule (3 warm-up + cosine), EMA decay 0.998.
  - `train.py` (validation run on seed 42, evaluates at epochs 1, 5, 10, …; saves best_normal/best_ema/last):
    options `--model --run-name --size --lr-backbone --epochs --stop-epoch (20) --pool --P --K --camera-aware --llrd`.
  - `train_final.py` (no evaluation; saves the EMA at `--stop-epoch` (15) with an `input_size` key):
    `--model --train-csv --out --seed --size --lr-backbone --epochs --stop-epoch --pool --P --K --camera-aware --llrd`.
    Final models = all of train.csv → `weights/{name}.pth`.
- `reid/rerank.py` — k-reciprocal re-ranking (k1=6, k2=2, λ=0.3); `re_ranking_streaming(q, g, topk=100)`
  processes **one query at a time** against the static gallery (allowed: answers #28/#38 and a Telegram answer on 23.09).
- `run_inference.py` — `--images --query --gallery --out`, `--mode fast|accurate` (default fast), `--threshold 0.72`,
  `--topk 100`, `--k1 6 --k2 2`, `--no-rerank`, `--base-weights weights/base.pth`, `--small-weights weights/small.pth`.
  Crop by box → longer side 384 → test transform. fast = Base, no flip. accurate = Base 0.6 + Small 0.4 (√w-scaled,
  glued) with flip. Writes embeddings.npy, submission.csv (no header, query + 10), candidates.csv (header; rows only
  above threshold), all_candidates.csv (top-1 for every query, for tuning).
  **TODO:** switch to the new refusal rule (section 9) and to the extract() contract.
- `reid/evaluate.py` = the organizers' official script, **unmodified**.

### Weights on the laptop
| File | What |
|---|---|
| `weights/base.pth`, `weights/small.pth` | FINAL models (all 1,541 cars, EMA at epoch 15) |
| `runs/convnext_b_v6_ema/best_ema.pth` | validation Base (seed 42) |
| `runs/convnext_s_v2_ema/best_ema.pth` | validation Small (seed 42) |
| `runs/val_vit/best_ema.pth` | validation ViT (seed 42, lr 5e-5, EMA at epoch 20) |
| `runs/split7/base.pth`, `small.pth` | seed-7 models (train_final, EMA at 15) |
| `runs/split7_vit/vit.pth` | seed-7 ViT (produced by the queue) |
| `runs/val_base_320/best_ema.pth` | rejected 320 experiment |
Pipeline output folders: `runs/speed_base_noflip` (seed-42 fast mode), `runs/split7/fast` (seed-7 fast mode),
`runs/split7/pipeline` (seed-7 accurate), `submission/` (v2 files on the public test, pushed).

---

## 6. Scoring and rules (details in docs/ORGANIZER_ANSWERS.md)

**45%** mAP@10 from submission.csv · **20%** speed (10% latency batch 1, full cycle: ≤40 ms full, 40–80 linear,
>80 zero; 10% throughput: best FPS of batch 1/8/16/32, ≥100 full, 50–100 linear, <50 zero; measured on
**RTX A5000 24 GB, CUDA driver 12.2**; weights > 2 GB → no speed score) · **15%** engineering quality
(reproducibility, architecture, Docker, stack per ТЗ §6, docs) · **10%** refusal = 0.7·F1 + 0.3·TNR ·
**10%** defense (pitch). Tie-breakers (0 points): web UI (upload, ranking, confidence, **export**), Grad-CAM,
**ANN search (FAISS/HNSW) for ~10^6 gallery**, deep error analysis.

Allowed: camera_id for sampling/validation only (#2); re-ranking inside one query's top-K incl. a heavier model
(#28), not timed (#31); the gallery is a static base that can be pre-built (#38); single-image TTA (#38); any
public weights/datasets with source + version + checksum (#46, #47; VeRi OK); FP16/ONNX/TensorRT (#36).
Forbidden: query expansion, clustering over queries, anything using other queries (#38, checked in the code #40);
test labels; plate features (finalists are re-tested with plates painted over, #48); network at run time (#39).
Docker build may use the network; pinned versions; weights in the repo or downloaded in the Dockerfile with sha256 (#39).
Batch run: images folder + CSVs in, 3 files out; time limit ≈ latency × n × 3 (~4 min) (#40).
ТЗ §6 requires: microservices (inference backend / DB / client), Python backend, a DB for embeddings (pgvector/FAISS/Milvus),
OpenAPI/Swagger, a browser thin client, Dockerfile + docker-compose one command. ТЗ §12 docs: architecture,
methods, build steps, val metrics, **threshold justification**, all libraries/datasets **with versions**.
ТЗ §11 presentation: method + backbone choice, viewpoint/lighting handling, user-journey diagram, scaling plan,
honest error analysis; **slides 7–11 in the strict template** (slide 10: delete unused team cards).
ТЗ §13 submission links: open repo, presentation, **working prototype**, docs.

---

## 7. Results history (mAP@10, %)

34 (old model) → 38.3 (frozen backbone) → 65.8 (Small fine-tuned) → 68.7 (Base) → 70.2 (soft-margin triplet)
→ 75.5 (Base + EMA, quick metric 0.7554 at epoch 15) → **80.4** (accurate: Base+Small + flip + streaming re-rank, seed 42).
Honest seed-7: accurate ≈ 77.0, fast ≈ 76.3 (76.04 in the multisize run).
Fast mode (Base, no flip, re-rank): **78.98** (seed 42), cosine only 75.05. Speed (earlier benchmark, RTX 4050):
fast 38 ms/car GPU, 218 ms CPU; accurate 130 ms GPU, 546 ms CPU (model only; NOT the official protocol).

Quick training metric by epoch (EMA): Base 256: 1:0.4077, 5:0.588, 10:0.7354, **15:0.7554**, 20:0.7452,
25:0.7284, 30:0.7181. Small EMA at 15: 0.7306.

---

## 8. Experiments done (keep for EXPERIMENTS.md)

| Experiment | Result | Decision |
|---|---|---|
| Strong light/night augmentation | worse | rejected (`STRONG_LIGHT_AUG=False`) |
| VeRi pre-training (earlier) | not run; rejected for visible plates | **reopened** (answers #46/#47) |
| Rectangular input 224×288 | worse | rejected |
| Model soup | worse | rejected |
| CosFace head | overfits | rejected |
| Colour test | — | numbers in earlier logs **(unverified)** |
| All-queries re-ranking | forbidden | replaced by streaming re-ranking |
| Multi-size TTA (no training) | seed 42: best +0.56 (224+256+320), 288 −0.04, 320 −1.11; seed 7: 288 +1.21, 320 +1.34, 256+288+320 +1.59 | rejected: no combo ≥ +0.7 on both splits |
| **320 input (trained)** | quick EMA ep15 0.7627 (bar 0.7624); real pipeline seed 42: 78.25 @320, 78.52 @352, 77.30 @288 vs 78.98 | **rejected** (worse with re-ranking, ~1.6× slower) |
| **DINOv3 ViT-B** (lr 5e-5, EMA ep20) | quick: normal 5:0.6741 10:0.7306 15:0.7443 20:0.7537; EMA 5:0.4098 10:0.6757 15:0.7431 **20:0.7601** (still rising) | kept for testing |
| ViT as fast model (no flip + re-rank) | **79.14** vs Base 78.98 | tie on accuracy; speed via official benchmark |
| **Ensemble test** seed 42 (flip + re-rank) | Base 79.28 · ViT 80.37 · Base0.6+Small0.4 80.40 · **Base0.5+ViT0.5 82.94 (+2.54)** · Base0.6+ViT0.4 82.28 · Base0.4+Small0.3+ViT0.3 82.87 · Base0.5+Small0.25+ViT0.25 81.67 | passes on seed 42 |
| **Ensemble test seed 7** (flip + re-rank) | Base 77.15 · ViT 77.28 · Base0.6+Small0.4 76.93 · **Base0.5+ViT0.5 81.29 (+4.36)** · Base0.6+ViT0.4 80.21 · Base0.4+Small0.3+ViT0.3 80.61 · Base0.5+Small0.25+ViT0.25 79.56 | **GATE PASSED on both splits → Base 0.5 + ViT 0.5 is the new accurate ensemble** (Small drops out) |
| **Refusal study** (0.7·F1+0.3·TNR, 20% strangers) | seed 42: cosine@0.72 = 0.932, cosine re-tuned (0.78) 0.948, **cos+gap 0.957** (plateau 0.830–0.890); seed 7: cosine@0.72 0.942, cosine re-tuned 0.957, **cos+gap 0.962** (plateau 0.798–0.889); PR-AUC best for cos+gap on both | **adopted: cos+gap, threshold 0.86** for the current fast Base model |

"gap" = top-1 cosine minus the best cosine among the OTHER gallery items (one query only). "cos+gap" = the sum.
The threshold must be re-studied whenever the fast model or pipeline changes (`experiments/threshold_study.py`).

Grad-CAM (`gradcam.py`, validation Base): it looks at the car, not the background; plates stay cold; the Chery match
(rear → front) used an orange side sticker (instance detail); the Tiguan false match relied on the VW logo (brand, not
identity); the Transporter matched grey vs black (colour ignored). Wrong matches at sim 0.63 fall below the threshold
(refused); the Tiguan at 0.73 passes. `docs/gradcam/` images: committed **(unverified)**.

---

## 9. The experiment queue (running now)

`queue.ps1` (resumable: jobs whose log says finished are skipped; logs in `runs/queue_logs/<name>.txt`, UTF-16):
1. `eval_vit_fast` ✅ (79.14)
2. `vit_s7` ✅ seed-7 ViT → `runs/split7_vit/vit.pth` (finished 16:57)
3. `eval_ens_s7` ✅ Base0.5+ViT0.5 = 81.29 vs Base+Small 76.93 (+4.36) → ViT gate passed on both splits
4. `vit_e30` ✅ ViT, full 30 epochs (seed 42): quick EMA 0.7686 at ep30 (vs 0.7601 for the ViT EMA at ep20); real-pipeline check = `eval_vit_e30` / `eval_ens_e30`
5. `vit_llrd` ✅ ViT, lr 1e-4 + layer-wise decay 0.75, 30 epochs: quick EMA 0.7586 (not better than vit_e30)
6. `base_cam` ✅ Base + camera-aware batches (A3): quick EMA 0.7463 (worse than 0.7554)
7. `base_gem` ✅ Base + GeM (A4): quick EMA 0.7516 at ep15, 0.7383 at ep20 (not better)
8. `base_p32` ⛔ stopped: batch 128 does not fit in 6 GB (5.7/6.0 GB dedicated + 2.6 GB shared memory; 493–1116 s/epoch vs 88 s). Log kept as runs/queue_logs/base_p32_stopped.txt
9–15. real-pipeline evals: `eval_base_cam/gem/p32` (multisize_test --sizes 256; compare with 78.98),
   `eval_vit_e30/llrd` (compare with 79.14), `eval_ens_e30/llrd` (compare with 82.94)
Outputs: `runs/multisize_splits_<run>.csv`, `runs/ensemble_<splits>_<vitdir>.csv`.

---

## 10. Scripts added recently

- `experiments/multisize_test.py --weights --splits --backbone --sizes` (real pipeline: cosine and re-rank mAP@10)
- `experiments/ensemble_test.py --splits --base --small --vit` (accurate mode, fixed weight grid)
- `experiments/threshold_study.py <run dir> --splits` (refusal signals and thresholds, self-checks against official code)
- `experiments/official_bench.py --model --weights --decode pil|pil-draft|gpu --workers --channels-last` — copies the
  official speed protocol and prints the estimated speed points; appends to `docs/speed_log.csv`. **Run only when the
  GPU is idle.** In the sandbox, decoding one 1080p JPEG took ~33 ms → decoding is the likely bottleneck → try nvJPEG (gpu).
- `experiments/collect_runs.py` → `docs/experiments_raw.md` (all logs and CSVs in one file, for EXPERIMENTS.md).
  Patch 13 (robust to old log formats) was **sent but not yet applied** when this was written.
- `docs/ORGANIZER_ANSWERS.md`, `docs/RULES_CHECKLIST.md` (rewritten with answer numbers), `docs/API_CONTRACT.md` (old).

---

## 11. Master plan (nothing dropped; the gates decide)

**A. Accuracy (45%)**
A1 ViT ensemble ✅ passed on both splits (+2.54 / +4.36); the final ViT on all data (train_final --model vit --lr-backbone 5e-5 --stop-epoch 20, or the better A2 recipe) is still TODO · A2 ViT recipe (30 epochs, LLRD) in the queue · A3 camera-aware batches
in the queue · A4 GeM in the queue · A5 P=32 in the queue · A6 ConvNeXt-Large (1-epoch fit test in 6 GB first; ensemble
member or teacher only) · A7 letterbox input (keep the aspect ratio, grey padding) · A8 label cleaning (flag suspicious training
labels, check by eye) · A9 **VeRi / external data** (joint training; ~5× longer; list in README; check Grad-CAM for plate
attention because of #48) · A10 **two-stage search**: the fast model builds embeddings + top-K, the heavy ensemble re-scores
the top-50 (allowed #28, not timed #31; must fit the ~4 min total #40; declare in README) · A11 **distillation** of the
ensemble into the fast model · A12 re-tune the re-ranking (k1, k2, λ, top-K) after the models change · A13 gallery-side
smoothing (allowed via #38) · A14 synthetic data (VehicleX; low priority).
After each round: confirm winners on seed 7 → combine winners → final retrains on all data → **freeze**.

**B. Refusal (10%)** B1 objective 0.7·F1 + 0.3·TNR ✅ · B2 20% strangers ✅ · B3 cos+gap signal ✅ adopted (0.86 for the fast
Base) · B4 README justification (plateau overlap on 2 splits) · re-run for the final pipeline.

**C. Speed (20%)** C1 official benchmark (script ready; run when the GPU is idle) · C2 fast decoding (PIL draft / nvJPEG)
— check accuracy on validation before adopting · C3 fp16 + channels_last, torch.compile / TensorRT / ONNX · C4 inference
weights without the classifier head, saved in fp16 · C5 choose the fast model by speed × accuracy (Base vs ViT) ·
C6 rent an RTX A5000 for real numbers.

**D. Docker / reproducibility (mandatory)** D1 the **extractor.py / extract() contract** (mentioned in #43, not found in the
dataset archive, ТЗ, README or example_submission.zip → question drafted for the moderator; **unverified** whether sent) ·
D2 **torch build compatible with CUDA driver 12.2** (a cu132 build will NOT run there → speed score 0) · D3 pinned versions
+ sha256 for our weights and the DINOv3 weights · D4 offline test (`--network none`) · D5 total run time within the limit ·
D6 outputs match example_submission.zip (checked: our formats already match) and pass evaluate.py · D7 clean-machine test.

**E. Docs** README (architecture, methods, one-command build, val metrics on both splits, threshold reasoning, re-ranking
and two-stage description, all libraries + datasets + versions, fast vs accurate) · EXPERIMENTS.md (from
docs/experiments_raw.md) · error analysis · Grad-CAM · limitations · RULES_CHECKLIST · ORGANIZER_ANSWERS.

**F. Service (engineering 15% + tie-breakers)** FastAPI inference backend + vector DB (pgvector or FAISS) + browser UI,
OpenAPI/Swagger, docker-compose one command, offline; input = image **+ bbox** (#45); UI: upload, ranking view, confidence,
**export**; ANN (FAISS/HNSW) demo at ~10^6 gallery; a **hosted prototype link** (ТЗ §13; how long it must stay online is
unanswered). Fix `API_CONTRACT.md`: remove `plate_number` (bad optics under the no-plate rule), make the bbox required,
update the threshold example (it says 0.60).

**G. Presentation** PDF/PPTX per ТЗ §11; slides 7–11 in the strict template (Telegram 23.09); demo video.

Suggested order: finish the queue (ViT decided: kept) → official speed benchmark (C1) → D1/D2 early (risky) →
A6–A11 → final retrains → freeze → pipeline + Docker → service → docs → presentation → clean-machine test →
upload the links early.

---

## 12. Recent mistakes to avoid repeating

- Patch 9 declared `--lr-backbone` twice in `train_final.py` (argparse crash); fixed in patch 10.
- `collect_runs.py` v1 assumed every log had the new columns; crashed on old logs; fixed in patch 13.
- A claim that the ViT was faster than Base came from unequal conditions (warm-up); retracted.
- The service's weight in the score was misstated as "only a tie-breaker"; ТЗ §6/§9 make it part of engineering quality.
- Always compare in the real pipeline (re-ranking), on both splits.

---

## 13. Open questions / waiting on

- Moderator: the extractor.py contract (drafted; sending **unverified**). Earlier questions about camera_id and gallery
  processing are already answered by #2 and #38.
- Telegram: what exactly the upload button should contain (a mentor is checking); how long the prototype must stay online.
- Whether the GitHub repo is public.
- Drive backups for runs/split7, runs/val_vit, the crops, the logs.
