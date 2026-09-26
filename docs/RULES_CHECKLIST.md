# Rules checklist — ФАЛЬКОН.Tech (ЛЦТ 2026, task 7)

Sources: the ТЗ, the dataset README, the official `evaluate.py`, the official Q&A table (answers cited as #N,
summarized in [`ORGANIZER_ANSWERS.md`](ORGANIZER_ANSWERS.md)), Telegram announcements.
Every idea was checked here before it was tried. Status as of 26 Sep 2026.

## Hard rules (violation = disqualification)
| Rule | Status | How we comply / evidence |
|---|---|---|
| No licence-plate features (ТЗ §9, #48) | ✅ | Plate-masking self-test: 79.14 → 79.64 (split 42), 78.00 → 77.71 (split 7) with plates painted over — no measurable drop; Grad-CAM: plate zone cold (`docs/gradcam/`) |
| Reproducible from the submitted code (ТЗ §9) | ✅ | Docker build from a fresh clone + offline run reproduces our output (same top-1 for 100% of queries, same refusals); weights in a public release with sha256; training commands in the README; pinned versions |
| No test labels used (#38) | ✅ | we never had them; no tuning on the public test |

## Inference rules
| Rule | Status | How we comply |
|---|---|---|
| Each query processed alone, stream protocol (#38, #40) | ✅ | `falcon/search.py`: only the current query + the static gallery |
| Re-ranking / heavier model inside one query's top-K (#28, #38) | ✅ | stage 2 re-orders the fast top-100; k-reciprocal re-ranking per query; described in the README (#12) |
| No query expansion / clustering over queries (#38) | ✅ | never used; the forbidden all-queries reference script was removed from the repo |
| Single-image TTA (#38) | ✅ | horizontal flip of the same image (stage 2 only) |
| Offline run, all weights inside the image (#39) | ✅ | tested with `docker run --network none` |
| Weights ≤ 2 GB incl. ensembles (#37) | ✅ | 330 MB (two fp16 files); `.dockerignore` keeps any other weight file out of the image |
| No OOM on the test set (ТЗ §7) | ✅ | full public test in Docker; peak VRAM of the timed model 726 MB |
| Whole run within ≈ latency × n × 3 (#40) | ✅ | 86.5 s of ≈ 152 s on the laptop |
| Docker GPU with CUDA driver 12.2 (#30) | ✅ (by design) | torch 2.14.0 CUDA 12.6 build, asserted at build time (CUDA 12.x minor-version compatibility); tested on drivers of CUDA 13.x (laptop, Colab) — the exact 12.2 driver was not available to us |
| Timed `extract()` = full cycle (#31) | ✅ | `falcon.extract(image, bbox)`; our copy of the protocol: `tools/speed_bench.py` |
| Determinism (#31) | ✅ | two runs identical (max diff 0.0) on two machines |

## Data & external resources
| Rule | Status | How we comply |
|---|---|---|
| Public pretrained weights with source, version, checksum (#39, #46; ТЗ §7) | ✅ | DINOv3 timm ids + Hugging Face revisions in the README; our weights with sha256 in `weights/manifest.json` |
| Licences respected | ✅ | DINOv3 License copy shipped with the weights (`weights/LICENSE_DINOv3.md`) |
| External datasets public and listed (#46, #47) | ✅ | none used by the submission (VeRi tested and rejected — listed in the README) |
| `camera_id` only for PK sampling and validation, never a model input (#2) | ✅ | validation splits + junk filter only |

## Output format (#18–#29, О-2; checked against `example_submission.zip` and `evaluate.py`)
| File | Status |
|---|---|
| `submission.csv` — no header, query_id + 10 gallery_ids for EVERY query | ✅ |
| `candidates.csv` — header, refusal = no row, only the top confidence counts | ✅ |
| `embeddings.npy` — float32, (n_query + n_gallery, D), queries then gallery in CSV order | ✅ (1860 × 768) |
| Threshold stated and justified in the README (#26) | ✅ 0.809 (plateau overlap on two splits) |

## Architecture & delivery (ТЗ §6, §11–§13; #41)
| Requirement | Status |
|---|---|
| Microservices: inference backend / DB / client | ✅ api (FastAPI) / db (PostgreSQL + pgvector) / web (nginx) |
| Backend in Python, Linux, open source | ✅ |
| Vector DB for embeddings + metadata | ✅ pgvector with HNSW |
| OpenAPI/Swagger for all endpoints; browser thin client | ✅ `/api/docs` |
| Dockerfile + docker-compose, one-command start, offline | ✅ batch (tested) · service (compose; full Docker test planned before upload) |
| Pinned versions (`==`), weights with sha256 (#39) | ✅ |
| README: architecture, methods, build steps, validation metrics, threshold reasoning, all libraries/datasets with versions (ТЗ §12) | ✅ |
| Presentation PDF/PPTX (ТЗ §11), slides 7–11 in the strict template (Telegram) | 🔄 in progress |
| Submission links: open repo, presentation, working prototype, docs (ТЗ §13) | 🔄 before the deadline |
