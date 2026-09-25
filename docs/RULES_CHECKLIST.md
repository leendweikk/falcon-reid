# Rules checklist — ФАЛЬКОН.Tech (ЛЦТ 2026, task 7)

Sources: ТЗ, dataset README, official `evaluate.py`, the official Q&A table (answers cited as #N,
summarized in `ORGANIZER_ANSWERS.md`), Telegram announcements.
Every new idea is checked here BEFORE we try it. Before submission we go through every line.

## Hard rules (violation = disqualification)
| Rule | Status | How we comply |
|---|---|---|
| No license-plate features (ТЗ §9, #48) | OK | Organizers' images have blurred plates; Grad-CAM shows the plate zone stays cold. Finalists are re-tested with plates painted over (#48) |
| Reproducible from submitted code (ТЗ §9) | TODO | Train/infer scripts, fixed seeds, pinned `requirements.txt`, README steps, clean-machine test |
| No test labels used (#38) | OK | We never had them; no tuning on the public test |

## Inference rules
| Rule | Status | How we comply |
|---|---|---|
| Each query independent, stream protocol (#38, #40) | OK | Only the current query + the static gallery are used |
| k-reciprocal / heavier re-ranking inside one query's top-K (#28, #38) | OK | `re_ranking_streaming`, top-100, one query at a time; described in README (#12) |
| Gallery-side processing (gallery = static base, may be pre-built) (#38) | OK to try | A13 gallery smoothing uses gallery items only |
| No query expansion / clustering over queries (#38) | OK | All-queries re-ranking rejected on 23.09 |
| Single-image TTA (#38) | OK | Horizontal flip of the same image |
| Runs offline, all weights inside the image (#39) | TODO | Backbones built with `pretrained=False`, our weights loaded from local files |
| Total weights ≤ 2 GB incl. ensembles/rerankers (#37) | OK | Base ~355 MB + Small ~200 MB + ViT ~330 MB; classifier heads can be dropped |
| No OOM on the test set (ТЗ §7) | TODO | Test in Docker on the full public test |
| Whole run within ~latency × n × 3 (#40) | TODO | Measure end-to-end incl. re-ranking |
| Docker GPU works with CUDA driver 12.2 (#30) | TODO | torch build compatible with driver 12.2 |

## Data & external resources
| Rule | Status | How we comply |
|---|---|---|
| Public pretrained weights with source, version, checksum (#39, #46; ТЗ §7) | TODO (list) | DINOv3 ConvNeXt-B/S + ViT-B/16 (timm tags `*.dinov3_lvd1689m`) — URL + sha256 in README |
| External datasets allowed if public and reproducible, plates OK (#46, #47) | OK to try | VeRi (A9) — must be listed in README with source/version |
| camera_id only for PK-sampling and validation, never model input (#2) | OK | `PKSampler(camera_aware=True)`, split building, junk filter |

## Output format (#18–#29, О-2; checked against example_submission.zip)
| File | Status |
|---|---|
| `submission.csv` — no header, query_id + 10 gallery_ids for EVERY query | OK |
| `candidates.csv` — header, refusal = no row, only top confidence counts | OK |
| `embeddings.npy` — float32, (n_q + n_g, D), queries then gallery in CSV order | OK |
| Threshold stated and justified in README (#26) | TODO — objective 0.7·F1 + 0.3·TNR (#15) at 20% open-set (#17) |

## Architecture & delivery (ТЗ §6, §11–§13; #41)
| Requirement | Status |
|---|---|
| Microservices: inference backend / DB / client | TODO |
| Backend in Python/C++/Rust, Linux | OK (Python) |
| Vector DB (pgvector / FAISS / Milvus) for embeddings + metadata | TODO |
| OpenAPI/Swagger for all endpoints; browser thin client | TODO |
| Dockerfile + docker-compose, one-command start, offline | TODO |
| Pinned versions (`==`), weights with sha256 (#39) | PARTIAL (requirements.txt pinned) |
| README: architecture, methods, build steps, val metrics, threshold reasoning, ALL libraries/datasets with versions (ТЗ §12) | TODO |
| Presentation PDF/PPTX: method + backbone choice, viewpoint/lighting, user-journey diagram, scaling plan, honest error analysis; slides 7–11 in the strict template (ТЗ §11, Telegram 23.09) | TODO |
| Submission links: repo (open), presentation, **working prototype**, docs (ТЗ §13); upload links early | TODO |

## Open questions to organizers
- `extractor.py` contract (#43 mentions it; not found in materials) — ask
- How long the demo prototype must stay online — unanswered in Telegram
