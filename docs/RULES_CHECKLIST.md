# Rules checklist — ФАЛЬКОН.Tech (ЛЦТ 2026, task 7)

Sources: technical brief (ТЗ), dataset README, official evaluate.py, organizers' answers in Telegram.
Every new idea is checked here BEFORE we try it. Before submission we go through every line.

## Hard rules (violation = disqualification)
| Rule | Status | How we comply |
|---|---|---|
| No license-plate features (plates are blurred; using residues forbidden) | OK | Only organizers' data (plates blurred). VeRi (visible plates) rejected. |
| Reproducible from submitted code | TODO | Train/infer scripts in repo, fixed seeds, requirements.txt with versions, README steps |
| No test labels used | OK | We never had them |

## Inference rules
| Rule | Status | How we comply |
|---|---|---|
| Each query processed independently (stream) | OK | Flip-TTA (own image), ensemble (same image), DBA (gallery only), streaming re-ranking (1 query + static gallery) |
| No query expansion / anything using other test queries | OK | All-queries re-ranking rejected (organizers' answer, 23.09) |
| Runs offline, one command, all weights included | TODO | Load backbone weights from local files, not the internet |
| Total weights <= 2 GB | OK | Base ~355 MB + Small ~200 MB |
| No out-of-memory on test set | TODO | Test in Docker on full test set |

## Data & external resources
| Rule | Status | How we comply |
|---|---|---|
| Public pretrained weights allowed, listed in README | TODO (list) | DINOv3 ConvNeXt via timm / Hugging Face |
| No closed/proprietary/non-reproducible data | OK | Only organizers' train.csv |
| camera_id | OK | Used only for validation split and analysis, never as model input |

## Output format (checked with official evaluate.py)
| File | Status |
|---|---|
| submission.csv — no header, query_id + 10 gallery_ids | OK |
| candidates.csv — header, top-1 + confidence, refusal = no row | OK |
| embeddings.npy — one 2D array, queries then gallery, CSV order | OK |
| Refusal threshold justified in README | TODO (have the sweep data) |

## Architecture & delivery
| Requirement | Status |
|---|---|
| Microservices: inference backend / DB / client | TODO |
| Backend in Python/C++/Rust, Linux | OK (Python) |
| Vector DB (pgvector / FAISS / Milvus) | TODO |
| OpenAPI/Swagger for all endpoints | TODO |
| Dockerfile + docker-compose, one-command start | TODO |
| README: architecture, methods, how to run, metrics, threshold reasoning, all libraries + versions | TODO |
| Presentation: template slides 7–11 unchanged in design | TODO (teammate) |
| Submission: repo link, presentation link, prototype link, docs link | TODO |

## Open questions to organizers
- Research-license datasets (VeRi) allowed? — asked, no answer -> not used
- Test hardware (GPU/CPU)? — asked, no answer -> must run on both
- How long must the demo server stay up? — asked by other team, no answer