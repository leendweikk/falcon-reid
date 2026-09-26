# experiments/ — research scripts behind the decisions

These scripts produced the numbers in `docs/EXPERIMENTS.md` (validation splits only). They are kept for
transparency and are **not** used by the submission: the submission is `falcon/` (+ `reid/` model code),
run by the Dockerfile. Every inference-time comparison here processes each query alone against the gallery
(answer #38); an early all-queries re-ranking comparison (forbidden, reference only) was removed from the repo.

Main ones: `stage_tests.py` (two-stage search), `ensemble_test.py`, `multisize_test.py`,
`threshold_study.py` (refusal signals), `plate_mask_test.py` (plate-masking self-test, answer #48),
`prepare_veri.py` (VeRi preparation — the VeRi experiment was rejected), `official_bench.py` (speed).

- `legacy/` — superseded tools from the Base + Small era (the old `run_inference.py` pipeline, the old benchmark,
  model soup, old threshold scripts). **The submission's only entry point is `python -m falcon.predict`.**
- `queues/` — PowerShell queues that ran the validation experiments overnight on the Windows laptop
  (run from the project root: `powershell -ExecutionPolicy Bypass -File experiments/queues/queue.ps1`).
