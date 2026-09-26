> **LEGACY** (Base + Small era, 24–25 Sep). Current speed numbers: `docs/speed_log_falcon.csv` and the README.

# Speed benchmark

Hardware: GPU = NVIDIA GeForce RTX 4050 Laptop GPU; CPU = 8 threads; torch 2.14.0+cu132

- Image decode + crop + resize (CPU, per vehicle): **14.4 ms**
- Re-ranking (streaming, top-100, gallery 750): **12.2 ms per query**

`model_ms_batch1` = feature for ONE vehicle; `total_ms_batch1` = + decode/crop/resize; `model_fps_batch32` = images/second in batches of 32.

| device   | option                      |   model_ms_batch1 |   total_ms_batch1 |   model_fps_batch32 |
|:---------|:----------------------------|------------------:|------------------:|--------------------:|
| CUDA     | Base only, no flip          |              23.7 |              38.1 |               237.4 |
| CUDA     | Base only + flip            |              53.5 |              68   |               124.9 |
| CUDA     | Ensemble, no flip           |              59.3 |              73.7 |               148.2 |
| CUDA     | Ensemble + flip (submitted) |             115.6 |             130.1 |                75.3 |
| CPU      | Base only, no flip          |             203.3 |             217.7 |                 8   |
| CPU      | Base only + flip            |             327.8 |             342.2 |                 4   |
| CPU      | Ensemble, no flip           |             268.6 |             283   |                 4.9 |
| CPU      | Ensemble + flip (submitted) |             531.9 |             546.3 |                 2.4 |
