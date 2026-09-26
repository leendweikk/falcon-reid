# Error analysis — final pipeline, validation split 42

Produced by `python error_analysis.py runs/val42_final` on the final two-stage pipeline
(`falcon/config_val42.json`, models trained **without** the 300 validation cars). The official answers allow and
recommend doing this on a held-out part of `train.csv`, since the test labels are hidden (#50).

**932 scored queries** (queries of the 50 "stranger" cars have no pair and are excluded, as in the official mAP):
mAP@10 82.25 · Rank-1 78.11 · Rank-5 93.99. The correct car is missing from the top-5 in **56 queries (6.0%)**.

## Why failures happen

In **91% of the failures a wrong car is more similar than the best true match** — the lookalike problem: another
vehicle of the same make, model and colour, often seen from the same angle. Median similarity (stage-1 vectors):
best true match 0.276 vs best wrong match 0.425 in failures; best true match 0.571 in successes. In other words,
when the model fails it is usually because the true photo of the car looks genuinely different (other side,
other light), not because the ranking is noisy.

## Accuracy by condition

| Brightness of the query (0 dark – 255 bright) | mAP@10 | Failure rate |
|---|---|---|
| Q1 darkest (17 – 80) | **0.764** | 8.2% |
| Q2 (80 – 102) | 0.810 | 6.4% |
| Q3 (102 – 124) | 0.856 | 3.9% |
| Q4 brightest (124 – 177) | 0.860 | 5.6% |

| Box area of the query (pixels) | mAP@10 | Failure rate |
|---|---|---|
| Q1 smallest (48,600 – 250,233) | 0.850 | 3.9% |
| Q2 | 0.807 | 6.9% |
| Q3 | 0.799 | 9.4% |
| Q4 largest (444,854 – 1,062,271) | 0.834 | 3.9% |

| True matches in the gallery | Queries | mAP@10 |
|---|---|---|
| 1 | 569 | 0.848 |
| 2 | 262 | 0.757 |
| 3 | 72 | 0.879 |
| 4 | 16 | 0.883 |
| 5–6 | 13 | 0.63 |

Reading: **dark photos are the clearest weakness** (−10 points vs bright). A stronger synthetic night/glare
augmentation was tried and made things worse, so the realistic fix is more real night data. Box size matters
less than expected; the medium sizes are the hardest, likely because they mix views and partial occlusions.

## Typical failure cases (picture grids)

Each grid shows the query, its top-5 and the best true match. Files: `docs/error_analysis/worst_01.jpg` …
`worst_20.jpg` (the 20 worst queries), `summary.txt`, `per_query.csv`.

![worst 1](error_analysis/worst_01.jpg)
![worst 2](error_analysis/worst_02.jpg)
![worst 3](error_analysis/worst_03.jpg)

What we see in the 20 worst queries (checked by eye; several rows are different photos of the same query car,
so the counts are approximate and only describe these 20):
- **Two cars inside one box (≈ 5 of 20).** The box covers the target car *and* another car in front of or behind
  it (for example a white-and-orange car-sharing SUV parked next to a dark sedan). The model then describes the
  more visible car, and the "true" gallery photo shows the other one. This is an annotation ambiguity, not
  something a better embedding can fully fix.
- **Front ↔ rear (≈ 5 of 20).** Query from behind, true match from the front (or the reverse): very few shared
  visible parts. The top-5 are then lookalikes seen from the same side as the query.
- **Night, headlights and brake lights (≈ 3 of 20).** Glare hides the body details the model relies on — the
  same weakness as the brightness table above.
- **Fleet lookalikes.** Car-sharing and taxi fleets use identical cars with identical liveries; for these queries
  the top-5 is full of the same model in the same colours. Without the plate, some of these pairs are close to
  indistinguishable — worth saying openly to an operator.
- **Probable label problems (≈ 4 of 20).** In a few grids the "true" photo shows a clearly different vehicle
  (for example a sedan whose true match is a black minivan). We did not remove such pairs (training or
  validation) and do not claim a percentage.

## What the model looks at (Grad-CAM)

`docs/gradcam/` (made with `gradcam.py` on a validation ConvNeXt-Base): the heat sits on the car, not the
background; the plate zone stays cold. In a correct rear → front match the model used an orange side sticker
(an instance detail); in a wrong match it relied on the brand logo (brand, not identity). This agrees with the
plate-masking test (no drop with plates painted over).
