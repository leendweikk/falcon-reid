# Experiments — everything we tried, kept or rejected

Rule for every change (see README → Validation protocol): measured in the **real pipeline** with the organizers'
`evaluate.py` on our validation split 42, kept only if it wins by **≥ 0.7 mAP@10** there **and** still wins on
split 7 (seed-to-seed noise ≈ 0.7). Numbers are mAP@10 in %, as measured at the time.

## Progress of the main number (split 42)

| Step | mAP@10 |
|---|---|
| First model (ResNet50-IBN, small batch, old laptop GPU) | 34.0 |
| Frozen generic backbone, no fine-tuning | 38.3 |
| DINOv3 ConvNeXt-Small fine-tuned | 65.8 |
| DINOv3 ConvNeXt-Base fine-tuned | 68.7 |
| + soft-margin triplet loss | 70.2 |
| + EMA of the weights (quick training metric) | 75.5 |
| + k-reciprocal re-ranking, one query at a time (fast Base, real pipeline) | 78.98 |
| DINOv3 ViT-B/16 as the fast model (real pipeline) | 79.14 |
| Ensemble Base + ViT with flip TTA (not timed-feasible alone: ~85 ms) | 82.94 |
| **Final: two-stage (fast ViT top-100 → ensemble re-orders)** | **82.25** (split 7: 80.29) |

## Kept

| Idea | Evidence | Why it stays |
|---|---|---|
| DINOv3 backbones fine-tuned with BNNeck + ID + triplet | 34 → 68.7 | foundation-model features transfer to vehicles |
| Soft-margin batch-hard triplet | 68.7 → 70.2 | never goes flat, keeps teaching |
| EMA of the weights (decay 0.998), saved at the plateau epoch | quick metric 75.5 | smoother, better model |
| k-reciprocal re-ranking, streaming (one query vs gallery, top-100) | fast Base: cosine 75.05 → 78.98 | allowed (#28, #38); strong gain |
| ViT-B/16 (20 epochs, lr 5e-5) as the timed model | s42 79.14 vs Base 78.98; s7 78.00 vs 76.04; 33.8 vs 50.8 ms | best accuracy AND fits the 40 ms bar |
| Ensemble Base 0.5 + ViT 0.5 with flip | s42 82.94 (+2.54 vs Base+Small), s7 81.29 (+4.36) | passed the gate on both splits |
| Two-stage: ViT top-100 → ensemble re-orders (stage 2 not timed, #31) | s42 82.25 vs 79.36, s7 80.29 vs 78.15 | ensemble accuracy with full speed points |
| Stage-2 variant a) Base flip + ViT flip | b) reuse ViT vectors −0.46 / −0.35; c) Base without flip −0.19 / −1.51 | a) is best; run time 50% of the limit |
| Refusal signal cos+gap | s42 0.957 vs cosine 0.948; s7 0.962 vs 0.957 (fast Base) | better on both splits, higher PR-AUC |
| Threshold 0.809 from the overlap of both splits' plateaus | 0.9669 on both splits (20% open-set weighting) | not tuned to one split |
| Decode workers = one per CPU thread (max 16) | laptop 107 → 135 FPS, identical outputs | throughput is CPU-decode bound |

## Rejected (measured)

| Idea | Result | Decision |
|---|---|---|
| Strong night/glare augmentation | worse than mild colour jitter | rejected |
| Rectangular input 224 × 288 | worse | rejected |
| Input 320 (trained) | 78.25 @320, 78.52 @352 vs 78.98 @256; ~1.6× slower | rejected |
| Multi-size TTA (no training) | best combos +0.56 (s42) / +1.59 (s7): none ≥ +0.7 on both | rejected |
| Model soup | worse | rejected |
| CosFace head | overfits | rejected |
| ViT 30 epochs | s42 80.27 vs 79.14 but s7 77.06 vs 78.00 | failed the gate |
| ViT with layer-wise LR decay | 78.97 vs 79.14 | rejected |
| Camera-aware PK batches (camera_id only for sampling, #2) | 77.23 vs 78.98 | rejected |
| GeM pooling | 78.45 vs 78.98 | rejected |
| P = 32 cars per batch | does not fit in 6 GB (memory spill, 5–12× slower) | not testable on our hardware |
| Re-tuned re-ranking (k1 = 4, k2 = 2, λ = 0.5) | s42 +0.90 but s7 −0.12 | failed the gate; defaults kept |
| Relational distillation of the ensemble into the ViT | +0.39 (s42) / +0.03 (s7) | below the bar |
| Three-model ensembles with ConvNeXt-Small | Base 0.4 + Small 0.3 + ViT 0.3: 82.87 / 80.61 < Base + ViT | Small dropped |
| **VeRi-776 as extra training data** (49,357 photos, plates blurred by a public detector) | fast: +0.25 / +0.08; **two-stage s42: 82.02 vs 82.25 (−0.23)** | rejected: does not help our cars |
| JPEG draft (reduced-size) decoding | Colab: 32.3 ms / 48.8 FPS vs 36.6 / 47.2 (triggers only for boxes ≥ 768 px) | not used |
| Query expansion / re-ranking over all queries at once | — | **forbidden** by answer #38; never used in the submission |

## Considered and not tried (with reasons)

| Idea | Reason |
|---|---|
| `torch.compile` / TensorRT / ONNX | the timed model already meets both speed bars; compilation adds build and determinism risk for no points |
| INT8 quantization / pruning | typically costs accuracy on re-ID; speed already at full points |
| ConvNeXt-Large as a stage-2 member | does not train in 6 GB; no budget for a rented GPU |
| Training on unlabelled test images (pseudo-labels) | uses test data transductively — against the spirit of answer #38; test cameras are the same city network anyway |
| Synthetic data (VehicleX) | no time |
