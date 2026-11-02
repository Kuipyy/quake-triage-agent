# Model card

Skeleton following Mitchell et al., *Model Cards for Model Reporting*. Each week fills its section; empty
sections stay `TBD` rather than being guessed. Numbers must come from `reports/*.json`, not from memory.

## Model details
- **Task:** bi-temporal change detection; predicts a per-pixel damage probability from a pre/post image pair.
- **Baseline (Week 2):** early-fusion U-Net, 6 input channels, ResNet34 encoder, ImageNet weights. Checkpoint `baseline_best.pt` (epoch 33, Kaggle GPU).
- **Siamese variant (Week 3, primary model):** shared ResNet34 encoder, |pre−post| feature differences fed to the decoder at each scale. Checkpoint `siamese_best.pt` (epoch 13 of 23, early stopping, Kaggle GPU, single run).
- **Uncertainty (Week 3):** 8-view TTA (std over views) chosen over MC Dropout; see `docs/decisions.md` and `reports/uncertainty_*.json`.
- **Checkpoint, config hash, git SHA, seed:** TBD (written by the training script).

## Intended use
- Research and decision-support prototype for ranking buildings for field verification and team dispatch.
- **Not** validated for real search-and-rescue operations. Outputs are advice with uncertainty, not decisions.

## Training data
- KATE-CD train split (404 samples); see `docs/data_card.md`.
- Heavily imbalanced (about 3.7 % damaged pixels): Dice + Focal loss, threshold tuned on validation F1.

## Evaluation
| Level | Data | Metrics | Result |
|---|---|---|---|
| Pixel | KATE-CD test (38), baseline, thr 0.2 (val-tuned) | F1 0.502 [0.381–0.592], precision 0.635, recall 0.414, IoU 0.335 (`reports/metrics_baseline_f1.json`) | thr 0.5 (untuned): F1 0.490 |
| Pixel | KATE-CD test (38), **Siamese**, thr 0.2 (val-tuned) | F1 0.602 [0.497–0.681], precision 0.732, recall 0.511, IoU 0.430 (`reports/metrics_siamese_f1.json`) | thr 0.5 (untuned): F1 0.586 |
| Comparison | paired bootstrap, Siamese − baseline | ΔF1 +0.104 [+0.038, +0.166]; ΔIoU +0.099 [+0.037, +0.155]; Δrecall +0.099 [+0.015, +0.175]; Δprecision +0.096 [−0.003, +0.196] (`reports/comparison_siamese_vs_baseline_f1.json`) | single run per model |
| Calibration | KATE-CD test, Siamese | ECE: single 0.0118, MC Dropout 0.0062, **TTA 0.0079**; Brier TTA 0.0152 (best); uncertainty/error ratio TTA 20.9 (`reports/uncertainty_siamese_test.json`) | MC Dropout inference-only |
| Pixel (high-recall profile) | KATE-CD test, Siamese, F2-val-tuned thr 0.02 | recall 0.588, precision 0.598, F1 0.593 (`reports/metrics_siamese_f2.json`) | threshold unstable, see decisions |
| Building | Demo area + HOTOSM destroyed labels | precision, recall, F1 | TBD |

## Limitations and risks
- Single earthquake, so generalization to other regions and building types is not guaranteed.
- No damage-free tiles in training data: elevated false-positive risk on undamaged areas.
- Domain shift between KATE-CD and the Vantor demo imagery (sensor, resolution, season): TBD.
- HOTOSM labels are incomplete, which biases building-level precision downwards.

## Ethics
- Wrong priorities can cost lives; every score is reported with uncertainty and low-confidence or poor-quality
  inputs go to a field-verification queue instead of producing a decision.
