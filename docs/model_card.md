# Model card

Skeleton following Mitchell et al., *Model Cards for Model Reporting*. Each week fills its section; empty
sections stay `TBD` rather than being guessed. Numbers must come from `reports/*.json`, not from memory.

## Model details
- **Task:** bi-temporal change detection; predicts a per-pixel damage probability from a pre/post image pair.
- **Baseline (Week 2):** early-fusion U-Net, 6 input channels, ResNet34 encoder, ImageNet weights. Checkpoint `baseline_best.pt` (epoch 33, Kaggle GPU).
- **Siamese variant (Week 3):** shared encoder, feature differences fed to the decoder at each scale. TBD
- **Uncertainty (Week 3):** MC Dropout vs 8-view TTA compared on the baseline (`reports/uncertainty_baseline_*.json`); final choice pending the dropout-trained model and the Siamese comparison.
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
| Pixel | KATE-CD test (38), baseline, thr 0.5 | F1 0.490 [0.370–0.582], precision 0.657 [0.551–0.760], recall 0.391 [0.259–0.511], IoU 0.325 [0.227–0.410] (`reports/metrics_baseline.json`) | Siamese: TBD |
| Calibration | KATE-CD test, baseline | ECE: single 0.0186, MC Dropout 0.0113, TTA 0.0131 (MC Dropout applied at inference only; `reports/uncertainty_baseline_test.json`) | Final method: TBD |
| Building | Demo area + HOTOSM destroyed labels | precision, recall, F1 | TBD |

## Limitations and risks
- Single earthquake, so generalization to other regions and building types is not guaranteed.
- No damage-free tiles in training data: elevated false-positive risk on undamaged areas.
- Domain shift between KATE-CD and the Vantor demo imagery (sensor, resolution, season): TBD.
- HOTOSM labels are incomplete, which biases building-level precision downwards.

## Ethics
- Wrong priorities can cost lives; every score is reported with uncertainty and low-confidence or poor-quality
  inputs go to a field-verification queue instead of producing a decision.
