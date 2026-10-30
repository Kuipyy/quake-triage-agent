# Model card

Skeleton following Mitchell et al., *Model Cards for Model Reporting*. Each week fills its section; empty
sections stay `TBD` rather than being guessed. Numbers must come from `reports/*.json`, not from memory.

## Model details
- **Task:** bi-temporal change detection; predicts a per-pixel damage probability from a pre/post image pair.
- **Baseline (Week 2):** early-fusion U-Net, 6 input channels, ResNet34 encoder, ImageNet weights. TBD
- **Siamese variant (Week 3):** shared encoder, feature differences fed to the decoder at each scale. TBD
- **Uncertainty (Week 3):** TBD (MC Dropout or test-time augmentation / ensemble; see `docs/review/`).
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
| Pixel | KATE-CD test (38) | IoU, F1, precision, recall, ECE, with bootstrap CI | TBD |
| Building | Demo area + HOTOSM destroyed labels | precision, recall, F1 | TBD |

## Limitations and risks
- Single earthquake, so generalization to other regions and building types is not guaranteed.
- No damage-free tiles in training data: elevated false-positive risk on undamaged areas.
- Domain shift between KATE-CD and the Vantor demo imagery (sensor, resolution, season): TBD.
- HOTOSM labels are incomplete, which biases building-level precision downwards.

## Ethics
- Wrong priorities can cost lives; every score is reported with uncertainty and low-confidence or poor-quality
  inputs go to a field-verification queue instead of producing a decision.
