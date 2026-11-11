# Model card

Following Mitchell et al., *Model Cards for Model Reporting*. Numbers come from `reports/*.json`; anything not
measured says so instead of being guessed. The checkpoints are **not** distributed with this repository.

## Model details
- **Task:** bi-temporal change detection; predicts a per-pixel damage probability from a pre/post image pair.
- **Baseline (Week 2):** early-fusion U-Net, 6 input channels, ResNet34 encoder, ImageNet weights. Checkpoint `baseline_best.pt` (epoch 33, Kaggle GPU).
- **Siamese variant (Week 3, primary model):** shared ResNet34 encoder, |pre−post| feature differences fed to the decoder at each scale. Checkpoint `siamese_best.pt` (epoch 13 of 23, early stopping, Kaggle GPU, single run).
- **Uncertainty (Week 3):** 8-view TTA (std over views) chosen over MC Dropout; see `docs/decisions.md` and `reports/uncertainty_*.json`.
- **Size:** about 24.5 M parameters each (ResNet34 encoder + decoder); checkpoint files about 98 MB, not in the repository.
- **Checkpoints (SHA-256, first 16 hex):** `siamese_best.pt` `ab88d2cf7ecd4a02`, `baseline_best.pt` `f484c31d164d6b2c`. Evaluation provenance: git SHA `64beaad` (dirty tree), seed not recorded; one training run per model.
- **Inference:** 8-view TTA gives probability and uncertainty (std over views); the operating threshold 0.2 was chosen on validation. A quality guard (blank-pixel share, saturated/cloud-proxy share, pre/post alignment by block-wise SSIM; thresholds in `configs/default.yaml`) blocks decisions on bad image pairs.

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
| Building (SIMULATED) | Antakya demo set, 5,895 buildings, 732 HOTOSM destroyed | precision 0.81, recall 0.81, F1 0.81 for the *simulated* detector (miss 20 %, false alarm 3 %) | These follow from the chosen noise rates by construction. They measure planning robustness, **not** model accuracy (`reports/metrics_building.json`) |
| Building (real model) | none | not measured | No building-level ground truth for the real model: OSM contains almost none of the destroyed buildings (8 overlap), and the Kilis crop has 1 labelled building |
| Error analysis | KATE-CD test, Siamese, 10 worst tiles | all 10 are missed damage (6 with IoU 0, 5 of those with nothing predicted) | Tiles, not buildings; causes beyond what is visible were not tested (`reports/error_analysis.json`) |

## Limitations and risks
- Single earthquake, so generalization to other regions and building types is not guaranteed.
- No damage-free tiles in training data: elevated false-positive risk on undamaged areas.
- Domain shift between KATE-CD (512×512 tiles) and the Vantor Kilis crop (EPSG:32637, 0.31 m/px, 2023-01-26 / 2023-02-07) is not measured; the Siamese model runs on the crop but there are no labels, so this is no accuracy evidence.
- Single run per model and 38 test tiles: confidence intervals are wide (Siamese F1 0.60, 95 % CI 0.50–0.68); the threshold choice is validation-dependent (the high-recall F2 threshold was unstable).
- Uncertainty is validated on KATE-CD only (uncertainty / error ratio 20.9 for TTA); the building-level uncertainty in the demo is simulated and independent of correctness.
- HOTOSM labels are incomplete, which biases building-level precision downwards.

## Ethics
- Wrong priorities can cost lives; every score is reported with uncertainty and low-confidence or poor-quality
  inputs go to a field-verification queue instead of producing a decision.

## System components around the model

| Component | What it is | Measured | Source |
|---|---|---|---|
| Damage gate + TOPSIS/AHP | Only buildings above the damage threshold (0.2) are candidates; weights 0.35 / 0.35 / 0.15 / 0.15 (damage, persons, floors, distance to depot) | planning on simulated detections: VRP collects 26.0 priority / 3,106 persons vs 22.9 / 2,749 (highest priority first), 12.1 / 430 (nearest first), 11.7 / 899 (random mean) | `reports/metrics_planning.json` |
| Route planner | OR-Tools prize-collecting VRP, 4 teams, 6 h, 30 min per building, debris and closed-road rules, walking leg | simulated detector; depot is a scenario assumption | `docs/decisions.md` |
| RAG | multilingual-e5-small + BM25 over AFAD TAMP and INSARAG (458 chunks) | recall@1 0.80, MRR 0.80 on 10 questions | `reports/metrics_rag.json` |
| LLM agent | router + 9 tools + sitrep; Gemini primary, Groq fallback | 38 scripted scenarios, one cold run: task success 34 / 38, tool selection 45 / 48 turns, 0 forbidden-behaviour violations, protocol answers citing the gold page 7 / 8 | `reports/agent_eval.json` |

The agent numbers are the combined effect of the damage gate, the citation normalisation and the protocol-retrieval fix compared with the first run (30 / 38); the gate's effect alone was not isolated. Scenarios were written by the party that wrote the agent, single run, LLM output not deterministic.
