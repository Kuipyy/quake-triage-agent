# QuakeTriage

**Agentic post-earthquake building damage assessment and rescue prioritization.**

> ⚠️ Research prototype only. Not validated for operational use in real search-and-rescue operations.

## Overview

QuakeTriage answers the question:
> "The damage map is ready. Where should I send my limited rescue teams first, and by which route? Which buildings is the model uncertain about?"

Given pre/post-earthquake satellite imagery, QuakeTriage:
1. Detects damaged buildings (change detection model with uncertainty)
2. Scores buildings by estimated occupancy and damage severity
3. Prioritizes sites using TOPSIS multi-criteria decision analysis
4. Plans optimal team routes with OR-Tools (prize-collecting VRP)
5. Grounds decisions in AFAD TAMP and INSARAG guidelines via RAG
6. Presents results through an interactive Gradio interface

## Architecture

```mermaid
flowchart TD
    U["User question (TR/EN)"] --> A["LangGraph agent"]
    A --> Q["check_image_quality"]
    Q -- "poor quality" --> V["field-verification queue"]
    Q --> D["detect_damage (probability + uncertainty)"]
    D --> S["score_buildings"]
    S --> O["estimate_occupancy"]
    O --> P["prioritize_sites (TOPSIS)"]
    P --> R["plan_routes (OR-Tools)"]
    R --> L["lookup_protocol (RAG, cited)"]
    L --> W["write_sitrep"]
    A -. "what-if" .-> C["update_constraints"]
    C -. "re-run changed steps" .-> R
```

Planned design; components are added week by week (see `PLAN.md`).

## Status

Work in progress — 8-week development plan (Nov–Dec 2026).

## Results

Generated from `reports/*.json` by `make report`; do not edit by hand.

<!-- results:begin -->
| Report | Metric | Value |
|---|---|---|
| `comparison_siamese_vs_baseline_f1` | `threshold_mode` | f1 |
| `comparison_siamese_vs_baseline_f1` | `selected_on` | validation |
| `comparison_siamese_vs_baseline_f1` | `evaluated_on` | test |
| `comparison_siamese_vs_baseline_f1` | `n_test` | 38 |
| `comparison_siamese_vs_baseline_f1` | `models.early_fusion.threshold` | 0.2 |
| `comparison_siamese_vs_baseline_f1` | `models.early_fusion.metrics.precision` | 0.6354 |
| `comparison_siamese_vs_baseline_f1` | `models.early_fusion.metrics.recall` | 0.4145 |
| `comparison_siamese_vs_baseline_f1` | `models.early_fusion.metrics.f1` | 0.5017 |
| `comparison_siamese_vs_baseline_f1` | `models.early_fusion.metrics.iou` | 0.3348 |
| `comparison_siamese_vs_baseline_f1` | `models.siamese.threshold` | 0.2 |
| `comparison_siamese_vs_baseline_f1` | `models.siamese.metrics.precision` | 0.7319 |
| `comparison_siamese_vs_baseline_f1` | `models.siamese.metrics.recall` | 0.5108 |
| `comparison_siamese_vs_baseline_f1` | `models.siamese.metrics.f1` | 0.6017 |
| `comparison_siamese_vs_baseline_f1` | `models.siamese.metrics.iou` | 0.4303 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.f1.metric` | f1 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.f1.mean_diff` | 0.1038 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.f1.ci95_lo` | 0.03829 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.f1.ci95_hi` | 0.1657 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.f1.p_not_better` | 0 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.iou.metric` | iou |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.iou.mean_diff` | 0.09857 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.iou.ci95_lo` | 0.03736 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.iou.ci95_hi` | 0.155 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.iou.p_not_better` | 0 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.precision.metric` | precision |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.precision.mean_diff` | 0.09649 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.precision.ci95_lo` | -0.003213 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.precision.ci95_hi` | 0.1961 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.precision.p_not_better` | 0.03 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.recall.metric` | recall |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.recall.mean_diff` | 0.09901 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.recall.ci95_lo` | 0.01546 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.recall.ci95_hi` | 0.1749 |
| `comparison_siamese_vs_baseline_f1` | `paired_bootstrap_siamese_minus_baseline.recall.p_not_better` | 0.011 |
| `comparison_siamese_vs_baseline_f2` | `threshold_mode` | f2 |
| `comparison_siamese_vs_baseline_f2` | `selected_on` | validation |
| `comparison_siamese_vs_baseline_f2` | `evaluated_on` | test |
| `comparison_siamese_vs_baseline_f2` | `n_test` | 38 |
| `comparison_siamese_vs_baseline_f2` | `models.early_fusion.threshold` | 0.01 |
| `comparison_siamese_vs_baseline_f2` | `models.early_fusion.metrics.precision` | 0.5238 |
| `comparison_siamese_vs_baseline_f2` | `models.early_fusion.metrics.recall` | 0.5072 |
| `comparison_siamese_vs_baseline_f2` | `models.early_fusion.metrics.f1` | 0.5154 |
| `comparison_siamese_vs_baseline_f2` | `models.early_fusion.metrics.iou` | 0.3472 |
| `comparison_siamese_vs_baseline_f2` | `models.siamese.threshold` | 0.02 |
| `comparison_siamese_vs_baseline_f2` | `models.siamese.metrics.precision` | 0.5983 |
| `comparison_siamese_vs_baseline_f2` | `models.siamese.metrics.recall` | 0.5881 |
| `comparison_siamese_vs_baseline_f2` | `models.siamese.metrics.f1` | 0.5932 |
| `comparison_siamese_vs_baseline_f2` | `models.siamese.metrics.iou` | 0.4217 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.f1.metric` | f1 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.f1.mean_diff` | 0.08053 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.f1.ci95_lo` | 0.03199 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.f1.ci95_hi` | 0.1351 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.f1.p_not_better` | 0 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.iou.metric` | iou |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.iou.mean_diff` | 0.07666 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.iou.ci95_lo` | 0.03157 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.iou.ci95_hi` | 0.1238 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.iou.p_not_better` | 0 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.precision.metric` | precision |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.precision.mean_diff` | 0.07431 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.precision.ci95_lo` | -0.005027 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.precision.ci95_hi` | 0.1536 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.precision.p_not_better` | 0.039 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.recall.metric` | recall |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.recall.mean_diff` | 0.08349 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.recall.ci95_lo` | 0.006421 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.recall.ci95_hi` | 0.1628 |
| `comparison_siamese_vs_baseline_f2` | `paired_bootstrap_siamese_minus_baseline.recall.p_not_better` | 0.016 |
| `metrics_baseline` | `checkpoint` | baseline_best.pt |
| `metrics_baseline` | `model` | early_fusion |
| `metrics_baseline` | `split` | test |
| `metrics_baseline` | `encoder` | resnet34 |
| `metrics_baseline` | `trained_epoch` | 33 |
| `metrics_baseline` | `best_val_f1` | 0.5196 |
| `metrics_baseline` | `threshold` | 0.5 |
| `metrics_baseline` | `threshold_selection.mode` | ckpt |
| `metrics_baseline` | `n_samples` | 38 |
| `metrics_baseline` | `metrics.precision` | 0.6573 |
| `metrics_baseline` | `metrics.recall` | 0.3908 |
| `metrics_baseline` | `metrics.f1` | 0.4901 |
| `metrics_baseline` | `metrics.iou` | 0.3246 |
| `metrics_baseline` | `bootstrap_ci.precision.mean` | 0.6599 |
| `metrics_baseline` | `bootstrap_ci.precision.ci95_lo` | 0.5513 |
| `metrics_baseline` | `bootstrap_ci.precision.ci95_hi` | 0.7598 |
| `metrics_baseline` | `bootstrap_ci.recall.mean` | 0.3886 |
| `metrics_baseline` | `bootstrap_ci.recall.ci95_lo` | 0.259 |
| `metrics_baseline` | `bootstrap_ci.recall.ci95_hi` | 0.5114 |
| `metrics_baseline` | `bootstrap_ci.f1.mean` | 0.4854 |
| `metrics_baseline` | `bootstrap_ci.f1.ci95_lo` | 0.3702 |
| `metrics_baseline` | `bootstrap_ci.f1.ci95_hi` | 0.5818 |
| `metrics_baseline` | `bootstrap_ci.iou.mean` | 0.3224 |
| `metrics_baseline` | `bootstrap_ci.iou.ci95_lo` | 0.2272 |
| `metrics_baseline` | `bootstrap_ci.iou.ci95_hi` | 0.4102 |
| `metrics_baseline` | `inference_secs` | 3.6 |
| `metrics_baseline_f1` | `checkpoint` | baseline_best.pt |
| `metrics_baseline_f1` | `model` | early_fusion |
| `metrics_baseline_f1` | `split` | test |
| `metrics_baseline_f1` | `encoder` | resnet34 |
| `metrics_baseline_f1` | `trained_epoch` | 33 |
| `metrics_baseline_f1` | `best_val_f1` | 0.5196 |
| `metrics_baseline_f1` | `threshold` | 0.2 |
| `metrics_baseline_f1` | `threshold_selection.mode` | f1 |
| `metrics_baseline_f1` | `threshold_selection.selected_on` | validation |
| `metrics_baseline_f1` | `threshold_selection.objective` | F1 |
| `metrics_baseline_f1` | `threshold_selection.value` | 0.521 |
| `metrics_baseline_f1` | `n_samples` | 38 |
| `metrics_baseline_f1` | `metrics.precision` | 0.6354 |
| `metrics_baseline_f1` | `metrics.recall` | 0.4145 |
| `metrics_baseline_f1` | `metrics.f1` | 0.5017 |
| `metrics_baseline_f1` | `metrics.iou` | 0.3348 |
| `metrics_baseline_f1` | `bootstrap_ci.precision.mean` | 0.638 |
| `metrics_baseline_f1` | `bootstrap_ci.precision.ci95_lo` | 0.5293 |
| `metrics_baseline_f1` | `bootstrap_ci.precision.ci95_hi` | 0.7381 |
| `metrics_baseline_f1` | `bootstrap_ci.recall.mean` | 0.4125 |
| `metrics_baseline_f1` | `bootstrap_ci.recall.ci95_lo` | 0.281 |
| `metrics_baseline_f1` | `bootstrap_ci.recall.ci95_hi` | 0.5366 |
| `metrics_baseline_f1` | `bootstrap_ci.f1.mean` | 0.4973 |
| `metrics_baseline_f1` | `bootstrap_ci.f1.ci95_lo` | 0.3813 |
| `metrics_baseline_f1` | `bootstrap_ci.f1.ci95_hi` | 0.592 |
| `metrics_baseline_f1` | `bootstrap_ci.iou.mean` | 0.3328 |
| `metrics_baseline_f1` | `bootstrap_ci.iou.ci95_lo` | 0.2356 |
| `metrics_baseline_f1` | `bootstrap_ci.iou.ci95_hi` | 0.4204 |
| `metrics_baseline_f1` | `inference_secs` | 3 |
| `metrics_baseline_f2` | `checkpoint` | baseline_best.pt |
| `metrics_baseline_f2` | `model` | early_fusion |
| `metrics_baseline_f2` | `split` | test |
| `metrics_baseline_f2` | `encoder` | resnet34 |
| `metrics_baseline_f2` | `trained_epoch` | 33 |
| `metrics_baseline_f2` | `best_val_f1` | 0.5196 |
| `metrics_baseline_f2` | `threshold` | 0.01 |
| `metrics_baseline_f2` | `threshold_selection.mode` | f2 |
| `metrics_baseline_f2` | `threshold_selection.selected_on` | validation |
| `metrics_baseline_f2` | `threshold_selection.objective` | F2 |
| `metrics_baseline_f2` | `threshold_selection.value` | 0.5508 |
| `metrics_baseline_f2` | `n_samples` | 38 |
| `metrics_baseline_f2` | `metrics.precision` | 0.5238 |
| `metrics_baseline_f2` | `metrics.recall` | 0.5072 |
| `metrics_baseline_f2` | `metrics.f1` | 0.5154 |
| `metrics_baseline_f2` | `metrics.iou` | 0.3472 |
| `metrics_baseline_f2` | `bootstrap_ci.precision.mean` | 0.5255 |
| `metrics_baseline_f2` | `bootstrap_ci.precision.ci95_lo` | 0.4158 |
| `metrics_baseline_f2` | `bootstrap_ci.precision.ci95_hi` | 0.6229 |
| `metrics_baseline_f2` | `bootstrap_ci.recall.mean` | 0.5059 |
| `metrics_baseline_f2` | `bootstrap_ci.recall.ci95_lo` | 0.3532 |
| `metrics_baseline_f2` | `bootstrap_ci.recall.ci95_hi` | 0.6389 |
| `metrics_baseline_f2` | `bootstrap_ci.f1.mean` | 0.5124 |
| `metrics_baseline_f2` | `bootstrap_ci.f1.ci95_lo` | 0.4081 |
| `metrics_baseline_f2` | `bootstrap_ci.f1.ci95_hi` | 0.6033 |
| `metrics_baseline_f2` | `bootstrap_ci.iou.mean` | 0.346 |
| `metrics_baseline_f2` | `bootstrap_ci.iou.ci95_lo` | 0.2563 |
| `metrics_baseline_f2` | `bootstrap_ci.iou.ci95_hi` | 0.432 |
| `metrics_baseline_f2` | `inference_secs` | 3.1 |
| `metrics_building` | `SIMULATED_DETECTOR` | True |
| `metrics_building` | `disclaimer` | damage_prob/uncertainty are simulated from HOTOSM labels; these numbers measure planning-pipeline robustness to detector noise, NOT model accuracy. Recall/precision follow from the chosen miss/false-alarm rates by construction. queue_share is NOT informative here: simulated uncertainty does not depend on whether the detector is right (on KATE-CD real uncertainty is 14-21x higher on wrong pixels, reports/uncertainty_*.json). |
| `metrics_building` | `n_buildings` | 5895 |
| `metrics_building` | `n_destroyed_truth` | 732 |
| `metrics_building` | `demo_set.zone_buffer_m` | 100 |
| `metrics_building` | `demo_set.osm_footprints_overlapping_hotosm` | 8 |
| `metrics_building` | `demo_set.note` | destroyed footprints come from HOTOSM; others are nearby OSM buildings |
| `metrics_building` | `base_scenario.miss_rate` | 0.2 |
| `metrics_building` | `base_scenario.false_alarm_rate` | 0.03 |
| `metrics_building` | `base_scenario.size_exponent` | 1 |
| `metrics_building` | `base_scenario.seed` | 0 |
| `metrics_building` | `base_scenario.metrics.n` | 5895 |
| `metrics_building` | `base_scenario.metrics.n_truth` | 732 |
| `metrics_building` | `base_scenario.metrics.tp` | 593 |
| `metrics_building` | `base_scenario.metrics.fp` | 141 |
| `metrics_building` | `base_scenario.metrics.fn` | 139 |
| `metrics_building` | `base_scenario.metrics.precision` | 0.8079 |
| `metrics_building` | `base_scenario.metrics.recall` | 0.8101 |
| `metrics_building` | `base_scenario.metrics.f1` | 0.809 |
| `metrics_building` | `base_scenario.metrics.persons_in_destroyed` | 1.252e+04 |
| `metrics_building` | `base_scenario.metrics.person_recall` | 0.9275 |
| `metrics_building` | `base_scenario.queue_share` | 0.019 |
| `metrics_building` | `population.residents_day` | 6.177e+04 |
| `metrics_building` | `population.estimated_persons_night` | 5.463e+04 |
| `metrics_building` | `n_seeds` | 5 |
| `metrics_eda` | `train.n_samples` | 404 |
| `metrics_eda` | `train.mean_damaged_ratio` | 0.03669 |
| `metrics_eda` | `train.median_damaged_ratio` | 0.02069 |
| `metrics_eda` | `train.pct_with_any_damage` | 1 |
| `metrics_eda` | `validation.n_samples` | 44 |
| `metrics_eda` | `validation.mean_damaged_ratio` | 0.03204 |
| `metrics_eda` | `validation.median_damaged_ratio` | 0.01701 |
| `metrics_eda` | `validation.pct_with_any_damage` | 1 |
| `metrics_eda` | `test.n_samples` | 38 |
| `metrics_eda` | `test.mean_damaged_ratio` | 0.02825 |
| `metrics_eda` | `test.median_damaged_ratio` | 0.02032 |
| `metrics_eda` | `test.pct_with_any_damage` | 1 |
| `metrics_rag` | `n_questions` | 10 |
| `metrics_rag` | `n_chunks` | 458 |
| `metrics_rag` | `recall_at_1` | 0.8 |
| `metrics_rag` | `recall_at_3` | 0.8 |
| `metrics_rag` | `recall_at_5` | 0.8 |
| `metrics_rag` | `mrr` | 0.8 |
| `metrics_rag` | `embedding_model` | intfloat/multilingual-e5-small |
| `metrics_rag` | `hybrid` | True |
| `metrics_rag` | `chunk_chars` | 700 |
| `metrics_siamese` | `checkpoint` | siamese_best.pt |
| `metrics_siamese` | `model` | siamese |
| `metrics_siamese` | `split` | test |
| `metrics_siamese` | `encoder` | resnet34 |
| `metrics_siamese` | `trained_epoch` | 13 |
| `metrics_siamese` | `best_val_f1` | 0.5547 |
| `metrics_siamese` | `threshold` | 0.5 |
| `metrics_siamese` | `threshold_selection.mode` | ckpt |
| `metrics_siamese` | `n_samples` | 38 |
| `metrics_siamese` | `metrics.precision` | 0.7764 |
| `metrics_siamese` | `metrics.recall` | 0.4706 |
| `metrics_siamese` | `metrics.f1` | 0.586 |
| `metrics_siamese` | `metrics.iou` | 0.4144 |
| `metrics_siamese` | `bootstrap_ci.precision.mean` | 0.7789 |
| `metrics_siamese` | `bootstrap_ci.precision.ci95_lo` | 0.6977 |
| `metrics_siamese` | `bootstrap_ci.precision.ci95_hi` | 0.8469 |
| `metrics_siamese` | `bootstrap_ci.recall.mean` | 0.471 |
| `metrics_siamese` | `bootstrap_ci.recall.ci95_lo` | 0.3521 |
| `metrics_siamese` | `bootstrap_ci.recall.ci95_hi` | 0.5691 |
| `metrics_siamese` | `bootstrap_ci.f1.mean` | 0.5852 |
| `metrics_siamese` | `bootstrap_ci.f1.ci95_lo` | 0.4791 |
| `metrics_siamese` | `bootstrap_ci.f1.ci95_hi` | 0.6668 |
| `metrics_siamese` | `bootstrap_ci.iou.mean` | 0.4152 |
| `metrics_siamese` | `bootstrap_ci.iou.ci95_lo` | 0.315 |
| `metrics_siamese` | `bootstrap_ci.iou.ci95_hi` | 0.5001 |
| `metrics_siamese` | `inference_secs` | 4.7 |
| `metrics_siamese_f1` | `checkpoint` | siamese_best.pt |
| `metrics_siamese_f1` | `model` | siamese |
| `metrics_siamese_f1` | `split` | test |
| `metrics_siamese_f1` | `encoder` | resnet34 |
| `metrics_siamese_f1` | `trained_epoch` | 13 |
| `metrics_siamese_f1` | `best_val_f1` | 0.5547 |
| `metrics_siamese_f1` | `threshold` | 0.2 |
| `metrics_siamese_f1` | `threshold_selection.mode` | f1 |
| `metrics_siamese_f1` | `threshold_selection.selected_on` | validation |
| `metrics_siamese_f1` | `threshold_selection.objective` | F1 |
| `metrics_siamese_f1` | `threshold_selection.value` | 0.5583 |
| `metrics_siamese_f1` | `n_samples` | 38 |
| `metrics_siamese_f1` | `metrics.precision` | 0.7319 |
| `metrics_siamese_f1` | `metrics.recall` | 0.5108 |
| `metrics_siamese_f1` | `metrics.f1` | 0.6017 |
| `metrics_siamese_f1` | `metrics.iou` | 0.4303 |
| `metrics_siamese_f1` | `bootstrap_ci.precision.mean` | 0.7345 |
| `metrics_siamese_f1` | `bootstrap_ci.precision.ci95_lo` | 0.6556 |
| `metrics_siamese_f1` | `bootstrap_ci.precision.ci95_hi` | 0.8051 |
| `metrics_siamese_f1` | `bootstrap_ci.recall.mean` | 0.5115 |
| `metrics_siamese_f1` | `bootstrap_ci.recall.ci95_lo` | 0.3864 |
| `metrics_siamese_f1` | `bootstrap_ci.recall.ci95_hi` | 0.6112 |
| `metrics_siamese_f1` | `bootstrap_ci.f1.mean` | 0.6012 |
| `metrics_siamese_f1` | `bootstrap_ci.f1.ci95_lo` | 0.4972 |
| `metrics_siamese_f1` | `bootstrap_ci.f1.ci95_hi` | 0.6808 |
| `metrics_siamese_f1` | `bootstrap_ci.iou.mean` | 0.4313 |
| `metrics_siamese_f1` | `bootstrap_ci.iou.ci95_lo` | 0.3308 |
| `metrics_siamese_f1` | `bootstrap_ci.iou.ci95_hi` | 0.5161 |
| `metrics_siamese_f1` | `inference_secs` | 4.2 |
| `metrics_siamese_f2` | `checkpoint` | siamese_best.pt |
| `metrics_siamese_f2` | `model` | siamese |
| `metrics_siamese_f2` | `split` | test |
| `metrics_siamese_f2` | `encoder` | resnet34 |
| `metrics_siamese_f2` | `trained_epoch` | 13 |
| `metrics_siamese_f2` | `best_val_f1` | 0.5547 |
| `metrics_siamese_f2` | `threshold` | 0.02 |
| `metrics_siamese_f2` | `threshold_selection.mode` | f2 |
| `metrics_siamese_f2` | `threshold_selection.selected_on` | validation |
| `metrics_siamese_f2` | `threshold_selection.objective` | F2 |
| `metrics_siamese_f2` | `threshold_selection.value` | 0.5893 |
| `metrics_siamese_f2` | `n_samples` | 38 |
| `metrics_siamese_f2` | `metrics.precision` | 0.5983 |
| `metrics_siamese_f2` | `metrics.recall` | 0.5881 |
| `metrics_siamese_f2` | `metrics.f1` | 0.5932 |
| `metrics_siamese_f2` | `metrics.iou` | 0.4217 |
| `metrics_siamese_f2` | `bootstrap_ci.precision.mean` | 0.5998 |
| `metrics_siamese_f2` | `bootstrap_ci.precision.ci95_lo` | 0.5256 |
| `metrics_siamese_f2` | `bootstrap_ci.precision.ci95_hi` | 0.6685 |
| `metrics_siamese_f2` | `bootstrap_ci.recall.mean` | 0.5894 |
| `metrics_siamese_f2` | `bootstrap_ci.recall.ci95_lo` | 0.4498 |
| `metrics_siamese_f2` | `bootstrap_ci.recall.ci95_hi` | 0.6978 |
| `metrics_siamese_f2` | `bootstrap_ci.f1.mean` | 0.5929 |
| `metrics_siamese_f2` | `bootstrap_ci.f1.ci95_lo` | 0.4976 |
| `metrics_siamese_f2` | `bootstrap_ci.f1.ci95_hi` | 0.6659 |
| `metrics_siamese_f2` | `bootstrap_ci.iou.mean` | 0.4227 |
| `metrics_siamese_f2` | `bootstrap_ci.iou.ci95_lo` | 0.3312 |
| `metrics_siamese_f2` | `bootstrap_ci.iou.ci95_hi` | 0.4992 |
| `metrics_siamese_f2` | `inference_secs` | 4.1 |
| `rag_ablation` | `n_questions` | 10 |
| `uncertainty_baseline_test` | `model` | early_fusion |
| `uncertainty_baseline_test` | `checkpoint` | baseline_best.pt |
| `uncertainty_baseline_test` | `split` | test |
| `uncertainty_baseline_test` | `threshold` | 0.5 |
| `uncertainty_baseline_test` | `T` | 10 |
| `uncertainty_baseline_test` | `dropout_p` | 0.1 |
| `uncertainty_baseline_test` | `n` | 38 |
| `uncertainty_baseline_test` | `results.single.f1` | 0.4901 |
| `uncertainty_baseline_test` | `results.single.ece` | 0.0186 |
| `uncertainty_baseline_test` | `results.single.brier` | 0.02124 |
| `uncertainty_baseline_test` | `results.mc_dropout.f1` | 0.487 |
| `uncertainty_baseline_test` | `results.mc_dropout.ece` | 0.01133 |
| `uncertainty_baseline_test` | `results.mc_dropout.brier` | 0.02021 |
| `uncertainty_baseline_test` | `results.mc_dropout.uncertainty_error_ratio` | 5.196 |
| `uncertainty_baseline_test` | `results.mc_dropout.mean_uncertainty` | 0.01705 |
| `uncertainty_baseline_test` | `results.tta.f1` | 0.4665 |
| `uncertainty_baseline_test` | `results.tta.ece` | 0.01309 |
| `uncertainty_baseline_test` | `results.tta.brier` | 0.01902 |
| `uncertainty_baseline_test` | `results.tta.uncertainty_error_ratio` | 20.46 |
| `uncertainty_baseline_test` | `results.tta.mean_uncertainty` | 0.01203 |
| `uncertainty_baseline_test` | `note` | MC dropout applied at inference only (checkpoint trained without dropout). |
| `uncertainty_baseline_validation` | `model` | early_fusion |
| `uncertainty_baseline_validation` | `checkpoint` | baseline_best.pt |
| `uncertainty_baseline_validation` | `split` | validation |
| `uncertainty_baseline_validation` | `threshold` | 0.5 |
| `uncertainty_baseline_validation` | `T` | 10 |
| `uncertainty_baseline_validation` | `dropout_p` | 0.1 |
| `uncertainty_baseline_validation` | `n` | 44 |
| `uncertainty_baseline_validation` | `results.single.f1` | 0.5196 |
| `uncertainty_baseline_validation` | `results.single.ece` | 0.02467 |
| `uncertainty_baseline_validation` | `results.single.brier` | 0.02702 |
| `uncertainty_baseline_validation` | `results.mc_dropout.f1` | 0.5208 |
| `uncertainty_baseline_validation` | `results.mc_dropout.ece` | 0.01629 |
| `uncertainty_baseline_validation` | `results.mc_dropout.brier` | 0.02479 |
| `uncertainty_baseline_validation` | `results.mc_dropout.uncertainty_error_ratio` | 7.161 |
| `uncertainty_baseline_validation` | `results.mc_dropout.mean_uncertainty` | 0.01983 |
| `uncertainty_baseline_validation` | `results.tta.f1` | 0.5006 |
| `uncertainty_baseline_validation` | `results.tta.ece` | 0.01593 |
| `uncertainty_baseline_validation` | `results.tta.brier` | 0.02296 |
| `uncertainty_baseline_validation` | `results.tta.uncertainty_error_ratio` | 13.88 |
| `uncertainty_baseline_validation` | `results.tta.mean_uncertainty` | 0.01941 |
| `uncertainty_baseline_validation` | `note` | MC dropout applied at inference only (checkpoint trained without dropout). |
| `uncertainty_siamese_test` | `model` | siamese |
| `uncertainty_siamese_test` | `checkpoint` | siamese_best.pt |
| `uncertainty_siamese_test` | `split` | test |
| `uncertainty_siamese_test` | `threshold` | 0.5 |
| `uncertainty_siamese_test` | `T` | 10 |
| `uncertainty_siamese_test` | `dropout_p` | 0.1 |
| `uncertainty_siamese_test` | `n` | 38 |
| `uncertainty_siamese_test` | `results.single.f1` | 0.586 |
| `uncertainty_siamese_test` | `results.single.ece` | 0.01182 |
| `uncertainty_siamese_test` | `results.single.brier` | 0.01687 |
| `uncertainty_siamese_test` | `results.mc_dropout.f1` | 0.575 |
| `uncertainty_siamese_test` | `results.mc_dropout.ece` | 0.00619 |
| `uncertainty_siamese_test` | `results.mc_dropout.brier` | 0.01664 |
| `uncertainty_siamese_test` | `results.mc_dropout.uncertainty_error_ratio` | 5.707 |
| `uncertainty_siamese_test` | `results.mc_dropout.mean_uncertainty` | 0.01717 |
| `uncertainty_siamese_test` | `results.tta.f1` | 0.5862 |
| `uncertainty_siamese_test` | `results.tta.ece` | 0.00785 |
| `uncertainty_siamese_test` | `results.tta.brier` | 0.01522 |
| `uncertainty_siamese_test` | `results.tta.uncertainty_error_ratio` | 20.91 |
| `uncertainty_siamese_test` | `results.tta.mean_uncertainty` | 0.01 |
| `uncertainty_siamese_test` | `note` | MC dropout applied at inference only (checkpoint trained without dropout). |
| `uncertainty_siamese_validation` | `model` | siamese |
| `uncertainty_siamese_validation` | `checkpoint` | siamese_best.pt |
| `uncertainty_siamese_validation` | `split` | validation |
| `uncertainty_siamese_validation` | `threshold` | 0.5 |
| `uncertainty_siamese_validation` | `T` | 10 |
| `uncertainty_siamese_validation` | `dropout_p` | 0.1 |
| `uncertainty_siamese_validation` | `n` | 44 |
| `uncertainty_siamese_validation` | `results.single.f1` | 0.5547 |
| `uncertainty_siamese_validation` | `results.single.ece` | 0.01825 |
| `uncertainty_siamese_validation` | `results.single.brier` | 0.02321 |
| `uncertainty_siamese_validation` | `results.mc_dropout.f1` | 0.5488 |
| `uncertainty_siamese_validation` | `results.mc_dropout.ece` | 0.01028 |
| `uncertainty_siamese_validation` | `results.mc_dropout.brier` | 0.02204 |
| `uncertainty_siamese_validation` | `results.mc_dropout.uncertainty_error_ratio` | 6.982 |
| `uncertainty_siamese_validation` | `results.mc_dropout.mean_uncertainty` | 0.02072 |
| `uncertainty_siamese_validation` | `results.tta.f1` | 0.5616 |
| `uncertainty_siamese_validation` | `results.tta.ece` | 0.01101 |
| `uncertainty_siamese_validation` | `results.tta.brier` | 0.02012 |
| `uncertainty_siamese_validation` | `results.tta.uncertainty_error_ratio` | 16.51 |
| `uncertainty_siamese_validation` | `results.tta.mean_uncertainty` | 0.01536 |
| `uncertainty_siamese_validation` | `note` | MC dropout applied at inference only (checkpoint trained without dropout). |
| `vantor_stac_summary` | `event_id` | Kahramanmaras-turkey-earthquake-23 |
| `vantor_stac_summary` | `event_url` | https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/collection.json |
| `vantor_stac_summary` | `title` | Turkey and Syria Earthquake 2023 |
<!-- results:end -->

## Data Sources & Licenses

| Source | Use | License |
|--------|-----|---------|
| KATE-CD | Model training & evaluation | Citation required (Earthquake Damage Assessment with SAMCD) |
| Vantor (Maxar) Open Data | Georeferenced demo imagery | **CC BY-NC 4.0** — non-commercial only |
| HOTOSM Destroyed Buildings | Building-level validation labels | ODbL |
| OSM / HOTOSM Buildings & Roads | Building footprints, road network | ODbL |
| WorldPop Turkey 100m | Population estimation | CC BY 4.0 |
| AFAD TAMP | RAG (TR) | Public document |
| INSARAG Guidelines 2020 | RAG (EN) | Public document |

## Attribution

- **Imagery:** satellite imagery © Vantor (formerly Maxar), released under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) through the Maxar Open Data Program.
  Non-commercial use only.
- **Map data:** © OpenStreetMap contributors and Humanitarian OpenStreetMap Team (HOTOSM), ODbL.
- **Population:** WorldPop, CC BY 4.0.
- **Training data:** KATE-CD; please cite *Earthquake Damage Assessment with SAMCD* (see `cscrs/kate-cd`).

## Documentation

See [`docs/README.md`](docs/README.md) for the documentation map: decisions, data card, model card, glossary.
Cite this repository via [`CITATION.cff`](CITATION.cff).

## Setup

```bash
# Install uv (https://docs.astral.sh/uv/)
uv sync --all-extras  # plain `uv sync` drops the torch/agent extras
cp .env.example .env  # fill in your API keys
uv run pytest -q
```

## Limitations

- Population estimates are approximations (WorldPop 100m, pre-earthquake).
- Model trained on a single earthquake dataset (KATE-CD); generalization not guaranteed.
- Model accuracy is reported on KATE-CD only. The Kilis Vantor crop is a pipeline demo (different sensors/off-nadir angles, almost no damage labels); Antakya building damage uses HOTOSM destroyed-building labels as *simulated* detector output.
- Vantor imagery is CC BY-NC 4.0 — commercial use requires a different imagery source.
- INSARAG Guidelines 2020 used; new version effective January 2027.
