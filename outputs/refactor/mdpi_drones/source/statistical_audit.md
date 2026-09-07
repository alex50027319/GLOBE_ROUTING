# Statistical audit

Primary design: 5 training seeds × 14 scenarios × 200 episodes. Episode outcomes are aggregated by seed and scenario; scenario means are then macro-averaged within seed. Confidence intervals use the five independent seed summaries.

Latency: warm-up 50, 2,000 repeats for each seed/variant/component, batch 1, one CPU thread, synchronized CUDA boundaries. The primary value is the mean of seed-specific p95 statistics. Paired effects use matched checkpoints. Bootstrap uses a deterministic seed. Exact sign-flip enumeration has 32 assignments, so the smallest two-sided p-value is 0.0625.

Known aggregation conflict: `full/analysis_8method_and_ablation/csv/external_overall_metrics.csv` pools/weights data differently from `synthesis/combined_comparison/summaries/statistics_overall.csv`. This changes connected PDR slightly and changes failure-penalized energy materially. The manuscript consistently uses the synthesis scenario-macro table. Alternate values are not cherry-picked.

Fast acceptance gate: the 95% lower confidence bound for connected PDR and deadline ratio must be at least −0.005. Fast connected-PDR change is −0.01809 with CI [−0.03606, −0.00011], so it fails. The conclusion does not depend on a p-value threshold.

Remaining weaknesses: five seeds, aggregated rather than raw A100 timing traces, non-randomized variant order not documented in the primary archive, and a dirty source state recorded by the synthesis manifest.

