# Claims and evidence ledger

| Manuscript claim | Evidence file | Verification |
|---|---|---|
| Exact connected PDR = 0.90529 | `tables/statistics_overall.csv` | Direct row |
| Exact deadline ratio = 0.83764 | same | Direct row |
| Exact vs Evo connected improvement = 0.01841 | `tables/paired_effects_overall.csv` | Direct paired row |
| Fused CUDA p95 = 5.836 ms | `tables/a100_runtime_benchmarks.csv` plus final report | Five-seed summary |
| Legacy CUDA p95 = 10.004 ms | same | Five-seed summary |
| Fusion reduction = 41.66% | `(10.004249−5.836040)/10.004249` | Recalculated |
| CPU p95 = 2.208 ms | same | Five-seed summary |
| Fast CUDA p95 = 2.005 ms | same | Five-seed summary |
| Fast PDR degradation = −0.01809 | `full/ablation/validation/fast_vs_exact_overall.csv` | Direct paired row |
| Early Exit skip = 74.2747%, mismatches = 0 | final latency report and figure data | 43,467 decisions / 14,000 episodes |
| Exact DtoH scalar events = 1,050 | `tables/a100_exact_operators.csv` | Operator count, 30 calls |

