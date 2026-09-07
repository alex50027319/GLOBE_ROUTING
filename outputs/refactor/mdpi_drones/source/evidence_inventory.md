# Evidence inventory

Cutoff date: 2026-09-05. Primary branch: `refactor/globev2`.

| Evidence | Source | Status | Paper use |
|---|---|---|---|
| 8-method routing outcomes | `artifacts/final_paper_simulation/synthesis/combined_comparison/` | Verified complete by manifest; 112,000 episode rows | Primary routing result |
| Exact--Fast gate | `artifacts/final_paper_simulation/full/ablation/validation/fast_vs_exact_overall.csv` | Five paired seeds | Non-inferiority decision |
| Current A100 timing | `artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/results/` | Same session, five checkpoints | Primary decision-latency result |
| A100 operator profile | `.../globev2_colab_cli_20260905/profile/results/` | Seed 42, 30 calls | Bottleneck attribution only |
| Exact implementation | `implementations/lite_globe/models/student_policy.py` | `SwitchGlobePolicy` alias and fused `decide()` inspected | Equations and Algorithm 1 |
| Research lineage | `docs/method_history.md` | Audited; overbroad universal-dominance sentence not adopted | Method history |
| Latency conclusions | `docs/latency_optimization/final_report.md` | Primary synthesis | Tables and limitations |
| Literature | live Notion database + `literature_comparison.csv` | 45 accessible records | Related work and metric audit |
| Official class | `Definitions/mdpi.cls` | SHA-256 `e389b10d10a5a26d2a7e9d86e7dc8b74cf5007049487e05d3a5d661244872590`; header 13 March 2026 | Drones layout |

The routing synthesis manifest records a dirty source state. Internal row validation is complete, but a clean-tag rerun is required for the strongest public reproducibility claim.

