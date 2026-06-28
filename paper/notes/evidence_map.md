# Evidence Map

| Claim | Current Evidence | File(s) | Status |
| --- | --- | --- | --- |
| Phase 12 achieves best overall PDR and deadline delivery among implemented methods | 14 scenarios, 5 seeds, merged external baseline table | `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Phase_12_Full_Result_Analysis.md`, `ResearchAIWorkspace/artifacts/lite_globe/phase12/tables/risk_switch_results.md` | Supported |
| Phase 12 reduces delay p95 vs Predictive Geographic, Evo-QGeo, IQMR, and DRAMA | Overall table | Same as above | Supported |
| Phase 12 uses fewer input bytes than Predictive Geographic, Evo-QGeo, and DRAMA | Overall table | Same as above | Supported as input-footprint proxy |
| Global teacher is offline/training-time only | Phase 3/4 code and notes | `run_phase3.py`, `run_phase4.py`, `teacher_gnn.py`, `distillation.py` | Supported |
| Deployed student uses local observations | Student policy and adapter code | `student_policy.py`, `policy_adapter.py`, observation code | Supported |
| Phase 13 P+ improves over Phase 12 at full scale | Only smoke-tested | `ResearchAIWorkspace/artifacts/lite_globe/phase13_smoke` | Not yet supported |
| Link-loss gate, energy tie, drop suppression each improve performance | Planned ablations exist | Phase 13 code and notes | Needs full ablation |
| Method has lower communication overhead than DRAMA | Input bytes are lower; explicit control bytes absent | Phase 12 tables | Partially supported |
| Method beats AODV/OLSR | Baselines not implemented | None | Not supported |
| Method improves real radio energy use | Energy proxy only | Phase 12 statistics | Partially supported |
| Method architecture diagram is paper-appropriate | Replaced presentation infographic with grayscale vector PDF | `paper/figures/method_overview_pub.pdf` | Supported |

## Interpretation Rule

Use strong wording only for supported Phase 12 results. Use cautious wording for Phase 13 P+ until full results are available.
