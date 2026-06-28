# Workspace Audit

## Found Code

- `ResearchAIWorkspace/implementations/lite_globe/run_phase12.py`
- `ResearchAIWorkspace/implementations/lite_globe/run_phase13.py`
- `ResearchAIWorkspace/implementations/lite_globe/experiments/phase12_campaign.py`
- `ResearchAIWorkspace/implementations/lite_globe/experiments/phase13_campaign.py`
- `ResearchAIWorkspace/implementations/lite_globe/models/student_policy.py`
- `ResearchAIWorkspace/implementations/lite_globe/models/teacher_gnn.py`
- `ResearchAIWorkspace/implementations/lite_globe/algorithms/ppo.py`
- `ResearchAIWorkspace/implementations/lite_globe/algorithms/distillation.py`

## Found Simulation Results

- Full Phase 12 results:
  - `ResearchAIWorkspace/artifacts/lite_globe/phase12/raw/episodes.csv`
  - `ResearchAIWorkspace/artifacts/lite_globe/phase12/raw/seed_summaries.csv`
  - `ResearchAIWorkspace/artifacts/lite_globe/phase12/summaries/statistics.csv`
  - `ResearchAIWorkspace/artifacts/lite_globe/phase12/summaries/paired_effects.csv`
  - `ResearchAIWorkspace/artifacts/lite_globe/phase12/tables/risk_switch_results.md`
  - `ResearchAIWorkspace/artifacts/lite_globe/phase12/tables/risk_switch_paired_effects.md`
- Phase 13 smoke results:
  - `ResearchAIWorkspace/artifacts/lite_globe/phase13_smoke`

## Found Figures

- `ResearchAIWorkspace/artifacts/lite_globe/phase12/figures/risk_switch_pdr.png`
- `ResearchAIWorkspace/artifacts/lite_globe/phase12/figures/risk_switch_delay_p95.png`
- `ResearchAIWorkspace/artifacts/lite_globe/phase12/figures/risk_switch_input_bytes.png`
- `paper/figures/method_overview_pub.pdf` generated as a publication-style replacement for the original presentation infographic

## Found Research Notes

- `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/00_Overview.md`
- `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Phase_12_Full_Result_Analysis.md`
- `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Phase_13_Risk_Switch_Lite_GLOBE_P_Plus.md`
- `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Comparison_Methods_Guide.md`
- `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Professor_Presentation_2026-06-28.md`

## Missing Critical Evidence

- Full Phase 13 P+ multi-seed validation.
- Full Phase 13 component ablation analysis.
- Explicit control-byte overhead per delivered packet.
- AODV and OLSR baselines under the same simulator, if the paper wants to claim comparison against classical MANET protocols.
- Complete BibTeX metadata for local PDF papers.
- Realistic radio/energy model validation beyond the current proxy.

## Usable for Paper

- Phase 12 full-run table and figures.
- Phase 12 scenario-level interpretation.
- Phase 13 method equations and implemented design.
- Teacher-student KD formulation from existing implementation notes.
- Baseline list for currently implemented methods.

## Not Yet Usable

- Claims that Phase 13 P+ is fully superior to all baselines.
- Claims against AODV/OLSR.
- Claims of real-world energy savings.
- Claims of online communication-free superiority without explicit control-byte measurement.
