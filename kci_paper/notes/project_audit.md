# Project Audit Report

## 1. Current Working Directory
- Path: `/Users/alex/Documents/GLOBE_ROUTING`

## 2. Detected Project Structure
- `paper/`: contains current IEEE LaTeX skeleton, section sources, references library, and compiled graphics.
- `paper_kci/`: contains KCI Korean draft utilizing standard `kotex` and KICS layout conventions.
- `ResearchAIWorkspace/`: contains Obsidian vault database, python scripts, implementation sources, test cases, and simulation artifacts.
- `kci_paper/`: new folder allocated for this integrated KCI submission package.

## 3. Found Source Code
- `ResearchAIWorkspace/implementations/lite_globe/`:
  - `env/`: contains gymnasium-style custom FANET network environment definitions.
  - `models/`: GNN actor-critic models (`teacher_gnn.py`), MLP local student models (`student_mlp.py`), and risk PRS models.
  - `utils/`: geometric calculators, Kalman filter based link lifetime predictors, and logging trackers.
  - `run_phase1.py` through `run_phase13.py`: orchestration run-scripts for all stages of learning, distillation, optimization, and evaluation.

## 4. Found Simulation Scripts
- `run_phase1.py` (Gym environment & GPSR)
- `run_phase3.py` (Teacher MAPPO rollout)
- `run_phase4.py` (Dataset collection & KL distillation)
- `run_phase5.py` (Local fine-tuning interface)
- `run_phase9.py` (Risk-aware validation runs)
- `run_phase12.py` / `run_phase13.py` (Full statistical evaluations and P+ validation)

## 5. Found Model Implementations
- Global teacher model with 2-layer Edge-conditioned Message Passing GNN.
- Local student model with permutation-invariant 2-layer MLP and action masking.
- Predictive Risk Switch logic incorporating RSSI margin, queue length, current-link lifetime, and onward k-stability.

## 6. Found Evaluation Scripts
- `run_phase6.py` (multi-seed evaluation)
- `run_phase12.py` (Phase 12 full-scale baseline comparisons)

## 7. Found Experimental Results
- `ResearchAIWorkspace/artifacts/lite_globe/phase12/tables/overall_results.tex`
- CSV statistics summaries for PDR, delay-p95, and input control bytes under `ResearchAIWorkspace/artifacts/lite_globe/phase12/summaries/`.

## 8. Found Figures and Visualizations
- `paper/figures/phase12_pdr.png`
- `paper/figures/phase12_delay_p95.png`
- `paper/figures/phase12_input_bytes.png`
- `paper/figures/method_overview_pub.pdf`

## 9. Found Obsidian / Markdown Research Notes
- `ResearchAIWorkspace/vault/04_GLOBE_PlusPlus/` contains:
  - `novelty_claims.md` (claims verification)
  - `problem_formulation.md` (Dec-POMDP outlines)
  - `reviewer_attack_list.md` (critic defensive strategies)

## 10. Found Existing Paper Drafts
- LaTeX drafts located inside `paper/` and `paper_kci/` (compiled successfully to PDF).

## 11. Found References
- BibTeX references configured in `paper/references.bib` and `paper_kci/references.bib`.

## 12. Missing Critical Materials
- **Phase 13 P+ statistical sweep**: While the code is fully implemented and smoke tested, the final 5-seed statistics values for Phase 13 P+ (such as PDR improvements under link-loss) are not yet fully generated in the tables. This will be marked as a future TODO.

## 13. What Can Be Used for KCI Paper
- Dec-POMDP formulation equations.
- Global PPO training loss formulas and soft-KL distillation formulation.
- Multi-seed comparison tables and plots for PDR, latency, and bytes overhead.
- Systematic descriptions of the Danger Score, Safety Utility, and PRS switcher logic.

## 14. What Cannot Yet Be Claimed
- Superiority of the P+ extension over Evo-QGeo under heavy link-loss at full-scale statistical sweep (only Phase 12 results can be strictly claimed).
