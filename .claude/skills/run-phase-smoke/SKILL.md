---
name: run-phase-smoke
description: Run the code-level smoke pytest suite for a given Lite-GLOBE phase number (not the checkpoint-dependent run_phaseN.py --smoke CLI flag). Use when the user asks to smoke test, sanity check, or verify a phase's code without needing trained checkpoints.
---

Given a phase number N (default: all phases if none given):

1. Run:
   ```bash
   pytest tests/lite_globe/test_phase${N}_*.py -v
   ```
   If no test file matches that exact pattern, list `tests/lite_globe/` and
   pick the closest match (e.g. phase 13's file is
   `test_phase13_risk_switch_plus.py`).

2. Report pass/fail counts plainly.

3. If the user actually wants the full `run_phaseN.py --smoke` training/eval
   pipeline (not just code verification), tell them it needs upstream
   checkpoints at `artifacts/lite_globe/phase{8,11,12}/checkpoints/seed_*/*.pt`
   depending on the phase, and check whether those exist before running it.

4. Never run `scripts/colab_run.py` as part of this skill — that provisions a
   paid Colab GPU/TPU session and needs explicit separate confirmation.
