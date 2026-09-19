---
name: verify-switchglobe
description: Run focused or full code-level verification for SwitchGLOBE without starting checkpoint-dependent training.
---

# Verify SwitchGLOBE

Choose the smallest test scope that covers the request:

- final policy: `python -m pytest tests/lite_globe/test_switchglobe.py`
- environment: `python -m pytest tests/lite_globe/test_environment.py`
- external comparison: `python -m pytest tests/lite_globe/test_external_comparison.py tests/lite_globe/test_external_baselines.py`
- full suite: `python -m pytest tests/lite_globe`

Report command, pass/fail count, elapsed time, and failing test names. This skill is code-level
verification; do not substitute `train_switchglobe_pipeline.py --smoke`, which executes a reduced
training lineage and needs a separate request.
