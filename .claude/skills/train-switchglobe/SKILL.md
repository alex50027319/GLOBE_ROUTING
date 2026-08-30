---
name: train-switchglobe
description: Preflight and execute the reproducible Teacher-to-SwitchGLOBE training lineage when the user requests training or a pipeline smoke run.
---

# Train SwitchGLOBE

Read `scripts/train_switchglobe_pipeline.py` and `config/switchglobe.yaml` before execution.

Preflight:

- confirm device, smoke/full mode, `--start-at`, `--stop-after`, and artifacts root;
- verify upstream checkpoints when starting after `teacher`;
- run focused code tests;
- show the four stage output directories.

Canonical commands:

```bash
python scripts/train_switchglobe_pipeline.py --device cpu --smoke --resume
python scripts/train_switchglobe_pipeline.py --device auto --resume
```

Run only when the user asked to execute training. Preserve partial stages with `--resume` and do
not call a smoke result full. Verify final manifest and all five seeds before reporting completion.
