---
name: run-external-comparison
description: Preflight and execute fair, resumable SwitchGLOBE external-baseline comparisons, including seed chunks and checkpoint validation.
---

# Run External Comparison

1. Read the external config, baseline registry, and simulation protocol.
2. Verify one SwitchGLOBE checkpoint per requested seed.
3. Pass the current checkpoint path explicitly; do not rely on the legacy CLI default.
4. Use `--seed` repeatedly for independent chunks and `--resume` for recoverability.
5. Keep smoke and full output trees separate.

Example:

```bash
python -m implementations.lite_globe.run_external_comparison \
  --device auto --resume --seed 42 \
  --switchglobe-checkpoint-dir artifacts/switchglobe/final/checkpoints \
  --output-dir artifacts/external_comparison
```

Run training/evaluation only when requested. Afterward, validate method registry coverage,
paired evaluation seeds, expected rows, manifest completeness, and deployment-cost outputs.
