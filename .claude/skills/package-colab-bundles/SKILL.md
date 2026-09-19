---
name: package-colab-bundles
description: Build and verify SwitchGLOBE or external-comparison Colab ZIP bundles without starting a remote Colab session.
---

# Package Colab Bundles

Select the package matching the request:

```bash
python scripts/package_switchglobe_colab.py
python scripts/package_external_comparison_colab.py \
  --checkpoint-dir artifacts/switchglobe/final/checkpoints
```

Before packaging, inspect required inputs and confirm that the external bundle has all five
SwitchGLOBE checkpoints. After packaging, run `unzip -t` and inspect the archive list for secrets,
caches, unrelated artifacts, and missing README/config files. Packaging does not authorize remote
upload or GPU execution.
