from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path("/content/SwitchGLOBE_model_selection")
root.mkdir(parents=True, exist_ok=True)
for archive in (
    Path("/content/switchglobe_source.zip"),
    Path("/content/switchglobe_checkpoints.zip"),
):
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "gymnasium>=1.0", "pyyaml>=6.0"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(root)],
    check=True,
)
import torch
payload = {
    "root": str(root),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
if not torch.cuda.is_available() or "A100" not in str(payload["cuda_device"]):
    raise RuntimeError(f"A100 validation failed: {payload}")
print(json.dumps(payload, indent=2))
