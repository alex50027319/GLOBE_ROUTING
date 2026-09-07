from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

root = Path("/content/SwitchGLOBE_model_selection")
common = [
    "--episodes", "200", "--device", "cuda",
    "--phase7-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase7/checkpoints"),
    "--phase8-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase8/checkpoints"),
    "--phase11-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase11/checkpoints"),
    "--phase12-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"),
    "--fast-dir", str(root / "artifacts/final_paper_simulation/full/ablation/fast_training/checkpoints"),
    "--zip-results",
]
status = []
for seed in (42, 77, 123, 314, 2718):
    output = Path(f"/content/model_selection_seed_{seed}")
    archive = output.with_suffix(".zip")
    if archive.is_file():
        status.append({"seed": seed, "status": "already_complete", "zip": str(archive)})
        continue
    command = [
        sys.executable, "-m", "implementations.lite_globe.run_model_selection_ablation",
        "--seed", str(seed), "--output-dir", str(output), *common,
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    Path(f"/content/model_selection_seed_{seed}.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"seed {seed} failed; see /content/model_selection_seed_{seed}.log")
    status.append({"seed": seed, "status": "complete", "zip": str(archive)})
    print(json.dumps(status[-1]), flush=True)
print(json.dumps({"complete": True, "seeds": status}, indent=2))
