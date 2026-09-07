from __future__ import annotations

from pathlib import Path
import subprocess
import sys

root = Path("/content/SwitchGLOBE_model_selection")
output = Path("/content/model_selection_latency")
command = [
    sys.executable, "-m", "implementations.lite_globe.run_model_selection_latency",
    "--phase7-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase7/checkpoints"),
    "--phase8-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase8/checkpoints"),
    "--phase11-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase11/checkpoints"),
    "--phase12-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"),
    "--fast-dir", str(root / "artifacts/final_paper_simulation/full/ablation/fast_training/checkpoints"),
    "--output-dir", str(output), "--warmup", "50", "--repeats", "2000",
]
completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
Path("/content/model_selection_latency.log").write_text(
    completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
    encoding="utf-8",
)
if completed.returncode != 0:
    raise RuntimeError("latency benchmark failed; see /content/model_selection_latency.log")
import shutil
archive = shutil.make_archive(str(output), "zip", root_dir=output)
print(archive)
