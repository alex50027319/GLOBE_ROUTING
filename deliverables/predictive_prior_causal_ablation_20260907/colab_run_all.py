from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

root = Path("/content/PredictivePriorCausalAblation")
status = []
for seed in (42, 77, 123, 314, 2718):
    output = Path(f"/content/predictive_prior_causal_seed_{seed}")
    archive = output.with_suffix(".zip")
    if archive.is_file():
        status.append({"seed": seed, "status": "already_complete", "zip": str(archive)})
        continue
    command = [
        sys.executable, "-m", "implementations.lite_globe.run_predictive_prior_causal_ablation",
        "--seed", str(seed),
        "--teacher-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase7/checkpoints"),
        "--legacy-dir", str(root / "ResearchAIWorkspace/artifacts/lite_globe/phase11/checkpoints"),
        "--output-dir", str(output), "--device", "cuda",
        "--train-episodes-per-scenario-rollout", "100",
        "--evaluation-episodes", "200", "--epochs", "120",
        "--batch-size", "512", "--learning-rate", "0.005",
        "--warmup", "50", "--latency-repeats", "2000", "--zip-results",
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    Path(f"/content/predictive_prior_causal_seed_{seed}.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"seed {seed} failed; see /content/predictive_prior_causal_seed_{seed}.log")
    status.append({"seed": seed, "status": "complete", "zip": str(archive)})
    print(json.dumps(status[-1]), flush=True)
print(json.dumps({"complete": True, "seeds": status}, indent=2))
