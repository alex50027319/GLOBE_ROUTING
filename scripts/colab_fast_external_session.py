"""Run one FastSwitchGLOBE full seed inside an already provisioned Colab VM."""

from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import zipfile


SEEDS = {42, 77, 123, 314, 2718}
bundle = Path("/content/fast_external_comparison_colab_bundle.zip")
seed_path = Path("/content/fast_external_seed.txt")
source = Path("/content/SwitchGLOBE_fast_external")
output = Path("/content/fast_external_comparison")

if not bundle.is_file() or not seed_path.is_file():
    raise FileNotFoundError("bundle and seed file must be uploaded before execution")
seed = int(seed_path.read_text(encoding="utf-8").strip())
if seed not in SEEDS:
    raise ValueError(f"unsupported seed {seed}; expected one of {sorted(SEEDS)}")

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if "A100" not in gpu:
    raise RuntimeError(f"A100 required, found {gpu}")
print(f"GPU VERIFIED: {gpu}", flush=True)

marker = source / ".fast_external_installed"
bundle_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != bundle_sha256:
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as archive:
        unsafe = [
            name for name in archive.namelist()
            if Path(name).is_absolute() or ".." in Path(name).parts
        ]
        if unsafe:
            raise ValueError(f"unsafe bundle paths: {unsafe[:3]}")
        archive.extractall(source)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-lite-globe.txt"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
        cwd=source,
        check=True,
    )
    marker.write_text(bundle_sha256, encoding="utf-8")

command = [
    sys.executable,
    "-m",
    "implementations.lite_globe.run_fast_external_comparison",
    "--device",
    "cuda",
    "--seed",
    str(seed),
    "--resume",
    "--pretrained-fast-checkpoint-dir",
    "artifacts/final_paper_simulation/full/ablation/fast_training/checkpoints",
    "--zip-results",
    "--output-dir",
    str(output),
]
print("+ " + " ".join(command), flush=True)
completed = subprocess.run(
    command, cwd=source, capture_output=True, text=True
)
if completed.stdout:
    print(completed.stdout, flush=True)
if completed.stderr:
    print(completed.stderr, file=sys.stderr, flush=True)
completed.check_returncode()
result = output / "full" / f"fast_seeds_{seed}.zip"
if not result.is_file():
    raise FileNotFoundError(result)
print(f"RESULT_READY={result}", flush=True)
