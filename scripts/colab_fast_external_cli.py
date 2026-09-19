"""Colab-CLI cell: install the uploaded bundle and run one requested full seed.

Before executing this file with ``colab exec -f``, upload:
  /content/fast_external_comparison_colab_bundle.zip
  /content/fast_external_seed.txt
and mount Drive at /content/drive.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ALLOWED_SEEDS = {42, 77, 123, 314, 2718}
BUNDLE = Path("/content/fast_external_comparison_colab_bundle.zip")
SEED_FILE = Path("/content/fast_external_seed.txt")
SOURCE = Path("/content/SwitchGLOBE_colab")
DRIVE_OUTPUT = Path("/content/drive/MyDrive/SwitchGLOBE/fast_external_comparison")


def run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


if not BUNDLE.is_file():
    raise FileNotFoundError(f"upload bundle first: {BUNDLE}")
if not SEED_FILE.is_file():
    raise FileNotFoundError(f"upload seed file first: {SEED_FILE}")
seed = int(SEED_FILE.read_text(encoding="utf-8").strip())
if seed not in ALLOWED_SEEDS:
    raise ValueError(f"seed must be one of {sorted(ALLOWED_SEEDS)}, got {seed}")
if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Google Drive is not mounted at /content/drive")

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
print(f"GPU: {gpu}", flush=True)
if "A100" not in gpu:
    raise RuntimeError(f"A100 required by this manual, found: {gpu}")

if SOURCE.exists():
    shutil.rmtree(SOURCE)
SOURCE.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as archive:
    unsafe = [
        name
        for name in archive.namelist()
        if Path(name).is_absolute() or ".." in Path(name).parts
    ]
    if unsafe:
        raise ValueError(f"unsafe bundle paths: {unsafe[:3]}")
    archive.extractall(SOURCE)

run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-lite-globe.txt", cwd=SOURCE)
run(sys.executable, "-m", "pip", "install", "-q", "-e", ".", cwd=SOURCE)
run(
    sys.executable,
    "-m",
    "implementations.lite_globe.run_fast_external_comparison",
    "--device",
    "cuda",
    "--seed",
    str(seed),
    "--resume",
    "--zip-results",
    "--output-dir",
    str(DRIVE_OUTPUT),
    cwd=SOURCE,
)

result = DRIVE_OUTPUT / "full" / f"fast_seeds_{seed}.zip"
if not result.is_file():
    raise FileNotFoundError(f"expected result ZIP was not created: {result}")
print(f"FAST EXTERNAL FULL SEED {seed} COMPLETE: {result}", flush=True)
