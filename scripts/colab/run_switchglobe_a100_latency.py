"""Run and package the current-commit SwitchGLOBE A100 latency benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch


CONTENT = Path("/content")
ROOT = CONTENT / "SwitchGLOBE_globev2"
CHECKPOINT_ROOT = CONTENT / "switchglobe_checkpoints"
OUTPUT = CONTENT / "switchglobe_globev2_a100_20260905"
ARCHIVE = CONTENT / "switchglobe_globev2_a100_20260905.zip"
SOURCE_COMMIT = "29a8b4f768fc573cd273c7ea5dba463de17b80eb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "NONE"
    hardware = {
        "requested_accelerator": "A100",
        "cuda_available": gpu_available,
        "gpu_name": gpu_name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "source_git_commit": SOURCE_COMMIT,
        "torch_num_threads": 1,
        "batch_size": 1,
    }
    (OUTPUT / "hardware.json").write_text(
        json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not gpu_available or "A100" not in gpu_name.upper():
        raise RuntimeError(f"A100 verification failed: cuda={gpu_available}, name={gpu_name!r}")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    command = [
        sys.executable, "-m", "implementations.lite_globe.run_latency_benchmark",
        "--checkpoint-dir", str(CHECKPOINT_ROOT / "phase12"),
        "--fast-checkpoint-dir", str(CHECKPOINT_ROOT / "fast"),
        "--output-dir", str(OUTPUT),
        "--include-fast", "--include-early-exit",
        "--warmup", "50", "--repeats", "2000", "--zip-results",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True,
    )
    (OUTPUT / "benchmark_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (OUTPUT / "benchmark_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, command, completed.stdout, completed.stderr
        )

    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "execution_backend": "google_colab_cli",
        "requested_accelerator": "A100",
        "verified_gpu_name": gpu_name,
        "source_git_commit": SOURCE_COMMIT,
        "torch_num_threads": 1,
        "batch_size": 1,
        "elapsed_seconds": time.time() - started,
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True, stdout=(OUTPUT / "pip_freeze.txt").open("w", encoding="utf-8"),
        check=True,
    )

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", root_dir=OUTPUT)
    print(json.dumps({
        "complete": True,
        "archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "gpu_name": gpu_name,
        "elapsed_seconds": time.time() - started,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
