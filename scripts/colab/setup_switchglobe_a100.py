"""Prepare a self-contained SwitchGLOBE checkout on a Colab runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys


CONTENT = Path("/content")
ROOT = CONTENT / "SwitchGLOBE_globev2"
CHECKPOINT_ROOT = CONTENT / "switchglobe_checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    source_archive = CONTENT / "switchglobe_globev2_source.zip"
    exact_archive = CONTENT / "phase12_checkpoints.tar.gz"
    fast_archive = CONTENT / "fast_checkpoints.tar.gz"
    for path in (source_archive, exact_archive, fast_archive):
        if not path.is_file():
            raise FileNotFoundError(path)

    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    (CHECKPOINT_ROOT / "phase12").mkdir(parents=True)
    (CHECKPOINT_ROOT / "fast").mkdir(parents=True)

    run(["unzip", "-q", str(source_archive), "-d", str(ROOT)])
    run([
        "tar", "-xzf", str(exact_archive), "-C",
        str(CHECKPOINT_ROOT / "phase12"), "--strip-components=2",
    ])
    run([
        "tar", "-xzf", str(fast_archive), "-C",
        str(CHECKPOINT_ROOT / "fast"), "--strip-components=1",
    ])
    # Archives produced on macOS may contain AppleDouble ``._*`` sidecars.
    # They are not checkpoints and should not inflate the integrity count.
    for sidecar in CHECKPOINT_ROOT.rglob("._*"):
        sidecar.unlink()

    # Colab already supplies CUDA-enabled torch/numpy/matplotlib. Install only
    # the small missing runtime packages, then install this checkout without
    # changing the pre-provisioned torch build.
    run([sys.executable, "-m", "pip", "install", "-q", "gymnasium>=1.0", "pyyaml>=6.0"])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], cwd=ROOT)

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("GPU:", gpu)
    print("Python:", sys.version.replace("\n", " "))
    for path in (source_archive, exact_archive, fast_archive):
        print(f"SHA256 {path.name}: {sha256(path)}")
    print("Exact checkpoint files:", len(list(
        (CHECKPOINT_ROOT / "phase12").glob("seed_*/risk_switch_lite_globe_p.pt")
    )))
    print("Fast checkpoint files:", len(list(
        (CHECKPOINT_ROOT / "fast").glob("seed_*/fast_switchglobe.pt")
    )))


if __name__ == "__main__":
    main()
