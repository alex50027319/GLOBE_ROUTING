"""Remote bootstrapper script for Lite-GLOBE executions on Google Colab."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main() -> int:
    print("=== Colab Bootstrapper Started ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")

    # 1. Load arguments/configuration from colab_args.json
    args_file = Path("colab_args.json")
    if not args_file.exists():
        print("Error: colab_args.json not found in the current directory.")
        return 1

    with open(args_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    bundle_filename = config.get("bundle_filename")
    run_module = config.get("run_module", "implementations.lite_globe.run_phase13")
    script_args = config.get("args", [])
    output_dir_str = config.get("output_dir", "artifacts/lite_globe/phase13")
    results_zip_filename = config.get("results_zip", "phase13_results.zip")
    resume_archive_filename = config.get("resume_archive")

    # 2. Find and extract the workspace bundle
    bundle_path = Path(bundle_filename) if bundle_filename else None
    if not bundle_path or not bundle_path.exists():
        # Fallback: look for any zip bundle in the current directory
        zip_files = list(Path(".").glob("*_colab_bundle.zip"))
        if zip_files:
            bundle_path = zip_files[0]
            print(f"Auto-detected bundle: {bundle_path}")
        else:
            print("Error: No zip bundle found to extract.")
            return 1

    print(f"Extracting {bundle_path}...")
    with zipfile.ZipFile(bundle_path, "r") as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete.")

    # A previous short-session ZIP contains the checkpoint/progress tree under
    # its original workspace-relative path. Overlay it after the clean bundle.
    if resume_archive_filename:
        resume_archive = Path(resume_archive_filename)
        if not resume_archive.is_file():
            print(f"Error: resume archive not found: {resume_archive}")
            return 1
        print(f"Restoring resumable state from {resume_archive}...")
        with zipfile.ZipFile(resume_archive, "r") as zip_ref:
            zip_ref.extractall(".")
        print("Resume state restored.")

    # 3. Pip install dependencies
    requirements_file = Path("requirements-lite-globe.txt")
    if requirements_file.exists():
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
            check=True,
        )
    else:
        print("Warning: requirements-lite-globe.txt not found. Skipping dependency installation.")

    # Install the workspace in editable mode
    if Path("pyproject.toml").exists():
        print("Installing workspace package in editable mode...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
            check=True,
        )
    else:
        print("Warning: pyproject.toml not found. Skipping package installation.")

    # 4. Execute the main run module
    cmd = [sys.executable, "-m", run_module] + script_args
    print(f"Executing remote command: {' '.join(cmd)}")
    sys.stdout.flush()
    sys.stderr.flush()

    try:
        subprocess.run(cmd, check=True)
        print("Remote command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Remote command exited with code {e.returncode}")
        return e.returncode

    # 5. Zip output results for downloading
    output_dir = Path(output_dir_str)
    if output_dir.exists():
        print(f"Zipping results from {output_dir} into {results_zip_filename}...")
        with zipfile.ZipFile(
            results_zip_filename, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as results_zip:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = Path(root) / file
                    results_zip.write(file_path, file_path.relative_to(Path(".")))
        print(f"Created {results_zip_filename}")
    else:
        print(f"Warning: Output directory {output_dir} not found. Nothing to zip.")

    print("=== Colab Bootstrapper Finished Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
