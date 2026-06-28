"""Orchestrate Google Colab execution for Lite-GLOBE phases using google-colab-cli."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Lite-GLOBE phase experiments on Google Colab using colab-cli"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=13,
        help="Lite-GLOBE phase number to execute (default: 13)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="T4",
        help="GPU accelerator variant (T4, L4, G4, H100, A100). Default: T4",
    )
    parser.add_argument(
        "--tpu",
        type=str,
        default=None,
        help="TPU accelerator variant (v5e1, v6e1). Overrides --gpu if specified.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU/TPU.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Colab session name. Defaults to colab-routing-phase<phase>.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the Colab session alive after running (do not stop it).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke test mode (faster, less episodes).",
    )
    return parser.parse_args()


def get_colab_bin() -> str:
    """Find the colab CLI binary path in the local virtual environment, or system PATH."""
    venv_bin = ROOT / ".venv" / "bin" / "colab"
    if venv_bin.exists():
        return str(venv_bin)
    return "colab"


def is_session_active(colab_bin: str, session_name: str) -> bool:
    """Check if the given session is already active on the Colab backend."""
    try:
        result = subprocess.run(
            [colab_bin, "sessions"],
            capture_output=True,
            text=True,
            check=True,
        )
        return session_name in result.stdout
    except Exception as e:
        print(f"Warning: Failed to check active sessions ({e}). Assuming not active.")
        return False


def main() -> int:
    args = parse_args()
    colab_bin = get_colab_bin()

    # Define defaults and names
    phase = args.phase
    session_name = args.session or f"colab-routing-phase{phase}"
    bundle_filename = f"phase{phase}_colab_bundle.zip"
    bundle_path = ROOT / "artifacts" / "lite_globe" / bundle_filename
    results_zip_filename = f"phase{phase}_results.zip"
    local_results_path = ROOT / "artifacts" / "lite_globe" / results_zip_filename
    remote_content_dir = "content"

    print(f"=== Starting Colab Orchestration for Phase {phase} ===")
    print(f"Session Name: {session_name}")

    # 1. Package the project bundle locally
    packager_script = ROOT / "scripts" / f"package_phase{phase}_colab.py"
    if not packager_script.exists():
        print(f"Error: Packaging script for phase {phase} not found at {packager_script}")
        return 1

    print(f"Running packager: {packager_script.name}...")
    try:
        subprocess.run(
            [sys.executable, str(packager_script)],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Error: Packaging bundle failed.")
        return 1

    if not bundle_path.exists():
        print(f"Error: Bundle zip was not created at {bundle_path}")
        return 1

    # 2. Setup Colab session
    if is_session_active(colab_bin, session_name):
        print(f"Session '{session_name}' is already active. Reusing it.")
    else:
        new_cmd = [colab_bin, "new", "-s", session_name]
        if args.cpu:
            print("Provisioning a CPU session...")
        elif args.tpu:
            print(f"Provisioning a TPU ({args.tpu}) session...")
            new_cmd.extend(["--tpu", args.tpu])
        else:
            print(f"Provisioning a GPU ({args.gpu}) session...")
            new_cmd.extend(["--gpu", args.gpu])

        try:
            subprocess.run(new_cmd, check=True)
            print("Session provisioned successfully.")
        except subprocess.CalledProcessError:
            print("Error: Failed to provision Colab session. Please check your authentication state with 'colab sessions'.")
            return 1

    # 3. Create execution config arguments for the remote bootstrap script
    device = "cpu" if args.cpu else "cuda"
    output_dir = f"artifacts/lite_globe/phase{phase}"
    if args.smoke:
        output_dir += "_smoke"

    remote_args = ["--device", device, "--resume", "--output-dir", output_dir]
    if args.smoke:
        remote_args.append("--smoke")

    config_data = {
        "bundle_filename": bundle_filename,
        "run_module": f"implementations.lite_globe.run_phase{phase}",
        "args": remote_args,
        "output_dir": output_dir,
        "results_zip": results_zip_filename,
    }

    local_config_path = ROOT / "colab_args.json"
    with open(local_config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # 4. Upload files, execute, download results
    success = False
    try:
        # Upload config, bootstrap script, and bundle zip
        print("Uploading workspace config and scripts to Colab VM...")
        subprocess.run(
            [
                colab_bin,
                "upload",
                "-s",
                session_name,
                str(local_config_path),
                f"{remote_content_dir}/colab_args.json",
            ],
            check=True,
        )
        subprocess.run(
            [
                colab_bin,
                "upload",
                "-s",
                session_name,
                str(ROOT / "scripts" / "colab_bootstrap.py"),
                f"{remote_content_dir}/colab_bootstrap.py",
            ],
            check=True,
        )
        print(f"Uploading project bundle {bundle_filename} (this may take a moment)...")
        subprocess.run(
            [
                colab_bin,
                "upload",
                "-s",
                session_name,
                str(bundle_path),
                f"{remote_content_dir}/{bundle_filename}",
            ],
            check=True,
        )

        # Run execution
        print("Starting remote execution...")
        bootstrap_path = ROOT / "scripts" / "colab_bootstrap.py"
        subprocess.run(
            [colab_bin, "exec", "-s", session_name, "-f", str(bootstrap_path)],
            check=True,
        )

        # Download results
        print("Downloading results ZIP...")
        local_results_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                colab_bin,
                "download",
                "-s",
                session_name,
                f"{remote_content_dir}/{results_zip_filename}",
                str(local_results_path),
            ],
            check=True,
        )
        print(f"Results successfully saved locally to: {local_results_path}")
        success = True

    except subprocess.CalledProcessError as e:
        print(f"Execution failed during a Colab CLI operation: {e}")
    finally:
        # Clean up local config temp file
        if local_config_path.exists():
            os.remove(local_config_path)

        # Stop session if not keeping
        if not args.keep:
            print(f"Stopping Colab session '{session_name}' to save compute units...")
            try:
                subprocess.run([colab_bin, "stop", "-s", session_name], check=True)
                print("Session stopped.")
            except subprocess.CalledProcessError:
                print("Warning: Failed to stop Colab session. Please check with 'colab sessions' and stop it manually if needed.")
        else:
            print(f"Keeping Colab session '{session_name}' active. Remember to run 'colab stop -s {session_name}' later!")

    if success:
        print("=== Colab Orchestration Completed Successfully ===")
        return 0
    else:
        print("=== Colab Orchestration Failed ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
