"""Per-seed confirmatory model-selection ablation for an A100 Colab session.

The final-model variants use the Phase 8/11/12 checkpoints.  Phase 7
PPO/KD variants form a separate matched-architecture historical audit and
must not be interpreted as a causal retraining ablation of the final model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation import episode_row, evaluate_policy_results, generalization_summary
from .evaluation.reporting import write_csv
from .experiments.ablation_campaign import (
    AblationConfig,
    build_variant_policies,
    FAST_SWITCHGLOBE,
    GEO_RESIDUAL,
    PREDICTIVE_NO_SWITCH,
    PREDICTIVE_PRIOR_ONLY,
    SWITCHGLOBE_EXACT,
)
from .models import GlobalTeacherActorCritic, LocalStudentPolicy
from .models.policy_adapter import StudentPolicyAdapter
from .models.teacher_adapter import TeacherPolicyAdapter
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance
from .scenarios import phase9_evaluation_scenarios
from .utils import load_checkpoint


SEEDS = (42, 77, 123, 314, 2718)
HISTORICAL_UNTRAINED = "Historical Untrained Local"
HISTORICAL_NO_KD = "Historical PPO-only Local (No KD)"
HISTORICAL_KD_ONLY = "Historical KD-only Local"
HISTORICAL_KD_PPO = "Historical KD+PPO Local"
GLOBAL_TEACHER = "Privileged Global Teacher"
METHODS = (
    HISTORICAL_UNTRAINED,
    HISTORICAL_NO_KD,
    HISTORICAL_KD_ONLY,
    HISTORICAL_KD_PPO,
    GLOBAL_TEACHER,
    GEO_RESIDUAL,
    PREDICTIVE_PRIOR_ONLY,
    PREDICTIVE_NO_SWITCH,
    SWITCHGLOBE_EXACT,
    FAST_SWITCHGLOBE,
)
PHASE7_FILES = {
    HISTORICAL_UNTRAINED: "untrained_student.pt",
    HISTORICAL_NO_KD: "ppo_only_student.pt",
    HISTORICAL_KD_ONLY: "kd_only_student.pt",
    HISTORICAL_KD_PPO: "kd_plus_ppo_student.pt",
    GLOBAL_TEACHER: "global_teacher.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--phase7-dir", type=Path, required=True)
    parser.add_argument("--phase8-dir", type=Path, required=True)
    parser.add_argument("--phase11-dir", type=Path, required=True)
    parser.add_argument("--phase12-dir", type=Path, required=True)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def _phase7_models(
    root: Path, *, seed: int, max_nodes: int, device: torch.device,
) -> tuple[dict[str, LocalStudentPolicy], GlobalTeacherActorCritic]:
    directory = root / f"seed_{seed}"
    locals_: dict[str, LocalStudentPolicy] = {}
    for method in (
        HISTORICAL_UNTRAINED, HISTORICAL_NO_KD,
        HISTORICAL_KD_ONLY, HISTORICAL_KD_PPO,
    ):
        model = LocalStudentPolicy(max_nodes, hidden_dim=64)
        load_checkpoint(directory / PHASE7_FILES[method], model, map_location=device)
        model.to(device).eval()
        locals_[method] = model
    teacher = GlobalTeacherActorCritic(max_nodes, hidden_dim=64)
    load_checkpoint(directory / PHASE7_FILES[GLOBAL_TEACHER], teacher, map_location=device)
    teacher.to(device).eval()
    return locals_, teacher


def main() -> int:
    args = parse_args()
    if args.episodes != 200:
        raise ValueError("confirmatory protocol requires exactly 200 episodes/scenario")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    scenarios = phase9_evaluation_scenarios(args.seed)
    if len(scenarios) != 14:
        raise ValueError(f"expected 14 scenarios, got {len(scenarios)}")
    max_nodes = scenarios[0].config.max_nodes
    final_policies = build_variant_policies(
        AblationConfig(training_seeds=(args.seed,), evaluation_episodes=args.episodes),
        seed=args.seed, max_nodes=max_nodes,
        phase8_checkpoint_dir=args.phase8_dir,
        phase11_checkpoint_dir=args.phase11_dir,
        switchglobe_checkpoint_dir=args.phase12_dir,
        fast_checkpoint_dir=args.fast_dir,
        device=device,
    )
    local_models, teacher_model = _phase7_models(
        args.phase7_dir, seed=args.seed, max_nodes=max_nodes, device=device,
    )
    local_policies = {
        method: StudentPolicyAdapter(
            model, device=device, deterministic=True,
            force_forward_if_available=True,
        )
        for method, model in local_models.items()
    }
    episode_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        evaluation_seeds = list(range(
            1_100_000 + scenario_index * 10_000,
            1_100_000 + scenario_index * 10_000 + args.episodes,
        ))
        env = FanetRoutingEnv(scenario.config)
        policies = {
            **local_policies,
            GLOBAL_TEACHER: TeacherPolicyAdapter(
                env, teacher_model, device=device, deterministic=True,
            ),
            GEO_RESIDUAL: final_policies[GEO_RESIDUAL],
            PREDICTIVE_PRIOR_ONLY: final_policies[PREDICTIVE_PRIOR_ONLY],
            PREDICTIVE_NO_SWITCH: final_policies[PREDICTIVE_NO_SWITCH],
            SWITCHGLOBE_EXACT: final_policies[SWITCHGLOBE_EXACT],
            FAST_SWITCHGLOBE: final_policies[FAST_SWITCHGLOBE],
        }
        for method in METHODS:
            results = evaluate_policy_results(
                env, policies[method], evaluation_seeds,
                reset_options=scenario.reset_options,
            )
            episode_rows.extend(
                episode_row(
                    result, method=method, scenario=scenario.name,
                    training_seed=args.seed,
                )
                for result in results
            )
            summary_rows.append(generalization_summary(
                results, method=method, scenario=scenario.name,
                training_seed=args.seed,
            ))
    expected = len(METHODS) * 14 * args.episodes
    if len(episode_rows) != expected or len(summary_rows) != len(METHODS) * 14:
        raise RuntimeError("result cardinality mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_episodes.csv", episode_rows)
    write_csv(args.output_dir / "seed_summaries.csv", summary_rows)
    checkpoint_paths = {
        f"phase7_{method}": args.phase7_dir / f"seed_{args.seed}" / filename
        for method, filename in PHASE7_FILES.items()
    }
    checkpoint_paths.update({
        "phase8_normal": args.phase8_dir / f"seed_{args.seed}" / "geo_residual_kd.pt",
        "phase11_predictive": args.phase11_dir / f"seed_{args.seed}" / "lite_globe_p.pt",
        "phase12_switch": args.phase12_dir / f"seed_{args.seed}" / "risk_switch_lite_globe_p.pt",
        "fast": args.fast_dir / f"seed_{args.seed}" / "fast_switchglobe.pt",
    })
    effective_config = {
        "training_seed": args.seed, "evaluation_episodes_per_scenario": args.episodes,
        "evaluation_seed_formula": "1100000 + scenario_index*10000 + episode_index",
        "scenario_count": 14, "methods": list(METHODS), "device": str(device),
        "deterministic": True, "force_forward_if_available": True,
        "phase7_interpretation": "historical matched-architecture KD audit; not a causal final-model retraining ablation",
    }
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": "switchglobe_confirmatory_model_selection",
        "episode_rows": len(episode_rows), "seed_summary_rows": len(summary_rows),
        "config": effective_config, "config_sha256": config_sha256(effective_config),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **git_provenance(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if args.zip_results:
        archive = Path(shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir))
        manifest["result_zip"] = str(archive)
        manifest["result_zip_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
