"""Matched-budget causal audit of predictive-prior supervision signals.

Every deployable variant has the same ``LiteGlobePStudentPolicy`` scoring
equation, default initialization, disabled neural residual, optimizer, state
dataset, update count, and deterministic routing adapter. Only the labels in
the training objective differ. Oracle and risk-oracle rollouts define the
common state distribution; the global teacher labels those states offline but
never controls a rollout used for the no-teacher variants.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .algorithms.distillation import forward_kl_loss
from .data import concatenate_datasets, generate_teacher_dataset, split_by_episode_group
from .env.fanet_env import FanetRoutingEnv
from .evaluation import episode_row, evaluate_policy_results, generalization_summary
from .evaluation.interleaved_latency import InterleavedSpec, benchmark_interleaved
from .evaluation.reporting import write_csv
from .models import GlobalTeacherActorCritic, LiteGlobePStudentPolicy
from .models.policy_adapter import StudentPolicyAdapter
from .models.tensor_observation import observation_to_tensors
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance
from .scenarios import (
    phase9_curriculum, phase9_evaluation_scenarios,
    phase9_hole_training_scenarios, phase9_predictive_training_scenarios,
    phase9_predictive_link_loss_training_scenarios,
)
from .utils import load_checkpoint, save_checkpoint, seed_everything


SEEDS = (42, 77, 123, 314, 2718)
MANUAL = "Fixed Manual Prior"
SHORTEST_ONLY = "Shortest-Oracle Only"
RISK_ONLY = "Risk-Oracle Only"
NO_TEACHER = "Shortest+Risk (No Teacher)"
TEACHER_ONLY = "Teacher Distillation Only"
FULL = "Teacher+Shortest+Risk"
LEGACY = "Existing Phase11 Predictive Prior"
METHODS = (MANUAL, SHORTEST_ONLY, RISK_ONLY, NO_TEACHER, TEACHER_ONLY, FULL, LEGACY)
TRAINED_METHODS = (SHORTEST_ONLY, RISK_ONLY, NO_TEACHER, TEACHER_ONLY, FULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--train-episodes-per-scenario-rollout", type=int, default=100)
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--latency-repeats", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def _training_scenarios(seed: int):
    return [
        *phase9_curriculum(seed),
        *phase9_hole_training_scenarios(seed),
        *phase9_predictive_training_scenarios(seed),
        *phase9_predictive_link_loss_training_scenarios(seed),
    ]


def _teacher(root: Path, *, seed: int, max_nodes: int, device: torch.device):
    model = GlobalTeacherActorCritic(max_nodes, hidden_dim=64)
    load_checkpoint(root / f"seed_{seed}" / "global_teacher.pt", model, map_location=device)
    return model.to(device).eval()


def _collect_common_dataset(
    teacher: GlobalTeacherActorCritic, *, seed: int, episodes: int,
    device: torch.device,
):
    datasets = []
    for scenario_index, scenario in enumerate(_training_scenarios(seed)):
        for rollout_index, rollout in enumerate(("oracle", "risk_oracle")):
            start = seed + 2_000_000 + scenario_index * 20_000 + rollout_index * 10_000
            datasets.append(generate_teacher_dataset(
                FanetRoutingEnv(scenario.config), teacher,
                episode_seeds=list(range(start, start + episodes)),
                scenario_id=f"{scenario.name}_{rollout}_rollout",
                reset_options=scenario.reset_options, rollout_policy=rollout,
                device=device,
            ))
    return concatenate_datasets(datasets)


def _new_model(max_nodes: int, *, seed: int, device: torch.device):
    seed_everything(seed)
    model = LiteGlobePStudentPolicy(
        max_nodes, hidden_dim=64, initial_prior_strength=8.0,
        initial_forwardability_strength=0.05,
        initial_predictive_strength=(0.75, 3.0, 0.25, 6.0),
        initial_break_penalty=18.0, initial_residual_bound=1.5,
        margin_gate=0.04, lifetime_gate=0.20, onward_gate=0.20,
    )
    model.set_residual_weight(0.0)
    return model.to(device)


def _batch_observation(batch: dict[str, torch.Tensor], device: torch.device):
    keys = (
        "self_features", "neighbor_features", "edge_features", "packet_features",
        "action_mask", "candidate_forwardability", "candidate_risk_features",
    )
    return {key: batch[key].to(device) for key in keys if key in batch}


def _loss(
    model: LiteGlobePStudentPolicy, batch: dict[str, torch.Tensor],
    *, method: str, device: torch.device,
) -> torch.Tensor:
    observation = _batch_observation(batch, device)
    output = model(observation)
    shortest = torch.nn.functional.cross_entropy(
        output.masked_logits, batch["oracle_actions"].to(device)
    )
    risk = torch.nn.functional.cross_entropy(
        output.masked_logits, batch["risk_oracle_actions"].to(device)
    )
    teacher_kl, _, _ = forward_kl_loss(
        batch["teacher_logits"].to(device), output.logits,
        observation["action_mask"], temperature=1.0,
    )
    teacher_hard = torch.nn.functional.cross_entropy(
        output.masked_logits, batch["selected_actions"].to(device)
    )
    if method == SHORTEST_ONLY:
        return shortest
    if method == RISK_ONLY:
        return risk
    if method == NO_TEACHER:
        return 0.30 * shortest + 1.00 * risk
    if method == TEACHER_ONLY:
        return teacher_kl + 0.05 * teacher_hard
    if method == FULL:
        return teacher_kl + 0.05 * teacher_hard + 0.30 * shortest + 1.00 * risk
    raise ValueError(method)


def _train(
    model: LiteGlobePStudentPolicy, train, validation, *, method: str,
    epochs: int, batch_size: int, learning_rate: float, seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    trainable = {
        "log_prior_strength", "log_forwardability_strength",
        "log_predictive_strength", "log_break_penalty",
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=1e-5,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, generator=generator,
    )
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    curve = []
    for epoch in range(epochs):
        model.train(); total = samples = 0
        for batch in train_loader:
            objective = _loss(model, batch, method=method, device=device)
            optimizer.zero_grad(set_to_none=True); objective.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            count = int(batch["selected_actions"].shape[0])
            total += float(objective.detach().item()) * count; samples += count
        model.eval(); validation_total = validation_samples = 0
        with torch.inference_mode():
            for batch in validation_loader:
                objective = _loss(model, batch, method=method, device=device)
                count = int(batch["selected_actions"].shape[0])
                validation_total += float(objective.item()) * count
                validation_samples += count
        curve.append({
            "method": method, "epoch": epoch + 1,
            "train_objective": total / samples,
            "validation_objective": validation_total / validation_samples,
        })
    model.eval()
    return curve


def _audit_labels(model, dataset, *, method: str, device: torch.device):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    counts = {"samples": 0, "teacher": 0, "shortest": 0, "risk": 0}
    with torch.inference_mode():
        for batch in loader:
            output = model(_batch_observation(batch, device))
            action = torch.argmax(output.masked_logits, dim=-1)
            count = int(action.numel()); counts["samples"] += count
            counts["teacher"] += int((action == batch["selected_actions"].to(device)).sum())
            counts["shortest"] += int((action == batch["oracle_actions"].to(device)).sum())
            counts["risk"] += int((action == batch["risk_oracle_actions"].to(device)).sum())
    strengths = {
        "geographic_strength": float(torch.nn.functional.softplus(model.log_prior_strength).item()),
        "break_penalty": float(torch.nn.functional.softplus(model.log_break_penalty).item()),
        "predictive_weight": float(model.predictive_weight.item()),
        "residual_weight": float(model.residual_weight.item()),
    }
    forward = torch.nn.functional.softplus(model.log_forwardability_strength).detach().cpu().tolist()
    predictive = torch.nn.functional.softplus(model.log_predictive_strength).detach().cpu().tolist()
    return {
        "method": method, **strengths,
        "forwardability_binary_strength": forward[0],
        "forwardability_count_strength": forward[1],
        "margin_strength": predictive[0], "lifetime_strength": predictive[1],
        "queue_headroom_strength": predictive[2], "onward_lifetime_strength": predictive[3],
        "test_samples": counts["samples"],
        "test_teacher_action_agreement": counts["teacher"] / counts["samples"],
        "test_shortest_action_agreement": counts["shortest"] / counts["samples"],
        "test_risk_action_agreement": counts["risk"] / counts["samples"],
    }


def _latency(
    models: dict[str, LiteGlobePStudentPolicy], *, seed: int,
    warmup: int, repeats: int,
):
    summary_rows, raw_rows = [], []
    scenario = phase9_evaluation_scenarios(seed)[0]
    observation, _ = FanetRoutingEnv(scenario.config).reset(
        seed=1_099_999, options=scenario.reset_options
    )
    for device in (torch.device("cpu"), torch.device("cuda")):
        specs = []
        for method in METHODS:
            replica = copy.deepcopy(models[method]).to(device).eval()
            policy = StudentPolicyAdapter(replica, device=device, force_forward_if_available=True)
            specs.append(InterleavedSpec(
                method, "end_to_end_policy", device,
                lambda policy=policy: policy.act_with_metadata(observation),
            ))
        summaries, timings = benchmark_interleaved(
            specs, warmup=warmup, repeats=repeats,
            order_seed=seed + (500_000 if device.type == "cuda" else 0),
        )
        summary_rows.extend({"training_seed": seed, **row} for row in summaries)
        raw_rows.extend({"training_seed": seed, **row} for row in timings)
    return summary_rows, raw_rows


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.smoke:
        train_episodes, evaluation_episodes, epochs = 3, 2, 2
        warmup, latency_repeats = 2, 5
    else:
        train_episodes = args.train_episodes_per_scenario_rollout
        evaluation_episodes, epochs = args.evaluation_episodes, args.epochs
        warmup, latency_repeats = args.warmup, args.latency_repeats
        if evaluation_episodes != 200:
            raise ValueError("confirmatory run requires 200 evaluation episodes/scenario")
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    scenarios = phase9_evaluation_scenarios(args.seed)
    if len(scenarios) != 14 or len(_training_scenarios(args.seed)) != 14:
        raise ValueError("protocol requires 14 training and 14 evaluation scenarios")
    max_nodes = scenarios[0].config.max_nodes
    teacher = _teacher(args.teacher_dir, seed=args.seed, max_nodes=max_nodes, device=device)
    dataset = _collect_common_dataset(
        teacher, seed=args.seed, episodes=train_episodes, device=device,
    )
    split = split_by_episode_group(dataset, seed=args.seed + 2_500_000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save(args.output_dir / "common_labeled_dataset.npz")
    base = _new_model(max_nodes, seed=args.seed + 2_600_000, device=device)
    models = {MANUAL: copy.deepcopy(base)}
    curves = []
    for index, method in enumerate(TRAINED_METHODS):
        model = copy.deepcopy(base)
        curves.extend(_train(
            model, split.train, split.validation, method=method,
            epochs=epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + 2_700_000 + index, device=device,
        ))
        models[method] = model
    legacy = LiteGlobePStudentPolicy(max_nodes, hidden_dim=64)
    load_checkpoint(
        args.legacy_dir / f"seed_{args.seed}" / "lite_globe_p.pt",
        legacy, map_location=device,
    )
    legacy.set_residual_weight(0.0); legacy.to(device).eval(); models[LEGACY] = legacy
    for method, model in models.items():
        save_checkpoint(
            args.output_dir / "checkpoints" / f"{method.lower().replace(' ', '_').replace('+', '_plus_')}.pt",
            model, metadata={"method": method, "training_seed": args.seed},
        )
    parameter_rows = [
        {"training_seed": args.seed, **_audit_labels(model, split.test, method=method, device=device)}
        for method, model in models.items()
    ]
    episode_rows, summary_rows = [], []
    policies = {
        method: StudentPolicyAdapter(model, device=device, force_forward_if_available=True)
        for method, model in models.items()
    }
    for scenario_index, scenario in enumerate(scenarios):
        evaluation_seeds = list(range(
            1_100_000 + scenario_index * 10_000,
            1_100_000 + scenario_index * 10_000 + evaluation_episodes,
        ))
        env = FanetRoutingEnv(scenario.config)
        for method in METHODS:
            results = evaluate_policy_results(
                env, policies[method], evaluation_seeds,
                reset_options=scenario.reset_options,
            )
            episode_rows.extend(episode_row(
                result, method=method, scenario=scenario.name,
                training_seed=args.seed,
            ) for result in results)
            summary_rows.append(generalization_summary(
                results, method=method, scenario=scenario.name,
                training_seed=args.seed,
            ))
    latency_summary, latency_raw = _latency(
        models, seed=args.seed, warmup=warmup, repeats=latency_repeats,
    ) if torch.cuda.is_available() else ([], [])
    write_csv(args.output_dir / "raw_episodes.csv", episode_rows)
    write_csv(args.output_dir / "seed_summaries.csv", summary_rows)
    write_csv(args.output_dir / "training_curves.csv", curves)
    write_csv(args.output_dir / "learned_parameters_and_test_agreement.csv", parameter_rows)
    write_csv(args.output_dir / "latency_seed_summary.csv", latency_summary)
    write_csv(args.output_dir / "latency_raw.csv", latency_raw)
    config = {
        "training_seed": args.seed, "methods": list(METHODS),
        "training_scenarios": [scenario.name for scenario in _training_scenarios(args.seed)],
        "evaluation_scenarios": [scenario.name for scenario in scenarios],
        "common_state_rollouts": ["shortest_oracle", "risk_oracle"],
        "train_episodes_per_scenario_rollout": train_episodes,
        "evaluation_episodes_per_scenario": evaluation_episodes,
        "epochs": epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "warmup": warmup,
        "latency_repeats": latency_repeats, "residual_weight": 0.0,
        "objective_weights": {"teacher_kl": 1.0, "teacher_action": 0.05, "shortest": 0.30, "risk": 1.0},
    }
    checkpoints = {
        "teacher": args.teacher_dir / f"seed_{args.seed}" / "global_teacher.pt",
        "legacy": args.legacy_dir / f"seed_{args.seed}" / "lite_globe_p.pt",
    }
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": "predictive_prior_matched_causal_ablation",
        "mode": "smoke" if args.smoke else "full",
        "config": config, "config_sha256": config_sha256(config),
        "dataset_samples": len(dataset), "train_samples": len(split.train),
        "validation_samples": len(split.validation), "test_samples": len(split.test),
        "episode_rows": len(episode_rows), "seed_summary_rows": len(summary_rows),
        "latency_raw_rows": len(latency_raw),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoints),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **git_provenance(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if args.zip_results:
        archive = Path(shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir))
        manifest["result_zip_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
