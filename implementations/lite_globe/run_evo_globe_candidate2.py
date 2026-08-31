"""Run the frozen seed-level gate for cost-to-go SwitchGLOBE distillation."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation import measure_policy_cost
from .evaluation.reporting import write_csv
from .experiments.evo_globe_campaign import (
    CostToGoDistillationConfig,
    evaluate_cost_to_go_candidate,
    train_cost_to_go_switchglobe,
)
from .models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from .models.policy_adapter import StudentPolicyAdapter
from .scenarios import phase9_evaluation_scenarios
from .utils.checkpoint import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--archive-member")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-scenario", type=int, default=50)
    parser.add_argument("--episode-seed-base", type=int, default=1_100_000)
    parser.add_argument("--dataset-episodes-per-scenario", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--return-action-coefficient", type=float, default=0.20)
    parser.add_argument("--latency-rounds", type=int, default=5)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evo_globe/candidate2_gate/seed_42"),
    )
    return parser.parse_args()


def _checkpoint_payload(args: argparse.Namespace) -> tuple[dict, dict[str, str]]:
    if (args.checkpoint is None) == (args.checkpoint_archive is None):
        raise ValueError("provide exactly one of --checkpoint or --checkpoint-archive")
    if args.checkpoint is not None:
        raw = args.checkpoint.read_bytes()
        source = str(args.checkpoint)
    else:
        if not args.archive_member:
            raise ValueError("--archive-member is required with --checkpoint-archive")
        with tarfile.open(args.checkpoint_archive, "r:gz") as archive:
            extracted = archive.extractfile(args.archive_member)
            if extracted is None:
                raise FileNotFoundError(args.archive_member)
            raw = extracted.read()
        source = f"{args.checkpoint_archive}:{args.archive_member}"
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    return payload, {"checkpoint_source": source, "checkpoint_sha256": sha256(raw).hexdigest()}


def _build_reference(payload: dict, *, max_nodes: int) -> SwitchGlobePolicy:
    model = SwitchGlobePolicy(
        GeographicResidualStudentPolicy(max_nodes, hidden_dim=64),
        LiteGlobePStudentPolicy(max_nodes, hidden_dim=64),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval()


def _latency_rows(
    reference: SwitchGlobePolicy,
    candidate: SwitchGlobePolicy,
    scenario,
    *,
    rounds: int,
    warmup: int,
    repeats: int,
) -> list[dict]:
    env = FanetRoutingEnv(scenario.config)
    observation, _ = env.reset(seed=1_099_999, options=scenario.reset_options)
    policies = {
        "SwitchGLOBE_exact": StudentPolicyAdapter(
            reference, device="cpu", force_forward_if_available=True
        ),
        "SwitchGLOBE_cost_to_go": StudentPolicyAdapter(
            candidate, device="cpu", force_forward_if_available=True
        ),
    }
    rows = []
    for round_index in range(rounds):
        order = tuple(policies) if round_index % 2 == 0 else tuple(reversed(policies))
        for method in order:
            cost = measure_policy_cost(
                policies[method],
                observation,
                model=policies[method].model,
                device="cpu",
                warmup=warmup,
                repeats=repeats,
            )
            rows.append({"round": round_index, "method": method, **cost.to_dict()})
    return rows


def main() -> int:
    args = parse_args()
    if args.latency_rounds <= 0:
        raise ValueError("--latency-rounds must be positive")
    device = torch.device(args.device)
    scenarios = phase9_evaluation_scenarios(args.seed)
    payload, provenance = _checkpoint_payload(args)
    reference = _build_reference(payload, max_nodes=scenarios[0].config.max_nodes)
    config = CostToGoDistillationConfig(
        dataset_episodes_per_scenario=args.dataset_episodes_per_scenario,
        epochs=args.epochs,
        return_action_coefficient=args.return_action_coefficient,
    )
    candidate, training = train_cost_to_go_switchglobe(
        reference, config, seed=args.seed, device=device
    )
    exact = evaluate_cost_to_go_candidate(
        reference,
        scenarios,
        episode_seed_base=args.episode_seed_base,
        episodes_per_scenario=args.episodes_per_scenario,
        device=device,
    )
    cost_to_go = evaluate_cost_to_go_candidate(
        candidate,
        scenarios,
        episode_seed_base=args.episode_seed_base,
        episodes_per_scenario=args.episodes_per_scenario,
        device=device,
    )
    latency = _latency_rows(
        reference.cpu(),
        candidate.cpu(),
        scenarios[0],
        rounds=args.latency_rounds,
        warmup=args.latency_warmup,
        repeats=args.latency_repeats,
    )
    exact_p95 = np.asarray([
        row["latency_p95_ms"] for row in latency
        if row["method"] == "SwitchGLOBE_exact"
    ])
    candidate_p95 = np.asarray([
        row["latency_p95_ms"] for row in latency
        if row["method"] == "SwitchGLOBE_cost_to_go"
    ])
    summary = {
        "seed": args.seed,
        "scenario_count": len(scenarios),
        "episodes_per_scenario": args.episodes_per_scenario,
        **provenance,
        "training": training,
        "exact": {key: value for key, value in exact.items() if key != "scenario_rows"},
        "cost_to_go": {
            key: value for key, value in cost_to_go.items() if key != "scenario_rows"
        },
        "delta": {
            key: cost_to_go[key] - exact[key]
            for key in (
                "connected_pair_pdr",
                "deadline_delivery_ratio",
                "energy_per_delivered_packet",
            )
        },
        "latency": {
            "exact_median_p95_ms": float(np.median(exact_p95)),
            "candidate_median_p95_ms": float(np.median(candidate_p95)),
            "candidate_over_exact_ratio": float(
                np.median(candidate_p95) / np.median(exact_p95)
            ),
        },
    }
    scenario_rows = []
    for exact_row, candidate_row in zip(
        exact["scenario_rows"], cost_to_go["scenario_rows"], strict=True
    ):
        if exact_row["scenario"] != candidate_row["scenario"]:
            raise RuntimeError("paired scenario order changed")
        row = {"scenario": exact_row["scenario"]}
        for metric in (
            "connected_pair_pdr",
            "deadline_delivery_ratio",
            "energy_per_delivered_packet",
        ):
            row[f"exact_{metric}"] = exact_row[metric]
            row[f"candidate_{metric}"] = candidate_row[metric]
            row[f"delta_{metric}"] = candidate_row[metric] - exact_row[metric]
        scenario_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scenario_metrics.csv", scenario_rows)
    write_csv(args.output_dir / "latency_rounds.csv", latency)
    save_checkpoint(
        args.output_dir / "cost_to_go_switchglobe.pt",
        candidate,
        metadata={"training": training, **provenance},
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
