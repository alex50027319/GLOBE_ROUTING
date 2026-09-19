"""Six-variant SwitchGLOBE ablation evaluation, in the baseline row schema.

This harness evaluates already-trained checkpoints; it performs no
calibration and no FastSwitchGLOBE training of its own (see
``run_fast_switchglobe.py`` for that). Every variant is evaluated with the
same 14 ``phase9_evaluation_scenarios``, the same evaluation-seed formula,
and the same ``episode_row``/``generalization_summary`` schema the external
baseline comparison uses, so both tables share downstream statistics code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import episode_row, evaluate_policy_results, generalization_summary
from ..models import (
    FastSwitchGlobePolicy,
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import phase9_evaluation_scenarios
from ..utils import load_checkpoint
from .external_comparison_campaign import load_switchglobe
from .latency_optimization_campaign import checkpoint_path as fast_checkpoint_path


GEO_RESIDUAL = "Geo-Residual Student"
PREDICTIVE_PRIOR_ONLY = "Predictive Prior Only"
PREDICTIVE_NO_SWITCH = "Predictive Student (No Switch)"
SWITCHGLOBE_EXACT = "SwitchGLOBE Exact"
FAST_SWITCHGLOBE = "FastSwitchGLOBE"
FAST_SWITCHGLOBE_TOP2 = "FastSwitchGLOBE + Top-2"

ABLATION_METHODS = (
    GEO_RESIDUAL,
    PREDICTIVE_PRIOR_ONLY,
    PREDICTIVE_NO_SWITCH,
    SWITCHGLOBE_EXACT,
    FAST_SWITCHGLOBE,
    FAST_SWITCHGLOBE_TOP2,
)


@dataclass(frozen=True)
class AblationConfig:
    training_seeds: tuple[int, ...] = (42, 77, 123, 314, 2718)
    evaluation_episodes: int = 200
    hidden_dim: int = 64
    fast_hidden_dim: int = 32


def _load_phase8(
    *, seed: int, checkpoint_dir: Path, max_nodes: int, hidden_dim: int,
    device: torch.device,
) -> GeographicResidualStudentPolicy:
    model = GeographicResidualStudentPolicy(max_nodes, hidden_dim=hidden_dim)
    load_checkpoint(
        checkpoint_dir / f"seed_{seed}" / "geo_residual_kd.pt",
        model, map_location=device,
    )
    model.eval()
    return model


def _load_phase11(
    *, seed: int, checkpoint_dir: Path, max_nodes: int, hidden_dim: int,
    device: torch.device,
) -> LiteGlobePStudentPolicy:
    model = LiteGlobePStudentPolicy(max_nodes, hidden_dim=hidden_dim)
    load_checkpoint(
        checkpoint_dir / f"seed_{seed}" / "lite_globe_p.pt",
        model, map_location=device,
    )
    model.eval()
    return model


def _load_fast(
    *, seed: int, checkpoint_dir: Path, max_nodes: int, hidden_dim: int,
    device: torch.device,
) -> FastSwitchGlobePolicy:
    path = fast_checkpoint_path(checkpoint_dir, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"FastSwitchGLOBE checkpoint not found for seed {seed} at {path}; "
            "run run_fast_switchglobe.py first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("complete") or int(payload.get("training_seed", -1)) != seed:
        raise ValueError(f"FastSwitchGLOBE checkpoint at {path} is incomplete or seed-mismatched")
    model = FastSwitchGlobePolicy(max_nodes, hidden_dim=hidden_dim)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model


def build_variant_policies(
    config: AblationConfig, *, seed: int, max_nodes: int,
    phase8_checkpoint_dir: Path, phase11_checkpoint_dir: Path,
    switchglobe_checkpoint_dir: Path, fast_checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, StudentPolicyAdapter]:
    """Load one Adapter per named ablation variant for a training seed."""

    phase8 = _load_phase8(
        seed=seed, checkpoint_dir=phase8_checkpoint_dir, max_nodes=max_nodes,
        hidden_dim=config.hidden_dim, device=device,
    )
    phase11 = _load_phase11(
        seed=seed, checkpoint_dir=phase11_checkpoint_dir, max_nodes=max_nodes,
        hidden_dim=config.hidden_dim, device=device,
    )
    predictive_only = LiteGlobePStudentPolicy(max_nodes, hidden_dim=config.hidden_dim)
    predictive_only.load_state_dict(phase11.state_dict())
    predictive_only.set_residual_weight(0.0)
    predictive_only.eval()
    switchglobe_exact = load_switchglobe(
        switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
        hidden_dim=config.hidden_dim, device=device,
    )
    fast_model = _load_fast(
        seed=seed, checkpoint_dir=fast_checkpoint_dir, max_nodes=max_nodes,
        hidden_dim=config.fast_hidden_dim, device=device,
    )
    fast_model_top2 = FastSwitchGlobePolicy(max_nodes, hidden_dim=config.fast_hidden_dim)
    fast_model_top2.load_state_dict(fast_model.state_dict())
    fast_model_top2.to(device).eval()
    return {
        GEO_RESIDUAL: StudentPolicyAdapter(
            phase8, device=device, force_forward_if_available=True,
        ),
        PREDICTIVE_PRIOR_ONLY: StudentPolicyAdapter(
            predictive_only, device=device, force_forward_if_available=True,
        ),
        PREDICTIVE_NO_SWITCH: StudentPolicyAdapter(
            phase11, device=device, force_forward_if_available=True,
        ),
        SWITCHGLOBE_EXACT: switchglobe_exact,
        FAST_SWITCHGLOBE: StudentPolicyAdapter(
            fast_model, device=device, force_forward_if_available=True,
            enable_fast_failover=False,
        ),
        FAST_SWITCHGLOBE_TOP2: StudentPolicyAdapter(
            fast_model_top2, device=device, force_forward_if_available=True,
            enable_fast_failover=True,
        ),
    }


def run_ablation_campaign(
    config: AblationConfig, *, phase8_checkpoint_dir: Path,
    phase11_checkpoint_dir: Path, switchglobe_checkpoint_dir: Path,
    fast_checkpoint_dir: Path, device: torch.device | str = "cpu",
) -> dict[str, list[dict[str, Any]]]:
    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        scenarios = phase9_evaluation_scenarios(seed)
        max_nodes = scenarios[0].config.max_nodes
        policies = build_variant_policies(
            config, seed=seed, max_nodes=max_nodes,
            phase8_checkpoint_dir=phase8_checkpoint_dir,
            phase11_checkpoint_dir=phase11_checkpoint_dir,
            switchglobe_checkpoint_dir=switchglobe_checkpoint_dir,
            fast_checkpoint_dir=fast_checkpoint_dir,
            device=device,
        )
        for scenario_index, scenario in enumerate(scenarios):
            evaluation_seeds = list(range(
                1_100_000 + scenario_index * 10_000,
                1_100_000 + scenario_index * 10_000 + config.evaluation_episodes,
            ))
            env = FanetRoutingEnv(scenario.config)
            for method in ABLATION_METHODS:
                results = evaluate_policy_results(
                    env, policies[method], evaluation_seeds,
                    reset_options=scenario.reset_options,
                )
                episode_rows.extend(
                    episode_row(result, method=method, scenario=scenario.name, training_seed=seed)
                    for result in results
                )
                summary_rows.append(
                    generalization_summary(results, method=method, scenario=scenario.name, training_seed=seed)
                )
    return {"episodes": episode_rows, "seed_summaries": summary_rows}
