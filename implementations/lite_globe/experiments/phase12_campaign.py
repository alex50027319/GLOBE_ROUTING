"""Historical Phase 12 implementation of the final SwitchGLOBE campaign."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from ..baselines import GpsrPolicy, PredictiveGeographicPolicy
from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import (
    episode_row,
    evaluate_policy_results,
    generalization_summary,
)
from ..models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import (
    phase9_curriculum,
    phase9_evaluation_scenarios,
    phase9_hole_calibration_scenarios,
    phase9_predictive_calibration_scenarios,
    phase9_predictive_link_loss_calibration_scenarios,
)
from ..utils import load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class Phase12Config:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    calibration_episodes_per_stage: int
    calibration_pdr_tolerance: float
    switch_thresholds: tuple[float, ...]
    margin_gates: tuple[float, ...]
    lifetime_gates: tuple[float, ...]
    onward_gates: tuple[float, ...]
    include_link_loss_calibration: bool = False


PHASE12_METHODS = (
    "GPSR",
    "Predictive Geographic",
    "Phase 8 Geo-Residual KD",
    "Lite-GLOBE-P predictive-prior only",
    "Lite-GLOBE-P no-switch",
    "SwitchGLOBE",
)


def _load_phase8(
    *,
    training_seed: int,
    phase8_checkpoint_dir: Path,
    max_nodes: int,
    hidden_dim: int,
    device: torch.device,
) -> GeographicResidualStudentPolicy:
    model = GeographicResidualStudentPolicy(max_nodes, hidden_dim=hidden_dim)
    load_checkpoint(
        phase8_checkpoint_dir
        / f"seed_{training_seed}"
        / "geo_residual_kd.pt",
        model,
        map_location=device,
    )
    return model


def _load_phase11(
    *,
    training_seed: int,
    phase11_checkpoint_dir: Path,
    max_nodes: int,
    hidden_dim: int,
    device: torch.device,
) -> LiteGlobePStudentPolicy:
    model = LiteGlobePStudentPolicy(max_nodes, hidden_dim=hidden_dim)
    load_checkpoint(
        phase11_checkpoint_dir
        / f"seed_{training_seed}"
        / "lite_globe_p.pt",
        model,
        map_location=device,
    )
    return model


def _risk_switch_policy(
    phase8: GeographicResidualStudentPolicy,
    phase11: LiteGlobePStudentPolicy,
) -> SwitchGlobePolicy:
    predictive = LiteGlobePStudentPolicy(
        phase11.max_nodes,
        hidden_dim=phase11.hidden_dim,
    )
    predictive.load_state_dict(phase11.state_dict())
    predictive.set_residual_weight(0.0)
    return SwitchGlobePolicy(
        phase8,
        predictive,
    )


def _measure_candidate(
    model: SwitchGlobePolicy,
    scenarios,
    *,
    seed: int,
    episodes_per_stage: int,
    device: torch.device,
) -> dict[str, float]:
    delivered = 0
    deadline_met = 0
    episodes = 0
    delay = 0.0
    energy = 0.0
    input_bytes = 0.0
    for index, scenario in enumerate(scenarios):
        start = seed + 1_370_000 + index * 10_000
        results = evaluate_policy_results(
            FanetRoutingEnv(scenario.config),
            StudentPolicyAdapter(
                model,
                device=device,
                force_forward_if_available=True,
            ),
            list(range(start, start + episodes_per_stage)),
            reset_options=scenario.reset_options,
        )
        delivered += sum(result.delivered for result in results)
        deadline_met += sum(result.deadline_met for result in results)
        episodes += len(results)
        delay += sum(
            result.steps for result in results if result.delivered
        )
        energy += sum(
            result.transmission_energy_proxy for result in results
        )
        input_bytes += sum(result.policy_input_bytes for result in results)
    return {
        "pdr": delivered / episodes,
        "deadline_delivery_ratio": deadline_met / episodes,
        "mean_success_delay": delay / max(delivered, 1),
        "mean_energy": energy / episodes,
        "mean_policy_input_bytes": input_bytes / episodes,
    }


def _candidate_switch_parameters(config: Phase12Config):
    for switch_threshold in config.switch_thresholds:
        for margin_gate in config.margin_gates:
            for lifetime_gate in config.lifetime_gates:
                for onward_gate in config.onward_gates:
                    yield {
                        "switch_threshold": switch_threshold,
                        "margin_gate": margin_gate,
                        "lifetime_gate": lifetime_gate,
                        "onward_gate": onward_gate,
                    }


def _calibrate_switch(
    model: RiskSwitchLiteGlobePStudentPolicy,
    config: Phase12Config,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generic = phase9_curriculum(seed)
    holes = phase9_hole_calibration_scenarios(seed)
    predictive = phase9_predictive_calibration_scenarios(seed)
    if config.include_link_loss_calibration:
        predictive = predictive + phase9_predictive_link_loss_calibration_scenarios(seed)
    calibration_seed = seed + 1_000

    phase8_like = _risk_switch_policy(
        model.normal_policy,
        model.predictive_policy,
    )
    phase8_like.set_switch_parameters(
        switch_threshold=3.0,
        margin_gate=0.0,
        lifetime_gate=0.0,
        onward_gate=0.0,
    )
    baseline_generic = _measure_candidate(
        phase8_like,
        generic,
        seed=calibration_seed,
        episodes_per_stage=config.calibration_episodes_per_stage,
        device=device,
    )
    baseline_holes = _measure_candidate(
        phase8_like,
        holes,
        seed=calibration_seed + 1_000,
        episodes_per_stage=config.calibration_episodes_per_stage,
        device=device,
    )
    measurements = []
    for params in _candidate_switch_parameters(config):
        model.set_switch_parameters(**params)
        generic_result = _measure_candidate(
            model,
            generic,
            seed=calibration_seed,
            episodes_per_stage=config.calibration_episodes_per_stage,
            device=device,
        )
        hole_result = _measure_candidate(
            model,
            holes,
            seed=calibration_seed + 1_000,
            episodes_per_stage=config.calibration_episodes_per_stage,
            device=device,
        )
        predictive_result = _measure_candidate(
            model,
            predictive,
            seed=calibration_seed + 2_000,
            episodes_per_stage=config.calibration_episodes_per_stage,
            device=device,
        )
        measurements.append(
            {
                **params,
                "generic_pdr": generic_result["pdr"],
                "hole_pdr": hole_result["pdr"],
                "predictive_pdr": predictive_result["pdr"],
                "deadline_delivery_ratio": generic_result[
                    "deadline_delivery_ratio"
                ],
                "mean_success_delay": generic_result[
                    "mean_success_delay"
                ],
                "mean_energy": generic_result["mean_energy"],
                "mean_policy_input_bytes": generic_result[
                    "mean_policy_input_bytes"
                ],
            }
        )
    feasible = [
        item
        for item in measurements
        if item["generic_pdr"]
        >= baseline_generic["pdr"] - config.calibration_pdr_tolerance
        and item["hole_pdr"]
        >= baseline_holes["pdr"] - config.calibration_pdr_tolerance
    ]
    best = max(
        feasible,
        key=lambda item: (
            item["predictive_pdr"],
            item["generic_pdr"],
            item["hole_pdr"],
            item["deadline_delivery_ratio"],
            -item["mean_success_delay"],
            -item["mean_energy"],
            -item["mean_policy_input_bytes"],
        ),
    )
    model.set_switch_parameters(
        switch_threshold=best["switch_threshold"],
        margin_gate=best["margin_gate"],
        lifetime_gate=best["lifetime_gate"],
        onward_gate=best["onward_gate"],
    )
    return {
        **best,
        "phase8_like_generic_pdr": baseline_generic["pdr"],
        "phase8_like_hole_pdr": baseline_holes["pdr"],
        "candidate_count": float(len(measurements)),
    }


def _policy(
    method: str,
    model,
    env: FanetRoutingEnv,
    device: torch.device,
):
    if method == "GPSR":
        return GpsrPolicy(env.drop_action)
    if method == "Predictive Geographic":
        return PredictiveGeographicPolicy(env.drop_action)
    assert model is not None
    return StudentPolicyAdapter(
        model,
        device=device,
        force_forward_if_available=True,
    )


def run_phase12_campaign(
    config: Phase12Config,
    *,
    phase8_checkpoint_dir: Path,
    phase11_checkpoint_dir: Path,
    output_checkpoint_dir: Path | None = None,
    device: torch.device | str = "cpu",
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Calibrate risk-switch thresholds and evaluate paired scenarios."""

    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for training_seed in config.training_seeds:
        max_nodes = phase9_curriculum(training_seed)[0].config.max_nodes
        phase8 = _load_phase8(
            training_seed=training_seed,
            phase8_checkpoint_dir=phase8_checkpoint_dir,
            max_nodes=max_nodes,
            hidden_dim=config.hidden_dim,
            device=device,
        )
        phase11 = _load_phase11(
            training_seed=training_seed,
            phase11_checkpoint_dir=phase11_checkpoint_dir,
            max_nodes=max_nodes,
            hidden_dim=config.hidden_dim,
            device=device,
        )
        predictive_only = LiteGlobePStudentPolicy(
            max_nodes,
            hidden_dim=config.hidden_dim,
        )
        predictive_only.load_state_dict(phase11.state_dict())
        predictive_only.set_residual_weight(0.0)
        risk_switch = _risk_switch_policy(phase8, phase11)
        checkpoint = (
            output_checkpoint_dir
            / f"seed_{training_seed}"
            / "switchglobe.pt"
            if output_checkpoint_dir is not None
            else None
        )
        metrics_path = (
            checkpoint.parent / "training_metrics.json"
            if checkpoint is not None
            else None
        )
        if (
            resume
            and checkpoint is not None
            and checkpoint.is_file()
            and metrics_path is not None
            and metrics_path.is_file()
        ):
            load_checkpoint(checkpoint, risk_switch, map_location=device)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = {
                "training_seed": training_seed,
                **_calibrate_switch(
                    risk_switch,
                    config,
                    seed=training_seed,
                    device=device,
                ),
            }
            if checkpoint is not None and metrics_path is not None:
                save_checkpoint(
                    checkpoint,
                    risk_switch,
                    metadata={
                        "phase": 12,
                        "training_seed": training_seed,
                        "method": "SwitchGLOBE",
                    },
                )
                metrics_path.write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        training_rows.append(metrics)
        models = {
            "GPSR": None,
            "Predictive Geographic": None,
            "Phase 8 Geo-Residual KD": phase8,
            "Lite-GLOBE-P predictive-prior only": predictive_only,
            "Lite-GLOBE-P no-switch": phase11,
            "SwitchGLOBE": risk_switch,
        }
        for scenario_index, scenario in enumerate(
            phase9_evaluation_scenarios(training_seed)
        ):
            evaluation_seeds = list(
                range(
                    1_100_000 + scenario_index * 10_000,
                    1_100_000
                    + scenario_index * 10_000
                    + config.evaluation_episodes,
                )
            )
            env = FanetRoutingEnv(scenario.config)
            for method in PHASE12_METHODS:
                results = evaluate_policy_results(
                    env,
                    _policy(method, models[method], env, device),
                    evaluation_seeds,
                    reset_options=scenario.reset_options,
                )
                episode_rows.extend(
                    episode_row(
                        result,
                        method=method,
                        scenario=scenario.name,
                        training_seed=training_seed,
                    )
                    for result in results
                )
                summary_rows.append(
                    generalization_summary(
                        results,
                        method=method,
                        scenario=scenario.name,
                        training_seed=training_seed,
                    )
                )
    return {
        "episodes": episode_rows,
        "seed_summaries": summary_rows,
        "training": training_rows,
    }
