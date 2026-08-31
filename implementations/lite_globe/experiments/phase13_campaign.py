"""Phase 13: evaluate Risk-Switch Lite-GLOBE-P+ and ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    RiskSwitchLiteGlobePPlusStudentPolicy,
    RiskSwitchLiteGlobePStudentPolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import (
    phase9_curriculum,
    phase9_evaluation_scenarios,
    phase9_hole_calibration_scenarios,
    phase9_predictive_calibration_scenarios,
)
from ..utils import load_checkpoint, save_checkpoint
from .phase12_campaign import _load_phase8, _load_phase11, _risk_switch_policy


@dataclass(frozen=True)
class Phase13Config:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    calibration_episodes_per_stage: int
    calibration_pdr_tolerance: float
    switch_thresholds: tuple[float, ...]
    margin_gates: tuple[float, ...]
    lifetime_gates: tuple[float, ...]
    onward_gates: tuple[float, ...]
    topk_onward_gates: tuple[float, ...]
    redundancy_gates: tuple[float, ...]
    loss_keep_gates: tuple[float, ...]
    predictive_margins: tuple[float, ...]
    energy_tie_weights: tuple[float, ...]
    drop_suppression_bonuses: tuple[float, ...]


PHASE13_METHODS = (
    "GPSR",
    "Predictive Geographic",
    "Phase 8 Geo-Residual KD",
    "Lite-GLOBE-P no-switch",
    "Risk-Switch Lite-GLOBE-P",
    "Risk-Switch Lite-GLOBE-P+",
    "P+ ablation no link-loss gate",
    "P+ ablation no energy tie",
    "P+ ablation no drop suppression",
    "P+ ablation top1 onward",
)


def _risk_switch_plus_policy(
    phase8: GeographicResidualStudentPolicy,
    phase11: LiteGlobePStudentPolicy,
) -> RiskSwitchLiteGlobePPlusStudentPolicy:
    predictive = LiteGlobePStudentPolicy(
        phase11.max_nodes,
        hidden_dim=phase11.hidden_dim,
    )
    predictive.load_state_dict(phase11.state_dict())
    predictive.set_residual_weight(0.0)
    return RiskSwitchLiteGlobePPlusStudentPolicy(phase8, predictive)


def _clone_plus(
    model: RiskSwitchLiteGlobePPlusStudentPolicy,
) -> RiskSwitchLiteGlobePPlusStudentPolicy:
    cloned = _risk_switch_plus_policy(model.normal_policy, model.predictive_policy)
    cloned.load_state_dict(model.state_dict())
    return cloned


def _measure_candidate(
    model: RiskSwitchLiteGlobePPlusStudentPolicy,
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
    switch_steps = 0.0
    agent_drops = 0
    for index, scenario in enumerate(scenarios):
        start = seed + 1_530_000 + index * 10_000
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
        agent_drops += sum(result.drop_reason == "agent_drop" for result in results)
        episodes += len(results)
        delay += sum(result.steps for result in results if result.delivered)
        energy += sum(result.transmission_energy_proxy for result in results)
        input_bytes += sum(result.policy_input_bytes for result in results)
        switch_steps += sum(result.switch_steps for result in results)
    return {
        "pdr": delivered / episodes,
        "deadline_delivery_ratio": deadline_met / episodes,
        "agent_drop_rate": agent_drops / episodes,
        "mean_success_delay": delay / max(delivered, 1),
        "mean_energy": energy / episodes,
        "mean_policy_input_bytes": input_bytes / episodes,
        "mean_switch_steps": switch_steps / episodes,
    }


def _candidate_switch_parameters(config: Phase13Config):
    for switch_threshold in config.switch_thresholds:
        for margin_gate in config.margin_gates:
            for lifetime_gate in config.lifetime_gates:
                for onward_gate in config.onward_gates:
                    for topk_onward_gate in config.topk_onward_gates:
                        for redundancy_gate in config.redundancy_gates:
                            for loss_keep_gate in config.loss_keep_gates:
                                for predictive_margin in config.predictive_margins:
                                    for energy_tie_weight in config.energy_tie_weights:
                                        for drop_bonus in config.drop_suppression_bonuses:
                                            yield {
                                                "switch_threshold": switch_threshold,
                                                "margin_gate": margin_gate,
                                                "lifetime_gate": lifetime_gate,
                                                "onward_gate": onward_gate,
                                                "topk_onward_gate": topk_onward_gate,
                                                "redundancy_gate": redundancy_gate,
                                                "loss_keep_gate": loss_keep_gate,
                                                "predictive_margin": predictive_margin,
                                                "energy_tie_weight": energy_tie_weight,
                                                "drop_suppression_bonus": drop_bonus,
                                            }


def _calibrate_plus(
    model: RiskSwitchLiteGlobePPlusStudentPolicy,
    config: Phase13Config,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generic = phase9_curriculum(seed)
    holes = phase9_hole_calibration_scenarios(seed)
    predictive = phase9_predictive_calibration_scenarios(seed)
    calibration_seed = seed + 1_000

    phase8_like = _clone_plus(model)
    phase8_like.set_switch_parameters(
        switch_threshold=3.0,
        margin_gate=0.0,
        lifetime_gate=0.0,
        onward_gate=0.0,
        topk_onward_gate=0.0,
        redundancy_gate=0.0,
        loss_keep_gate=0.0,
        predictive_margin=1.0,
        energy_tie_weight=0.0,
        drop_suppression_bonus=0.0,
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
                "agent_drop_rate": generic_result["agent_drop_rate"],
                "mean_success_delay": generic_result["mean_success_delay"],
                "mean_energy": generic_result["mean_energy"],
                "mean_policy_input_bytes": generic_result[
                    "mean_policy_input_bytes"
                ],
                "mean_switch_steps": generic_result["mean_switch_steps"],
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
            -item["agent_drop_rate"],
            -item["mean_success_delay"],
            -item["mean_energy"],
            -item["mean_policy_input_bytes"],
            -item["mean_switch_steps"],
        ),
    )
    model.set_switch_parameters(
        switch_threshold=best["switch_threshold"],
        margin_gate=best["margin_gate"],
        lifetime_gate=best["lifetime_gate"],
        onward_gate=best["onward_gate"],
        topk_onward_gate=best["topk_onward_gate"],
        redundancy_gate=best["redundancy_gate"],
        loss_keep_gate=best["loss_keep_gate"],
        predictive_margin=best["predictive_margin"],
        energy_tie_weight=best["energy_tie_weight"],
        drop_suppression_bonus=best["drop_suppression_bonus"],
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


def _ablation_models(
    plus: RiskSwitchLiteGlobePPlusStudentPolicy,
) -> dict[str, RiskSwitchLiteGlobePPlusStudentPolicy]:
    models = {
        "Risk-Switch Lite-GLOBE-P+": plus,
        "P+ ablation no link-loss gate": _clone_plus(plus),
        "P+ ablation no energy tie": _clone_plus(plus),
        "P+ ablation no drop suppression": _clone_plus(plus),
        "P+ ablation top1 onward": _clone_plus(plus),
    }
    models["P+ ablation no link-loss gate"].loss_keep_gate.fill_(0.0)
    models["P+ ablation no energy tie"].energy_tie_weight.fill_(0.0)
    models["P+ ablation no drop suppression"].drop_suppression_bonus.fill_(0.0)
    models["P+ ablation top1 onward"].topk_onward_gate.fill_(0.0)
    models["P+ ablation top1 onward"].redundancy_gate.fill_(0.0)
    return models


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write resumable state without leaving a valid-looking partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_signature(config: Phase13Config, training_seed: int) -> dict[str, Any]:
    """Identify state that must match before cached work may be reused."""

    signature = {
        "phase": 13,
        "resume_schema_version": 1,
        "training_seed": training_seed,
        "config": asdict(config),
        "methods": list(PHASE13_METHODS),
        "evaluation_scenarios": [
            scenario.name
            for scenario in phase9_evaluation_scenarios(training_seed)
        ],
    }
    # Normalize tuples to their on-disk JSON representation before comparing.
    return json.loads(json.dumps(signature, ensure_ascii=False))


def _ensure_run_signature(seed_dir: Path, signature: dict[str, Any]) -> None:
    signature_path = seed_dir / "run_signature.json"
    if signature_path.is_file():
        cached = _read_json(signature_path)
        if cached != signature:
            raise ValueError(
                "Phase 13 resume state does not match the requested config for "
                f"{seed_dir.name}; use a new output directory"
            )
        return
    _write_json_atomic(signature_path, signature)


def _calibration_key(item: dict[str, float]) -> tuple[float, ...]:
    return (
        item["predictive_pdr"],
        item["generic_pdr"],
        item["hole_pdr"],
        item["deadline_delivery_ratio"],
        -item["agent_drop_rate"],
        -item["mean_success_delay"],
        -item["mean_energy"],
        -item["mean_policy_input_bytes"],
        -item["mean_switch_steps"],
    )


def _calibrate_plus_resumable(
    model: RiskSwitchLiteGlobePPlusStudentPolicy,
    config: Phase13Config,
    *,
    seed: int,
    device: torch.device,
    progress_dir: Path,
    max_new_candidates: int | None,
) -> tuple[dict[str, float] | None, int]:
    """Evaluate the canonical grid incrementally without changing selection."""

    generic = phase9_curriculum(seed)
    holes = phase9_hole_calibration_scenarios(seed)
    predictive = phase9_predictive_calibration_scenarios(seed)
    calibration_seed = seed + 1_000
    baseline_path = progress_dir / "baselines.json"
    if baseline_path.is_file():
        baselines = _read_json(baseline_path)
    else:
        phase8_like = _clone_plus(model)
        phase8_like.set_switch_parameters(
            switch_threshold=3.0,
            margin_gate=0.0,
            lifetime_gate=0.0,
            onward_gate=0.0,
            topk_onward_gate=0.0,
            redundancy_gate=0.0,
            loss_keep_gate=0.0,
            predictive_margin=1.0,
            energy_tie_weight=0.0,
            drop_suppression_bonus=0.0,
        )
        baselines = {
            "generic": _measure_candidate(
                phase8_like,
                generic,
                seed=calibration_seed,
                episodes_per_stage=config.calibration_episodes_per_stage,
                device=device,
            ),
            "holes": _measure_candidate(
                phase8_like,
                holes,
                seed=calibration_seed + 1_000,
                episodes_per_stage=config.calibration_episodes_per_stage,
                device=device,
            ),
        }
        _write_json_atomic(baseline_path, baselines)

    candidates = list(_candidate_switch_parameters(config))
    candidate_dir = progress_dir / "candidates"
    measurements: list[dict[str, float]] = []
    new_candidates = 0
    for index, params in enumerate(candidates):
        candidate_path = candidate_dir / f"candidate_{index:03d}.json"
        if candidate_path.is_file():
            measurement = _read_json(candidate_path)
            if any(measurement.get(key) != value for key, value in params.items()):
                raise ValueError(
                    f"cached Phase 13 candidate {index} has incompatible parameters"
                )
            measurements.append(measurement)
            continue
        if (
            max_new_candidates is not None
            and new_candidates >= max_new_candidates
        ):
            break
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
        measurement = {
            **params,
            "generic_pdr": generic_result["pdr"],
            "hole_pdr": hole_result["pdr"],
            "predictive_pdr": predictive_result["pdr"],
            "deadline_delivery_ratio": generic_result[
                "deadline_delivery_ratio"
            ],
            "agent_drop_rate": generic_result["agent_drop_rate"],
            "mean_success_delay": generic_result["mean_success_delay"],
            "mean_energy": generic_result["mean_energy"],
            "mean_policy_input_bytes": generic_result[
                "mean_policy_input_bytes"
            ],
            "mean_switch_steps": generic_result["mean_switch_steps"],
        }
        _write_json_atomic(candidate_path, measurement)
        measurements.append(measurement)
        new_candidates += 1

    if len(measurements) != len(candidates):
        return None, new_candidates

    feasible = [
        item
        for item in measurements
        if item["generic_pdr"]
        >= baselines["generic"]["pdr"] - config.calibration_pdr_tolerance
        and item["hole_pdr"]
        >= baselines["holes"]["pdr"] - config.calibration_pdr_tolerance
    ]
    if not feasible:
        raise ValueError("no feasible Phase 13 calibration candidate")
    # measurements are in canonical grid order. max() therefore preserves the
    # original first-wins tie behavior even when candidates ran in chunks.
    best = max(feasible, key=_calibration_key)
    model.set_switch_parameters(
        switch_threshold=best["switch_threshold"],
        margin_gate=best["margin_gate"],
        lifetime_gate=best["lifetime_gate"],
        onward_gate=best["onward_gate"],
        topk_onward_gate=best["topk_onward_gate"],
        redundancy_gate=best["redundancy_gate"],
        loss_keep_gate=best["loss_keep_gate"],
        predictive_margin=best["predictive_margin"],
        energy_tie_weight=best["energy_tie_weight"],
        drop_suppression_bonus=best["drop_suppression_bonus"],
    )
    return (
        {
            **best,
            "phase8_like_generic_pdr": baselines["generic"]["pdr"],
            "phase8_like_hole_pdr": baselines["holes"]["pdr"],
            "candidate_count": float(len(measurements)),
        },
        new_candidates,
    )


def run_phase13_campaign(
    config: Phase13Config,
    *,
    phase8_checkpoint_dir: Path,
    phase11_checkpoint_dir: Path,
    phase12_checkpoint_dir: Path | None = None,
    output_checkpoint_dir: Path | None = None,
    device: torch.device | str = "cpu",
    resume: bool = False,
    max_calibration_candidates: int | None = None,
    max_evaluation_units: int | None = None,
) -> dict[str, Any]:
    """Calibrate P+ safeguards and evaluate final-method ablations.

    A work unit is one calibration candidate or one scenario-method pair. When
    limits are supplied, completed units are cached below the seed checkpoint
    directory and later invocations produce the same rows in canonical order.
    """

    device = torch.device(device)
    if (
        max_calibration_candidates is not None
        and max_calibration_candidates < 1
    ):
        raise ValueError("max_calibration_candidates must be positive")
    if max_evaluation_units is not None and max_evaluation_units < 1:
        raise ValueError("max_evaluation_units must be positive")
    if (
        (
            resume
            or max_calibration_candidates is not None
            or max_evaluation_units is not None
        )
        and output_checkpoint_dir is None
    ):
        raise ValueError("resumable Phase 13 runs require output_checkpoint_dir")

    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    new_calibration_candidates = 0
    new_evaluation_units = 0
    completed_calibration_candidates = 0
    expected_calibration_candidates = (
        len(list(_candidate_switch_parameters(config)))
        * len(config.training_seeds)
    )
    completed_evaluation_units = 0
    expected_evaluation_units = (
        len(phase9_evaluation_scenarios(0))
        * len(PHASE13_METHODS)
        * len(config.training_seeds)
    )
    complete = True
    resumable = (
        resume
        or max_calibration_candidates is not None
        or max_evaluation_units is not None
    )
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
        risk_switch = _risk_switch_policy(phase8, phase11)
        if phase12_checkpoint_dir is not None:
            phase12_checkpoint = (
                phase12_checkpoint_dir
                / f"seed_{training_seed}"
                / "risk_switch_lite_globe_p.pt"
            )
            if phase12_checkpoint.is_file():
                load_checkpoint(
                    phase12_checkpoint,
                    risk_switch,
                    map_location=device,
                )
        plus = _risk_switch_plus_policy(phase8, phase11)
        checkpoint = (
            output_checkpoint_dir
            / f"seed_{training_seed}"
            / "risk_switch_lite_globe_p_plus.pt"
            if output_checkpoint_dir is not None
            else None
        )
        metrics_path = (
            checkpoint.parent / "training_metrics.json"
            if checkpoint is not None
            else None
        )
        seed_dir = checkpoint.parent if checkpoint is not None else None
        if seed_dir is not None and resumable:
            seed_dir.mkdir(parents=True, exist_ok=True)
            _ensure_run_signature(
                seed_dir,
                _run_signature(config, training_seed),
            )
        if (
            resume
            and checkpoint is not None
            and checkpoint.is_file()
            and metrics_path is not None
            and metrics_path.is_file()
        ):
            load_checkpoint(checkpoint, plus, map_location=device)
            metrics = _read_json(metrics_path)
        else:
            if seed_dir is not None and resumable:
                remaining_candidates = (
                    None
                    if max_calibration_candidates is None
                    else max(
                        max_calibration_candidates - new_calibration_candidates,
                        0,
                    )
                )
                calibrated, processed = _calibrate_plus_resumable(
                    plus,
                    config,
                    seed=training_seed,
                    device=device,
                    progress_dir=seed_dir / "calibration_progress",
                    max_new_candidates=remaining_candidates,
                )
                new_calibration_candidates += processed
                if calibrated is None:
                    complete = False
                    candidate_dir = seed_dir / "calibration_progress" / "candidates"
                    completed_calibration_candidates += len(
                        list(candidate_dir.glob("candidate_*.json"))
                    )
                    break
                metrics = {
                    "training_seed": training_seed,
                    **calibrated,
                }
            else:
                metrics = {
                    "training_seed": training_seed,
                    **_calibrate_plus(
                        plus,
                        config,
                        seed=training_seed,
                        device=device,
                    ),
                }
            if checkpoint is not None and metrics_path is not None:
                save_checkpoint(
                    checkpoint,
                    plus,
                    metadata={
                        "phase": 13,
                        "training_seed": training_seed,
                        "method": "Risk-Switch Lite-GLOBE-P+",
                    },
                )
                _write_json_atomic(metrics_path, metrics)
        calibration_progress = (
            seed_dir / "calibration_progress"
            if seed_dir is not None
            else None
        )
        if calibration_progress is not None and calibration_progress.is_dir():
            completed_calibration_candidates += len(
                list(
                    (calibration_progress / "candidates").glob(
                        "candidate_*.json"
                    )
                )
            )
        else:
            completed_calibration_candidates += len(
                list(_candidate_switch_parameters(config))
            )
        training_rows.append(metrics)
        models: dict[str, Any] = {
            "GPSR": None,
            "Predictive Geographic": None,
            "Phase 8 Geo-Residual KD": phase8,
            "Lite-GLOBE-P no-switch": phase11,
            "Risk-Switch Lite-GLOBE-P": risk_switch,
            **_ablation_models(plus),
        }
        for scenario_index, scenario in enumerate(
            phase9_evaluation_scenarios(training_seed)
        ):
            evaluation_seeds = list(
                range(
                    1_300_000 + scenario_index * 10_000,
                    1_300_000
                    + scenario_index * 10_000
                    + config.evaluation_episodes,
                )
            )
            env = FanetRoutingEnv(scenario.config)
            for method_index, method in enumerate(PHASE13_METHODS):
                unit_path = (
                    seed_dir
                    / "evaluation_progress"
                    / f"scenario_{scenario_index:02d}"
                    / f"method_{method_index:02d}.json"
                    if seed_dir is not None and resumable
                    else None
                )
                if resume and unit_path is not None and unit_path.is_file():
                    unit = _read_json(unit_path)
                    if (
                        unit.get("scenario") != scenario.name
                        or unit.get("method") != method
                        or int(unit.get("training_seed")) != training_seed
                    ):
                        raise ValueError(
                            f"incompatible cached Phase 13 evaluation unit: {unit_path}"
                        )
                else:
                    if (
                        max_evaluation_units is not None
                        and new_evaluation_units >= max_evaluation_units
                    ):
                        complete = False
                        break
                    results = evaluate_policy_results(
                        env,
                        _policy(method, models[method], env, device),
                        evaluation_seeds,
                        reset_options=scenario.reset_options,
                    )
                    unit = {
                        "training_seed": training_seed,
                        "scenario": scenario.name,
                        "method": method,
                        "episodes": [
                            episode_row(
                                result,
                                method=method,
                                scenario=scenario.name,
                                training_seed=training_seed,
                            )
                            for result in results
                        ],
                        "summary": generalization_summary(
                            results,
                            method=method,
                            scenario=scenario.name,
                            training_seed=training_seed,
                        ),
                    }
                    if unit_path is not None:
                        _write_json_atomic(unit_path, unit)
                    new_evaluation_units += 1
                episode_rows.extend(unit["episodes"])
                summary_rows.append(unit["summary"])
                completed_evaluation_units += 1
            if not complete:
                break
        if not complete:
            break
    return {
        "episodes": episode_rows,
        "seed_summaries": summary_rows,
        "training": training_rows,
        "complete": complete,
        "progress": {
            "new_calibration_candidates": new_calibration_candidates,
            "completed_calibration_candidates": completed_calibration_candidates,
            "expected_calibration_candidates": expected_calibration_candidates,
            "new_evaluation_units": new_evaluation_units,
            "completed_evaluation_units": completed_evaluation_units,
            "expected_evaluation_units": expected_evaluation_units,
        },
    }
