"""Offline divergence calibration for a proposed risk-margin early-exit gate.

This module never changes SwitchGLOBE Exact's actions. ``GateCalibrationAdapter``
runs the unmodified ``RiskSwitchLiteGlobePStudentPolicy.decide`` on every step
(so simulated trajectories are identical to a plain ``StudentPolicyAdapter``)
and only *observes*, for a swept set of candidate danger-score cutoffs, how
often a cheap analytic gate (normal-branch danger score alone, no
predictive_policy forward pass) would have skipped the predictive branch, and
how often that skip would have changed the actual routing decision. See
``docs/method_history.md`` for why this is calibration, not a value-preserving
optimization: the switch condition's ``safer_predictive`` term depends on
``predictive_action``, so no per-step gate can be proven bit-exact without
running the predictive network.

A margin is an *absolute* cutoff on the normal branch's own danger score
(``skip iff normal_danger <= margin``), not an offset from a seed's
calibrated ``switch_threshold``: each of the five training seeds calibrates
its own threshold independently, so an offset-from-threshold margin would
silently mean a different risk tolerance per seed. ``margin=0.0`` is the
threshold-agnostic rule "skip only when the normal branch already clears
every one of its own safety gates (margin, lifetime, onward all pass)."
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from numpy.typing import NDArray
import numpy as np

from ..env.fanet_env import FanetRoutingEnv
from ..models.policy_adapter import PolicyDecision, StudentPolicyAdapter
from ..models.student_policy import RiskSwitchLiteGlobePStudentPolicy
from ..scenarios import phase9_evaluation_scenarios
from .evaluator import run_episode
from .records import episode_row
from .reporting import write_csv

DEFAULT_GATE_MARGINS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def margin_key(margin: float) -> str:
    """Stable column-name suffix for one candidate safety margin."""

    return f"{margin:.4f}"


class GateCalibrationAdapter(StudentPolicyAdapter):
    """Replay SwitchGLOBE Exact while logging a candidate early-exit gate.

    Actions are always SwitchGLOBE Exact's real actions (``decide()`` is
    called unmodified and both branches are always evaluated); the gate is
    only a passive counter recorded into ``episode_diagnostics()``.
    """

    def __init__(
        self,
        model: RiskSwitchLiteGlobePStudentPolicy,
        *,
        gate_margins: tuple[float, ...] = DEFAULT_GATE_MARGINS,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(model, RiskSwitchLiteGlobePStudentPolicy):
            raise TypeError(
                "GateCalibrationAdapter requires a RiskSwitchLiteGlobePStudentPolicy"
            )
        if not gate_margins:
            raise ValueError("gate_margins must be non-empty")
        super().__init__(
            model,
            device=device,
            deterministic=True,
            force_forward_if_available=True,
        )
        self.gate_margins = tuple(gate_margins)

    @torch.inference_mode()
    def act_with_metadata(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> PolicyDecision:
        tensors = self._tensors(observation)
        switch_decision = self.model.decide(tensors)
        diagnostics = switch_decision.diagnostics
        if diagnostics:
            keys = tuple(diagnostics)
            values = (
                torch.stack([diagnostics[key].reshape(()) for key in keys])
                .detach()
                .cpu()
                .tolist()
            )
            for key, value in zip(keys, values, strict=True):
                self._episode_diagnostics[key] = (
                    self._episode_diagnostics.get(key, 0.0) + float(value)
                )

        risk = tensors.get("candidate_risk_features")
        if risk is None:
            raise ValueError(
                "GateCalibrationAdapter requires candidate_risk_features"
            )
        max_nodes = self.model.max_nodes
        normal_action = int(switch_decision.normal_action.reshape(()).item())
        predictive_action = int(
            switch_decision.predictive_action.reshape(()).item()
        )
        switch = bool(switch_decision.switch.reshape(()).item())
        disagreement = normal_action != predictive_action
        normal_index = min(normal_action, max_nodes - 1)
        normal_risk = risk[normal_index].unsqueeze(0)
        normal_danger = float(
            self.model._danger_score(normal_risk).reshape(()).item()
        )

        for margin in self.gate_margins:
            # ``margin`` is an absolute cutoff on the normal branch's own
            # danger score, not an offset from this seed's calibrated
            # ``switch_threshold``: each seed calibrates its own threshold
            # independently (see docs/method_history.md), so an
            # offset-from-threshold margin would silently mean a different
            # risk tolerance per seed. margin=0.0 is the threshold-agnostic
            # "normal branch clears every safety gate" rule.
            would_skip = normal_danger <= margin
            key = margin_key(margin)
            self._episode_diagnostics[f"gate_skip_steps__{key}"] = (
                self._episode_diagnostics.get(f"gate_skip_steps__{key}", 0.0)
                + float(would_skip)
            )
            outcome_divergence = would_skip and switch and disagreement
            self._episode_diagnostics[
                f"gate_outcome_divergence_steps__{key}"
            ] = (
                self._episode_diagnostics.get(
                    f"gate_outcome_divergence_steps__{key}", 0.0
                )
                + float(outcome_divergence)
            )
        self._episode_diagnostics["gate_decision_steps"] = (
            self._episode_diagnostics.get("gate_decision_steps", 0.0) + 1.0
        )

        probabilities = switch_decision.output.probabilities
        action = int(torch.argmax(probabilities).item())
        if self.force_forward_if_available and action == self.model.drop_action:
            candidate_mask = tensors["action_mask"][:max_nodes].to(torch.bool)
            if torch.any(candidate_mask):
                candidate_probabilities = probabilities[:max_nodes].masked_fill(
                    ~candidate_mask, -1.0
                )
                action = int(torch.argmax(candidate_probabilities).item())
        input_bytes = self._switch_input_bytes_from_decision(
            observation, switch_decision
        )
        return PolicyDecision(action=action, input_bytes=input_bytes)


@dataclass(frozen=True)
class GateCalibrationConfig:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    gate_margins: tuple[float, ...] = DEFAULT_GATE_MARGINS


def run_gate_calibration(
    config: GateCalibrationConfig,
    *,
    switchglobe_checkpoint_dir: Path,
    device: torch.device | str = "cpu",
) -> dict[str, list[dict[str, Any]]]:
    """Replay SwitchGLOBE Exact checkpoints and log the candidate gate.

    Requires already-trained, already-calibrated SwitchGLOBE Exact
    checkpoints (``switchglobe_checkpoint_dir/seed_<seed>/switchglobe.pt``);
    this function trains nothing.
    """

    from ..experiments.external_comparison_campaign import load_switchglobe
    from ..scenarios import phase9_curriculum

    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    for training_seed in config.training_seeds:
        max_nodes = phase9_curriculum(training_seed)[0].config.max_nodes
        base_adapter = load_switchglobe(
            switchglobe_checkpoint_dir,
            seed=training_seed,
            max_nodes=max_nodes,
            hidden_dim=config.hidden_dim,
            device=device,
        )
        adapter = GateCalibrationAdapter(
            base_adapter.model,
            gate_margins=config.gate_margins,
            device=device,
        )
        for scenario_index, scenario in enumerate(
            phase9_evaluation_scenarios(training_seed)
        ):
            env = FanetRoutingEnv(scenario.config)
            evaluation_seeds = list(
                range(
                    1_100_000 + scenario_index * 10_000,
                    1_100_000
                    + scenario_index * 10_000
                    + config.evaluation_episodes,
                )
            )
            for seed in evaluation_seeds:
                result = run_episode(
                    env, adapter, seed=seed, reset_options=scenario.reset_options
                )
                row = episode_row(
                    result,
                    method="SwitchGLOBE Exact",
                    scenario=scenario.name,
                    training_seed=training_seed,
                )
                gate_diagnostics = adapter.episode_diagnostics()
                for key, value in gate_diagnostics.items():
                    if key.startswith("gate_"):
                        row[key] = value
                episode_rows.append(row)
    return {"episodes": episode_rows}


def aggregate_gate_calibration(
    episode_rows: list[dict[str, Any]],
    gate_margins: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Per (scenario, margin) and overall skip / outcome-divergence rates.

    Rates are exposure-weighted (summed steps, not mean-of-episode-means) and
    also reported per training seed so seed-to-seed spread is visible, per
    this project's seed-as-statistical-unit convention.
    """

    from collections import defaultdict

    scenarios = sorted({row["scenario"] for row in episode_rows})
    seeds = sorted({row["training_seed"] for row in episode_rows})
    rows: list[dict[str, Any]] = []
    for margin in gate_margins:
        key = margin_key(margin)
        skip_key = f"gate_skip_steps__{key}"
        divergence_key = f"gate_outcome_divergence_steps__{key}"
        for scope, scenario_filter in (
            ("overall", None),
            *((scenario, scenario) for scenario in scenarios),
        ):
            per_seed: dict[int, tuple[float, float, float]] = {}
            for seed in seeds:
                matching = [
                    row
                    for row in episode_rows
                    if row["training_seed"] == seed
                    and (scenario_filter is None or row["scenario"] == scenario_filter)
                ]
                steps = sum(row["gate_decision_steps"] for row in matching)
                skips = sum(row[skip_key] for row in matching)
                divergences = sum(row[divergence_key] for row in matching)
                per_seed[seed] = (steps, skips, divergences)
            total_steps = sum(steps for steps, _, _ in per_seed.values())
            total_skips = sum(skips for _, skips, _ in per_seed.values())
            total_divergences = sum(
                divergences for _, _, divergences in per_seed.values()
            )
            seed_skip_rates = [
                skips / steps if steps else 0.0
                for steps, skips, _ in per_seed.values()
            ]
            seed_divergence_rates = [
                divergences / skips if skips else 0.0
                for _, skips, divergences in per_seed.values()
            ]
            rows.append(
                {
                    "gate_margin": key,
                    "scope": scope,
                    "decision_steps": total_steps,
                    "skip_rate": total_skips / total_steps if total_steps else 0.0,
                    "outcome_divergence_rate_of_skipped": (
                        total_divergences / total_skips if total_skips else 0.0
                    ),
                    "outcome_divergence_rate_of_all_steps": (
                        total_divergences / total_steps if total_steps else 0.0
                    ),
                    "seed_count": len(seeds),
                    "mean_skip_rate_across_seeds": (
                        sum(seed_skip_rates) / len(seed_skip_rates)
                        if seed_skip_rates
                        else 0.0
                    ),
                    "mean_outcome_divergence_rate_across_seeds": (
                        sum(seed_divergence_rates) / len(seed_divergence_rates)
                        if seed_divergence_rates
                        else 0.0
                    ),
                }
            )
    return rows


def write_gate_calibration_artifacts(
    output_dir: Path,
    *,
    episode_rows: list[dict[str, Any]],
    gate_margins: tuple[float, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw = output_dir / "raw"
    summaries = output_dir / "summaries"
    raw.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_gate_calibration(episode_rows, gate_margins)
    write_csv(raw / "episodes.csv", episode_rows)
    write_csv(summaries / "gate_margin_statistics.csv", aggregates)
    manifest = {
        "schema_version": 1,
        "purpose": (
            "offline divergence calibration for a proposed early-exit gate; "
            "not a full-run PDR/latency result"
        ),
        "episode_rows": len(episode_rows),
        "gate_margins": [margin_key(margin) for margin in gate_margins],
        **metadata,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
