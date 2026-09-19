"""EarlyExitSwitchGlobePolicy must match SwitchGLOBE Exact whenever it acts.

The early exit is only taken when the normal branch's own candidate is not
DROP, a candidate exists, and its danger score is exactly zero -- see
artifacts/gated_switchglobe/calibration for the empirical validation (zero
divergence across the full 5-seed x 14-scenario x 200-episode evaluation
set) that motivates this specific condition.
"""

from __future__ import annotations

import numpy as np
import torch

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models import (
    EarlyExitSwitchGlobePolicy,
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.models.policy_adapter import StudentPolicyAdapter
from implementations.lite_globe.scenarios import (
    predictive_break_config,
    predictive_break_options,
)


def _shared_policies(max_nodes: int) -> tuple[SwitchGlobePolicy, EarlyExitSwitchGlobePolicy]:
    """Two wrappers around the *same* normal/predictive weight objects."""

    normal = GeographicResidualStudentPolicy(max_nodes, hidden_dim=32)
    predictive = LiteGlobePStudentPolicy(
        max_nodes,
        hidden_dim=32,
        initial_predictive_strength=(0.75, 3.0, 0.25, 6.0),
        initial_break_penalty=18.0,
        initial_residual_bound=1.5,
    )
    predictive.set_residual_weight(0.0)
    kwargs = dict(
        switch_threshold=0.05,
        margin_gate=0.04,
        lifetime_gate=0.20,
        onward_gate=0.20,
    )
    exact = SwitchGlobePolicy(normal, predictive, **kwargs)
    early_exit = EarlyExitSwitchGlobePolicy(normal, predictive, **kwargs)
    return exact, early_exit


def test_safe_step_skips_predictive_network_and_matches_exact(
    line_positions,
) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=4,
        area_size=10.0,
        communication_radius=1.1,
        max_queue_size=8,
        min_speed=0.0,
        max_speed=0.0,
        include_forwardability=True,
        include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    exact, early_exit = _shared_policies(config.max_nodes)

    # Force this step into the "normal branch already fully safe" regime
    # regardless of the randomly-initialized network's own risk read: the
    # early-exit condition only needs candidate_risk_features, not a
    # particular scenario.
    safe_risk = torch.zeros_like(
        torch.as_tensor(observation["candidate_risk_features"])
    )
    safe_risk[:, 0] = 10.0  # margin
    safe_risk[:, 1] = 10.0  # lifetime
    safe_risk[:, 3] = 10.0  # onward
    observation = {**observation, "candidate_risk_features": safe_risk.numpy()}

    exact_action = StudentPolicyAdapter(
        exact, force_forward_if_available=True
    ).act(observation)

    # normal_policy and predictive_policy are the *same* nn.Module objects
    # shared by both wrappers (see _shared_policies), so hooks are only
    # registered after exercising `exact` -- otherwise its own decide() call
    # would also increment these counters.
    counts = {"normal": 0, "predictive": 0}
    early_exit.normal_policy.register_forward_hook(
        lambda *_: counts.__setitem__("normal", counts["normal"] + 1)
    )
    early_exit.predictive_policy.register_forward_hook(
        lambda *_: counts.__setitem__("predictive", counts["predictive"] + 1)
    )
    early_exit_action = StudentPolicyAdapter(
        early_exit, force_forward_if_available=True
    ).act(observation)

    assert early_exit_action == exact_action
    assert counts["normal"] == 1
    assert counts["predictive"] == 0


def test_risky_step_still_runs_predictive_network_and_matches_exact() -> None:
    config = predictive_break_config(42)
    reset_options = predictive_break_options(0.0)
    observation, _ = FanetRoutingEnv(config).reset(seed=42, options=reset_options)
    exact, early_exit = _shared_policies(config.max_nodes)

    exact_action = StudentPolicyAdapter(
        exact, force_forward_if_available=True
    ).act(observation)

    counts = {"predictive": 0}
    early_exit.predictive_policy.register_forward_hook(
        lambda *_: counts.__setitem__("predictive", counts["predictive"] + 1)
    )
    early_exit_action = StudentPolicyAdapter(
        early_exit, force_forward_if_available=True
    ).act(observation)

    assert early_exit_action == exact_action
    assert counts["predictive"] == 1


def test_drop_candidate_never_takes_the_fast_path(line_positions) -> None:
    """A DROP pick must always fall back, even if risk looks safe everywhere.

    ``candidate_risk_features`` is indexed at the clamped normal action, so a
    DROP pick would otherwise read an unrelated candidate row and could look
    spuriously safe -- this is exactly the edge case the ``normal_is_drop``
    guard exists for.
    """

    config = FanetConfig(
        num_nodes=3,
        max_nodes=4,
        area_size=10.0,
        communication_radius=1.1,
        max_queue_size=8,
        min_speed=0.0,
        max_speed=0.0,
        include_forwardability=True,
        include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    exact, early_exit = _shared_policies(config.max_nodes)

    action_mask = np.array(observation["action_mask"], copy=True)
    action_mask[: config.max_nodes] = 0.0  # no real candidate is selectable
    action_mask[config.max_nodes] = 1.0  # DROP stays available
    safe_risk = torch.zeros_like(
        torch.as_tensor(observation["candidate_risk_features"])
    )
    safe_risk[:, 0] = 10.0
    safe_risk[:, 1] = 10.0
    safe_risk[:, 3] = 10.0
    observation = {
        **observation,
        "action_mask": action_mask,
        "candidate_risk_features": safe_risk.numpy(),
    }

    exact_decision = exact.decide(
        {k: torch.as_tensor(v) for k, v in observation.items()}
    )

    counts = {"predictive": 0}
    early_exit.predictive_policy.register_forward_hook(
        lambda *_: counts.__setitem__("predictive", counts["predictive"] + 1)
    )
    early_exit_decision = early_exit.decide(
        {k: torch.as_tensor(v) for k, v in observation.items()}
    )

    assert bool(exact_decision.normal_action.item() == config.max_nodes)
    assert counts["predictive"] == 1
    assert torch.equal(
        early_exit_decision.output.probabilities, exact_decision.output.probabilities
    )
