"""SwitchGLOBE policy tests (historically developed as Phase 12)."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models import (
    EvoFusionSwitchGlobePolicy,
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.models.policy_adapter import (
    StudentPolicyAdapter,
)
from implementations.lite_globe.models.tensor_observation import (
    observation_to_tensors,
)
from implementations.lite_globe.scenarios import (
    predictive_break_config,
    predictive_break_options,
)


def _policy(max_nodes: int) -> SwitchGlobePolicy:
    normal = GeographicResidualStudentPolicy(max_nodes, hidden_dim=32)
    predictive = LiteGlobePStudentPolicy(
        max_nodes,
        hidden_dim=32,
        initial_predictive_strength=(0.75, 3.0, 0.25, 6.0),
        initial_break_penalty=18.0,
        initial_residual_bound=1.5,
    )
    predictive.set_residual_weight(0.0)
    return SwitchGlobePolicy(
        normal,
        predictive,
        switch_threshold=0.05,
        margin_gate=0.04,
        lifetime_gate=0.20,
        onward_gate=0.20,
    )


def _fusion_policy(
    max_nodes: int,
    *,
    minimum_fusion_weight: float = 0.0,
    maximum_fusion_weight: float = 0.25,
) -> EvoFusionSwitchGlobePolicy:
    exact = _policy(max_nodes)
    return EvoFusionSwitchGlobePolicy(
        deepcopy(exact.normal_policy),
        deepcopy(exact.predictive_policy),
        switch_threshold=float(exact.switch_threshold.item()),
        margin_gate=float(exact.margin_gate.item()),
        lifetime_gate=float(exact.lifetime_gate.item()),
        onward_gate=float(exact.onward_gate.item()),
        minimum_fusion_weight=minimum_fusion_weight,
        maximum_fusion_weight=maximum_fusion_weight,
    )


def test_zero_weight_evo_fusion_exactly_matches_switchglobe() -> None:
    config = predictive_break_config(42)
    observation, _ = FanetRoutingEnv(config).reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    exact = _policy(config.max_nodes).eval()
    fusion = EvoFusionSwitchGlobePolicy(
        deepcopy(exact.normal_policy),
        deepcopy(exact.predictive_policy),
        switch_threshold=float(exact.switch_threshold.item()),
        margin_gate=float(exact.margin_gate.item()),
        lifetime_gate=float(exact.lifetime_gate.item()),
        onward_gate=float(exact.onward_gate.item()),
        minimum_fusion_weight=0.0,
        maximum_fusion_weight=0.0,
    ).eval()

    tensors = observation_to_tensors(observation)
    exact_decision = exact.decide(tensors)
    fusion_decision = fusion.decide(tensors)

    torch.testing.assert_close(
        fusion_decision.output.logits,
        exact_decision.output.logits,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(fusion_decision.switch, exact_decision.switch)
    assert torch.equal(
        fusion_decision.predictive_action,
        exact_decision.predictive_action,
    )
    assert set(fusion.state_dict()) == set(exact.state_dict())
    fusion.load_state_dict(exact.state_dict(), strict=True)


def test_evo_fusion_correction_is_bounded() -> None:
    config = predictive_break_config(42)
    observation, _ = FanetRoutingEnv(config).reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    fusion = _fusion_policy(
        config.max_nodes,
        minimum_fusion_weight=0.25,
        maximum_fusion_weight=0.25,
    ).eval()
    tensors = observation_to_tensors(observation)
    normal = fusion.normal_policy(tensors)
    pure = fusion.predictive_policy(tensors)
    fused = fusion._predictive_output(tensors, normal)
    bound = float(
        torch.nn.functional.softplus(
            fusion.predictive_policy.log_residual_bound
        ).item()
    )
    correction = torch.abs(
        fused.logits[: config.max_nodes]
        - pure.logits[: config.max_nodes]
    )
    assert torch.max(correction).item() <= 0.25 * bound + 1e-6
    torch.testing.assert_close(
        fused.logits[config.max_nodes],
        pure.logits[config.max_nodes],
    )


def test_evo_fusion_rejects_invalid_configuration() -> None:
    exact = _policy(4)
    with pytest.raises(ValueError, match="fusion weights"):
        EvoFusionSwitchGlobePolicy(
            exact.normal_policy,
            exact.predictive_policy,
            minimum_fusion_weight=0.5,
            maximum_fusion_weight=0.25,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_evo_fusion_cpu_and_cuda_choose_the_same_action() -> None:
    config = predictive_break_config(42)
    observation, _ = FanetRoutingEnv(config).reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    cpu = _fusion_policy(config.max_nodes).eval()
    cuda = deepcopy(cpu).to("cuda").eval()
    cpu_decision = cpu.decide(observation_to_tensors(observation))
    cuda_decision = cuda.decide(
        observation_to_tensors(observation, device="cuda")
    )

    assert int(torch.argmax(cpu_decision.output.masked_logits).item()) == int(
        torch.argmax(cuda_decision.output.masked_logits).item()
    )
    torch.testing.assert_close(
        cuda_decision.output.logits.cpu(),
        cpu_decision.output.logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_risk_switch_uses_predictive_branch_on_break_trap() -> None:
    config = predictive_break_config(42)
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    policy = _policy(config.max_nodes)
    action = StudentPolicyAdapter(
        policy,
        force_forward_if_available=True,
    ).act(observation)
    assert action == 3


def test_risk_switch_can_be_disabled_to_match_phase8() -> None:
    config = predictive_break_config(42)
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    policy = _policy(config.max_nodes)
    phase8_action = StudentPolicyAdapter(
        policy.normal_policy,
        force_forward_if_available=True,
    ).act(observation)
    policy.set_switch_parameters(
        switch_threshold=3.0,
        margin_gate=0.0,
        lifetime_gate=0.0,
        onward_gate=0.0,
    )
    switched_action = StudentPolicyAdapter(
        policy,
        force_forward_if_available=True,
    ).act(observation)
    assert switched_action == phase8_action


def test_risk_switch_observation_bytes_are_below_full_predictive_when_safe(
    line_positions,
) -> None:
    from implementations.lite_globe.env.config import FanetConfig

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
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    policy.set_switch_parameters(
        switch_threshold=3.0,
        margin_gate=0.0,
        lifetime_gate=0.0,
        onward_gate=0.0,
    )
    switch_bytes = StudentPolicyAdapter(policy).observation_bytes(observation)
    predictive = deepcopy(policy.predictive_policy)
    predictive.set_residual_weight(0.0)
    predictive_bytes = StudentPolicyAdapter(predictive).observation_bytes(
        observation
    )
    assert switch_bytes < predictive_bytes


def test_fused_decision_runs_each_branch_once(line_positions) -> None:
    from implementations.lite_globe.env.config import FanetConfig

    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0,
        communication_radius=1.1, max_queue_size=8,
        min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    adapter = StudentPolicyAdapter(policy, force_forward_if_available=True)
    counts = {"normal": 0, "predictive": 0}
    normal_hook = policy.normal_policy.register_forward_hook(
        lambda *_: counts.__setitem__("normal", counts["normal"] + 1)
    )
    predictive_hook = policy.predictive_policy.register_forward_hook(
        lambda *_: counts.__setitem__(
            "predictive", counts["predictive"] + 1
        )
    )
    try:
        decision = adapter.act_with_metadata(observation)
    finally:
        normal_hook.remove()
        predictive_hook.remove()
    assert decision.action < config.max_nodes
    assert counts == {"normal": 1, "predictive": 1}


def test_fused_metadata_matches_legacy_byte_accounting(line_positions) -> None:
    from implementations.lite_globe.env.config import FanetConfig

    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0,
        communication_radius=1.1, max_queue_size=8,
        min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    adapter = StudentPolicyAdapter(policy, force_forward_if_available=True)
    legacy_bytes = adapter.observation_bytes(observation)
    decision = adapter.act_with_metadata(observation)
    assert decision.input_bytes == legacy_bytes
    assert all(
        torch.isfinite(parameter).all() for parameter in policy.parameters()
    )


def test_buffered_adapter_matches_eager_action(line_positions) -> None:
    from implementations.lite_globe.env.config import FanetConfig

    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0,
        communication_radius=1.1, max_queue_size=8,
        min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    eager = StudentPolicyAdapter(
        deepcopy(policy), force_forward_if_available=True
    )
    buffered = StudentPolicyAdapter(
        deepcopy(policy), force_forward_if_available=True,
        reuse_tensor_buffer=True,
    )
    assert (
        eager.act_with_metadata(observation)
        == buffered.act_with_metadata(observation)
    )
