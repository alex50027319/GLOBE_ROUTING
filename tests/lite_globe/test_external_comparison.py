"""Protocol fidelity, neural resume, and external campaign contracts."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from implementations.lite_globe.baselines.aodv import AodvPolicy
from implementations.lite_globe.baselines.common import ProtocolSnapshot
from implementations.lite_globe.baselines.gat_gru_ddqn import GatGruDdqnPolicy
from implementations.lite_globe.baselines.olsr import OlsrPolicy
from implementations.lite_globe.baselines.rdqn_herp import RdqnHerpAdaptedPolicy
from implementations.lite_globe.baselines.registry import COMPARISON_METHODS, EXTERNAL_METHODS, build_method
from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation.external_comparison_reporting import SCENARIOS, paired_effects, validate_rows


def _snapshot(adjacency: np.ndarray, *, step: int = 0, current: int = 0, destination: int = 2) -> ProtocolSnapshot:
    return ProtocolSnapshot(adjacency.astype(np.bool_), np.zeros((adjacency.shape[0], 2), dtype=np.float32), step, current, destination)


def _observation() -> dict[str, np.ndarray]:
    env = FanetRoutingEnv(FanetConfig(num_nodes=3, max_nodes=4, communication_radius=1.1, min_speed=0.0, max_speed=0.0,
                                      include_forwardability=True, include_risk_features=True))
    observation, _ = env.reset(seed=1, options={"positions": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32), "source": 0, "destination": 2})
    return observation


def test_aodv_message_route_expiry_duplicate_and_link_break() -> None:
    adjacency = np.asarray([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=bool)
    policy = AodvPolicy(4, discovery_ttl=1, route_lifetime=2)
    policy.protocol_tick(_snapshot(adjacency, destination=3))
    assert policy.discover(0, 3)
    assert policy.control_messages > 0 and policy.control_bytes > 0
    assert policy.duplicate_rreq_suppressed > 0
    route = policy.routes[(0, 3)]
    broken = adjacency.copy()
    broken[0, route.next_hop] = broken[route.next_hop, 0] = False
    policy.protocol_tick(_snapshot(broken, step=1, destination=3))
    assert not route.valid and policy.route_errors >= 1
    policy.protocol_tick(_snapshot(broken, step=3, destination=3))
    assert not route.valid


def test_olsr_mpr_control_accounting_and_stale_expiry() -> None:
    adjacency = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    policy = OlsrPolicy(3, hello_interval=10, tc_interval=10, hold_time=2)
    policy.protocol_tick(_snapshot(adjacency))
    assert policy.mprs[0] == {1}
    assert policy._route(0, 2) == 1
    assert policy.control_messages > 0 and policy.control_bytes > 0
    policy.protocol_tick(_snapshot(np.zeros_like(adjacency), step=2))
    assert policy._route(0, 2) is None


def test_rdqn_herp_tiers_nstep_target_update_and_resume() -> None:
    torch.manual_seed(2)
    observation = _observation()
    policy = RdqnHerpAdaptedPolicy(4, hidden_dim=16, replay_capacity=32, n_step=2, target_interval=1)
    assert [policy.herp_tier(0.0, False, x) for x in (0.1, 0.7)] == [0, 1]
    assert policy.herp_tier(1.0, True, 0.0) == 2
    before = deepcopy(policy.online.state_dict())
    for index in range(4):
        policy.observe(observation, 1, 1.0 if index == 3 else 0.1, observation, index == 3, 0.6)
    loss = policy.learn(batch_size=2)
    assert loss is not None and np.isfinite(loss)
    assert any(not torch.equal(before[key], value) for key, value in policy.online.state_dict().items())
    assert all(torch.equal(policy.online.state_dict()[key], policy.target.state_dict()[key]) for key in policy.online.state_dict())
    action = policy.act(observation)
    restored = RdqnHerpAdaptedPolicy(4, hidden_dim=16, replay_capacity=32, n_step=2, target_interval=1)
    restored.load_checkpoint_state(policy.checkpoint_state())
    assert restored.act(observation) == action
    assert restored.environment_steps == policy.environment_steps and restored.updates == policy.updates


def test_gat_gru_dynamic_mask_hidden_reset_and_resume() -> None:
    observation = _observation()
    policy = GatGruDdqnPolicy(4, hidden_dim=16, replay_capacity=16)
    action = policy.act(observation)
    assert observation["action_mask"][action] == 1
    assert policy.online.hidden is not None
    state = policy.checkpoint_state()
    restored = GatGruDdqnPolicy(4, hidden_dim=16, replay_capacity=16)
    restored.load_checkpoint_state(state)
    assert restored.online.hidden is not None
    restored.reset(3)
    assert restored.online.hidden is None
    masked = {key: value.copy() for key, value in observation.items()}
    masked["action_mask"][:] = 0
    masked["action_mask"][-1] = 1
    assert restored.act(masked) == 4


def test_summary_contract_and_paired_effect_excludes_switchglobe() -> None:
    rows = []
    for method in COMPARISON_METHODS:
        for scenario in SCENARIOS:
            rows.append({
                "method": method, "scenario": scenario, "training_seed": 42,
                "connected_pair_pdr": 0.8, "deadline_delivery_ratio": 0.7,
                "p95_success_delay": 4.0, "energy_per_delivered_packet": 1.2,
                "decision_latency_p95_ms": 0.2, "mean_policy_input_bytes": 128.0,
            })
    validate_rows(rows, training_seeds=(42,))
    effects = paired_effects(rows)
    assert effects
    assert {row["baseline"] for row in effects} == set(EXTERNAL_METHODS)


def test_registry_policies_are_mask_safe_and_seed_deterministic() -> None:
    observation = _observation()
    adjacency = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    for method in EXTERNAL_METHODS:
        policy = build_method(method, max_nodes=4, hidden_dim=16, device="cpu")
        actions = []
        for _ in range(2):
            policy.reset(123)
            tick = getattr(policy, "protocol_tick", None)
            if tick is not None:
                tick(_snapshot(adjacency))
            actions.append(policy.act(observation))
        assert actions[0] == actions[1], method
        assert observation["action_mask"][actions[0]] == 1, method
