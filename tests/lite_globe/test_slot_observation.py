"""Contract tests for local-slot observation compaction (E-2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    LocalStudentPolicy,
)
from implementations.lite_globe.models.slot_observation import (
    CANDIDATE_KEYS,
    SlotCompactedPolicy,
    compact_observation,
    observation_nbytes,
)


def _config(**overrides) -> FanetConfig:
    base = dict(
        num_nodes=8,
        max_nodes=32,
        area_size=10.0,
        communication_radius=4.0,
        max_episode_steps=12,
        packet_ttl=12,
        min_speed=0.05,
        max_speed=0.20,
        include_node_ids=False,
        include_forwardability=True,
        include_risk_features=True,
        seed=42,
    )
    base.update(overrides)
    return FanetConfig(**base)


def _observation(seed: int, **overrides):
    env = FanetRoutingEnv(_config(**overrides))
    observation, _ = env.reset(seed=seed, options={"require_connected": True})
    return env, observation


def _to_tensors(observation, device="cpu"):
    tensors = {
        key: torch.as_tensor(np.asarray(value), device=device)
        for key, value in observation.items()
    }
    tensors["action_mask"] = tensors["action_mask"].to(torch.bool)
    for key, value in tensors.items():
        if key != "action_mask" and not torch.is_floating_point(value):
            tensors[key] = value.to(torch.float32)
    return tensors


# --------------------------------------------------------------------------
# No parameter depends on max_nodes, so checkpoints are width-agnostic.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [LocalStudentPolicy, GeographicResidualStudentPolicy, LiteGlobePStudentPolicy],
)
def test_no_parameter_depends_on_max_nodes(factory):
    wide = factory(32, hidden_dim=64)
    narrow = factory(8, hidden_dim=64)
    wide_shapes = {k: tuple(v.shape) for k, v in wide.state_dict().items()}
    narrow_shapes = {k: tuple(v.shape) for k, v in narrow.state_dict().items()}
    assert wide_shapes == narrow_shapes
    narrow.load_state_dict(wide.state_dict())  # must not raise


# --------------------------------------------------------------------------
# Compaction is a semantics-preserving re-indexing.
# --------------------------------------------------------------------------


def test_compaction_packs_valid_candidates_into_leading_slots():
    env, observation = _observation(seed=3)
    valid = np.flatnonzero(observation["action_mask"][:32])
    compacted, mapping = compact_observation(observation, max_slots=8)

    assert mapping.global_ids == tuple(int(v) for v in valid)
    assert mapping.truncated == 0
    assert int(compacted["action_mask"][: len(valid)].sum()) == len(valid)
    assert compacted["action_mask"][-1] == 1
    for key in CANDIDATE_KEYS:
        for slot, node in enumerate(mapping.global_ids):
            np.testing.assert_array_equal(
                compacted[key][slot], observation[key][node]
            )


@pytest.mark.parametrize("seed", [3, 5, 11, 19, 23])
def test_compaction_preserves_the_selected_next_hop(seed):
    """A narrow model on compacted input must choose the same global node."""

    env, observation = _observation(seed=seed)
    wide = GeographicResidualStudentPolicy(32, hidden_dim=64)
    torch.manual_seed(0)
    for parameter in wide.parameters():
        if parameter.dim() > 1:
            torch.nn.init.normal_(parameter, std=0.3)
    wide.eval()

    narrow = GeographicResidualStudentPolicy(8, hidden_dim=64)
    narrow.load_state_dict(wide.state_dict())
    narrow.eval()

    with torch.no_grad():
        wide_action = int(
            torch.argmax(wide(_to_tensors(observation)).masked_logits).item()
        )

    policy = SlotCompactedPolicy(narrow, env_drop_action=32, max_slots=8)
    policy.reset(seed)
    slot_action = policy.act(observation)
    assert slot_action == wide_action


@pytest.mark.parametrize("seed", [3, 5, 11, 19])
def test_compacted_logits_match_per_candidate(seed):
    """Per-candidate logits must be identical, not merely the argmax."""

    env, observation = _observation(seed=seed)
    wide = GeographicResidualStudentPolicy(32, hidden_dim=64)
    torch.manual_seed(1)
    for parameter in wide.parameters():
        if parameter.dim() > 1:
            torch.nn.init.normal_(parameter, std=0.3)
    wide.eval()
    narrow = GeographicResidualStudentPolicy(8, hidden_dim=64)
    narrow.load_state_dict(wide.state_dict())
    narrow.eval()

    compacted, mapping = compact_observation(observation, max_slots=8)
    with torch.no_grad():
        wide_logits = wide(_to_tensors(observation)).logits
        narrow_logits = narrow(_to_tensors(compacted)).logits

    for slot, node in enumerate(mapping.global_ids):
        assert narrow_logits[slot].item() == pytest.approx(
            wide_logits[node].item(), abs=1e-4
        )
    # The drop logit is produced from pooled context and must also agree.
    assert narrow_logits[-1].item() == pytest.approx(
        wide_logits[-1].item(), abs=1e-4
    )


def test_permutation_of_slots_does_not_change_candidate_logits():
    """The shared per-candidate encoder must be permutation-equivariant."""

    env, observation = _observation(seed=7)
    model = GeographicResidualStudentPolicy(8, hidden_dim=64)
    torch.manual_seed(2)
    for parameter in model.parameters():
        if parameter.dim() > 1:
            torch.nn.init.normal_(parameter, std=0.3)
    model.eval()

    compacted, mapping = compact_observation(observation, max_slots=8)
    occupied = len(mapping.global_ids)
    assert occupied >= 2

    permutation = np.arange(8)
    permutation[:occupied] = permutation[:occupied][::-1]
    shuffled = {
        key: (value[permutation] if key in CANDIDATE_KEYS else value.copy())
        for key, value in compacted.items()
    }
    shuffled["action_mask"] = compacted["action_mask"].copy()

    with torch.no_grad():
        base = model(_to_tensors(compacted)).logits
        moved = model(_to_tensors(shuffled)).logits

    for slot in range(occupied):
        assert moved[permutation[slot]].item() == pytest.approx(
            base[slot].item(), abs=1e-4
        )


# --------------------------------------------------------------------------
# Byte footprint: the whole point of the exercise.
# --------------------------------------------------------------------------


def test_compaction_reduces_policy_input_bytes():
    env, observation = _observation(seed=3)
    before = observation_nbytes(observation)
    compacted, _ = compact_observation(observation, max_slots=8)
    after = observation_nbytes(compacted)
    assert after < before
    # 32 -> 8 slots on the four candidate arrays is close to a 4x reduction.
    assert after < before * 0.4


def test_slot_policy_reports_its_own_input_bytes():
    env, observation = _observation(seed=3)
    model = GeographicResidualStudentPolicy(8, hidden_dim=64)
    policy = SlotCompactedPolicy(model, env_drop_action=32, max_slots=8)
    policy.reset(0)
    policy.act(observation)
    assert 0 < policy.last_input_bytes < observation_nbytes(observation)


# --------------------------------------------------------------------------
# Truncation must be explicit, never silent.
# --------------------------------------------------------------------------


def test_truncation_is_recorded_and_keeps_nearest_candidates():
    env, observation = _observation(seed=3, num_nodes=24, area_size=17.0,
                                    communication_radius=4.4)
    degree = int(observation["action_mask"][:32].sum())
    assert degree >= 3
    max_slots = degree - 2
    compacted, mapping = compact_observation(observation, max_slots=max_slots)

    assert mapping.truncated == 2
    assert len(mapping.global_ids) == max_slots
    kept = np.asarray(mapping.global_ids)
    all_valid = np.flatnonzero(observation["action_mask"][:32])
    dropped = np.setdiff1d(all_valid, kept)
    distances = observation["edge_features"][:, 0]
    assert distances[kept].max() <= distances[dropped].min() + 1e-6


def test_zero_node_ids_suppresses_global_identifiers():
    env, observation = _observation(seed=3, include_node_ids=True)
    assert not np.allclose(observation["packet_features"][4:6], 0.0)
    compacted, _ = compact_observation(observation, max_slots=8, zero_node_ids=True)
    np.testing.assert_allclose(compacted["packet_features"][4:6], 0.0)
    kept, _ = compact_observation(observation, max_slots=8, zero_node_ids=False)
    np.testing.assert_allclose(
        kept["packet_features"][4:6], observation["packet_features"][4:6]
    )


def test_switchglobe_decision_survives_compaction():
    """The full two-branch policy and its switch flag must be preserved.

    ``_switch_mask`` indexes ``candidate_risk_features`` by the argmax action,
    so it has to stay consistent once candidates are renumbered. This walks real
    trajectories from every evaluation scenario and requires both the selected
    next hop and the switch indicator to match the 32-wide reference exactly.
    """

    from implementations.lite_globe.models import SwitchGlobePolicy
    from implementations.lite_globe.scenarios import phase9_evaluation_scenarios

    def _build(width):
        torch.manual_seed(0)
        normal = GeographicResidualStudentPolicy(width, hidden_dim=64)
        predictive = LiteGlobePStudentPolicy(width, hidden_dim=64)
        for module in (normal, predictive):
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    torch.nn.init.normal_(parameter, std=0.3)
        return SwitchGlobePolicy(normal, predictive)

    wide = _build(32)
    wide.eval()
    narrow = _build(12)
    narrow.load_state_dict(wide.state_dict())
    narrow.eval()

    compared = 0
    for scenario in phase9_evaluation_scenarios(42):
        env = FanetRoutingEnv(scenario.config)
        for seed in range(6):
            try:
                observation, _ = env.reset(seed=seed, options=scenario.reset_options)
            except RuntimeError:
                continue
            for _ in range(4):
                compacted, mapping = compact_observation(observation, max_slots=12)
                with torch.no_grad():
                    reference = wide.decide(_to_tensors(observation))
                    slotted = narrow.decide(_to_tensors(compacted))

                wide_action = int(torch.argmax(reference.output.masked_logits))
                slot_action = int(torch.argmax(slotted.output.masked_logits))
                if slot_action == mapping.drop_slot:
                    resolved = 32
                else:
                    resolved = mapping.to_global(slot_action)
                assert resolved == wide_action, scenario.name
                assert bool(reference.switch.item()) == bool(slotted.switch.item())
                compared += 1

                valid = np.flatnonzero(
                    observation["action_mask"][: scenario.config.max_nodes]
                )
                if not valid.size:
                    break
                observation, _, terminated, truncated, _ = env.step(int(valid[0]))
                if terminated or truncated:
                    break
    assert compared > 200, compared


def test_slot_policy_rejects_width_mismatch():
    model = GeographicResidualStudentPolicy(32, hidden_dim=64)
    with pytest.raises(ValueError, match="max_nodes"):
        SlotCompactedPolicy(model, env_drop_action=32, max_slots=8)


def test_dead_end_returns_environment_drop_action():
    """With no valid candidate the wrapper must emit the environment DROP."""

    env, observation = _observation(seed=3)
    observation = {k: np.asarray(v).copy() for k, v in observation.items()}
    observation["action_mask"][:32] = 0
    model = GeographicResidualStudentPolicy(8, hidden_dim=64)
    policy = SlotCompactedPolicy(model, env_drop_action=32, max_slots=8)
    policy.reset(0)
    assert policy.act(observation) == 32
