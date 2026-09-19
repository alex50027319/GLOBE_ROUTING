"""Contract tests for beacon-driven observation degradation."""

from __future__ import annotations

import numpy as np
import pytest

from implementations.lite_globe.env.beacon import (
    BEACON_PAYLOAD_BYTES,
    BeaconCache,
    BeaconDegradation,
)
from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.scenarios.degradation_suite import (
    apply_degradation,
    degradation_grid,
    staleness_axis,
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
        include_forwardability=True,
        include_risk_features=True,
        seed=42,
    )
    base.update(overrides)
    return FanetConfig(**base)


def _greedy_action(observation, max_nodes: int, visited: set[int]) -> int:
    """Pick the unvisited valid neighbour with the most geographic progress.

    A first-valid-index policy loops within two hops, which ends episodes before
    beacon staleness can accumulate. Greedy forwarding produces the multi-hop
    trajectories these tests need.
    """

    valid = np.flatnonzero(observation["action_mask"][:max_nodes])
    candidates = [int(v) for v in valid if int(v) not in visited]
    if not candidates:
        return max_nodes  # DROP
    destination_delta = observation["packet_features"][:2]
    current_distance = float(np.linalg.norm(destination_delta))
    best, best_progress = candidates[0], -np.inf
    for node in candidates:
        neighbour_delta = observation["neighbor_features"][node][:2]
        progress = current_distance - float(
            np.linalg.norm(destination_delta - neighbour_delta)
        )
        if progress > best_progress:
            best, best_progress = node, progress
    return best


def _rollout(config: FanetConfig, *, seed: int, steps: int = 12):
    """Run a deterministic greedy rollout and collect every observation."""

    env = FanetRoutingEnv(config)
    observation, info = env.reset(seed=seed, options={"require_connected": True})
    frames = [{k: np.asarray(v).copy() for k, v in observation.items()}]
    infos = [dict(info)]
    visited = {env.packet.current}
    for _ in range(steps):
        action = _greedy_action(observation, config.max_nodes, visited)
        visited.add(action)
        observation, _, terminated, truncated, info = env.step(action)
        frames.append({k: np.asarray(v).copy() for k, v in observation.items()})
        infos.append(dict(info))
        if terminated or truncated:
            break
    return frames, infos


# --------------------------------------------------------------------------
# The default must be a bit-exact no-op so every existing artifact reproduces.
# --------------------------------------------------------------------------


def test_default_beacon_setting_is_disabled():
    assert FanetConfig().beacon == BeaconDegradation()
    assert not FanetConfig().beacon.enabled
    assert FanetRoutingEnv(_config()).beacon_cache is None


def test_disabled_degradation_is_bit_identical():
    """An explicitly-disabled setting must match the historical observations."""

    baseline, base_infos = _rollout(_config(), seed=7)
    explicit = _config(beacon=BeaconDegradation(beacon_period=1))
    degraded, deg_infos = _rollout(explicit, seed=7)

    assert len(baseline) == len(degraded)
    for before, after in zip(baseline, degraded):
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    for before, after in zip(base_infos, deg_infos):
        assert before["path"] == after["path"]
        assert before["delivered"] == after["delivered"]


def test_disabled_degradation_does_not_consume_rng():
    """Enabling the code path must not perturb the mobility RNG stream."""

    a, _ = _rollout(_config(), seed=11)
    b, _ = _rollout(_config(beacon=BeaconDegradation()), seed=11)
    np.testing.assert_array_equal(
        a[-1]["neighbor_features"], b[-1]["neighbor_features"]
    )


# --------------------------------------------------------------------------
# Degradation must actually degrade, and do so reproducibly.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings",
    [
        BeaconDegradation(beacon_period=5),
        BeaconDegradation(beacon_loss=0.3, dead_reckon=False),
        BeaconDegradation(position_noise_std=0.15, velocity_noise_std=0.15),
    ],
)
def test_degradation_changes_risk_features(settings):
    baseline, _ = _rollout(_config(), seed=3)
    degraded, _ = _rollout(_config(beacon=settings), seed=3)
    changed = any(
        not np.array_equal(b["candidate_risk_features"], d["candidate_risk_features"])
        for b, d in zip(baseline, degraded)
    )
    assert changed, f"{settings} produced identical risk features"


def test_one_step_dead_reckoning_is_exact_under_random_waypoint():
    """Document why beacon loss alone is nearly free in this simulator.

    Random Waypoint mobility is piecewise constant-velocity, and
    ``predicted_link_lifetime_steps`` assumes constant relative velocity. The
    two models coincide, so a one-step-stale dead-reckoned beacon reproduces the
    true position to floating-point precision and the risk features do not move.

    This is a property of the mobility model, not of the beacon code. Under an
    accelerating model (Gauss-Markov, Paparazzi) the same staleness would carry
    real error, which is the point of the degradation study.
    """

    from implementations.lite_globe.env.mobility import RandomWaypointMobility

    rng = np.random.default_rng(0)
    mobility = RandomWaypointMobility(8, 10.0, 0.6, 1.2, 1.0, 0.25)
    state = mobility.reset(rng)
    cache = BeaconCache(
        BeaconDegradation(beacon_period=100),  # refresh only at reset
        num_nodes=8,
        communication_radius=4.0,
        max_speed=1.2,
        time_step=1.0,
    )
    cache.reset(state.positions, state.velocities, rng)

    state = mobility.step(state, rng)
    cache.step(state.positions, state.velocities, rng)
    believed, _ = cache.believed_state()
    np.testing.assert_allclose(believed, state.positions, rtol=0, atol=1e-5)

    # Error only appears once nodes reach waypoints and change velocity.
    errors = []
    for _ in range(5):
        state = mobility.step(state, rng)
        cache.step(state.positions, state.velocities, rng)
        believed, _ = cache.believed_state()
        errors.append(float(np.max(np.linalg.norm(believed - state.positions, axis=1))))
    assert errors[-1] > 1.0, errors


def test_degradation_is_deterministic_under_seed():
    settings = BeaconDegradation(5, 0.3, 0.15, 0.15)
    first, _ = _rollout(_config(beacon=settings), seed=5)
    second, _ = _rollout(_config(beacon=settings), seed=5)
    for a, b in zip(first, second):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_observations_stay_valid_under_severe_degradation():
    """Degraded beliefs must not break the observation contract."""

    settings = BeaconDegradation(10, 0.3, 0.15, 0.15, degrade_topology=True)
    frames, _ = _rollout(_config(beacon=settings), seed=9)
    for frame in frames:
        for key, value in frame.items():
            assert np.all(np.isfinite(value)), key
        mask = frame["action_mask"]
        assert set(np.unique(mask)).issubset({0, 1})
        assert mask[-1] == 1  # DROP must remain available


def test_relay_knows_its_own_state_exactly():
    """Self features are never degraded; only neighbours come from beacons."""

    config = _config(beacon=BeaconDegradation(10, 0.0, 0.15, 0.15))
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(seed=13, options={"require_connected": True})
    current = env.packet.current
    np.testing.assert_allclose(
        observation["self_features"][:2],
        env.mobility.positions[current] / config.area_size,
        rtol=0,
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# Staleness must grow with the beacon period.
# --------------------------------------------------------------------------


def test_mean_beacon_age_grows_with_period():
    ages = []
    for period in (1, 5, 10):
        _, infos = _rollout(
            _config(beacon=BeaconDegradation(beacon_period=period)), seed=17, steps=10
        )
        # period 1 is the exact setting, which reports no beacon fields.
        ages.append(infos[-1].get("mean_beacon_age_steps", 0.0))
    assert ages[0] <= ages[1] <= ages[2]
    assert ages[0] == pytest.approx(0.0)
    assert ages[2] > 0.0


def test_beacon_control_accounting_is_reported():
    _, infos = _rollout(
        _config(beacon=BeaconDegradation(beacon_period=2)), seed=19, steps=8
    )
    final = infos[-1]
    assert final["beacon_messages"] > 0
    assert final["beacon_bytes"] == final["beacon_messages"] * BEACON_PAYLOAD_BYTES
    assert 0.0 <= final["beacon_delivery_ratio"] <= 1.0
    assert final["beacon_setting"] == "T2_L0_P0_V0"


def test_exact_setting_reports_no_beacon_fields():
    _, infos = _rollout(_config(), seed=21, steps=4)
    assert "beacon_bytes" not in infos[-1]


# --------------------------------------------------------------------------
# Topology degradation must be able to cause real link failures.
# --------------------------------------------------------------------------


def test_topology_degradation_can_cause_link_failure():
    """A wrong neighbour set must be able to produce a failed transmission.

    Staleness alone is not enough under Random Waypoint mobility (see
    ``test_one_step_dead_reckoning_is_exact_under_random_waypoint``), so this
    uses sensor noise, which is what actually moves a believed neighbour across
    the radio boundary.
    """

    settings = BeaconDegradation(
        beacon_period=5,
        position_noise_std=0.20,
        velocity_noise_std=0.20,
        degrade_topology=True,
    )
    reasons: set[str] = set()
    for seed in range(60):
        config = _config(beacon=settings, max_speed=1.2, min_speed=0.6)
        _, infos = _rollout(config, seed=seed, steps=12)
        reason = infos[-1]["drop_reason"]
        if reason:
            reasons.add(reason)
    assert "link_failure" in reasons, reasons


def test_feature_only_degradation_never_causes_link_failure():
    """Without ``degrade_topology`` the action mask stays exact."""

    settings = BeaconDegradation(
        beacon_period=10, position_noise_std=0.20, velocity_noise_std=0.20
    )
    for seed in range(40):
        config = _config(beacon=settings, max_speed=1.2, min_speed=0.6)
        _, infos = _rollout(config, seed=seed, steps=12)
        assert infos[-1]["drop_reason"] != "link_failure"


# --------------------------------------------------------------------------
# Grid construction.
# --------------------------------------------------------------------------


def test_degradation_grid_is_deduplicated_and_anchored():
    grid = degradation_grid()
    assert grid[0].label == "exact"
    assert grid[0].settings == BeaconDegradation()
    settings = [cell.settings for cell in grid]
    assert len(settings) == len(set(settings))
    assert {cell.axis for cell in grid} >= {"staleness", "loss", "noise", "combined"}


def test_apply_degradation_suffixes_names_and_sets_config():
    from implementations.lite_globe.scenarios.evaluation_suite import (
        EvaluationScenario,
    )

    base = [EvaluationScenario("heldout_medium", _config(), None, "in_distribution")]
    cell = staleness_axis()[2]
    out = apply_degradation(base, cell)
    assert out[0].name == f"heldout_medium__{cell.label}"
    assert out[0].config.beacon == cell.settings
    assert out[0].distribution.endswith("__staleness")


def test_beacon_cache_dead_reckoning_matches_constant_velocity():
    settings = BeaconDegradation(beacon_period=100)  # never refresh after reset
    cache = BeaconCache(
        settings,
        num_nodes=3,
        communication_radius=4.0,
        max_speed=1.0,
        time_step=1.0,
    )
    rng = np.random.default_rng(0)
    positions = np.zeros((3, 2), dtype=np.float32)
    velocities = np.ones((3, 2), dtype=np.float32)
    cache.reset(positions, velocities, rng)
    for _ in range(4):
        cache.step(positions, velocities, rng)
    believed, _ = cache.believed_state()
    np.testing.assert_allclose(believed, np.full((3, 2), 4.0), rtol=0, atol=1e-5)
