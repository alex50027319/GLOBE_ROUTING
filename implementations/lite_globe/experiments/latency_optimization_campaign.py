"""Train and evaluate exact and distilled SwitchGLOBE latency variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..algorithms.distillation import forward_kl_loss
from ..baselines.common import atomic_torch_save
from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import (
    episode_row, evaluate_policy_results, generalization_summary,
    profile_student_policy,
)
from ..models import FastSwitchGlobePolicy
from ..models.policy_adapter import StudentPolicyAdapter
from ..models.tensor_observation import observation_to_tensors
from ..provenance import config_sha256 as _shared_config_sha256
from ..scenarios import phase9_evaluation_scenarios
from .external_comparison_campaign import load_switchglobe, training_scenarios


ORIGINAL = "SwitchGLOBE Exact"
FAST = "FastSwitchGLOBE Distilled"


@dataclass(frozen=True)
class LatencyOptimizationConfig:
    training_seeds: tuple[int, ...] = (42, 77, 123, 314, 2718)
    dataset_episodes_per_scenario: int = 100
    evaluation_episodes: int = 200
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 1.0
    action_coefficient: float = 1.0
    switch_coefficient: float = 0.2
    hidden_dim: int = 32
    warmup: int = 50
    repeats: int = 500
    routing_step_duration_ms: float = 10.0
    benchmark_compile: bool = False
    enable_freshness_cache: bool = False
    freshness_cache_ttl_ms: float = 5.0
    freshness_cache_capacity: int = 128


class _ArrayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays, self.indices = arrays, indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[index]); item = {}
        for key in (
            "self_features", "neighbor_features", "edge_features",
            "packet_features", "candidate_forwardability",
            "candidate_risk_features",
        ):
            item[key] = torch.as_tensor(self.arrays[key][i], dtype=torch.float32)
        item["action_mask"] = torch.as_tensor(
            self.arrays["action_mask"][i], dtype=torch.bool
        )
        item["teacher_logits"] = torch.as_tensor(
            self.arrays["teacher_logits"][i], dtype=torch.float32
        )
        item["selected_action"] = torch.tensor(
            int(self.arrays["selected_action"][i]), dtype=torch.long
        )
        item["switch_target"] = torch.tensor(
            float(self.arrays["switch_target"][i]), dtype=torch.float32
        )
        return item


def _observation_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {
        key: batch[key].to(device)
        for key in (
            "self_features", "neighbor_features", "edge_features",
            "packet_features", "action_mask", "candidate_forwardability",
            "candidate_risk_features",
        )
    }


@torch.inference_mode()
def collect_teacher_data(
    teacher: StudentPolicyAdapter, scenarios, *, seed: int,
    episodes_per_scenario: int,
) -> dict[str, np.ndarray]:
    records: dict[str, list[Any]] = {key: [] for key in (
        "self_features", "neighbor_features", "edge_features", "packet_features",
        "action_mask", "candidate_forwardability", "candidate_risk_features",
        "teacher_logits", "selected_action", "switch_target", "episode_seed",
        "scenario_id",
    )}
    rng = np.random.default_rng(seed + 71_000)
    model = teacher.model
    for scenario_index, scenario in enumerate(scenarios):
        env = FanetRoutingEnv(scenario.config)
        for _ in range(episodes_per_scenario):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            observation, _ = env.reset(
                seed=episode_seed, options=scenario.reset_options
            )
            done = False
            while not done:
                tensors = observation_to_tensors(
                    observation, device=teacher.device
                )
                decision = model.decide(tensors)
                probabilities = decision.output.probabilities
                action = int(torch.argmax(probabilities).item())
                if (
                    action == model.drop_action
                    and teacher.force_forward_if_available
                ):
                    mask = tensors["action_mask"][: model.max_nodes].bool()
                    if torch.any(mask):
                        action = int(torch.argmax(
                            probabilities[: model.max_nodes].masked_fill(~mask, -1.0)
                        ).item())
                for key in (
                    "self_features", "neighbor_features", "edge_features",
                    "packet_features", "action_mask", "candidate_forwardability",
                    "candidate_risk_features",
                ):
                    records[key].append(np.array(observation[key], copy=True))
                records["teacher_logits"].append(
                    decision.output.logits.detach().cpu().numpy().copy()
                )
                records["selected_action"].append(action)
                records["switch_target"].append(
                    int(decision.switch.reshape(-1)[0].item())
                )
                records["episode_seed"].append(episode_seed)
                records["scenario_id"].append(scenario_index)
                observation, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
    return {key: np.asarray(values) for key, values in records.items()}


def _split_indices(arrays: dict[str, np.ndarray], *, seed: int):
    groups = np.asarray([
        f"{scenario}:{episode}"
        for scenario, episode in zip(
            arrays["scenario_id"], arrays["episode_seed"], strict=True
        )
    ])
    unique = np.unique(groups); rng = np.random.default_rng(seed + 91_000)
    unique = unique[rng.permutation(unique.size)]
    train_end = max(1, int(unique.size * 0.8))
    validation_end = max(train_end + 1, int(unique.size * 0.9))
    validation_end = min(validation_end, unique.size - 1)
    return tuple(
        np.flatnonzero(np.isin(groups, selected)).astype(np.int64)
        for selected in (
            unique[:train_end], unique[train_end:validation_end],
            unique[validation_end:],
        )
    )


@torch.inference_mode()
def _agreement(
    model: FastSwitchGlobePolicy, dataset: _ArrayDataset, *,
    device: torch.device, batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    actions = switches = count = 0; weighted_kl = 0.0
    for batch in loader:
        observation = _observation_batch(batch, device)
        output, switch_logit = model.forward_with_auxiliary(observation)
        kl, _, _ = forward_kl_loss(
            batch["teacher_logits"].to(device), output.logits,
            observation["action_mask"], temperature=1.0,
        )
        n = int(output.logits.shape[0]); count += n; weighted_kl += float(kl.item()) * n
        actions += int(torch.sum(
            torch.argmax(output.probabilities, dim=-1)
            == batch["selected_action"].to(device)
        ).item())
        switches += int(torch.sum(
            (switch_logit >= 0) == (batch["switch_target"].to(device) >= 0.5)
        ).item())
    return {
        "kl": weighted_kl / max(count, 1),
        "action_agreement": actions / max(count, 1),
        "switch_accuracy": switches / max(count, 1),
        "samples": count,
    }


def train_fast_policy(
    arrays: dict[str, np.ndarray], *, max_nodes: int,
    config: LatencyOptimizationConfig, seed: int, device: torch.device,
) -> tuple[FastSwitchGlobePolicy, dict[str, Any]]:
    torch.manual_seed(seed + 101_000)
    train_idx, validation_idx, test_idx = _split_indices(arrays, seed=seed)
    train = _ArrayDataset(arrays, train_idx)
    validation = _ArrayDataset(arrays, validation_idx)
    test = _ArrayDataset(arrays, test_idx)
    model = FastSwitchGlobePolicy(max_nodes, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 102_000),
    )
    best_state = None; best_validation = float("inf"); epochs = 0
    best_epoch = 0
    for epoch in range(config.epochs):
        model.train()
        for batch in loader:
            observation = _observation_batch(batch, device)
            output, switch_logit = model.forward_with_auxiliary(observation)
            kl, _, _ = forward_kl_loss(
                batch["teacher_logits"].to(device), output.logits,
                observation["action_mask"],
                temperature=config.temperature,
            )
            action_loss = torch.nn.functional.cross_entropy(
                output.masked_logits, batch["selected_action"].to(device)
            )
            switch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                switch_logit, batch["switch_target"].to(device)
            )
            loss = (
                kl + config.action_coefficient * action_loss
                + config.switch_coefficient * switch_loss
            )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epochs = epoch + 1; model.eval()
        metrics = _agreement(
            model, validation, device=device, batch_size=config.batch_size
        )
        if metrics["kl"] < best_validation:
            best_validation = metrics["kl"]
            best_epoch = epochs
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("FastSwitchGLOBE training produced no checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    return model, {
        "epochs": epochs, "best_epoch": best_epoch,
        "training_samples": len(train),
        "validation_samples": len(validation), "test_samples": len(test),
        "best_validation_kl": best_validation,
        "validation": _agreement(
            model, validation, device=device, batch_size=config.batch_size
        ),
        "test": _agreement(
            model, test, device=device, batch_size=config.batch_size
        ),
    }


def config_sha256(config: LatencyOptimizationConfig) -> str:
    """Hash the training-relevant config so resume can detect drift.

    Delegates to the shared ``provenance.config_sha256`` (identical
    ``json.dumps(asdict(config), sort_keys=True)`` serialization) so this
    stays byte-for-byte compatible with existing stored checkpoint hashes.
    """

    return _shared_config_sha256(config)


def checkpoint_path(root: Path, seed: int) -> Path:
    return root / f"seed_{seed}" / "fast_switchglobe.pt"


def train_or_load_fast(
    teacher: StudentPolicyAdapter, *, config: LatencyOptimizationConfig,
    seed: int, checkpoint_dir: Path, device: torch.device, resume: bool,
) -> tuple[StudentPolicyAdapter, dict[str, Any]]:
    path = checkpoint_path(checkpoint_dir, seed)
    expected_hash = config_sha256(config)
    if resume and path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        hash_matches = payload.get("config_sha256") == expected_hash
        if (
            payload.get("complete")
            and int(payload.get("training_seed", -1)) == seed
            and hash_matches
        ):
            model = FastSwitchGlobePolicy(
                teacher.model.max_nodes, hidden_dim=config.hidden_dim
            )
            model.load_state_dict(payload["model_state"])
            return StudentPolicyAdapter(
                model, device=device, force_forward_if_available=True,
                enable_fast_failover=True,
                enable_freshness_cache=config.enable_freshness_cache,
                freshness_cache_ttl_ms=config.freshness_cache_ttl_ms,
                freshness_cache_capacity=config.freshness_cache_capacity,
            ), {**payload["training"], "resumed": 1}
        if payload.get("complete") and not hash_matches:
            raise ValueError(
                f"refusing to resume seed {seed}: existing checkpoint at "
                f"{path} was trained with a different config "
                f"(stored={payload.get('config_sha256')!r}, "
                f"expected={expected_hash!r}); train to a new output "
                "directory instead of overwriting it"
            )
    arrays = collect_teacher_data(
        teacher, training_scenarios(seed), seed=seed,
        episodes_per_scenario=config.dataset_episodes_per_scenario,
    )
    model, training = train_fast_policy(
        arrays, max_nodes=teacher.model.max_nodes, config=config,
        seed=seed, device=device,
    )
    atomic_torch_save({
        "schema_version": 1, "complete": True, "training_seed": seed,
        "config": asdict(config), "config_sha256": expected_hash,
        "model_state": model.state_dict(),
        "training": training,
    }, path)
    return StudentPolicyAdapter(
        model, device=device, force_forward_if_available=True,
        enable_fast_failover=True,
        enable_freshness_cache=config.enable_freshness_cache,
        freshness_cache_ttl_ms=config.freshness_cache_ttl_ms,
        freshness_cache_capacity=config.freshness_cache_capacity,
    ), {**training, "resumed": 0}


def _load_original(
    root: Path, *, seed: int, max_nodes: int, device: torch.device,
    buffered: bool = False,
) -> StudentPolicyAdapter:
    loaded = load_switchglobe(
        root, seed=seed, max_nodes=max_nodes, hidden_dim=64, device=device
    )
    return StudentPolicyAdapter(
        loaded.model, device=device, force_forward_if_available=True,
        reuse_tensor_buffer=buffered,
    )


def run_latency_optimization(
    config: LatencyOptimizationConfig, *, switchglobe_checkpoint_dir: Path,
    checkpoint_dir: Path, device: torch.device | str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        scenarios = phase9_evaluation_scenarios(seed)
        max_nodes = scenarios[0].config.max_nodes
        teacher = _load_original(
            switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
            device=device,
        )
        fast, training = train_or_load_fast(
            teacher, config=config, seed=seed, checkpoint_dir=checkpoint_dir,
            device=device, resume=resume,
        )
        training_rows.append({"training_seed": seed, **training})
        cost_env = FanetRoutingEnv(scenarios[0].config)
        observation, _ = cost_env.reset(
            seed=1_099_999, options=scenarios[0].reset_options
        )
        runtime_specs: list[tuple[str, StudentPolicyAdapter]] = [
            ("original_eager_cpu", _load_original(
                switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
                device=torch.device("cpu"),
            )),
            ("original_buffered_cpu", _load_original(
                switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
                device=torch.device("cpu"), buffered=True,
            )),
        ]
        fast_cpu = FastSwitchGlobePolicy(max_nodes, config.hidden_dim)
        fast_cpu.load_state_dict(fast.model.state_dict())
        runtime_specs.append(("fast_eager_cpu", StudentPolicyAdapter(
            fast_cpu, device="cpu", force_forward_if_available=True,
            enable_fast_failover=True,
            enable_freshness_cache=config.enable_freshness_cache,
            freshness_cache_ttl_ms=config.freshness_cache_ttl_ms,
            freshness_cache_capacity=config.freshness_cache_capacity,
        )))
        if torch.cuda.is_available():
            runtime_specs.extend([
                ("original_eager_cuda", _load_original(
                    switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
                    device=torch.device("cuda"),
                )),
                ("original_buffered_cuda", _load_original(
                    switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
                    device=torch.device("cuda"), buffered=True,
                )),
            ])
            fast_cuda = FastSwitchGlobePolicy(max_nodes, config.hidden_dim)
            fast_cuda.load_state_dict(fast.model.state_dict())
            runtime_specs.append(("fast_eager_cuda", StudentPolicyAdapter(
                fast_cuda, device="cuda", force_forward_if_available=True,
                enable_fast_failover=True,
                enable_freshness_cache=config.enable_freshness_cache,
                freshness_cache_ttl_ms=config.freshness_cache_ttl_ms,
                freshness_cache_capacity=config.freshness_cache_capacity,
            )))
        for variant, policy in runtime_specs:
            for row in profile_student_policy(
                policy, observation, variant=variant,
                warmup=config.warmup, repeats=config.repeats,
            ):
                benchmark_rows.append({"training_seed": seed, **row.to_dict()})
        policies = {ORIGINAL: teacher, FAST: fast}
        for scenario_index, scenario in enumerate(scenarios):
            evaluation_seeds = list(range(
                1_100_000 + scenario_index * 10_000,
                1_100_000 + scenario_index * 10_000
                + config.evaluation_episodes,
            ))
            for method, policy in policies.items():
                results = evaluate_policy_results(
                    FanetRoutingEnv(scenario.config), policy, evaluation_seeds,
                    reset_options=scenario.reset_options,
                    routing_step_duration_ms=config.routing_step_duration_ms,
                )
                episode_rows.extend(
                    episode_row(
                        result, method=method, scenario=scenario.name,
                        training_seed=seed,
                    ) for result in results
                )
                summaries.append(generalization_summary(
                    results, method=method, scenario=scenario.name,
                    training_seed=seed,
                ))
    return {
        "episodes": episode_rows, "seed_summaries": summaries,
        "training": training_rows, "runtime_benchmarks": benchmark_rows,
    }
