"""Environment-facing adapter for a PyTorch Local Student."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic_ns
import numpy as np
import torch
from numpy.typing import NDArray

from .student_policy import (
    GeographicResidualStudentPolicy,
    FastSwitchGlobePolicy,
    LiteGlobePStudentPolicy,
    LocalStudentPolicy,
    RiskAwareGeographicResidualStudentPolicy,
    RiskSwitchLiteGlobePStudentPolicy,
)
from .tensor_observation import TensorObservationBuffer, observation_to_tensors


@dataclass(frozen=True)
class PolicyDecision:
    """Environment action plus metadata computed by the same inference pass."""

    action: int
    input_bytes: int
    backup_action: int | None = None


@dataclass(frozen=True)
class _FreshDecision:
    decision: PolicyDecision
    stored_at_ns: int
    observation: tuple[NDArray[np.generic], ...]


_FreshnessKey = tuple[bytes, bytes, bytes]


class StudentPolicyAdapter:
    """Expose deterministic or sampled actions through the baseline API."""

    def __init__(
        self,
        model: LocalStudentPolicy,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        force_forward_if_available: bool = False,
        reuse_tensor_buffer: bool = False,
        enable_fast_failover: bool = False,
        enable_freshness_cache: bool = False,
        freshness_cache_ttl_ms: float = 5.0,
        freshness_cache_capacity: int = 128,
    ) -> None:
        if enable_freshness_cache and not deterministic:
            raise ValueError("freshness cache requires deterministic actions")
        if enable_freshness_cache and not isinstance(
            model, FastSwitchGlobePolicy
        ):
            raise ValueError("freshness cache currently supports FastSwitchGLOBE only")
        if freshness_cache_ttl_ms <= 0:
            raise ValueError("freshness_cache_ttl_ms must be positive")
        if freshness_cache_capacity <= 0:
            raise ValueError("freshness_cache_capacity must be positive")
        self.model = model.to(device)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.force_forward_if_available = force_forward_if_available
        self.enable_fast_failover = enable_fast_failover
        self.enable_freshness_cache = enable_freshness_cache
        self.freshness_cache_ttl_ns = int(freshness_cache_ttl_ms * 1_000_000)
        self.freshness_cache_capacity = freshness_cache_capacity
        self._freshness_cache: OrderedDict[
            _FreshnessKey, _FreshDecision
        ] = OrderedDict()
        self._cache_clock_ns = monotonic_ns
        self.tensor_buffer = (
            TensorObservationBuffer(self.device) if reuse_tensor_buffer else None
        )
        self.generator = torch.Generator(device=self.device)
        self._episode_diagnostics: dict[str, float] = {}

    def _tensors(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> dict[str, torch.Tensor]:
        if self.tensor_buffer is not None:
            return self.tensor_buffer.convert(observation)
        return observation_to_tensors(observation, device=self.device)

    def reset(self, seed: int | None = None) -> None:
        self.generator.manual_seed(0 if seed is None else seed)
        self._episode_diagnostics = {}
        if self.enable_freshness_cache:
            self.clear_freshness_cache()

    def episode_diagnostics(self) -> dict[str, float]:
        """Return aggregated optional policy diagnostics for one episode."""

        return dict(self._episode_diagnostics)

    def clear_freshness_cache(self) -> None:
        """Invalidate every cached decision after a model or state reset."""

        self._freshness_cache.clear()

    def _increment_diagnostic(self, key: str) -> None:
        self._episode_diagnostics[key] = (
            self._episode_diagnostics.get(key, 0.0) + 1.0
        )

    def _freshness_key(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> _FreshnessKey:
        """Build a cheap lookup key; full arrays are verified before reuse."""

        return tuple(
            np.ascontiguousarray(observation[key]).tobytes()
            for key in ("self_features", "packet_features", "action_mask")
        )

    def _observation_snapshot(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> tuple[NDArray[np.generic], ...]:
        fields = getattr(self.model, "observation_fields", tuple(observation))
        return tuple(np.array(observation[key], copy=True) for key in fields)

    def _observation_matches(
        self,
        observation: dict[str, NDArray[np.generic]],
        snapshot: tuple[NDArray[np.generic], ...],
    ) -> bool:
        fields = getattr(self.model, "observation_fields", tuple(observation))
        return all(
            np.array_equal(observation[key], cached)
            for key, cached in zip(fields, snapshot, strict=True)
        )

    def _fresh_decision(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> tuple[_FreshnessKey | None, PolicyDecision | None]:
        if not self.enable_freshness_cache:
            return None, None
        key = self._freshness_key(observation)
        entry = self._freshness_cache.get(key)
        if entry is None:
            self._increment_diagnostic("freshness_cache_miss_steps")
            return key, None
        age = self._cache_clock_ns() - entry.stored_at_ns
        if age > self.freshness_cache_ttl_ns:
            del self._freshness_cache[key]
            self._increment_diagnostic("freshness_cache_miss_steps")
            self._increment_diagnostic("freshness_cache_stale_evictions")
            return key, None
        if not self._observation_matches(observation, entry.observation):
            del self._freshness_cache[key]
            self._increment_diagnostic("freshness_cache_miss_steps")
            self._increment_diagnostic("freshness_cache_state_evictions")
            return key, None
        self._freshness_cache.move_to_end(key)
        self._increment_diagnostic("freshness_cache_hit_steps")
        if entry.decision.backup_action is not None:
            self._increment_diagnostic("backup_available_steps")
        return key, entry.decision

    def _store_fresh_decision(
        self,
        key: _FreshnessKey | None,
        decision: PolicyDecision,
        observation: dict[str, NDArray[np.generic]],
    ) -> None:
        if key is None:
            return
        self._freshness_cache[key] = _FreshDecision(
            decision=decision,
            stored_at_ns=self._cache_clock_ns(),
            observation=self._observation_snapshot(observation),
        )
        self._freshness_cache.move_to_end(key)
        if len(self._freshness_cache) > self.freshness_cache_capacity:
            self._freshness_cache.popitem(last=False)
            self._increment_diagnostic("freshness_cache_capacity_evictions")

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        declared = getattr(self.model, "observation_fields", None)
        if declared is not None:
            return sum(
                int(observation[key].nbytes)
                for key in declared if key in observation
            )
        if isinstance(self.model, RiskSwitchLiteGlobePStudentPolicy):
            return self._risk_switch_observation_bytes(observation)
        keys = {
            "self_features",
            "neighbor_features",
            "packet_features",
            "action_mask",
        }
        if not (
            isinstance(self.model, LiteGlobePStudentPolicy)
            and float(self.model.residual_weight.item()) == 0.0
        ):
            keys.add("edge_features")
        if isinstance(self.model, GeographicResidualStudentPolicy):
            keys.add("candidate_forwardability")
        if isinstance(
            self.model,
            (RiskAwareGeographicResidualStudentPolicy, LiteGlobePStudentPolicy),
        ):
            keys.add("candidate_risk_features")
        return sum(
            int(observation[key].nbytes)
            for key in keys
            if key in observation
        )

    @torch.inference_mode()
    def _risk_switch_observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        phase8_adapter = StudentPolicyAdapter(
            self.model.normal_policy,
            device=self.device,
            deterministic=self.deterministic,
            force_forward_if_available=self.force_forward_if_available,
        )
        base_bytes = phase8_adapter.observation_bytes(observation)
        if "candidate_risk_features" not in observation:
            return base_bytes
        tensors = self._tensors(observation)
        normal_logits = self.model.normal_policy(tensors).logits.unsqueeze(0)
        predictive_logits = (
            self.model.predictive_policy(tensors).logits.unsqueeze(0)
        )
        adjust_fn = getattr(self.model, "_energy_adjusted_logits", None)
        if adjust_fn is not None:
            predictive_logits = adjust_fn(
                tensors,
                predictive_logits,
                unbatched=True,
            )
        switch = self.model._switch_mask(
            tensors,
            action_mask=tensors["action_mask"].unsqueeze(0),
            normal_logits=normal_logits,
            predictive_logits=predictive_logits,
            unbatched=True,
        )
        if bool(switch.item()):
            extra = int(observation["candidate_risk_features"].nbytes)
            return base_bytes + extra
        selected = phase8_adapter.act(observation)
        if selected < self.model.max_nodes:
            extra = int(
                observation["candidate_risk_features"][selected].nbytes
            )
            return base_bytes + extra
        return base_bytes

    @torch.inference_mode()
    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        return self.act_with_metadata(observation).action

    @torch.inference_mode()
    def act_with_metadata(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> PolicyDecision:
        """Choose one action and reuse its intermediates for all metadata."""

        cache_key, cached = self._fresh_decision(observation)
        if cached is not None:
            return cached
        tensors = self._tensors(observation)
        switch_decision = None
        if isinstance(self.model, RiskSwitchLiteGlobePStudentPolicy):
            switch_decision = self.model.decide(tensors)
            diagnostics = switch_decision.diagnostics
            probabilities = switch_decision.output.probabilities
        else:
            diagnostic_fn = getattr(self.model, "diagnostics", None)
            diagnostics = diagnostic_fn(tensors) if diagnostic_fn is not None else {}
            probabilities = self.model(tensors).probabilities
        if diagnostics:
            keys = tuple(diagnostics)
            values = torch.stack(
                [diagnostics[key].reshape(()) for key in keys]
            ).detach().cpu().tolist()
            for key, value in zip(keys, values, strict=True):
                self._episode_diagnostics[key] = (
                    self._episode_diagnostics.get(key, 0.0) + float(value)
                )
            self._episode_diagnostics["diagnostic_steps"] = (
                self._episode_diagnostics.get("diagnostic_steps", 0.0) + 1.0
            )
        if self.deterministic:
            action = int(torch.argmax(probabilities).item())
            if (
                self.force_forward_if_available
                and action == self.model.drop_action
            ):
                candidate_mask = tensors["action_mask"][
                    : self.model.max_nodes
                ].to(torch.bool)
                if torch.any(candidate_mask):
                    candidate_probabilities = probabilities[
                        : self.model.max_nodes
                    ].masked_fill(~candidate_mask, -1.0)
                    action = int(
                        torch.argmax(candidate_probabilities).item()
                    )
        else:
            sampled = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=self.generator,
            )
            action = int(sampled.item())
        input_bytes = (
            self._switch_input_bytes_from_decision(
                observation, switch_decision
            )
            if switch_decision is not None
            else self.observation_bytes(observation)
        )
        backup_action = self._backup_action(
            probabilities, tensors["action_mask"], primary_action=action
        )
        if backup_action is not None:
            self._increment_diagnostic("backup_available_steps")
        decision = PolicyDecision(
            action=action,
            input_bytes=input_bytes,
            backup_action=backup_action,
        )
        self._store_fresh_decision(cache_key, decision, observation)
        return decision

    def _backup_action(
        self,
        probabilities: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        primary_action: int,
    ) -> int | None:
        """Select a second valid next hop from the existing forward pass."""

        if not self.enable_fast_failover:
            return None
        candidate_mask = action_mask[: self.model.max_nodes].to(torch.bool).clone()
        if 0 <= primary_action < self.model.max_nodes:
            candidate_mask[primary_action] = False
        if not bool(torch.any(candidate_mask).item()):
            return None
        candidate_probabilities = probabilities[: self.model.max_nodes].masked_fill(
            ~candidate_mask, -1.0
        )
        return int(torch.argmax(candidate_probabilities).item())

    def resolve_decision(
        self,
        decision: PolicyDecision,
        live_action_mask: NDArray[np.generic] | torch.Tensor,
    ) -> int:
        """Resolve a cached primary/backup pair against the latest link mask.

        This method performs no model inference. It is intended to run directly
        before transmission, after the caller refreshes its local link state.
        """

        mask = torch.as_tensor(live_action_mask, device=self.device).reshape(-1)
        expected = self.model.max_nodes + 1
        if mask.numel() != expected:
            raise ValueError(
                f"live_action_mask must contain {expected} actions, got {mask.numel()}"
            )
        mask = mask.to(torch.bool)
        primary = int(decision.action)
        if not 0 <= primary <= self.model.drop_action:
            raise ValueError(f"primary action {primary} is out of range")
        if primary == self.model.drop_action or bool(mask[primary].item()):
            return primary
        backup = decision.backup_action
        if (
            backup is not None
            and 0 <= backup < self.model.max_nodes
            and bool(mask[backup].item())
        ):
            self._increment_diagnostic("fast_failover_steps")
            return backup
        self._increment_diagnostic("fast_failover_miss_steps")
        return self.model.drop_action

    def _switch_input_bytes_from_decision(
        self, observation: dict[str, NDArray[np.generic]], decision
    ) -> int:
        keys = {
            "self_features", "neighbor_features", "edge_features",
            "packet_features", "action_mask", "candidate_forwardability",
        }
        base = sum(
            int(observation[key].nbytes) for key in keys if key in observation
        )
        risk = observation.get("candidate_risk_features")
        if risk is None:
            return base
        if bool(decision.switch.reshape(-1)[0].item()):
            return base + int(risk.nbytes)
        selected = int(decision.normal_action.reshape(-1)[0].item())
        if self.force_forward_if_available and selected == self.model.drop_action:
            candidate_mask = observation["action_mask"][: self.model.max_nodes].astype(bool)
            if np.any(candidate_mask):
                values = decision.normal_probabilities[: self.model.max_nodes]
                mask = torch.as_tensor(candidate_mask, device=values.device)
                selected = int(torch.argmax(values.masked_fill(~mask, -1.0)).item())
        return base + (int(risk[selected].nbytes) if selected < self.model.max_nodes else 0)
