"""Environment-facing adapter for a PyTorch Local Student."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .student_policy import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    LocalStudentPolicy,
    RiskAwareGeographicResidualStudentPolicy,
    RiskSwitchLiteGlobePPlusStudentPolicy,
    RiskSwitchLiteGlobePStudentPolicy,
)
from .tensor_observation import observation_to_tensors


class StudentPolicyAdapter:
    """Expose deterministic or sampled actions through the baseline API."""

    def __init__(
        self,
        model: LocalStudentPolicy,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        force_forward_if_available: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.force_forward_if_available = force_forward_if_available
        self.generator = torch.Generator(device=self.device)
        self._episode_diagnostics: dict[str, float] = {}

    def reset(self, seed: int | None = None) -> None:
        self.generator.manual_seed(0 if seed is None else seed)
        self._episode_diagnostics = {}

    def episode_diagnostics(self) -> dict[str, float]:
        """Return aggregated optional policy diagnostics for one episode."""

        return dict(self._episode_diagnostics)

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        if isinstance(
            self.model,
            (
                RiskSwitchLiteGlobePStudentPolicy,
                RiskSwitchLiteGlobePPlusStudentPolicy,
            ),
        ):
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
        tensors = observation_to_tensors(observation, device=self.device)
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
            if (
                isinstance(self.model, RiskSwitchLiteGlobePPlusStudentPolicy)
                and "candidate_switch_features" in observation
            ):
                extra += int(observation["candidate_switch_features"].nbytes)
            return base_bytes + extra
        selected = phase8_adapter.act(observation)
        if selected < self.model.max_nodes:
            extra = int(
                observation["candidate_risk_features"][selected].nbytes
            )
            if (
                isinstance(self.model, RiskSwitchLiteGlobePPlusStudentPolicy)
                and "candidate_switch_features" in observation
            ):
                extra += int(
                    observation["candidate_switch_features"][selected].nbytes
                )
            return base_bytes + extra
        return base_bytes

    @torch.inference_mode()
    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        tensors = observation_to_tensors(observation, device=self.device)
        diagnostic_fn = getattr(self.model, "diagnostics", None)
        if diagnostic_fn is not None:
            for key, value in diagnostic_fn(tensors).items():
                self._episode_diagnostics[key] = (
                    self._episode_diagnostics.get(key, 0.0) + float(value)
                )
            self._episode_diagnostics["diagnostic_steps"] = (
                self._episode_diagnostics.get("diagnostic_steps", 0.0) + 1.0
            )
        probabilities = self.model(tensors).probabilities
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
            return action
        action = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=self.generator,
        )
        return int(action.item())
