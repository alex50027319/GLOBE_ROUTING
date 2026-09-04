"""Permutation-equivariant 1-hop Local Student policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from ..env.observation import (
    EDGE_FEATURES,
    NEIGHBOR_FEATURES,
    PACKET_FEATURES,
    SELF_FEATURES,
)
from .masking import masked_logits, masked_softmax


@dataclass(frozen=True)
class StudentPolicyOutput:
    """Masked policy tensors with a final explicit DROP action."""

    logits: Tensor
    masked_logits: Tensor
    probabilities: Tensor


@dataclass(frozen=True)
class SwitchGlobeDecision:
    """One exact SwitchGLOBE pass with reusable routing diagnostics."""

    output: StudentPolicyOutput
    diagnostics: dict[str, Tensor]
    switch: Tensor
    normal_action: Tensor
    predictive_action: Tensor
    normal_probabilities: Tensor


class LocalStudentPolicy(nn.Module):
    """Score each 1-hop neighbor with shared MLPs and mean context pooling."""

    def __init__(self, max_nodes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least 2")
        if hidden_dim not in {32, 48, 64}:
            raise ValueError("hidden_dim must be 32, 48, or 64")
        self.max_nodes = max_nodes
        self.drop_action = max_nodes
        self.hidden_dim = hidden_dim

        shared_dim = SELF_FEATURES + PACKET_FEATURES
        candidate_dim = shared_dim + NEIGHBOR_FEATURES + EDGE_FEATURES
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.drop_scorer = nn.Sequential(
            nn.Linear(shared_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentPolicyOutput:
        """Produce masked probabilities for a single or batched observation."""

        tensors, unbatched = self._validate_and_batch(observation)
        self_features = tensors["self_features"]
        neighbors = tensors["neighbor_features"]
        edges = tensors["edge_features"]
        packet = tensors["packet_features"]
        action_mask = tensors["action_mask"]
        candidate_mask = action_mask[:, : self.max_nodes]

        shared = torch.cat((self_features, packet), dim=-1)
        expanded_shared = shared.unsqueeze(1).expand(-1, self.max_nodes, -1)
        candidate_input = torch.cat(
            (expanded_shared, neighbors, edges), dim=-1
        )
        encoded = self.neighbor_encoder(candidate_input)

        valid = candidate_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp_min(1.0)
        context = pooled / counts

        expanded_context = context.unsqueeze(1).expand(-1, self.max_nodes, -1)
        candidate_logits = self.candidate_scorer(
            torch.cat((encoded, expanded_context), dim=-1)
        ).squeeze(-1)
        drop_logit = self.drop_scorer(
            torch.cat((shared, context), dim=-1)
        )
        logits = torch.cat((candidate_logits, drop_logit), dim=-1)
        valid_logits = masked_logits(logits, action_mask)
        probabilities = masked_softmax(logits, action_mask)

        if unbatched:
            logits = logits.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        return StudentPolicyOutput(logits, valid_logits, probabilities)

    def _validate_and_batch(
        self, observation: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], bool]:
        required = {
            "self_features",
            "neighbor_features",
            "edge_features",
            "packet_features",
            "action_mask",
        }
        missing = required.difference(observation)
        if missing:
            raise ValueError(f"observation is missing keys: {sorted(missing)}")

        self_features = observation["self_features"]
        unbatched = self_features.ndim == 1
        expected = {
            "self_features": (SELF_FEATURES,),
            "neighbor_features": (self.max_nodes, NEIGHBOR_FEATURES),
            "edge_features": (self.max_nodes, EDGE_FEATURES),
            "packet_features": (PACKET_FEATURES,),
            "action_mask": (self.max_nodes + 1,),
        }
        tensors: dict[str, Tensor] = {}
        batch_size: int | None = None
        for key, trailing_shape in expected.items():
            tensor = observation[key]
            expected_rank = len(trailing_shape) if unbatched else len(trailing_shape) + 1
            if tensor.ndim != expected_rank:
                raise ValueError(
                    f"{key} rank is {tensor.ndim}, expected {expected_rank}"
                )
            if tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
                raise ValueError(
                    f"{key} trailing shape is {tuple(tensor.shape)}, "
                    f"expected {trailing_shape}"
                )
            if unbatched:
                tensor = tensor.unsqueeze(0)
            elif batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError("observation batch dimensions do not match")
            tensors[key] = tensor

        if tensors["action_mask"].dtype != torch.bool:
            tensors["action_mask"] = tensors["action_mask"].to(torch.bool)
        for key in required - {"action_mask"}:
            if not torch.is_floating_point(tensors[key]):
                tensors[key] = tensors[key].to(torch.float32)
            if not torch.all(torch.isfinite(tensors[key])):
                raise ValueError(f"{key} contains non-finite values")
        return tensors, unbatched


class GeographicResidualStudentPolicy(LocalStudentPolicy):
    """Learn residual routing corrections on top of geographic progress."""

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 64,
        *,
        initial_prior_strength: float = 8.0,
        initial_forwardability_strength: float = 0.05,
    ) -> None:
        if initial_prior_strength <= 0:
            raise ValueError("initial_prior_strength must be positive")
        if initial_forwardability_strength <= 0:
            raise ValueError(
                "initial_forwardability_strength must be positive"
            )
        super().__init__(max_nodes=max_nodes, hidden_dim=hidden_dim)
        self.log_prior_strength = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_prior_strength)))
        )
        self.register_buffer("residual_weight", torch.tensor(1.0))
        self._residual_enabled = True
        self.log_forwardability_strength = nn.Parameter(
            torch.log(
                torch.expm1(
                    torch.full(
                        (2,),
                        initial_forwardability_strength,
                    )
                )
            )
        )
        nn.init.zeros_(self.candidate_scorer[-1].weight)
        nn.init.zeros_(self.candidate_scorer[-1].bias)
        nn.init.zeros_(self.drop_scorer[-1].weight)
        nn.init.constant_(self.drop_scorer[-1].bias, -4.0)

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentPolicyOutput:
        """Add a learnable GPSR-equivalent prior to learned candidate logits."""

        base = super().forward(observation)
        unbatched = observation["self_features"].ndim == 1
        neighbors = observation["neighbor_features"]
        packet = observation["packet_features"]
        action_mask = observation["action_mask"]
        if unbatched:
            neighbors = neighbors.unsqueeze(0)
            packet = packet.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            logits = base.logits.unsqueeze(0)
        else:
            logits = base.logits

        destination_delta = packet[:, :2].unsqueeze(1)
        neighbor_delta = neighbors[:, :, :2]
        current_distance = torch.linalg.vector_norm(
            destination_delta, dim=-1
        )
        next_distance = torch.linalg.vector_norm(
            destination_delta - neighbor_delta, dim=-1
        )
        geographic_progress = current_distance - next_distance
        strength = torch.nn.functional.softplus(self.log_prior_strength)
        forwardability = observation.get("candidate_forwardability")
        if forwardability is None:
            forwardability_bonus = torch.zeros_like(next_distance)
        else:
            if unbatched:
                forwardability = forwardability.unsqueeze(0)
            forwardability = forwardability.to(
                device=next_distance.device,
                dtype=next_distance.dtype,
            )
            forwardability_strength = torch.nn.functional.softplus(
                self.log_forwardability_strength
            )
            forwardability_bonus = torch.sum(
                forwardability * forwardability_strength,
                dim=-1,
            )
        candidate_logits = (
            self.residual_weight
            * (
                logits[:, : self.max_nodes]
                + forwardability_bonus
            )
            + strength * geographic_progress
        )
        drop_logits = (
            -20.0
            + self.residual_weight
            * (logits[:, self.max_nodes :] + 20.0)
        )
        combined = torch.cat(
            (candidate_logits, drop_logits), dim=-1
        )
        valid_logits = masked_logits(combined, action_mask)
        probabilities = masked_softmax(combined, action_mask)
        if unbatched:
            combined = combined.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        return StudentPolicyOutput(combined, valid_logits, probabilities)

    def set_residual_weight(self, weight: float) -> None:
        """Set the validated interpolation between GPSR and learned residual."""

        if not 0.0 <= weight <= 1.0:
            raise ValueError("residual weight must be in [0, 1]")
        self.residual_weight.fill_(weight)
        self._residual_enabled = weight > 0.0


class RiskAwareGeographicResidualStudentPolicy(
    GeographicResidualStudentPolicy
):
    """Add a small stability, energy, and queue prior to Phase 8."""

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 64,
        *,
        initial_prior_strength: float = 8.0,
        initial_forwardability_strength: float = 0.05,
        initial_risk_strength: float | Sequence[float] = 0.05,
    ) -> None:
        if isinstance(initial_risk_strength, (int, float)):
            risk_strengths = torch.full(
                (4,), float(initial_risk_strength)
            )
        else:
            if len(initial_risk_strength) != 4:
                raise ValueError(
                    "initial_risk_strength must contain four values"
                )
            risk_strengths = torch.tensor(
                list(initial_risk_strength), dtype=torch.float32
            )
        if torch.any(risk_strengths <= 0):
            raise ValueError("initial_risk_strength must be positive")
        super().__init__(
            max_nodes=max_nodes,
            hidden_dim=hidden_dim,
            initial_prior_strength=initial_prior_strength,
            initial_forwardability_strength=(
                initial_forwardability_strength
            ),
        )
        self.log_risk_strength = nn.Parameter(
            torch.log(
                torch.expm1(
                    risk_strengths
                )
            )
        )
        self.register_buffer("risk_weight", torch.tensor(1.0))

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentPolicyOutput:
        """Add positive deployable quality features to Phase 8 logits."""

        base = super().forward(observation)
        risk = observation.get("candidate_risk_features")
        if risk is None:
            return base
        unbatched = observation["self_features"].ndim == 1
        action_mask = observation["action_mask"]
        logits = base.logits
        if unbatched:
            risk = risk.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            logits = logits.unsqueeze(0)
        risk = risk.to(device=logits.device, dtype=logits.dtype)
        strengths = torch.nn.functional.softplus(
            self.log_risk_strength
        )
        bonus = torch.sum(risk * strengths, dim=-1)
        candidate_logits = (
            logits[:, : self.max_nodes]
            + self.risk_weight * bonus
        )
        combined = torch.cat(
            (candidate_logits, logits[:, self.max_nodes :]),
            dim=-1,
        )
        valid_logits = masked_logits(combined, action_mask)
        probabilities = masked_softmax(combined, action_mask)
        if unbatched:
            combined = combined.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        return StudentPolicyOutput(
            combined,
            valid_logits,
            probabilities,
        )

    def set_risk_weight(self, weight: float) -> None:
        """Set the calibrated contribution of predictive risk features."""

        if not 0.0 <= weight <= 1.0:
            raise ValueError("risk weight must be in [0, 1]")
        self.risk_weight.fill_(weight)


class LiteGlobePStudentPolicy(GeographicResidualStudentPolicy):
    """Predictive-prior residual policy for Lite-GLOBE-P.

    Phase 8 uses geographic progress as the dominant prior and lets the
    Student residual freely reshape that decision. Lite-GLOBE-P makes the
    deployable predictive prior explicit: link margin, current link lifetime,
    queue headroom, and onward link lifetime define the stable route score,
    while the learned residual is bounded so it can correct but not overturn
    a high-confidence link-break warning.
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 64,
        *,
        initial_prior_strength: float = 8.0,
        initial_forwardability_strength: float = 0.05,
        initial_predictive_strength: Sequence[float] = (
            0.75,
            3.00,
            0.25,
            6.00,
        ),
        initial_break_penalty: float = 18.0,
        initial_residual_bound: float = 1.5,
        lifetime_gate: float = 0.20,
        onward_gate: float = 0.20,
        margin_gate: float = 0.04,
    ) -> None:
        if len(initial_predictive_strength) != 4:
            raise ValueError(
                "initial_predictive_strength must contain four values"
            )
        predictive_strength = torch.tensor(
            list(initial_predictive_strength), dtype=torch.float32
        )
        if torch.any(predictive_strength <= 0):
            raise ValueError("initial_predictive_strength must be positive")
        if initial_break_penalty <= 0:
            raise ValueError("initial_break_penalty must be positive")
        if initial_residual_bound <= 0:
            raise ValueError("initial_residual_bound must be positive")
        for name, value in {
            "lifetime_gate": lifetime_gate,
            "onward_gate": onward_gate,
            "margin_gate": margin_gate,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        super().__init__(
            max_nodes=max_nodes,
            hidden_dim=hidden_dim,
            initial_prior_strength=initial_prior_strength,
            initial_forwardability_strength=(
                initial_forwardability_strength
            ),
        )
        self.log_predictive_strength = nn.Parameter(
            torch.log(torch.expm1(predictive_strength))
        )
        self.log_break_penalty = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_break_penalty)))
        )
        self.log_residual_bound = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_residual_bound)))
        )
        self.register_buffer("predictive_weight", torch.tensor(1.0))
        self.register_buffer("lifetime_gate", torch.tensor(lifetime_gate))
        self.register_buffer("onward_gate", torch.tensor(onward_gate))
        self.register_buffer("margin_gate", torch.tensor(margin_gate))

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentPolicyOutput:
        """Score candidates with a predictive prior plus bounded residual."""

        unbatched = observation["self_features"].ndim == 1
        neighbors = observation["neighbor_features"]
        packet = observation["packet_features"]
        action_mask = observation["action_mask"]
        if unbatched:
            neighbors = neighbors.unsqueeze(0)
            packet = packet.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        if self._residual_enabled:
            residual = LocalStudentPolicy.forward(self, observation)
            residual_logits = residual.logits
            if unbatched:
                residual_logits = residual_logits.unsqueeze(0)
        else:
            residual_logits = torch.zeros(
                action_mask.shape[0],
                self.max_nodes + 1,
                device=neighbors.device,
                dtype=neighbors.dtype,
            )

        destination_delta = packet[:, :2].unsqueeze(1)
        neighbor_delta = neighbors[:, :, :2]
        current_distance = torch.linalg.vector_norm(
            destination_delta, dim=-1
        )
        next_distance = torch.linalg.vector_norm(
            destination_delta - neighbor_delta, dim=-1
        )
        geographic_progress = current_distance - next_distance
        prior = (
            torch.nn.functional.softplus(self.log_prior_strength)
            * geographic_progress
        )

        forwardability = observation.get("candidate_forwardability")
        if forwardability is None:
            forwardability_bonus = torch.zeros_like(next_distance)
        else:
            if unbatched:
                forwardability = forwardability.unsqueeze(0)
            forwardability = forwardability.to(
                device=next_distance.device,
                dtype=next_distance.dtype,
            )
            forwardability_strength = torch.nn.functional.softplus(
                self.log_forwardability_strength
            )
            forwardability_bonus = torch.sum(
                forwardability * forwardability_strength,
                dim=-1,
            )

        risk = observation.get("candidate_risk_features")
        if risk is None:
            predictive_bonus = torch.zeros_like(next_distance)
            gate_penalty = torch.zeros_like(next_distance)
        else:
            if unbatched:
                risk = risk.unsqueeze(0)
            risk = risk.to(device=next_distance.device, dtype=next_distance.dtype)
            predictive_strength = torch.nn.functional.softplus(
                self.log_predictive_strength
            )
            predictive_bonus = torch.sum(
                risk * predictive_strength,
                dim=-1,
            )
            margin = risk[:, :, 0]
            lifetime = risk[:, :, 1]
            onward = risk[:, :, 3]
            gate_violation = (
                torch.relu(self.margin_gate - margin)
                + torch.relu(self.lifetime_gate - lifetime)
                + torch.relu(self.onward_gate - onward)
            )
            gate_penalty = (
                torch.nn.functional.softplus(self.log_break_penalty)
                * gate_violation
            )

        residual_bound = torch.nn.functional.softplus(
            self.log_residual_bound
        )
        bounded_residual = residual_bound * torch.tanh(
            residual_logits[:, : self.max_nodes]
        )
        candidate_logits = (
            prior
            + forwardability_bonus
            + self.predictive_weight * predictive_bonus
            - self.predictive_weight * gate_penalty
            + self.residual_weight * bounded_residual
        )
        drop_logits = (
            -20.0
            + self.residual_weight
            * torch.tanh(residual_logits[:, self.max_nodes :])
        )
        combined = torch.cat((candidate_logits, drop_logits), dim=-1)
        valid_logits = masked_logits(combined, action_mask)
        probabilities = masked_softmax(combined, action_mask)
        if unbatched:
            combined = combined.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        return StudentPolicyOutput(combined, valid_logits, probabilities)

    def set_predictive_weight(self, weight: float) -> None:
        """Set the calibrated contribution of predictive local features."""

        if not 0.0 <= weight <= 1.0:
            raise ValueError("predictive weight must be in [0, 1]")
        self.predictive_weight.fill_(weight)


class RiskSwitchLiteGlobePStudentPolicy(nn.Module):
    """Historical implementation class underlying SwitchGLOBE.

    The policy preserves the strong Phase 8 branch for ordinary routing and
    invokes a predictive-prior-only branch only when the Phase 8 next hop is
    locally risky. This avoids the Phase 11 failure mode where learned residuals
    occasionally override an otherwise correct predictive link-break decision.
    """

    def __init__(
        self,
        normal_policy: GeographicResidualStudentPolicy,
        predictive_policy: LiteGlobePStudentPolicy,
        *,
        switch_threshold: float = 0.05,
        margin_gate: float = 0.04,
        lifetime_gate: float = 0.20,
        onward_gate: float = 0.20,
    ) -> None:
        super().__init__()
        if normal_policy.max_nodes != predictive_policy.max_nodes:
            raise ValueError("normal and predictive policies must align")
        if not 0.0 <= switch_threshold <= 3.0:
            raise ValueError("switch_threshold must be in [0, 3]")
        for name, value in {
            "margin_gate": margin_gate,
            "lifetime_gate": lifetime_gate,
            "onward_gate": onward_gate,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.normal_policy = normal_policy
        self.predictive_policy = predictive_policy
        self.max_nodes = normal_policy.max_nodes
        self.drop_action = normal_policy.drop_action
        self.register_buffer(
            "switch_threshold", torch.tensor(switch_threshold)
        )
        self.register_buffer("margin_gate", torch.tensor(margin_gate))
        self.register_buffer("lifetime_gate", torch.tensor(lifetime_gate))
        self.register_buffer("onward_gate", torch.tensor(onward_gate))
        self.predictive_policy.set_residual_weight(0.0)

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentPolicyOutput:
        """Use Phase 8 unless the selected link is locally high-risk."""

        return self.decide(observation).output

    def decide(
        self, observation: Mapping[str, Tensor]
    ) -> SwitchGlobeDecision:
        """Compute the action distribution and diagnostics without repeats."""

        normal = self.normal_policy(observation)
        predictive = self.predictive_policy(observation)
        return self._decide_from_branches(
            observation, normal=normal, predictive=predictive
        )

    def _decide_from_branches(
        self,
        observation: Mapping[str, Tensor],
        *,
        normal: StudentPolicyOutput,
        predictive: StudentPolicyOutput,
    ) -> SwitchGlobeDecision:
        """Combine already-computed branch outputs (no repeated forward pass).

        Split out of ``decide()`` so a subclass that conditionally skips the
        predictive branch (see ``EarlyExitSwitchGlobePolicy``) can reuse this
        exact combination logic without calling ``normal_policy`` twice.
        """

        unbatched = observation["self_features"].ndim == 1
        action_mask = observation["action_mask"]
        normal_logits = normal.logits
        predictive_logits = predictive.logits
        if unbatched:
            action_mask = action_mask.unsqueeze(0)
            normal_logits = normal_logits.unsqueeze(0)
            predictive_logits = predictive_logits.unsqueeze(0)
        switch = self._switch_mask(
            observation,
            action_mask=action_mask,
            normal_logits=normal_logits,
            predictive_logits=predictive_logits,
            unbatched=unbatched,
        )
        combined = torch.where(
            switch.unsqueeze(-1),
            predictive_logits,
            normal_logits,
        )
        valid_logits = masked_logits(combined, action_mask)
        probabilities = masked_softmax(combined, action_mask)
        if unbatched:
            combined = combined.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        output = StudentPolicyOutput(combined, valid_logits, probabilities)
        normal_action = torch.argmax(
            masked_logits(normal_logits, action_mask), dim=-1
        )
        predictive_action = torch.argmax(
            masked_logits(predictive_logits, action_mask), dim=-1
        )
        diagnostics = self._diagnostics_from_decision(
            observation,
            action_mask=action_mask,
            normal_logits=normal_logits,
            predictive_logits=predictive_logits,
            normal_action=normal_action,
            predictive_action=predictive_action,
            switch=switch,
            unbatched=unbatched,
        )
        normal_probabilities = normal.probabilities
        return SwitchGlobeDecision(
            output=output,
            diagnostics=diagnostics,
            switch=switch,
            normal_action=normal_action,
            predictive_action=predictive_action,
            normal_probabilities=normal_probabilities,
        )

    def _switch_mask(
        self,
        observation: Mapping[str, Tensor],
        *,
        action_mask: Tensor,
        normal_logits: Tensor,
        predictive_logits: Tensor,
        unbatched: bool,
    ) -> Tensor:
        risk = observation.get("candidate_risk_features")
        if risk is None:
            return torch.zeros(
                action_mask.shape[0],
                dtype=torch.bool,
                device=normal_logits.device,
            )
        if unbatched:
            risk = risk.unsqueeze(0)
        risk = risk.to(device=normal_logits.device, dtype=normal_logits.dtype)
        normal_masked = masked_logits(normal_logits, action_mask)
        predictive_masked = masked_logits(predictive_logits, action_mask)
        normal_action = torch.argmax(normal_masked, dim=-1)
        predictive_action = torch.argmax(predictive_masked, dim=-1)

        batch = torch.arange(
            normal_action.shape[0],
            device=normal_logits.device,
        )
        normal_candidate = normal_action.clamp(max=self.max_nodes - 1)
        predictive_candidate = predictive_action.clamp(max=self.max_nodes - 1)
        normal_is_drop = normal_action == self.drop_action

        normal_risk = risk[batch, normal_candidate]
        predictive_risk = risk[batch, predictive_candidate]
        normal_danger = self._danger_score(normal_risk)
        predictive_danger = self._danger_score(predictive_risk)
        safety_gain = (
            predictive_risk[:, 0]
            + predictive_risk[:, 1]
            + predictive_risk[:, 3]
            - normal_risk[:, 0]
            - normal_risk[:, 1]
            - normal_risk[:, 3]
        )
        candidate_mask = action_mask[:, : self.max_nodes].to(torch.bool)
        has_candidate = torch.any(candidate_mask, dim=-1)
        high_risk = normal_danger > self.switch_threshold
        safer_predictive = (
            (predictive_action != normal_action)
            & (safety_gain > 0.10)
            & (predictive_danger < normal_danger)
        )
        return has_candidate & (normal_is_drop | high_risk | safer_predictive)

    def _danger_score(self, risk: Tensor) -> Tensor:
        margin = risk[:, 0]
        lifetime = risk[:, 1]
        onward = risk[:, 3]
        return (
            torch.relu(self.margin_gate - margin)
            + torch.relu(self.lifetime_gate - lifetime)
            + torch.relu(self.onward_gate - onward)
        )

    def diagnostics(self, observation: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return per-decision switch diagnostics without changing actions."""

        return self.decide(observation).diagnostics

    def _diagnostics_from_decision(
        self,
        observation: Mapping[str, Tensor],
        *,
        action_mask: Tensor,
        normal_logits: Tensor,
        predictive_logits: Tensor,
        normal_action: Tensor,
        predictive_action: Tensor,
        switch: Tensor,
        unbatched: bool,
    ) -> dict[str, Tensor]:
        """Derive diagnostics from logits already used for the action."""

        risk = observation.get("candidate_risk_features")
        if unbatched:
            if risk is not None:
                risk = risk.unsqueeze(0)
        if risk is None:
            zero = switch.to(normal_logits.dtype).sum() * 0.0
            return {"switch_steps": zero, "branch_disagreement_steps": zero}
        risk = risk.to(device=normal_logits.device, dtype=normal_logits.dtype)
        batch = torch.arange(normal_action.shape[0], device=normal_action.device)
        normal_risk = risk[batch, normal_action.clamp(max=self.max_nodes - 1)]
        predictive_risk = risk[batch, predictive_action.clamp(max=self.max_nodes - 1)]
        normal_danger = self._danger_score(normal_risk)
        predictive_danger = self._danger_score(predictive_risk)
        danger_reduction = torch.where(switch, normal_danger - predictive_danger, torch.zeros_like(normal_danger))
        return {
            "switch_steps": switch.to(normal_logits.dtype).sum(),
            "branch_disagreement_steps": (normal_action != predictive_action).to(normal_logits.dtype).sum(),
            "switch_danger_reduction": danger_reduction.sum(),
            "false_switch_steps": (switch & (normal_danger <= 0)).to(normal_logits.dtype).sum(),
            "missed_risk_steps": ((~switch) & (normal_danger > self.switch_threshold)).to(normal_logits.dtype).sum(),
            "mean_selected_danger": torch.where(switch, predictive_danger, normal_danger).sum(),
            "safe_forward_candidates": (normal_danger <= 0).to(normal_logits.dtype).sum(),
        }

    def set_switch_parameters(
        self,
        *,
        switch_threshold: float,
        margin_gate: float,
        lifetime_gate: float,
        onward_gate: float,
    ) -> None:
        """Set calibrated local risk-switch parameters."""

        for name, value, upper in (
            ("switch_threshold", switch_threshold, 3.0),
            ("margin_gate", margin_gate, 1.0),
            ("lifetime_gate", lifetime_gate, 1.0),
            ("onward_gate", onward_gate, 1.0),
        ):
            if not 0.0 <= value <= upper:
                raise ValueError(f"{name} must be in [0, {upper}]")
        self.switch_threshold.fill_(switch_threshold)
        self.margin_gate.fill_(margin_gate)
        self.lifetime_gate.fill_(lifetime_gate)
        self.onward_gate.fill_(onward_gate)


class SwitchGlobePolicy(RiskSwitchLiteGlobePStudentPolicy):
    """Canonical public name for the final Phase 12 deployment policy.

    The historical class name is retained above so that Phase 12 checkpoints
    and research-history scripts remain loadable without migration.
    """

    pass


class EarlyExitSwitchGlobePolicy(RiskSwitchLiteGlobePStudentPolicy):
    """SwitchGLOBE Exact with a calibrated, threshold-agnostic early exit.

    Loads the same ``normal_policy``/``predictive_policy`` weights and switch
    parameters as Exact and produces identical actions; the only difference
    is that ``predictive_policy`` is not evaluated when the normal branch's
    own chosen candidate is not DROP, a candidate exists, and its danger
    score (from ``candidate_risk_features`` alone, no predictive network) is
    exactly zero -- i.e. margin, lifetime, and onward all clear their
    calibrated gates.

    This is provably consistent with ``decide()``'s switch condition for
    every term except ``safer_predictive``, whose trigger depends on
    ``predictive_action`` and so cannot be ruled out analytically without
    running the predictive network. It was instead validated empirically: a
    read-only replay of the real 5-seed SwitchGLOBE Exact checkpoints over
    the full 14-scenario x 200-episode evaluation set found zero cases where
    skipping would have changed the routing decision (see
    ``artifacts/gated_switchglobe/calibration``). That is evidence, not
    proof -- re-verify before relying on this for a new checkpoint or
    scenario family.
    """

    def decide(
        self, observation: Mapping[str, Tensor]
    ) -> SwitchGlobeDecision:
        normal = self.normal_policy(observation)
        risk = observation.get("candidate_risk_features")
        if risk is None:
            predictive = self.predictive_policy(observation)
            return self._decide_from_branches(
                observation, normal=normal, predictive=predictive
            )

        unbatched = observation["self_features"].ndim == 1
        action_mask = observation["action_mask"]
        normal_logits = normal.logits
        risk_tensor = risk
        if unbatched:
            action_mask = action_mask.unsqueeze(0)
            normal_logits = normal_logits.unsqueeze(0)
            risk_tensor = risk_tensor.unsqueeze(0)

        normal_masked = masked_logits(normal_logits, action_mask)
        normal_action = torch.argmax(normal_masked, dim=-1)
        normal_is_drop = normal_action == self.drop_action
        candidate_mask = action_mask[:, : self.max_nodes].to(torch.bool)
        has_candidate = torch.any(candidate_mask, dim=-1)
        batch = torch.arange(normal_action.shape[0], device=normal_action.device)
        normal_candidate = normal_action.clamp(max=self.max_nodes - 1)
        risk_tensor = risk_tensor.to(
            device=normal_logits.device, dtype=normal_logits.dtype
        )
        normal_risk = risk_tensor[batch, normal_candidate]
        normal_danger = self._danger_score(normal_risk)
        can_skip = has_candidate & (~normal_is_drop) & (normal_danger <= 0.0)
        if not bool(torch.all(can_skip).item()):
            # normal_policy was already run above; do not recompute it.
            predictive = self.predictive_policy(observation)
            return self._decide_from_branches(
                observation, normal=normal, predictive=predictive
            )

        valid_logits = masked_logits(normal_logits, action_mask)
        probabilities = masked_softmax(normal_logits, action_mask)
        combined = normal_logits
        if unbatched:
            combined = combined.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        output = StudentPolicyOutput(combined, valid_logits, probabilities)
        switch = torch.zeros_like(can_skip)
        zero = switch.to(normal_logits.dtype).sum() * 0.0
        return SwitchGlobeDecision(
            output=output,
            diagnostics={"switch_steps": zero, "branch_disagreement_steps": zero},
            switch=switch,
            normal_action=normal_action,
            predictive_action=normal_action,
            normal_probabilities=normal.probabilities,
        )


class FastSwitchGlobePolicy(LocalStudentPolicy):
    """Single-pass deployable student distilled from final SwitchGLOBE actions."""

    observation_fields = (
        "self_features", "neighbor_features", "edge_features", "packet_features",
        "action_mask", "candidate_forwardability", "candidate_risk_features",
    )

    def __init__(self, max_nodes: int, hidden_dim: int = 32) -> None:
        super().__init__(max_nodes=max_nodes, hidden_dim=hidden_dim)
        shared_dim = SELF_FEATURES + PACKET_FEATURES
        candidate_dim = (
            shared_dim + NEIGHBOR_FEATURES + EDGE_FEATURES + 2 + 4
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.switch_head = nn.Sequential(
            nn.Linear(shared_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_with_auxiliary(
        self, observation: Mapping[str, Tensor]
    ) -> tuple[StudentPolicyOutput, Tensor]:
        tensors, unbatched = self._validate_and_batch(observation)
        forwardability = observation.get("candidate_forwardability")
        risk = observation.get("candidate_risk_features")
        if forwardability is None or risk is None:
            raise ValueError(
                "FastSwitchGLOBE requires forwardability and risk features"
            )
        if unbatched:
            forwardability = forwardability.unsqueeze(0)
            risk = risk.unsqueeze(0)
        forwardability = forwardability.to(
            device=tensors["self_features"].device, dtype=torch.float32
        )
        risk = risk.to(
            device=tensors["self_features"].device, dtype=torch.float32
        )
        if tuple(forwardability.shape[-2:]) != (self.max_nodes, 2):
            raise ValueError("candidate_forwardability shape is invalid")
        if tuple(risk.shape[-2:]) != (self.max_nodes, 4):
            raise ValueError("candidate_risk_features shape is invalid")
        shared = torch.cat(
            (tensors["self_features"], tensors["packet_features"]), dim=-1
        )
        expanded = shared.unsqueeze(1).expand(-1, self.max_nodes, -1)
        candidate_input = torch.cat(
            (
                expanded, tensors["neighbor_features"], tensors["edge_features"],
                forwardability, risk,
            ),
            dim=-1,
        )
        encoded = self.neighbor_encoder(candidate_input)
        candidate_mask = tensors["action_mask"][:, : self.max_nodes]
        valid = candidate_mask.unsqueeze(-1).to(encoded.dtype)
        context = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        candidate_logits = self.candidate_scorer(
            torch.cat(
                (encoded, context.unsqueeze(1).expand(-1, self.max_nodes, -1)),
                dim=-1,
            )
        ).squeeze(-1)
        drop_logit = self.drop_scorer(torch.cat((shared, context), dim=-1))
        logits = torch.cat((candidate_logits, drop_logit), dim=-1)
        valid_logits = masked_logits(logits, tensors["action_mask"])
        probabilities = masked_softmax(logits, tensors["action_mask"])
        switch_logit = self.switch_head(torch.cat((shared, context), dim=-1)).squeeze(-1)
        if unbatched:
            logits = logits.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
            switch_logit = switch_logit.squeeze(0)
        return StudentPolicyOutput(logits, valid_logits, probabilities), switch_logit

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentPolicyOutput:
        return self.forward_with_auxiliary(observation)[0]

    def diagnostics(self, observation: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Report the trained switch-head's activation rate for one decision.

        FastSwitchGLOBE has no separate predictive-branch network to gate at
        inference: the same fused scorer already produces the final action.
        ``switch_head`` is trained only as an auxiliary distillation target
        (matched against the teacher's binary switch label, see
        ``switch_accuracy`` in fast_training/training_metrics.csv) and is
        otherwise unused by ``forward``/act. This reports that head's belief
        using the same ``>= 0`` threshold as its training-time accuracy, so
        ``switch_activation_rate`` is measurable for FastSwitchGLOBE instead
        of silently staying at zero; it is a proxy for "the teacher would
        have switched here", not evidence a distinct branch executed.
        """

        _, switch_logit = self.forward_with_auxiliary(observation)
        switch = switch_logit >= 0
        return {"switch_steps": switch.to(switch_logit.dtype).sum()}
