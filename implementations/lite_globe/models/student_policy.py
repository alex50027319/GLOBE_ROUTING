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


class LocalStudentPolicy(nn.Module):
    """Score each 1-hop neighbor with shared MLPs and mean context pooling."""

    def __init__(self, max_nodes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least 2")
        if hidden_dim not in {32, 64}:
            raise ValueError("hidden_dim must be 32 or 64")
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
        if float(self.residual_weight.item()) > 0.0:
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
    """Hard-switch final policy for Risk-Switch Lite-GLOBE-P.

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

        normal = self.normal_policy(observation)
        predictive = self.predictive_policy(observation)
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
        return StudentPolicyOutput(combined, valid_logits, probabilities)

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


class RiskSwitchLiteGlobePPlusStudentPolicy(
    RiskSwitchLiteGlobePStudentPolicy
):
    """Risk-Switch Lite-GLOBE-P+ with deployable stability safeguards.

    Phase 12 switched to the predictive branch when the Phase 8 action looked
    locally risky. P+ keeps that conservative structure but adds the pieces
    needed for a paper-grade final method: link-loss-aware danger scoring,
    top-k onward stability, energy-aware predictive tie-breaking, and explicit
    drop suppression when at least one safe forward candidate is available.
    """

    def __init__(
        self,
        normal_policy: GeographicResidualStudentPolicy,
        predictive_policy: LiteGlobePStudentPolicy,
        *,
        switch_threshold: float = 0.08,
        margin_gate: float = 0.04,
        lifetime_gate: float = 0.20,
        onward_gate: float = 0.20,
        topk_onward_gate: float = 0.18,
        redundancy_gate: float = 0.18,
        loss_keep_gate: float = 0.82,
        predictive_margin: float = 0.08,
        energy_tie_weight: float = 0.35,
        drop_suppression_bonus: float = 8.0,
    ) -> None:
        super().__init__(
            normal_policy,
            predictive_policy,
            switch_threshold=switch_threshold,
            margin_gate=margin_gate,
            lifetime_gate=lifetime_gate,
            onward_gate=onward_gate,
        )
        for name, value in {
            "topk_onward_gate": topk_onward_gate,
            "redundancy_gate": redundancy_gate,
            "loss_keep_gate": loss_keep_gate,
            "predictive_margin": predictive_margin,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in {
            "energy_tie_weight": energy_tie_weight,
            "drop_suppression_bonus": drop_suppression_bonus,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.register_buffer(
            "topk_onward_gate", torch.tensor(topk_onward_gate)
        )
        self.register_buffer(
            "redundancy_gate", torch.tensor(redundancy_gate)
        )
        self.register_buffer("loss_keep_gate", torch.tensor(loss_keep_gate))
        self.register_buffer(
            "predictive_margin", torch.tensor(predictive_margin)
        )
        self.register_buffer(
            "energy_tie_weight", torch.tensor(energy_tie_weight)
        )
        self.register_buffer(
            "drop_suppression_bonus",
            torch.tensor(drop_suppression_bonus),
        )

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentPolicyOutput:
        """Use Phase 8 normally and P+ safeguards in high-risk states."""

        normal = self.normal_policy(observation)
        predictive = self.predictive_policy(observation)
        unbatched = observation["self_features"].ndim == 1
        action_mask = observation["action_mask"]
        normal_logits = normal.logits
        predictive_logits = predictive.logits
        if unbatched:
            action_mask = action_mask.unsqueeze(0)
            normal_logits = normal_logits.unsqueeze(0)
            predictive_logits = predictive_logits.unsqueeze(0)
        predictive_logits = self._energy_adjusted_logits(
            observation,
            predictive_logits,
            unbatched=unbatched,
        )
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
        combined = self._suppress_drop_when_forward_safe(
            observation,
            combined,
            action_mask=action_mask,
            unbatched=unbatched,
        )
        valid_logits = masked_logits(combined, action_mask)
        probabilities = masked_softmax(combined, action_mask)
        if unbatched:
            combined = combined.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        return StudentPolicyOutput(combined, valid_logits, probabilities)

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
        switch_features = self._switch_features(
            observation,
            template=risk,
            unbatched=unbatched,
        )
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
        normal_switch = switch_features[batch, normal_candidate]
        predictive_switch = switch_features[batch, predictive_candidate]
        normal_danger = self._danger_score(normal_risk, normal_switch)
        predictive_danger = self._danger_score(
            predictive_risk,
            predictive_switch,
        )
        safety_gain = (
            self._safety_score(predictive_risk, predictive_switch)
            - self._safety_score(normal_risk, normal_switch)
        )
        candidate_mask = action_mask[:, : self.max_nodes].to(torch.bool)
        has_candidate = torch.any(candidate_mask, dim=-1)
        high_risk = normal_danger > self.switch_threshold
        safer_predictive = (
            (predictive_action != normal_action)
            & (safety_gain > self.predictive_margin)
            & (predictive_danger < normal_danger)
        )
        return has_candidate & (normal_is_drop | high_risk | safer_predictive)

    def _switch_features(
        self,
        observation: Mapping[str, Tensor],
        *,
        template: Tensor,
        unbatched: bool,
    ) -> Tensor:
        switch_features = observation.get("candidate_switch_features")
        if switch_features is None:
            fallback = torch.zeros(
                template.shape[0],
                self.max_nodes,
                4,
                device=template.device,
                dtype=template.dtype,
            )
            fallback[:, :, 2:] = 1.0
            return fallback
        if unbatched:
            switch_features = switch_features.unsqueeze(0)
        return switch_features.to(device=template.device, dtype=template.dtype)

    def _danger_score(self, risk: Tensor, switch_features: Tensor) -> Tensor:
        margin = risk[:, 0]
        lifetime = risk[:, 1]
        onward = risk[:, 3]
        topk_onward = switch_features[:, 0]
        redundancy = switch_features[:, 1]
        link_keep = switch_features[:, 2]
        return (
            torch.relu(self.margin_gate - margin)
            + torch.relu(self.lifetime_gate - lifetime)
            + torch.relu(self.onward_gate - onward)
            + torch.relu(self.topk_onward_gate - topk_onward)
            + 0.5 * torch.relu(self.redundancy_gate - redundancy)
            + torch.relu(self.loss_keep_gate - link_keep)
        )

    def _safety_score(self, risk: Tensor, switch_features: Tensor) -> Tensor:
        return (
            risk[:, 0]
            + risk[:, 1]
            + risk[:, 3]
            + switch_features[:, 0]
            + 0.5 * switch_features[:, 1]
            + switch_features[:, 2]
        )

    def _energy_adjusted_logits(
        self,
        observation: Mapping[str, Tensor],
        logits: Tensor,
        *,
        unbatched: bool,
    ) -> Tensor:
        risk = observation.get("candidate_risk_features")
        if risk is None:
            return logits
        if unbatched:
            risk = risk.unsqueeze(0)
        risk = risk.to(device=logits.device, dtype=logits.dtype)
        switch_features = self._switch_features(
            observation,
            template=risk,
            unbatched=unbatched,
        )
        energy_efficiency = switch_features[:, :, 3]
        candidate_logits = (
            logits[:, : self.max_nodes]
            + self.energy_tie_weight * energy_efficiency
        )
        return torch.cat(
            (candidate_logits, logits[:, self.max_nodes :]),
            dim=-1,
        )

    def _suppress_drop_when_forward_safe(
        self,
        observation: Mapping[str, Tensor],
        logits: Tensor,
        *,
        action_mask: Tensor,
        unbatched: bool,
    ) -> Tensor:
        risk = observation.get("candidate_risk_features")
        if risk is None:
            return logits
        if unbatched:
            risk = risk.unsqueeze(0)
        risk = risk.to(device=logits.device, dtype=logits.dtype)
        switch_features = self._switch_features(
            observation,
            template=risk,
            unbatched=unbatched,
        )
        candidate_mask = action_mask[:, : self.max_nodes].to(torch.bool)
        safe = (
            candidate_mask
            & (risk[:, :, 0] >= self.margin_gate)
            & (risk[:, :, 1] >= self.lifetime_gate)
            & (risk[:, :, 3] >= self.onward_gate)
            & (switch_features[:, :, 0] >= self.topk_onward_gate)
            & (switch_features[:, :, 2] >= self.loss_keep_gate - 0.20)
        )
        has_safe = torch.any(safe, dim=-1)
        if not torch.any(has_safe):
            return logits
        adjusted = logits.clone()
        adjusted[:, self.drop_action] = torch.where(
            has_safe,
            adjusted[:, self.drop_action] - self.drop_suppression_bonus,
            adjusted[:, self.drop_action],
        )
        return adjusted

    @torch.inference_mode()
    def diagnostics(self, observation: Mapping[str, Tensor]) -> dict[str, float]:
        """Return step-level switch diagnostics for evaluation logging."""

        normal = self.normal_policy(observation)
        predictive = self.predictive_policy(observation)
        unbatched = observation["self_features"].ndim == 1
        action_mask = observation["action_mask"]
        normal_logits = normal.logits
        predictive_logits = predictive.logits
        if unbatched:
            action_mask = action_mask.unsqueeze(0)
            normal_logits = normal_logits.unsqueeze(0)
            predictive_logits = predictive_logits.unsqueeze(0)
        predictive_logits = self._energy_adjusted_logits(
            observation,
            predictive_logits,
            unbatched=unbatched,
        )
        switch = self._switch_mask(
            observation,
            action_mask=action_mask,
            normal_logits=normal_logits,
            predictive_logits=predictive_logits,
            unbatched=unbatched,
        )
        risk = observation.get("candidate_risk_features")
        if risk is None:
            return {
                "switch_steps": float(torch.sum(switch).item()),
                "safe_forward_candidates": 0.0,
                "mean_selected_danger": 0.0,
            }
        if unbatched:
            risk = risk.unsqueeze(0)
        risk = risk.to(device=normal_logits.device, dtype=normal_logits.dtype)
        switch_features = self._switch_features(
            observation,
            template=risk,
            unbatched=unbatched,
        )
        normal_action = torch.argmax(
            masked_logits(normal_logits, action_mask),
            dim=-1,
        )
        batch = torch.arange(normal_action.shape[0], device=normal_logits.device)
        selected = normal_action.clamp(max=self.max_nodes - 1)
        selected_danger = self._danger_score(
            risk[batch, selected],
            switch_features[batch, selected],
        )
        candidate_mask = action_mask[:, : self.max_nodes].to(torch.bool)
        safe = (
            candidate_mask
            & (risk[:, :, 0] >= self.margin_gate)
            & (risk[:, :, 1] >= self.lifetime_gate)
            & (risk[:, :, 3] >= self.onward_gate)
            & (switch_features[:, :, 0] >= self.topk_onward_gate)
        )
        return {
            "switch_steps": float(torch.sum(switch).item()),
            "safe_forward_candidates": float(torch.sum(safe).item()),
            "mean_selected_danger": float(torch.mean(selected_danger).item()),
        }

    def set_switch_parameters(
        self,
        *,
        switch_threshold: float,
        margin_gate: float,
        lifetime_gate: float,
        onward_gate: float,
        topk_onward_gate: float | None = None,
        redundancy_gate: float | None = None,
        loss_keep_gate: float | None = None,
        predictive_margin: float | None = None,
        energy_tie_weight: float | None = None,
        drop_suppression_bonus: float | None = None,
    ) -> None:
        """Set calibrated P+ switch and safeguard parameters."""

        super().set_switch_parameters(
            switch_threshold=switch_threshold,
            margin_gate=margin_gate,
            lifetime_gate=lifetime_gate,
            onward_gate=onward_gate,
        )
        updates = {
            "topk_onward_gate": topk_onward_gate,
            "redundancy_gate": redundancy_gate,
            "loss_keep_gate": loss_keep_gate,
            "predictive_margin": predictive_margin,
        }
        for name, value in updates.items():
            if value is None:
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            getattr(self, name).fill_(value)
        non_negative_updates = {
            "energy_tie_weight": energy_tie_weight,
            "drop_suppression_bonus": drop_suppression_bonus,
        }
        for name, value in non_negative_updates.items():
            if value is None:
                continue
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            getattr(self, name).fill_(value)
