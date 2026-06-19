"""Privileged two-layer message-passing Teacher actor-critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from ..env.global_observation import (
    GLOBAL_EDGE_FEATURES,
    GLOBAL_NODE_FEATURES,
    GLOBAL_PACKET_FEATURES,
)
from .masking import masked_logits, masked_softmax


@dataclass(frozen=True)
class TeacherOutput:
    """Policy and centralized value produced from the full FANET graph."""

    logits: Tensor
    masked_logits: Tensor
    probabilities: Tensor
    value: Tensor


class MessagePassingLayer(nn.Module):
    """Aggregate edge-conditioned neighbor messages with a masked mean."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + GLOBAL_EDGE_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        nodes: Tensor,
        adjacency: Tensor,
        edges: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        batch_size, max_nodes, hidden_dim = nodes.shape
        senders = nodes.unsqueeze(1).expand(
            batch_size, max_nodes, max_nodes, hidden_dim
        )
        messages = self.message(torch.cat((senders, edges), dim=-1))
        valid_edges = adjacency.unsqueeze(-1).to(messages.dtype)
        aggregated = (messages * valid_edges).sum(dim=2)
        degree = valid_edges.sum(dim=2).clamp_min(1.0)
        updated = self.update(torch.cat((nodes, aggregated / degree), dim=-1))
        return updated * node_mask.unsqueeze(-1).to(updated.dtype)


class GlobalTeacherActorCritic(nn.Module):
    """Full-graph policy used as a privileged reference during training."""

    def __init__(self, max_nodes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least 2")
        if hidden_dim not in {32, 64}:
            raise ValueError("hidden_dim must be 32 or 64")
        self.max_nodes = max_nodes
        self.drop_action = max_nodes
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(GLOBAL_NODE_FEATURES, hidden_dim),
            nn.ReLU(),
        )
        self.message_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim) for _ in range(2)]
        )
        candidate_dim = (
            hidden_dim * 2 + GLOBAL_EDGE_FEATURES + GLOBAL_PACKET_FEATURES
        )
        self.candidate_scorer = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.drop_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + GLOBAL_PACKET_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.drop_scorer[-1].bias, -2.0)
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim * 3 + GLOBAL_PACKET_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: Mapping[str, Tensor]) -> TeacherOutput:
        tensors, unbatched = self._validate_and_batch(observation)
        nodes = self.node_encoder(tensors["node_features"])
        adjacency = tensors["adjacency"]
        edges = tensors["edge_features"]
        node_mask = tensors["node_mask"]
        packet = tensors["packet_features"]
        action_mask = tensors["action_mask"]
        for layer in self.message_layers:
            nodes = layer(nodes, adjacency, edges, node_mask)

        current_indicator = tensors["node_features"][:, :, 5]
        destination_indicator = tensors["node_features"][:, :, 6]
        current = torch.sum(nodes * current_indicator.unsqueeze(-1), dim=1)
        destination = torch.sum(
            nodes * destination_indicator.unsqueeze(-1), dim=1
        )
        valid_nodes = node_mask.unsqueeze(-1).to(nodes.dtype)
        global_context = (nodes * valid_nodes).sum(dim=1) / valid_nodes.sum(
            dim=1
        ).clamp_min(1.0)
        current_edges = torch.einsum(
            "bn,bnme->bme", current_indicator, edges
        )

        expanded_current = current.unsqueeze(1).expand(-1, self.max_nodes, -1)
        expanded_packet = packet.unsqueeze(1).expand(-1, self.max_nodes, -1)
        candidate_logits = self.candidate_scorer(
            torch.cat(
                (expanded_current, nodes, current_edges, expanded_packet),
                dim=-1,
            )
        ).squeeze(-1)
        drop_logit = self.drop_scorer(
            torch.cat((current, global_context, packet), dim=-1)
        )
        logits = torch.cat((candidate_logits, drop_logit), dim=-1)
        valid_logits = masked_logits(logits, action_mask)
        probabilities = masked_softmax(logits, action_mask)
        value = self.value_network(
            torch.cat((global_context, current, destination, packet), dim=-1)
        ).squeeze(-1)

        if unbatched:
            logits = logits.squeeze(0)
            valid_logits = valid_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
            value = value.squeeze(0)
        return TeacherOutput(logits, valid_logits, probabilities, value)

    def _validate_and_batch(
        self, observation: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], bool]:
        expected = {
            "node_features": (self.max_nodes, GLOBAL_NODE_FEATURES),
            "adjacency": (self.max_nodes, self.max_nodes),
            "edge_features": (
                self.max_nodes,
                self.max_nodes,
                GLOBAL_EDGE_FEATURES,
            ),
            "node_mask": (self.max_nodes,),
            "packet_features": (GLOBAL_PACKET_FEATURES,),
            "action_mask": (self.max_nodes + 1,),
        }
        missing = set(expected).difference(observation)
        if missing:
            raise ValueError(f"global observation is missing keys: {sorted(missing)}")
        unbatched = observation["node_features"].ndim == 2
        tensors: dict[str, Tensor] = {}
        batch_size: int | None = None
        for key, trailing_shape in expected.items():
            tensor = observation[key]
            rank = len(trailing_shape) if unbatched else len(trailing_shape) + 1
            if tensor.ndim != rank:
                raise ValueError(f"{key} rank is {tensor.ndim}, expected {rank}")
            if tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
                raise ValueError(
                    f"{key} shape is {tuple(tensor.shape)}, expected *{trailing_shape}"
                )
            if unbatched:
                tensor = tensor.unsqueeze(0)
            elif batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError("global observation batch dimensions do not match")
            tensors[key] = tensor

        for key in ("adjacency", "node_mask", "action_mask"):
            tensors[key] = tensors[key].to(torch.bool)
        for key in ("node_features", "edge_features", "packet_features"):
            tensors[key] = tensors[key].to(torch.float32)
            if not torch.all(torch.isfinite(tensors[key])):
                raise ValueError(f"{key} contains non-finite values")
        return tensors, unbatched
