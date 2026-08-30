"""Strict-local GAT-GRU Double-DQN architecture control."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import Observation
from .rdqn_herp import RdqnHerpAdaptedPolicy, _batch


class GatGruQNetwork(nn.Module):
    """Candidate self-attention followed by a temporal GRU state."""

    def __init__(self, max_nodes: int, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.max_nodes = max_nodes
        self.input = nn.Linear(13, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.candidate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.drop = nn.Linear(hidden_dim, 1)
        self.hidden: Tensor | None = None

    def reset_hidden(self) -> None:
        self.hidden = None

    def reset_noise(self) -> None:
        """Compatibility with the shared DDQN trainer (no NoisyNet here)."""

        return None

    def forward(self, features: Tensor, mask: Tensor, *, update_hidden: bool = False) -> Tensor:
        encoded = torch.relu(self.input(features))
        attention_mask = mask.clone()
        empty = ~attention_mask.any(dim=1)
        attention_mask[empty, 0] = True
        attended, _ = self.attention(encoded, encoded, encoded, key_padding_mask=~attention_mask, need_weights=False)
        valid = mask.to(attended.dtype).unsqueeze(-1)
        pooled = (attended * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        hidden = self.hidden
        if hidden is None or hidden.shape[0] != pooled.shape[0]:
            hidden = torch.zeros_like(pooled)
        temporal = self.gru(pooled, hidden)
        if update_hidden:
            self.hidden = temporal.detach()
        expanded = temporal.unsqueeze(1).expand(-1, self.max_nodes, -1)
        candidate = self.candidate(torch.cat((attended, expanded), dim=-1)).squeeze(-1)
        return torch.cat((candidate, self.drop(temporal)), dim=-1)


class GatGruDdqnPolicy(RdqnHerpAdaptedPolicy):
    source = "architecture inspired by 10.1109/WCSP68525.2025.1010249"
    fidelity = "inspired architecture control; not SRRGD-DQN"

    def __init__(self, max_nodes: int, *, hidden_dim: int = 64, device: torch.device | str = "cpu", **kwargs) -> None:
        super().__init__(max_nodes, hidden_dim=hidden_dim, device=device, **kwargs)
        self.online = GatGruQNetwork(max_nodes, hidden_dim).to(self.device)
        self.target = GatGruQNetwork(max_nodes, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.online.reset_hidden()
        self.target.reset_hidden()

    def q_values(self, observation: Observation) -> Tensor:
        features, mask = _batch([observation], self.drop_action, self.device)
        self.online.eval()
        with torch.no_grad():
            values = self.online(features, mask, update_hidden=True)[0]
            full_mask = torch.cat((mask[0], torch.ones(1, dtype=torch.bool, device=self.device)))
            return values.masked_fill(~full_mask, -torch.inf)

    def checkpoint_state(self) -> dict:
        state = super().checkpoint_state()
        state["online_hidden"] = self.online.hidden
        state["target_hidden"] = self.target.hidden
        return state

    def load_checkpoint_state(self, state: dict) -> None:
        super().load_checkpoint_state(state)
        self.online.hidden = state.get("online_hidden")
        self.target.hidden = state.get("target_hidden")
