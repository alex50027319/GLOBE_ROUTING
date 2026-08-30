"""Single source of truth for external comparison method construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .aodv import AodvPolicy
from .evo_qgeo import EvoQGeoAdaptedPolicy
from .gat_gru_ddqn import GatGruDdqnPolicy
from .gpsr import GpsrPolicy
from .olsr import OlsrPolicy
from .rdqn_herp import RdqnHerpAdaptedPolicy


@dataclass(frozen=True)
class MethodSpec:
    name: str
    slug: str
    source: str
    fidelity: str
    trainable: bool
    control_plane: bool
    observation_fields: tuple[str, ...]
    hop_radius: str
    privileged_information: bool
    builder: Callable[..., object]

    def manifest_dict(self) -> dict:
        row = asdict(self)
        row.pop("builder")
        return row


def _greedy(max_nodes: int, **_: object) -> GpsrPolicy:
    return GpsrPolicy(max_nodes)


def _aodv(max_nodes: int, **_: object) -> AodvPolicy:
    return AodvPolicy(max_nodes)


def _olsr(max_nodes: int, **_: object) -> OlsrPolicy:
    return OlsrPolicy(max_nodes)


def _evo(max_nodes: int, **_: object) -> EvoQGeoAdaptedPolicy:
    return EvoQGeoAdaptedPolicy(max_nodes)


def _rdqn(max_nodes: int, *, hidden_dim: int, device: torch.device, **_: object) -> RdqnHerpAdaptedPolicy:
    return RdqnHerpAdaptedPolicy(max_nodes, hidden_dim=hidden_dim, device=device)


def _gat(max_nodes: int, *, hidden_dim: int, device: torch.device, **_: object) -> GatGruDdqnPolicy:
    return GatGruDdqnPolicy(max_nodes, hidden_dim=hidden_dim, device=device)


METHOD_REGISTRY: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in (
        MethodSpec("AODV", "aodv", "RFC 3561", "common-contract adaptation", False, True,
                   ("action_mask", "packet_features"), "message-disseminated multi-hop", False, _aodv),
        MethodSpec("OLSR", "olsr", "RFC 3626", "common-contract adaptation", False, True,
                   ("action_mask", "packet_features"), "message-disseminated multi-hop", False, _olsr),
        MethodSpec("Greedy Geographic", "greedy_geographic", "GPSR greedy mode only",
                   "partial: perimeter recovery not implemented", False, False,
                   ("neighbor_features", "packet_features", "action_mask"), "1-hop", False, _greedy),
        MethodSpec("Evo-QGeo (Adapted)", "evo_qgeo_adapted", "10.3390/drones10020150",
                   "common-contract adaptation", True, True,
                   ("self_features", "neighbor_features", "edge_features", "packet_features", "action_mask", "candidate_forwardability", "candidate_risk_features"),
                   "1-hop plus beacon Q summary", False, _evo),
        MethodSpec("RDQN-HERP (Adapted)", "rdqn_herp_adapted", "10.1109/TVT.2026.3668740",
                   "common-contract adaptation", True, False,
                   ("self_features", "neighbor_features", "edge_features", "packet_features", "action_mask"), "1-hop", False, _rdqn),
        MethodSpec("GAT-GRU-DDQN", "gat_gru_ddqn", "architecture inspired by 10.1109/WCSP68525.2025.1010249",
                   "inspired architecture control; not SRRGD-DQN", True, False,
                   ("self_features", "neighbor_features", "edge_features", "packet_features", "action_mask"), "1-hop", False, _gat),
    )
}

PROPOSED_METHOD = "SwitchGLOBE"
EXTERNAL_METHODS = tuple(METHOD_REGISTRY)
COMPARISON_METHODS = (*EXTERNAL_METHODS, PROPOSED_METHOD)


def build_method(name: str, *, max_nodes: int, hidden_dim: int = 64,
                 device: torch.device | str = "cpu") -> object:
    if name not in METHOD_REGISTRY:
        raise KeyError(f"unknown external method: {name}")
    return METHOD_REGISTRY[name].builder(
        max_nodes=max_nodes,
        hidden_dim=hidden_dim,
        device=torch.device(device),
    )
