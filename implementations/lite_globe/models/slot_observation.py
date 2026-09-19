"""Local-slot observation compaction and a size-agnostic policy wrapper.

Motivation
----------
The environment lays neighbour features out in a ``(max_nodes, F)`` array
indexed by *global node id*, and the action space is ``Discrete(max_nodes + 1)``
over those same global ids. Two consequences follow:

1. The network cannot be deployed on a swarm larger than ``max_nodes`` without
   retraining, even though the routing decision is purely local.
2. The policy reads ``max_nodes`` rows on every decision regardless of the
   relay's actual degree, which inflates the reported policy-input footprint and
   makes it incomparable with a protocol that only touches its real neighbours.

Key observation
---------------
``LocalStudentPolicy`` and every subclass score candidates with a *shared*
per-candidate encoder and pool context over valid candidates only. No parameter
tensor has a ``max_nodes`` dimension, which means

* the model is exactly permutation-equivariant over candidate slots, and
* a checkpoint trained with ``max_nodes=32`` loads unchanged into a model built
  with ``max_nodes=8``.

So compacting the observation into local slots is a *semantics-preserving*
re-indexing: the selected next hop is unchanged, while the tensor the policy
reads shrinks to the relay's real degree. No retraining is required.

Scope
-----
This module changes the observation layout and action indexing only. It does not
change any policy definition. The one behavioural caveat is argmax tie-breaking:
``torch.argmax`` resolves ties to the lowest index, and compaction renumbers the
candidates, so exactly-tied logits can select a different global node. Ties do
not occur for the trained checkpoints in this repository but the property is not
guaranteed, and :func:`compact_observation` is therefore documented as
action-preserving up to ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from numpy.typing import NDArray

#: Observation entries laid out as ``(max_nodes, F)`` and indexed by global id.
CANDIDATE_KEYS = (
    "neighbor_features",
    "edge_features",
    "candidate_forwardability",
    "candidate_risk_features",
)

#: Index into ``packet_features`` of the normalised source and destination ids.
#: These are global identifiers and must be suppressed for a size-agnostic
#: policy. The main evaluation scenarios already build observations with
#: ``include_node_ids=False``, which leaves both entries at zero.
NODE_ID_FEATURE_SLICE = slice(4, 6)


@dataclass(frozen=True)
class SlotMapping:
    """Mapping from compact local slots back to global node ids."""

    global_ids: tuple[int, ...]
    max_slots: int
    truncated: int

    @property
    def drop_slot(self) -> int:
        return self.max_slots

    def to_global(self, slot_action: int) -> int:
        """Translate a slot-indexed action back to an environment action.

        The drop slot maps to the environment's drop action, which the caller
        supplies because it depends on the environment's ``max_nodes``.
        """

        if slot_action == self.drop_slot:
            raise ValueError("drop must be translated by the caller")
        if not 0 <= slot_action < len(self.global_ids):
            raise ValueError(
                f"slot {slot_action} is outside the {len(self.global_ids)} "
                "occupied slots"
            )
        return self.global_ids[slot_action]


def compact_observation(
    observation: Mapping[str, NDArray[np.generic]],
    *,
    max_slots: int,
    zero_node_ids: bool = True,
) -> tuple[dict[str, NDArray[np.generic]], SlotMapping]:
    """Re-index a global-id observation into ``max_slots`` local slots.

    Valid candidates are packed into the leading slots in ascending global-id
    order. If the relay's degree exceeds ``max_slots`` the nearest candidates are
    kept, measured by the normalised link distance in ``edge_features``, and the
    number of dropped candidates is recorded on the mapping so truncation is
    never silent.

    Args:
        observation: A single (unbatched) environment observation.
        max_slots: Number of candidate slots in the compacted observation.
        zero_node_ids: Suppress the normalised global source/destination ids in
            ``packet_features``. Required for a genuinely size-agnostic policy.

    Returns:
        The compacted observation and the slot-to-global mapping.
    """

    if max_slots < 1:
        raise ValueError("max_slots must be at least 1")
    mask = np.asarray(observation["action_mask"])
    env_max_nodes = mask.shape[-1] - 1
    valid = np.flatnonzero(mask[:env_max_nodes]).astype(int)

    truncated = 0
    if valid.size > max_slots:
        edges = np.asarray(observation["edge_features"])
        # Column 0 is distance normalised by the communication radius.
        order = np.argsort(edges[valid, 0], kind="stable")
        keep = np.sort(valid[order[:max_slots]])
        truncated = int(valid.size - max_slots)
        valid = keep

    compacted: dict[str, NDArray[np.generic]] = {}
    for key in CANDIDATE_KEYS:
        if key not in observation:
            continue
        source = np.asarray(observation[key])
        out = np.zeros((max_slots, source.shape[1]), dtype=source.dtype)
        if valid.size:
            out[: valid.size] = source[valid]
        compacted[key] = out

    new_mask = np.zeros(max_slots + 1, dtype=mask.dtype)
    new_mask[: valid.size] = 1
    new_mask[max_slots] = 1  # DROP is always available.
    compacted["action_mask"] = new_mask

    compacted["self_features"] = np.asarray(observation["self_features"]).copy()
    packet = np.asarray(observation["packet_features"]).copy()
    if zero_node_ids:
        packet[NODE_ID_FEATURE_SLICE] = 0.0
    compacted["packet_features"] = packet

    return compacted, SlotMapping(
        global_ids=tuple(int(node) for node in valid),
        max_slots=max_slots,
        truncated=truncated,
    )


def observation_nbytes(observation: Mapping[str, NDArray[np.generic]]) -> int:
    """Total in-memory footprint of an observation, in bytes."""

    return sum(int(np.asarray(value).nbytes) for value in observation.values())


class SlotCompactedPolicy:
    """Run a slot-width student policy behind the global-id environment API.

    The wrapper compacts each observation into ``max_slots`` local slots, runs
    the wrapped model, and maps the chosen slot back to a global node id. It
    exposes the same ``reset``/``act`` protocol the evaluator expects, so it can
    be dropped into any existing campaign.

    Args:
        model: A ``LocalStudentPolicy`` subclass built with
            ``max_nodes == max_slots``. Because no parameter depends on
            ``max_nodes``, a checkpoint trained at any other width loads
            directly.
        env_drop_action: The environment's drop action, i.e. its ``max_nodes``.
        max_slots: Candidate slots presented to the model.
    """

    def __init__(
        self,
        model: Any,
        *,
        env_drop_action: int,
        max_slots: int,
        device: torch.device | str = "cpu",
        zero_node_ids: bool = True,
    ) -> None:
        if getattr(model, "max_nodes", max_slots) != max_slots:
            raise ValueError(
                f"model was built for max_nodes={model.max_nodes}, "
                f"but max_slots={max_slots}"
            )
        self.model = model
        self.env_drop_action = env_drop_action
        self.max_slots = max_slots
        self.device = torch.device(device)
        self.zero_node_ids = zero_node_ids
        self.last_input_bytes = 0
        self.last_truncated = 0
        self.total_truncated = 0

    def reset(self, seed: int | None = None) -> None:
        self.last_input_bytes = 0
        self.last_truncated = 0
        self.total_truncated = 0
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset(seed)

    def act(self, observation: Mapping[str, NDArray[np.generic]]) -> int:
        compacted, mapping = compact_observation(
            observation,
            max_slots=self.max_slots,
            zero_node_ids=self.zero_node_ids,
        )
        self.last_input_bytes = observation_nbytes(compacted)
        self.last_truncated = mapping.truncated
        self.total_truncated += mapping.truncated

        tensors = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in compacted.items()
        }
        tensors["action_mask"] = tensors["action_mask"].to(torch.bool)
        for key, value in tensors.items():
            if key != "action_mask" and not torch.is_floating_point(value):
                tensors[key] = value.to(torch.float32)

        with torch.no_grad():
            self.model.eval()
            output = self.model(tensors)
        slot_action = int(torch.argmax(output.masked_logits).item())
        if slot_action == mapping.drop_slot or not mapping.global_ids:
            return self.env_drop_action
        return mapping.to_global(slot_action)
