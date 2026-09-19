"""Conversion of NumPy Gymnasium observations to batched tensors."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


TensorObservation = dict[str, Tensor]


class TensorObservationBuffer:
    """Reuse fixed-shape device tensors across sequential decisions."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._buffers: TensorObservation = {}

    def convert(
        self, observation: Mapping[str, NDArray[np.generic] | Tensor]
    ) -> TensorObservation:
        if set(self._buffers) != set(observation):
            self._buffers = {}
        for key, value in observation.items():
            dtype = torch.bool if key == "action_mask" else torch.float32
            shape = tuple(value.shape)
            buffer = self._buffers.get(key)
            if buffer is None or tuple(buffer.shape) != shape or buffer.dtype != dtype:
                buffer = torch.empty(shape, dtype=dtype, device=self.device)
                self._buffers[key] = buffer
            source = torch.as_tensor(value, dtype=dtype)
            buffer.copy_(source, non_blocking=False)
        return self._buffers


def observation_to_tensors(
    observation: Mapping[str, NDArray[np.generic] | Tensor],
    *,
    device: torch.device | str | None = None,
) -> TensorObservation:
    """Convert one environment observation without changing its rank."""

    result: TensorObservation = {}
    for key, value in observation.items():
        tensor = torch.as_tensor(value, device=device)
        if key == "action_mask":
            tensor = tensor.to(dtype=torch.bool)
        else:
            tensor = tensor.to(dtype=torch.float32)
        result[key] = tensor
    return result
