"""Conversion of NumPy Gymnasium observations to batched tensors."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


TensorObservation = dict[str, Tensor]


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
