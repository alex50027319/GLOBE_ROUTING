"""Numerically stable structural action masking."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_logits(logits: Tensor, action_mask: Tensor) -> Tensor:
    """Replace invalid logits with negative infinity.

    Every batch row must retain at least one structurally valid action.
    """

    if logits.shape != action_mask.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match "
            f"mask shape {tuple(action_mask.shape)}"
        )
    mask = action_mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("each action-mask row must contain a valid action")
    if not torch.all(torch.isfinite(logits)):
        raise ValueError("logits contain non-finite values before masking")
    return logits.masked_fill(~mask, -torch.inf)


def masked_softmax(logits: Tensor, action_mask: Tensor) -> Tensor:
    """Return a normalized distribution with exact zero on invalid actions."""

    masked = masked_logits(logits, action_mask)
    probabilities = torch.softmax(masked, dim=-1)
    if not torch.all(torch.isfinite(probabilities)):
        raise ValueError("masked softmax produced non-finite probabilities")
    return probabilities
