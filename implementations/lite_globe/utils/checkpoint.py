"""Explicit model checkpoint save and restore."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "metadata": metadata or {}},
        destination,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return dict(checkpoint.get("metadata", {}))
