"""Serializable local-observation dataset with leakage-safe group splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


LOCAL_KEYS = (
    "self_features",
    "neighbor_features",
    "edge_features",
    "packet_features",
    "action_mask",
)
OPTIONAL_LOCAL_KEYS = (
    "candidate_forwardability",
    "candidate_risk_features",
)


class DistillationDataset(Dataset[dict[str, torch.Tensor]]):
    """Teacher targets paired only with deployable Student observations."""

    def __init__(self, arrays: dict[str, NDArray[np.generic]]) -> None:
        required = {
            *LOCAL_KEYS,
            "teacher_logits",
            "teacher_probabilities",
            "selected_actions",
            "episode_seeds",
            "episode_steps",
            "scenario_ids",
        }
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"dataset arrays are missing: {sorted(missing)}")
        sample_count = arrays["selected_actions"].shape[0]
        if sample_count == 0:
            raise ValueError("distillation dataset cannot be empty")
        for key in required:
            if arrays[key].shape[0] != sample_count:
                raise ValueError(f"{key} has inconsistent sample count")
        if (
            "oracle_actions" in arrays
            and arrays["oracle_actions"].shape[0] != sample_count
        ):
            raise ValueError("oracle_actions has inconsistent sample count")
        if (
            "risk_oracle_actions" in arrays
            and arrays["risk_oracle_actions"].shape[0] != sample_count
        ):
            raise ValueError(
                "risk_oracle_actions has inconsistent sample count"
            )
        for key in OPTIONAL_LOCAL_KEYS:
            if key in arrays and arrays[key].shape[0] != sample_count:
                raise ValueError(f"{key} has inconsistent sample count")
        if not np.allclose(
            arrays["teacher_probabilities"].sum(axis=-1), 1.0, atol=1e-5
        ):
            raise ValueError("teacher probabilities must sum to one")
        if np.any(
            arrays["teacher_probabilities"][
                arrays["action_mask"].astype(bool) == 0
            ]
            != 0
        ):
            raise ValueError("invalid actions must have zero Teacher probability")
        self.arrays = arrays

    def __len__(self) -> int:
        return int(self.arrays["selected_actions"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {}
        for key in LOCAL_KEYS:
            dtype = torch.bool if key == "action_mask" else torch.float32
            item[key] = torch.as_tensor(self.arrays[key][index], dtype=dtype)
        for key in OPTIONAL_LOCAL_KEYS:
            if key in self.arrays:
                item[key] = torch.as_tensor(
                    self.arrays[key][index], dtype=torch.float32
                )
        item["teacher_logits"] = torch.as_tensor(
            self.arrays["teacher_logits"][index], dtype=torch.float32
        )
        item["teacher_probabilities"] = torch.as_tensor(
            self.arrays["teacher_probabilities"][index], dtype=torch.float32
        )
        item["selected_actions"] = torch.as_tensor(
            self.arrays["selected_actions"][index], dtype=torch.long
        )
        if "oracle_actions" in self.arrays:
            item["oracle_actions"] = torch.as_tensor(
                self.arrays["oracle_actions"][index], dtype=torch.long
            )
        if "risk_oracle_actions" in self.arrays:
            item["risk_oracle_actions"] = torch.as_tensor(
                self.arrays["risk_oracle_actions"][index],
                dtype=torch.long,
            )
        return item

    @property
    def group_ids(self) -> NDArray[np.str_]:
        return np.char.add(
            np.char.add(self.arrays["scenario_ids"].astype(str), ":"),
            self.arrays["episode_seeds"].astype(str),
        )

    def subset(self, indices: NDArray[np.int64]) -> "DistillationDataset":
        return DistillationDataset(
            {key: value[indices] for key, value in self.arrays.items()}
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **self.arrays)

    @classmethod
    def load(cls, path: str | Path) -> "DistillationDataset":
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        return cls(arrays)


def concatenate_datasets(
    datasets: list[DistillationDataset],
) -> DistillationDataset:
    """Concatenate compatible datasets while preserving episode metadata."""

    if not datasets:
        raise ValueError("at least one dataset is required")
    keys = set(datasets[0].arrays)
    if any(set(dataset.arrays) != keys for dataset in datasets[1:]):
        raise ValueError("all datasets must contain the same arrays")
    return DistillationDataset(
        {
            key: np.concatenate(
                [dataset.arrays[key] for dataset in datasets], axis=0
            )
            for key in keys
        }
    )


@dataclass(frozen=True)
class DistillationSplit:
    train: DistillationDataset
    validation: DistillationDataset
    test: DistillationDataset


def split_by_episode_group(
    dataset: DistillationDataset,
    *,
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> DistillationSplit:
    """Assign complete scenario/episode groups to one split only."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum below one")
    groups = np.unique(dataset.group_ids)
    if groups.size < 3:
        raise ValueError("at least three episode groups are required")
    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(groups.size)]
    train_count = min(
        max(1, int(groups.size * train_fraction)),
        groups.size - 2,
    )
    validation_count = min(
        max(1, int(groups.size * validation_fraction)),
        groups.size - train_count - 1,
    )
    train_end = train_count
    validation_end = train_count + validation_count
    group_sets = (
        set(groups[:train_end]),
        set(groups[train_end:validation_end]),
        set(groups[validation_end:]),
    )
    if any(not group_set for group_set in group_sets):
        raise ValueError("each split must contain at least one episode group")
    indices = [
        np.flatnonzero(np.isin(dataset.group_ids, list(group_set))).astype(
            np.int64
        )
        for group_set in group_sets
    ]
    return DistillationSplit(
        train=dataset.subset(indices[0]),
        validation=dataset.subset(indices[1]),
        test=dataset.subset(indices[2]),
    )
