"""Offline forward-KL policy distillation for the Local Student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.distillation_dataset import DistillationDataset
from ..models.masking import masked_logits
from ..models.student_policy import LocalStudentPolicy


@dataclass(frozen=True)
class DistillationConfig:
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    temperature: float = 1.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    oracle_coefficient: float = 0.0
    risk_oracle_coefficient: float = 0.0
    teacher_action_coefficient: float = 0.0
    early_stopping_patience: int = 0
    minimum_improvement: float = 1e-5


@dataclass(frozen=True)
class DistillationMetrics:
    kl: float
    teacher_entropy: float
    action_agreement: float
    oracle_action_agreement: float | None = None
    risk_oracle_action_agreement: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class DistillationResult:
    epochs: int
    train: DistillationMetrics
    validation: DistillationMetrics


def forward_kl_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    action_mask: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute masked KL(Teacher || Student) and normalized distributions."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_masked = masked_logits(teacher_logits, action_mask) / temperature
    student_masked = masked_logits(student_logits, action_mask) / temperature
    teacher_log_probabilities = torch.log_softmax(teacher_masked, dim=-1)
    student_log_probabilities = torch.log_softmax(student_masked, dim=-1)
    teacher_probabilities = torch.exp(teacher_log_probabilities)
    valid = action_mask.to(torch.bool)
    log_ratio = torch.where(
        valid,
        teacher_log_probabilities - student_log_probabilities,
        torch.zeros_like(teacher_log_probabilities),
    )
    kl = torch.sum(
        teacher_probabilities * log_ratio,
        dim=-1,
    ).mean()
    return kl, teacher_probabilities, torch.exp(student_log_probabilities)


def _batch_observation(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    observation = {
        key: batch[key].to(device)
        for key in (
            "self_features",
            "neighbor_features",
            "edge_features",
            "packet_features",
            "action_mask",
        )
    }
    if "candidate_forwardability" in batch:
        observation["candidate_forwardability"] = batch[
            "candidate_forwardability"
        ].to(device)
    if "candidate_risk_features" in batch:
        observation["candidate_risk_features"] = batch[
            "candidate_risk_features"
        ].to(device)
    return observation


@torch.inference_mode()
def evaluate_distillation(
    model: LocalStudentPolicy,
    dataset: DistillationDataset,
    *,
    config: DistillationConfig,
    device: torch.device | str = "cpu",
) -> DistillationMetrics:
    device = torch.device(device)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    weighted_kl = 0.0
    weighted_entropy = 0.0
    agreements = 0
    oracle_agreements = 0
    risk_oracle_agreements = 0
    has_oracle = "oracle_actions" in dataset.arrays
    has_risk_oracle = "risk_oracle_actions" in dataset.arrays
    samples = 0
    for batch in loader:
        observation = _batch_observation(batch, device)
        output = model(observation)
        teacher_logits = batch["teacher_logits"].to(device)
        mask = observation["action_mask"]
        kl, teacher_probabilities, student_probabilities = forward_kl_loss(
            teacher_logits,
            output.logits,
            mask,
            temperature=config.temperature,
        )
        count = teacher_logits.shape[0]
        entropy = -torch.sum(
            teacher_probabilities
            * torch.log(teacher_probabilities.clamp_min(1e-8)),
            dim=-1,
        ).mean()
        agreements += int(
            torch.sum(
                torch.argmax(teacher_probabilities, dim=-1)
                == torch.argmax(student_probabilities, dim=-1)
            ).item()
        )
        if has_oracle:
            oracle_agreements += int(
                torch.sum(
                    torch.argmax(student_probabilities, dim=-1)
                    == batch["oracle_actions"].to(device)
                ).item()
            )
        if has_risk_oracle:
            risk_oracle_agreements += int(
                torch.sum(
                    torch.argmax(student_probabilities, dim=-1)
                    == batch["risk_oracle_actions"].to(device)
                ).item()
            )
        weighted_kl += float(kl.item()) * count
        weighted_entropy += float(entropy.item()) * count
        samples += count
    return DistillationMetrics(
        kl=weighted_kl / samples,
        teacher_entropy=weighted_entropy / samples,
        action_agreement=agreements / samples,
        oracle_action_agreement=(
            oracle_agreements / samples if has_oracle else None
        ),
        risk_oracle_action_agreement=(
            risk_oracle_agreements / samples
            if has_risk_oracle
            else None
        ),
    )


def _distillation_objective(
    model: LocalStudentPolicy,
    batch: dict[str, Tensor],
    observation: dict[str, Tensor],
    *,
    config: DistillationConfig,
    device: torch.device,
) -> Tensor:
    output = model(observation)
    kl, _, _ = forward_kl_loss(
        batch["teacher_logits"].to(device),
        output.logits,
        observation["action_mask"],
        temperature=config.temperature,
    )
    objective = kl
    if config.teacher_action_coefficient > 0:
        objective = objective + config.teacher_action_coefficient * (
            torch.nn.functional.cross_entropy(
                output.masked_logits,
                batch["selected_actions"].to(device),
            )
        )
    if config.oracle_coefficient > 0:
        if "oracle_actions" not in batch:
            raise ValueError(
                "oracle_coefficient requires oracle_actions in the dataset"
            )
        objective = objective + config.oracle_coefficient * (
            torch.nn.functional.cross_entropy(
                output.masked_logits,
                batch["oracle_actions"].to(device),
            )
        )
    if config.risk_oracle_coefficient > 0:
        if "risk_oracle_actions" not in batch:
            raise ValueError(
                "risk_oracle_coefficient requires risk_oracle_actions"
            )
        objective = (
            objective
            + config.risk_oracle_coefficient
            * torch.nn.functional.cross_entropy(
                output.masked_logits,
                batch["risk_oracle_actions"].to(device),
            )
        )
    return objective


@torch.inference_mode()
def _mean_objective(
    model: LocalStudentPolicy,
    dataset: DistillationDataset,
    *,
    config: DistillationConfig,
    device: torch.device,
) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    weighted = 0.0
    samples = 0
    for batch in loader:
        observation = _batch_observation(batch, device)
        objective = _distillation_objective(
            model,
            batch,
            observation,
            config=config,
            device=device,
        )
        count = batch["teacher_logits"].shape[0]
        weighted += float(objective.item()) * count
        samples += count
    return weighted / samples


def train_student_distillation(
    model: LocalStudentPolicy,
    train_dataset: DistillationDataset,
    validation_dataset: DistillationDataset,
    *,
    config: DistillationConfig,
    seed: int,
    device: torch.device | str = "cpu",
) -> DistillationResult:
    """Optimize the Student on offline Teacher logits."""

    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if (
        config.oracle_coefficient < 0
        or config.risk_oracle_coefficient < 0
        or config.teacher_action_coefficient < 0
    ):
        raise ValueError("auxiliary loss coefficients must be non-negative")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_state: dict[str, Tensor] | None = None
    best_objective = float("inf")
    stale_epochs = 0
    completed_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        for batch in loader:
            observation = _batch_observation(batch, device)
            loss = _distillation_objective(
                model,
                batch,
                observation,
                config=config,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            optimizer.step()
        completed_epochs = epoch + 1
        if config.early_stopping_patience > 0:
            validation_objective = _mean_objective(
                model,
                validation_dataset,
                config=config,
                device=device,
            )
            if (
                validation_objective
                < best_objective - config.minimum_improvement
            ):
                best_objective = validation_objective
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.early_stopping_patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return DistillationResult(
        epochs=completed_epochs,
        train=evaluate_distillation(
            model, train_dataset, config=config, device=device
        ),
        validation=evaluate_distillation(
            model, validation_dataset, config=config, device=device
        ),
    )
