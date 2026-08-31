"""Training algorithms for Lite-GLOBE policies."""

from .distillation import (
    DistillationConfig,
    DistillationResult,
    train_student_distillation,
)
from .student_finetune import (
    StudentFineTuneConfig,
    StudentFineTuneResult,
    fine_tune_student,
)
from .teacher_trainer import TeacherTrainingResult, train_teacher

__all__ = [
    "DistillationConfig",
    "DistillationResult",
    "StudentFineTuneConfig",
    "StudentFineTuneResult",
    "TeacherTrainingResult",
    "fine_tune_student",
    "train_student_distillation",
    "train_teacher",
]
