"""Shared work/finalization state for one RLM completion tree."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class RunPhase(StrEnum):
    """Whether the RLM may expand work or must finish from existing state."""

    WORK = "work"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class RunDeadlines:
    """Absolute monotonic deadlines shared by a root and its descendants."""

    work: float | None
    hard: float | None


@dataclass
class RunControl:
    """Mutable per-RLM phase with completion-tree-wide absolute deadlines.

    There is deliberately no sub-call counter here. Calls are admitted without
    regard to their ordinal during ``WORK`` and refused only after the local
    iteration reserve or the shared wall-clock reserve enters ``FINALIZE``.
    """

    FINALIZATION_REFUSAL: ClassVar[str] = (
        "SUBCALL_REFUSED: finalization phase is active. "
        "Use existing REPL variables and submit the best available answer."
    )

    max_iterations: int
    finalization_iterations: int
    deadlines: RunDeadlines
    current_iteration: int = 0
    phase: RunPhase = RunPhase.WORK

    @classmethod
    def create(
        cls,
        *,
        max_iterations: int,
        finalization_iterations: int,
        max_timeout: float | None,
        finalization_timeout: float,
        inherited_deadlines: RunDeadlines | None = None,
    ) -> RunControl:
        """Create root deadlines or a child view of inherited deadlines."""
        if inherited_deadlines is not None:
            deadlines = inherited_deadlines
        elif max_timeout is None:
            deadlines = RunDeadlines(work=None, hard=None)
        else:
            now = time.monotonic()
            hard = now + max_timeout
            deadlines = RunDeadlines(work=hard - finalization_timeout, hard=hard)
        control = cls(
            max_iterations=max_iterations,
            finalization_iterations=finalization_iterations,
            deadlines=deadlines,
        )
        control.refresh()
        return control

    def update_iteration(self, iteration: int) -> None:
        """Advance the local RLM's zero-based iteration and refresh its phase."""
        self.current_iteration = iteration
        self.refresh()

    def refresh(self) -> RunPhase:
        """Enter finalization when either the turn or work deadline is reached."""
        turn_boundary = self.max_iterations - self.finalization_iterations
        turns_exhausted = (
            self.finalization_iterations > 0 and self.current_iteration >= turn_boundary
        )
        time_exhausted = self.deadlines.work is not None and time.monotonic() >= self.deadlines.work
        if turns_exhausted or time_exhausted:
            self.phase = RunPhase.FINALIZE
        return self.phase

    def subcall_refusal(self) -> str | None:
        """Return an in-band refusal only after finalization has begun."""
        if self.refresh() is RunPhase.WORK:
            return None
        return self.FINALIZATION_REFUSAL

    def remaining_work_time(self) -> float | None:
        """Seconds before finalization time begins, clamped at zero."""
        if self.deadlines.work is None:
            return None
        return max(0.0, self.deadlines.work - time.monotonic())
