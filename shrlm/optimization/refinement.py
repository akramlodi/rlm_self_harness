"""Admission diagnostics for one bounded candidate-repair generation.

Refinement is deliberately not a second promotion metric.  A candidate can
enter the repair branch only after the ordinary, unconditional promotion gate
rejects it.  Conditional quality supplies evidence that the proposed mechanism
works when an answer is measurable; an increased configured failure rate
supplies evidence that the candidate introduced a separate reliability defect.
The repaired descendant must still pass the normal promotion rule on fresh
instances before it can become the incumbent.
"""

from dataclasses import dataclass
from typing import Any

from shrlm.experiment.config import RefinementSettings
from shrlm.optimization.validation import SPLIT_HELDOUT


@dataclass(frozen=True)
class RefinementDiagnostic:
    admitted: bool
    reasons: tuple[str, ...]
    baseline_conditional_quality: float | None
    candidate_conditional_quality: float | None
    conditional_quality_delta: float | None
    baseline_failure_rate: float
    candidate_failure_rate: float
    failure_rate_delta: float
    failure_causes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reasons": list(self.reasons),
            "baseline_conditional_quality": self.baseline_conditional_quality,
            "candidate_conditional_quality": self.candidate_conditional_quality,
            "conditional_quality_delta": self.conditional_quality_delta,
            "baseline_failure_rate": self.baseline_failure_rate,
            "candidate_failure_rate": self.candidate_failure_rate,
            "failure_rate_delta": self.failure_rate_delta,
            "failure_causes": list(self.failure_causes),
        }


def conditional_quality(summary: dict[str, Any]) -> float | None:
    quality = summary["splits"][SPLIT_HELDOUT].get("primary_quality")
    if quality is None:
        return None
    conditional = quality.get("conditional_on_measurement")
    if conditional is None:
        return None
    macro = conditional.get("macro_by_context_length")
    return float(macro) if macro is not None else _optional_float(conditional.get("mean"))


def configured_failure_rate(summary: dict[str, Any], causes: tuple[str, ...]) -> float:
    split = summary["splits"][SPLIT_HELDOUT]
    n_runs = int(split["n_runs"])
    if n_runs == 0:
        raise ValueError("refinement diagnostics require a non-empty held-out evaluation")
    counts = split.get("cause_counts")
    if counts is None:
        raise ValueError(
            "refinement diagnostics require cause_counts from the current validation protocol"
        )
    return sum(int(counts.get(cause, 0)) for cause in causes) / n_runs


def assess_refinement(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    settings: RefinementSettings,
) -> RefinementDiagnostic:
    """Classify a rejected candidate as promising-but-brittle or ineligible."""
    baseline_quality = conditional_quality(baseline_summary)
    candidate_quality = conditional_quality(candidate_summary)
    baseline_strata = _conditional_strata(baseline_summary)
    candidate_strata = _conditional_strata(candidate_summary)
    quality_delta = (
        candidate_quality - baseline_quality
        if baseline_quality is not None and candidate_quality is not None
        else None
    )
    baseline_failure = configured_failure_rate(baseline_summary, settings.failure_causes)
    candidate_failure = configured_failure_rate(candidate_summary, settings.failure_causes)
    failure_delta = candidate_failure - baseline_failure

    reasons: list[str] = []
    if baseline_strata != candidate_strata:
        reasons.append(
            "conditional quality is not supported on the same strata for baseline and candidate"
        )
    if quality_delta is None:
        reasons.append("conditional primary quality is unavailable for one or both subjects")
    elif quality_delta < settings.min_conditional_quality_delta:
        reasons.append(
            f"conditional-quality delta {quality_delta:+.6f} is below "
            f"{settings.min_conditional_quality_delta}"
        )
    if failure_delta < settings.min_failure_rate_delta:
        reasons.append(
            f"configured failure-rate delta {failure_delta:+.6f} is below "
            f"{settings.min_failure_rate_delta}"
        )
    if candidate_failure <= baseline_failure:
        reasons.append("candidate did not increase the configured failure rate")

    return RefinementDiagnostic(
        admitted=not reasons,
        reasons=tuple(reasons),
        baseline_conditional_quality=baseline_quality,
        candidate_conditional_quality=candidate_quality,
        conditional_quality_delta=quality_delta,
        baseline_failure_rate=baseline_failure,
        candidate_failure_rate=candidate_failure,
        failure_rate_delta=failure_delta,
        failure_causes=settings.failure_causes,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _conditional_strata(summary: dict[str, Any]) -> frozenset[str]:
    quality = summary["splits"][SPLIT_HELDOUT].get("primary_quality") or {}
    conditional = quality.get("conditional_on_measurement") or {}
    return frozenset((conditional.get("by_context_length") or {}).keys())


__all__ = [
    "RefinementDiagnostic",
    "assess_refinement",
    "conditional_quality",
    "configured_failure_rate",
]
