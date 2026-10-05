"""Verifier metric definitions and strict, read-only adapters for saved verdicts."""

import re
from typing import Any

from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import QualityDefinition, QualityMeasurement, Verdict

TERMINAL_ZERO = {
    cause.value: 0.0
    for cause in (
        VerifierCause.RUNTIME_ERROR,
        VerifierCause.RESOURCE_TERMINATED,
        VerifierCause.WRONG_FORMAT,
        VerifierCause.CONTENT_FILTERED,
    )
}
SET_QUALITY = QualityDefinition("f1", "v1", "higher", terminal_values=TERMINAL_ZERO)
OOLONG_QUALITY = QualityDefinition("score", "v1", "higher", terminal_values=TERMINAL_ZERO)
OOLONG_SCORE_RE = re.compile(
    r"score=(0\.\d+|1\.0+) exact=(True|False) "
    r"kind=(numeric|comparison|date|month_year|list|label|user|string)"
)


def legacy_quality(
    config: dict[str, Any], verdict: Verdict | None = None
) -> tuple[QualityDefinition | None, QualityMeasurement | None]:
    """Only known verifier contracts and exact historical detail grammars."""
    environment = config.get("environment")
    if environment in {"oolong_pairs", "graphwalks"}:
        from shrlm.environments.oolong_pairs import recorded_pair_metrics

        metrics = recorded_pair_metrics(verdict) if verdict is not None else None
        return SET_QUALITY, QualityMeasurement(
            SET_QUALITY.identifier, float(metrics["f1"])
        ) if metrics else None
    if environment in {"oolong", "oolong_synth", "oolong_real"}:
        match = OOLONG_SCORE_RE.fullmatch(verdict.detail) if verdict is not None else None
        value = float(match[1]) if match else None
        if (
            verdict is not None
            and verdict.cause is VerifierCause.NO_ANSWER
            and verdict.detail == "final line carried an explicit empty marker"
        ):
            value = 0.0
        return OOLONG_QUALITY, QualityMeasurement(
            OOLONG_QUALITY.identifier, value
        ) if value is not None else None
    return None, None
