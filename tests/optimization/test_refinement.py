from typing import Any

import pytest

from shrlm.experiment.config import RefinementSettings
from shrlm.optimization.refinement import assess_refinement


def summary(conditional: float | None, wrong_format: int, n_runs: int = 10) -> dict[str, Any]:
    return {
        "splits": {
            "heldout": {
                "n_runs": n_runs,
                "cause_counts": {"wrong_format": wrong_format},
                "primary_quality": {
                    "conditional_on_measurement": {
                        "mean": conditional,
                        "macro_by_context_length": None,
                    }
                },
            }
        }
    }


def settings() -> RefinementSettings:
    return RefinementSettings(
        enabled=True,
        promotion_size=5,
        min_conditional_quality_delta=0.1,
        min_failure_rate_delta=0.1,
        failure_causes=("wrong_format",),
    )


def test_promising_but_brittle_candidate_is_admitted() -> None:
    diagnostic = assess_refinement(summary(0.5, 1), summary(0.7, 3), settings())

    assert diagnostic.admitted
    assert diagnostic.conditional_quality_delta == pytest.approx(0.2)
    assert diagnostic.failure_rate_delta == pytest.approx(0.2)


def test_quality_gain_without_new_failure_is_not_admitted() -> None:
    diagnostic = assess_refinement(summary(0.5, 1), summary(0.7, 1), settings())

    assert not diagnostic.admitted
    assert "candidate did not increase" in " ".join(diagnostic.reasons)


def test_failure_without_conditional_gain_is_not_admitted() -> None:
    diagnostic = assess_refinement(summary(0.5, 1), summary(0.55, 3), settings())

    assert not diagnostic.admitted
    assert "conditional-quality delta" in " ".join(diagnostic.reasons)


def test_refinement_is_bounded_to_one_generation() -> None:
    try:
        RefinementSettings(enabled=True, promotion_size=5, max_generations=2)
    except ValueError as exc:
        assert "exactly 1" in str(exc)
    else:
        raise AssertionError("multi-generation refinement must be rejected")
