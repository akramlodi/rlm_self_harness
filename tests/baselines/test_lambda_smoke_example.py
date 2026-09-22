from typing import Any

import pytest

from examples.lambda_rlm_oolong_pairs_long_smoke import (
    B1_CONDITION,
    H0_STAR_CONDITION,
    LAMBDA_CONDITION,
    comparison_payload,
    parse_args,
    resolve_conditions,
    selected_conditions,
    summarize_entries,
    worst_case_spend,
)


def entry(
    instance_id: str,
    *,
    f1: float,
    passed: bool = False,
    cost: float = 0.01,
) -> dict[str, Any]:
    cause = None if passed else "mixed_set_error"
    return {
        "instance_id": instance_id,
        "passed": passed,
        "cause": cause,
        "cost": cost,
        "input_tokens": 100,
        "output_tokens": 20,
        "execution_time": 2.0,
        "verdict": {
            "passed": passed,
            "cause": cause,
            "gold": "[(1, 2)]",
            "produced": "[(1, 3)]",
            "detail": f"precision={f1:.3f} recall={f1:.3f} f1={f1:.3f} missing=1 extra=1",
        },
    }


def test_compare_flag_preserves_lambda_only_default() -> None:
    assert parse_args([]).compare_b1 is False
    assert parse_args([]).compare_h0_star is False
    assert parse_args(["--compare-b1"]).compare_b1 is True
    assert parse_args(["--compare-h0-star"]).compare_h0_star is True
    assert selected_conditions(False, False) == (LAMBDA_CONDITION,)
    assert selected_conditions(False, True) == (H0_STAR_CONDITION, LAMBDA_CONDITION)
    all_conditions = selected_conditions(True, True)
    assert all_conditions == (B1_CONDITION, H0_STAR_CONDITION, LAMBDA_CONDITION)
    assert worst_case_spend(4, 0.5, (LAMBDA_CONDITION,)) == 2.0
    assert worst_case_spend(4, 0.5, all_conditions) == 6.0


def test_explicit_conditions_and_context_length_support_isolated_diagnostics() -> None:
    args = parse_args(["--conditions", "h0_star", "--context-length", "short"])

    assert args.conditions == "h0_star"
    assert args.context_length == "short"
    assert resolve_conditions(args.conditions, args.compare_b1, args.compare_h0_star) == (
        H0_STAR_CONDITION,
    )
    assert resolve_conditions("lambda_rlm,h0_star", False, False) == (
        H0_STAR_CONDITION,
        LAMBDA_CONDITION,
    )


def test_explicit_conditions_reject_ambiguous_or_invalid_selections() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_conditions("h0_star", False, True)
    with pytest.raises(ValueError, match="unknown conditions"):
        resolve_conditions("unknown", False, False)
    with pytest.raises(ValueError, match="at least one"):
        resolve_conditions(",", False, False)
    with pytest.raises(ValueError, match="duplicates"):
        resolve_conditions("h0_star,h0_star", False, False)


def test_comparison_reports_matched_aggregates_and_deltas() -> None:
    h0_star = [entry("one", f1=0.1), entry("two", f1=0.3, passed=True)]
    lambda_rlm = [entry("one", f1=0.8, cost=0.02), entry("two", f1=0.6, cost=0.02)]

    payload = comparison_payload({H0_STAR_CONDITION: h0_star, LAMBDA_CONDITION: lambda_rlm})

    assert payload["conditions"][H0_STAR_CONDITION] == summarize_entries(h0_star)
    assert payload["conditions"][H0_STAR_CONDITION]["mean_f1"] == pytest.approx(0.2)
    assert payload["conditions"][H0_STAR_CONDITION]["exact_passes"] == 1
    assert payload["conditions"][LAMBDA_CONDITION]["mean_f1"] == pytest.approx(0.7)
    assert payload["per_instance"][0][H0_STAR_CONDITION]["f1"] == 0.1
    assert payload["per_instance"][0][LAMBDA_CONDITION]["f1"] == 0.8


def test_comparison_refuses_unmatched_instances() -> None:
    with pytest.raises(ValueError, match="not aligned by instance"):
        comparison_payload(
            {
                B1_CONDITION: [entry("one", f1=0.1)],
                LAMBDA_CONDITION: [entry("different", f1=0.8)],
            }
        )
