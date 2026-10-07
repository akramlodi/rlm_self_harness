"""Constructed controls prove execution plumbing, not autonomous discovery."""

import copy

import pytest

from examples.meta_harness_probe import ProbeBudget, build_variants, cases_for, run_case


@pytest.mark.parametrize("family", ["counts", "units"])
def test_authored_pair_executes_and_preserves_protected_pattern(tmp_path, family):
    variants = build_variants(tmp_path / "proposal", family)
    cases = cases_for(family)
    for variant, harness in variants.items():
        for name, case in cases.items():
            result = run_case(harness, family, case, tmp_path / variant / name)
            if variant == "paired":
                assert result["passed"]
                assert result["activated"]
            elif variant in {"baseline", "capability"}:
                assert result["passed"] == (name == "protected")
                assert not result["activated"]
            else:
                assert not result["passed"]

    renamed = copy.deepcopy(cases["target"])
    renamed["context"]["records"].reverse()
    for record in renamed["context"]["records"]:
        record["key"] = {"alpha": "omega", "beta": "delta"}[record["key"]]
    if family == "counts":
        renamed["expected"] = ["omega"]
    result = run_case(variants["paired"], family, renamed, tmp_path / "renamed")
    assert result["passed"] and result["activated"]


def test_probe_reserves_retries_and_refuses_excess_spend():
    budget = ProbeBudget(input_price=0.19, output_price=0.51, max_requests=2)
    request = {"messages": [{"role": "user", "content": "short"}], "max_completion_tokens": 4096}
    budget.reserve(request)
    budget.reserve(request)  # A retry reserves again before dispatch.
    assert budget.requests == 2 and budget.reserved_usd > 0
    with pytest.raises(RuntimeError, match="request limit"):
        budget.reserve(request)
    expensive = ProbeBudget(input_price=1000, output_price=1000)
    with pytest.raises(RuntimeError, match="spend limit"):
        expensive.reserve(request)
    assert expensive.requests == 0
