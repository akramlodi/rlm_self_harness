from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import shrlm.baselines.lambda_runner as lambda_runner_module
import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from rlm.utils.exceptions import TimeoutExceededError
from shrlm.baselines.lambda_runner import (
    LambdaRoundConfig,
    guarded_lambda_client,
    persist_interrupted_lambda_run,
    run_governed_lambda_round,
    run_lambda_round,
)
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    HardDeadlineExceeded,
    ValidationCaps,
)
from shrlm.optimization.driver import load_manifest
from shrlm.optimization.taxonomy import VerifierCause
from tests.mock_lm import MockLM
from tests.optimization.test_driver import ClientFactory, GoldVerifier

CAPS = ValidationCaps(
    max_depth=3,
    max_iterations=30,
    max_budget=0.1,
    max_timeout=10.0,
    candidate_budget=1.0,
)


def instances(count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "id": f"lambda-{index}",
            "prompt": f"short context {index}",
            "question": "Which answer is correct?",
            "gold": "RIGHT",
        }
        for index in range(1, count + 1)
    ]


def config_for(
    tmp_path: Path,
    *,
    caps: ValidationCaps = CAPS,
    verifier: GoldVerifier | None = None,
    **overrides: Any,
) -> LambdaRoundConfig:
    values: dict[str, Any] = {
        "round_index": 1,
        "instances": instances(),
        "verifier": verifier or GoldVerifier(),
        "out_dir": tmp_path,
        "backend": "openai",
        "backend_kwargs": {"model_name": "lambda-governance-test"},
        "max_budget": caps.max_budget,
        "max_timeout": caps.max_timeout,
    }
    values.update(overrides)
    return LambdaRoundConfig(**values)


def test_guarded_factory_wraps_and_always_restores_upstream_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = ClientFactory(["unused"])
    monkeypatch.setattr(upstream_lambda, "get_client", factory)

    with pytest.raises(RuntimeError, match="test failure"):
        with guarded_lambda_client(0.1) as guard:
            assert upstream_lambda.get_client is guard
            client = upstream_lambda.get_client("openai", {"model_name": "test"})
            assert client is guard.client
            raise RuntimeError("test failure")

    assert upstream_lambda.get_client is factory


def test_per_run_budget_termination_is_persisted_without_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = replace(CAPS, max_budget=0.0015)
    verifier = GoldVerifier()
    config = config_for(tmp_path, caps=caps, verifier=verifier, instances=instances(1))
    factory = ClientFactory(["2", "RIGHT"], cost_per_call=0.001)
    monkeypatch.setattr(upstream_lambda, "get_client", factory)

    entry = run_lambda_round(config)[0]

    assert factory.total_calls == 2
    assert verifier.calls == 0
    assert entry["passed"] is False
    assert entry["cause"] == VerifierCause.RESOURCE_TERMINATED.value
    assert entry["cost"] == pytest.approx(0.002)
    assert entry["usage_lower_bound"] is True
    assert "BudgetExceededError" in entry["verdict"]["detail"]

    no_op = ClientFactory([])
    monkeypatch.setattr(upstream_lambda, "get_client", no_op)
    assert run_lambda_round(config) == [entry]
    assert no_op.total_calls == 0


def test_in_run_timeout_is_persisted_without_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = GoldVerifier()
    config = config_for(tmp_path, verifier=verifier, instances=instances(1))

    def time_out(method: Any, prompt: str) -> Any:
        raise TimeoutExceededError(elapsed=11.0, timeout=10.0)

    monkeypatch.setattr(upstream_lambda.LambdaRLM, "completion", time_out)

    entry = run_lambda_round(config)[0]

    assert verifier.calls == 0
    assert entry["cause"] == VerifierCause.RESOURCE_TERMINATED.value
    assert entry["cost"] is None
    assert entry["usage_lower_bound"] is True
    assert "TimeoutExceededError" in entry["verdict"]["detail"]


def test_unrelated_runtime_error_remains_a_resumable_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_for(tmp_path, instances=instances(1))

    def crash(method: Any, prompt: str) -> Any:
        raise ValueError("unexpected implementation failure")

    monkeypatch.setattr(upstream_lambda.LambdaRLM, "completion", crash)

    with pytest.raises(ValueError, match="unexpected implementation failure"):
        run_lambda_round(config)

    assert load_manifest(tmp_path, 1) == []


def test_governed_round_trips_breaker_and_skips_remaining_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = replace(CAPS, candidate_budget=0.0015)
    config = config_for(tmp_path, caps=caps)
    factory = ClientFactory(["2", "RIGHT"], cost_per_call=0.001)
    monkeypatch.setattr(upstream_lambda, "get_client", factory)

    result = run_governed_lambda_round(config, CandidateSpendBreaker(caps))

    assert result.outcome == OUTCOME_OVER_BUDGET
    assert result.spent == pytest.approx(0.002)
    assert [entry["run_id"] for entry in result.entries] == ["lambda-1__a01"]
    assert result.skipped_run_ids == ["lambda-2__a01"]

    no_op = ClientFactory([])
    monkeypatch.setattr(upstream_lambda, "get_client", no_op)
    resumed = run_governed_lambda_round(config, CandidateSpendBreaker(caps))
    assert resumed.entries == result.entries
    assert resumed.outcome == OUTCOME_OVER_BUDGET
    assert resumed.skipped_run_ids == ["lambda-2__a01"]
    assert no_op.total_calls == 0


def test_escaped_deadline_persists_the_pending_run_and_charges_the_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = replace(CAPS, candidate_budget=0.5)
    config = config_for(tmp_path, caps=caps, instances=instances(1))
    calls = 0

    def deadline_once(fn: Any, deadline: float | None) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return fn()
        assert deadline is not None
        raise HardDeadlineExceeded(deadline)

    monkeypatch.setattr(lambda_runner_module, "call_with_hard_deadline", deadline_once)

    result = run_governed_lambda_round(config, CandidateSpendBreaker(caps))

    assert result.outcome == OUTCOME_COMPLETED
    assert result.spent == pytest.approx(caps.max_budget)
    assert result.skipped_run_ids == []
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry["cause"] == VerifierCause.RESOURCE_TERMINATED.value
    assert entry["cost"] is None
    assert entry["usage_lower_bound"] is True
    assert "HardDeadlineExceeded" in entry["verdict"]["detail"]


def test_interrupted_recovery_selects_first_pending_attempt(tmp_path: Path) -> None:
    config = config_for(tmp_path, attempts=2)
    first = persist_interrupted_lambda_run(config, HardDeadlineExceeded(1.0))
    second = persist_interrupted_lambda_run(config, HardDeadlineExceeded(1.0))

    assert first is not None and first["run_id"] == "lambda-1__a01"
    assert second is not None and second["run_id"] == "lambda-1__a02"
    assert [entry["run_id"] for entry in load_manifest(tmp_path, 1)] == [
        "lambda-1__a01",
        "lambda-1__a02",
    ]


def test_governed_round_requires_experiment_owned_limits(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    breaker = CandidateSpendBreaker(CAPS)

    with pytest.raises(ValueError, match="max_budget"):
        run_governed_lambda_round(replace(config, max_budget=None), breaker)
    with pytest.raises(ValueError, match="max_timeout"):
        run_governed_lambda_round(replace(config, max_timeout=5.0), breaker)


def test_costless_success_is_persisted_but_refused_by_breaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_for(tmp_path, instances=instances(1))
    client = MockLM(responses=["2", "RIGHT"])
    monkeypatch.setattr(upstream_lambda, "get_client", lambda backend, kwargs: client)

    with pytest.raises(ValueError, match="persisted no cost"):
        run_governed_lambda_round(config, CandidateSpendBreaker(CAPS))

    entries = load_manifest(tmp_path, 1)
    assert len(entries) == 1
    assert entries[0]["cost"] is None
    assert entries[0]["passed"] is True
