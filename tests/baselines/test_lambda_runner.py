import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from rlm.core.types import RLMChatCompletion
from rlm.utils.exceptions import BudgetExceededError
from shrlm.baselines.lambda_rlm import LambdaBaselineConfig, lambda_method_envelope
from shrlm.baselines.lambda_runner import (
    BudgetGuardClient,
    LambdaRoundConfig,
    run_lambda_round,
    validate_lambda_round,
)
from shrlm.environments.oolong_pairs import TASK_TEXTS, OolongEntry, build_prompt
from shrlm.optimization.bundle import round_dir
from shrlm.optimization.driver import RoundPersistenceError, load_manifest
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from tests.mock_lm import MockLM
from tests.optimization.test_driver import ClientFactory


@dataclass
class GoldVerifier:
    calls: int = 0

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        self.calls += 1
        gold = str(instance["gold"])
        return Verdict(
            passed=produced == gold,
            cause=None if produced == gold else VerifierCause.WRONG_VALUE,
            gold=gold,
            produced=produced,
            detail="" if produced == gold else "string mismatch",
        )


def make_config(tmp_path: Path, **overrides: Any) -> LambdaRoundConfig:
    values: dict[str, Any] = {
        "round_index": 1,
        "instances": [
            {
                "id": "lambda-pass",
                "prompt": "short context one",
                "question": "Which answer is correct?",
                "gold": "RIGHT",
            },
            {
                "id": "lambda-fail",
                "prompt": "short context two",
                "question": "Which answer is correct?",
                "gold": "RIGHT",
            },
        ],
        "verifier": GoldVerifier(),
        "out_dir": tmp_path,
        "backend": "openai",
        "backend_kwargs": {
            "model_name": "lambda-test",
            "sampling_args": {"temperature": 0.0},
        },
    }
    values.update(overrides)
    return LambdaRoundConfig(**values)


def test_persists_and_resumes_without_repeating_paid_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    first_factory = ClientFactory(["2", "RIGHT"])
    monkeypatch.setattr(upstream_lambda, "get_client", first_factory)

    first_entries = run_lambda_round(config, stop_after=1)

    path = round_dir(tmp_path, 1)
    assert len(first_entries) == 1
    assert first_factory.total_calls == 2
    assert json.loads((path / "method.json").read_text()) == lambda_method_envelope(config.method)
    assert (path / "instances.jsonl").exists()
    assert (path / first_entries[0]["trace_path"]).exists()
    assert first_entries[0]["passed"] is True

    second_factory = ClientFactory(["2", "WRONG"])
    monkeypatch.setattr(upstream_lambda, "get_client", second_factory)
    resumed_entries = run_lambda_round(config)

    assert len(resumed_entries) == 2
    assert second_factory.total_calls == 2
    assert resumed_entries[1]["passed"] is False
    assert resumed_entries[1]["cause"] == VerifierCause.WRONG_VALUE.value
    completion = RLMChatCompletion.from_dict(
        json.loads((path / resumed_entries[1]["trace_path"]).read_text())
    )
    assert completion.response == "WRONG"

    no_op_factory = ClientFactory([])
    monkeypatch.setattr(upstream_lambda, "get_client", no_op_factory)
    assert run_lambda_round(config) == resumed_entries
    assert no_op_factory.total_calls == 0
    assert load_manifest(tmp_path, 1) == resumed_entries


def test_rejects_method_or_instance_drift_on_resume(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert run_lambda_round(config, stop_after=0) == []

    changed_method = replace(
        config,
        method=LambdaBaselineConfig(context_window_chars=200_000),
    )
    with pytest.raises(RoundPersistenceError, match="method configurations"):
        run_lambda_round(changed_method, stop_after=0)

    changed_instances = replace(
        config,
        instances=[dict(config.instances[0], prompt="different bytes")],
    )
    with pytest.raises(RoundPersistenceError, match="identical instance list"):
        run_lambda_round(changed_instances, stop_after=0)


def test_refuses_a_modified_recorded_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, instances=[make_config(tmp_path).instances[0]])
    factory = ClientFactory(["2", "RIGHT"])
    monkeypatch.setattr(upstream_lambda, "get_client", factory)
    entry = run_lambda_round(config)[0]
    trace_path = round_dir(tmp_path, 1) / entry["trace_path"]
    trace_path.write_text(trace_path.read_text() + "tampered")

    with pytest.raises(RoundPersistenceError, match="was modified"):
        run_lambda_round(config)


def test_persists_pairwise_batch_audit_in_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = build_prompt(
        [
            OolongEntry(
                user_id=7,
                date=date(2023, 2, 2),
                instance="How many moons does Mars have ?",
                label=None,
            )
        ],
        TASK_TEXTS[1],
    )
    config = make_config(
        tmp_path,
        instances=[
            {
                "id": "oolong-audit",
                "prompt": prompt,
                "question": TASK_TEXTS[1],
                "task_id": 1,
                "gold": "No valid pairs found.",
            }
        ],
    )
    factory = ClientFactory(["0|N"])
    monkeypatch.setattr(upstream_lambda, "get_client", factory)

    entry = run_lambda_round(config)[0]
    trace = json.loads((round_dir(tmp_path, 1) / entry["trace_path"]).read_text())

    audit = trace["metadata"]["pairwise_audit"]
    assert audit["actual_model_calls"] == 1
    assert audit["batches"] == [
        {
            "attempts": [{"attempt": 1, "rejection": None, "response": "0|N"}],
            "batch_index": 0,
            "global_indices": [0],
            "predictions": {"0": "numeric value"},
        }
    ]


def test_validates_credentials_before_a_pending_paid_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = make_config(tmp_path, backend="openrouter")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_lambda_round(config)


def test_rejects_secret_backend_kwargs_before_writing(tmp_path: Path) -> None:
    config = make_config(tmp_path, backend_kwargs={"api_key": "do-not-persist"})

    with pytest.raises(ValueError, match="credential material"):
        run_lambda_round(config)

    assert not round_dir(tmp_path, 1).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_budget", 0),
        ("max_budget", -0.1),
        ("max_budget", True),
        ("max_timeout", 0),
        ("max_timeout", -1.0),
        ("max_timeout", False),
    ],
)
def test_rejects_invalid_lambda_limits(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = make_config(tmp_path, **{field: value})

    with pytest.raises(ValueError, match=field):
        validate_lambda_round(config)


def test_accepts_positive_or_unset_lambda_limits(tmp_path: Path) -> None:
    validate_lambda_round(make_config(tmp_path))
    validate_lambda_round(make_config(tmp_path, max_budget=0.1, max_timeout=30.0))


def test_budget_guard_allows_exact_budget_then_blocks_after_crossing() -> None:
    factory = ClientFactory(["one", "two", "three", "must-not-run"], cost_per_call=0.001)
    delegate = factory("openai", {"model_name": "guard-test"})
    client = BudgetGuardClient(delegate, max_budget=0.002)

    assert client.completion("first") == "one"
    assert client.completion("second") == "two"
    assert client.spent == pytest.approx(0.002)

    with pytest.raises(BudgetExceededError) as crossing:
        client.completion("third")

    assert crossing.value.spent == pytest.approx(0.003)
    assert crossing.value.budget == pytest.approx(0.002)
    assert client.budget_error is crossing.value
    assert factory.total_calls == 3

    with pytest.raises(BudgetExceededError) as blocked:
        client.completion("fourth")

    assert blocked.value is crossing.value
    assert factory.total_calls == 3
    assert factory.script == ["must-not-run"]


@pytest.mark.asyncio
async def test_budget_guard_enforces_async_calls() -> None:
    factory = ClientFactory(["over"], cost_per_call=0.002)
    delegate = factory("openai", {"model_name": "guard-test"})
    client = BudgetGuardClient(delegate, max_budget=0.001)

    with pytest.raises(BudgetExceededError, match="Budget exceeded"):
        await client.acompletion("prompt")

    assert factory.total_calls == 1


def test_budget_guard_delegates_usage_and_allows_unpriced_clients() -> None:
    delegate = MockLM(responses=["answer"])
    client = BudgetGuardClient(delegate, max_budget=0.1)

    assert client.completion("prompt") == "answer"
    assert client.spent is None
    assert client.get_usage_summary() == delegate.get_usage_summary()
    assert client.get_last_usage() == delegate.get_last_usage()


@pytest.mark.parametrize("value", [0, -0.1, True])
def test_budget_guard_rejects_invalid_caps(value: object) -> None:
    with pytest.raises(ValueError, match="max_budget"):
        BudgetGuardClient(MockLM(), value)
