import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from rlm.core.types import RLMChatCompletion
from shrlm.baselines.lambda_rlm import LambdaBaselineConfig, lambda_method_envelope
from shrlm.baselines.lambda_runner import LambdaRoundConfig, run_lambda_round
from shrlm.optimization.bundle import round_dir
from shrlm.optimization.driver import RoundPersistenceError, load_manifest
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
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
