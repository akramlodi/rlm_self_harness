import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from shrlm.baselines.lambda_rlm import (
    LAMBDA_RLM_DISPLAY_NAME,
    LAMBDA_RLM_METHOD_FORMAT,
    LAMBDA_RLM_METHOD_KIND,
    LAMBDA_RLM_SOURCE_SHA256,
    LAMBDA_RLM_UPSTREAM_REVISION,
    PAPER_RECONSTRUCTION_VERSION,
    LambdaBaselineConfig,
    lambda_input,
    lambda_method_hash,
    serialize_lambda_method,
    write_lambda_method_json,
)
from shrlm.baselines.upstream.lambda_rlm import LambdaRLM
from tests.optimization.test_driver import ClientFactory


def test_pins_paper_release_revision() -> None:
    assert LAMBDA_RLM_UPSTREAM_REVISION == "c70c6db2cf8498eaf05fa6999dc41088827158f4"


def test_pinned_source_hash_matches_upstream_file() -> None:
    source_path = Path(upstream_lambda.__file__)

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == LAMBDA_RLM_SOURCE_SHA256


def test_uses_upstream_defaults() -> None:
    config = LambdaBaselineConfig()

    assert config.context_window_chars == 100_000
    assert config.accuracy_target == 0.80
    assert config.a_leaf == 0.95
    assert config.a_compose == 0.90
    assert config.pairwise_max_attempts == 3


def test_extracts_only_non_gold_input() -> None:
    instance = {
        "prompt": "exact persisted prompt\n",
        "question": "Which nodes are reachable?",
        "answer_nodes": ["secret-node"],
        "gold_pairs": [[1, 2]],
    }

    value = lambda_input(instance)

    assert value.prompt == "exact persisted prompt\n"
    assert value.query == "Which nodes are reachable?"
    assert not hasattr(value, "answer_nodes")
    assert not hasattr(value, "gold_pairs")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt", 123, TypeError),
        ("question", ["not", "a", "string"], TypeError),
        ("prompt", "   ", ValueError),
        ("question", "", ValueError),
    ],
)
def test_rejects_invalid_input(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    instance = {
        "prompt": "context",
        "question": "question",
    }
    instance[field] = value

    with pytest.raises(error):
        lambda_input(instance)


def test_builds_upstream_method_without_calling_model() -> None:
    backend_kwargs = {
        "model_name": "test-model",
        "sampling_args": {"temperature": 0.0},
    }

    method = LambdaBaselineConfig().build(
        backend="openrouter",
        backend_kwargs=backend_kwargs,
        query="Test question",
    )

    assert isinstance(method, LambdaRLM)
    assert method.backend == "openrouter"
    assert method.backend_kwargs == backend_kwargs
    assert method.backend_kwargs is not backend_kwargs
    assert method.query == "Test question"
    assert method.context_window_chars == 100_000


def test_serializes_complete_method_identity() -> None:
    serialization = serialize_lambda_method(LambdaBaselineConfig())

    assert serialization == {
        "kind": LAMBDA_RLM_METHOD_KIND,
        "display_name": LAMBDA_RLM_DISPLAY_NAME,
        "upstream": {
            "repository": "https://github.com/lambda-calculus-LLM/lambda-RLM",
            "revision": LAMBDA_RLM_UPSTREAM_REVISION,
            "source_sha256": LAMBDA_RLM_SOURCE_SHA256,
        },
        "reconstruction": {
            "version": PAPER_RECONSTRUCTION_VERSION,
            "scope": "OOLONG-Pairs Algorithm 5: SPLIT-MAP-PARSE-FILTER-CROSS",
        },
        "runtime": {"environment": "local"},
        "configuration": {
            "context_window_chars": 100_000,
            "accuracy_target": 0.80,
            "a_leaf": 0.95,
            "a_compose": 0.90,
            "pairwise_max_batch_records": 256,
            "pairwise_max_batch_chars": 80_000,
            "pairwise_max_concurrency": 8,
            "pairwise_max_attempts": 3,
        },
    }


def test_method_hash_is_stable_and_configuration_sensitive() -> None:
    config = LambdaBaselineConfig()

    assert lambda_method_hash(config) == lambda_method_hash(config)
    assert len(lambda_method_hash(config)) == 64
    assert lambda_method_hash(config) != lambda_method_hash(
        replace(config, context_window_chars=200_000)
    )
    assert lambda_method_hash(config) != lambda_method_hash(
        replace(config, pairwise_max_attempts=2)
    )


def test_writes_method_identity_envelope(tmp_path: Path) -> None:
    config = LambdaBaselineConfig()
    path = tmp_path / "method.json"

    envelope = write_lambda_method_json(config, path)

    assert json.loads(path.read_text()) == envelope
    assert path.read_text().endswith("\n")
    assert envelope["format"] == LAMBDA_RLM_METHOD_FORMAT
    assert envelope["kind"] == LAMBDA_RLM_METHOD_KIND
    assert envelope["display_name"] == LAMBDA_RLM_DISPLAY_NAME
    assert envelope["hash"] == lambda_method_hash(config)
    assert envelope["method"] == serialize_lambda_method(config)


def test_runs_short_qa_through_upstream_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Call 1 selects QA in λ-RLM's task-detection prompt.
    # Call 2 answers the single leaf because this short input needs no split.
    factory = ClientFactory(["2", "RIGHT"])
    monkeypatch.setattr(upstream_lambda, "get_client", factory)

    method = LambdaBaselineConfig().build(
        backend="openrouter",
        backend_kwargs={
            "model_name": "test-model",
            "sampling_args": {"temperature": 0.0},
        },
        query="Which answer is correct?",
    )

    completion = method.completion("A short context containing the answer.")

    assert completion.response == "RIGHT"
    assert completion.root_model == "test-model"
    assert completion.prompt == "A short context containing the answer."
    assert completion.usage_summary.total_input_tokens == 20
    assert completion.usage_summary.total_output_tokens == 20
    assert factory.total_calls == 2
    assert factory.script == []
