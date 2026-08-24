import pytest

from shrlm.baselines.lambda_rlm import (
    LAMBDA_RLM_UPSTREAM_REVISION,
    LambdaBaselineConfig,
    lambda_input,
)
from shrlm.baselines.upstream.lambda_rlm import LambdaRLM


def test_pins_paper_release_revision() -> None:
    assert (
        LAMBDA_RLM_UPSTREAM_REVISION
        == "c70c6db2cf8498eaf05fa6999dc41088827158f4"
    )


def test_uses_upstream_defaults() -> None:
    config = LambdaBaselineConfig()

    assert config.context_window_chars == 100_000
    assert config.accuracy_target == 0.80
    assert config.a_leaf == 0.95
    assert config.a_compose == 0.90


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
