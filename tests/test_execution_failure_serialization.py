import pytest

from rlm.core.types import ExecutionFailure, RLMChatCompletion, UsageSummary


def test_execution_failure_round_trip_and_legacy_shape():
    completion = RLMChatCompletion("model", "prompt", "", UsageSummary({}), 1.0)
    legacy = completion.to_dict()
    assert "execution_failure" not in legacy
    assert RLMChatCompletion.from_dict(legacy).to_dict() == legacy
    completion.execution_failure = ExecutionFailure(
        "runtime_error", "TypeError", "bad tuple", "stack"
    )
    completion.error = "TypeError: bad tuple"
    restored = RLMChatCompletion.from_dict(completion.to_dict())
    assert restored.execution_failure == completion.execution_failure
    assert restored.error == completion.error


def test_invalid_failure_metadata_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Unknown execution failure"):
        ExecutionFailure("wrong_value", "TypeError", "bad", "stack")
