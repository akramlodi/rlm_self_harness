"""Transport-retry behavior for deficient HTTP-200 completion responses.

A provider (seen intermittently on OpenRouter) can return 200 with no usage
block, no choices, and sometimes an embedded error payload. The client must
retry a bounded number of times and then raise loudly -- never record an
unknown-cost call, never crash a multi-hour experiment on the first glitch.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import call, patch

import pytest

from rlm.clients.openai import (
    TRANSPORT_ATTEMPTS,
    OpenAIClient,
    _response_deficiency,
)


def good_response(prompt_tokens: int = 10, completion_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=None,
            model_extra=None,
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="ok", model_extra=None)
            )
        ],
        error=None,
        model_extra=None,
    )


def deficient_response(error: Any = None) -> SimpleNamespace:
    return SimpleNamespace(usage=None, choices=None, error=error, model_extra=None)


class ScriptedCreate:
    """chat.completions.create stand-in yielding scripted responses in order."""

    def __init__(self, responses: list[SimpleNamespace]):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return self.responses.pop(0)


def make_client(response: Any = None, sampling_args: dict[str, Any] | None = None) -> OpenAIClient:
    """An OpenRouter-shaped OpenAIClient with the SDK construction patched out.

    ``response``, when given, is what the mocked ``chat.completions.create``
    returns; ``sampling_args`` are forwarded to the constructor.
    """
    kwargs: dict[str, Any] = {}
    if sampling_args is not None:
        kwargs["sampling_args"] = sampling_args
    with patch("rlm.clients.openai.openai.OpenAI"), patch("rlm.clients.openai.openai.AsyncOpenAI"):
        client = OpenAIClient(
            api_key="test-key",
            model_name="test-model",
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )
    if response is not None:
        client.client.chat.completions.create.return_value = response
    return client


class TestResponseDeficiency:
    def test_good_response_is_not_deficient(self) -> None:
        assert _response_deficiency(good_response()) is None

    def test_missing_usage_and_choices_are_named(self) -> None:
        reason = _response_deficiency(deficient_response())
        assert reason is not None
        assert "no usage data received" in reason
        assert "no choices" in reason

    def test_embedded_provider_error_is_included_truncated(self) -> None:
        reason = _response_deficiency(deficient_response(error={"message": "upstream boom" * 100}))
        assert reason is not None
        assert "provider error payload" in reason
        assert len(reason) < 500


class TestTransportRetry:
    def test_recovers_after_one_deficient_response(self) -> None:
        client = make_client()
        scripted = ScriptedCreate([deficient_response(), good_response()])
        client.client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=scripted)),
        )
        with patch("rlm.clients.openai.time.sleep") as sleep:
            assert client.completion("hi") == "ok"
        assert scripted.calls == 2
        assert sleep.call_count == 1
        # The deficient attempt is never recorded: exactly one tracked call.
        assert client.model_call_counts["test-model"] == 1
        assert client.model_input_tokens["test-model"] == 10

    def test_raises_loudly_after_exhausting_attempts(self) -> None:
        client = make_client()
        scripted = ScriptedCreate(
            [deficient_response(error={"code": 502, "message": "provider down"})]
            * TRANSPORT_ATTEMPTS
        )
        client.client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=scripted)),
        )
        with patch("rlm.clients.openai.time.sleep"):
            with pytest.raises(ValueError) as excinfo:
                client.completion("hi")
        assert scripted.calls == TRANSPORT_ATTEMPTS
        message = str(excinfo.value)
        assert f"after {TRANSPORT_ATTEMPTS} attempts" in message
        assert "provider down" in message
        # Nothing was recorded for the unknowable-cost calls.
        assert client.model_call_counts["test-model"] == 0


class TestBackoffShape:
    def test_full_jitter_exponential_with_cap(self) -> None:
        from rlm.clients.openai import (
            _TRANSPORT_BACKOFF_CAP_SECONDS,
            _transport_backoff_seconds,
        )

        for attempt, ceiling in ((1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 16.0), (6, 30.0)):
            for _ in range(20):
                delay = _transport_backoff_seconds(attempt)
                assert 0 <= delay <= ceiling <= _TRANSPORT_BACKOFF_CAP_SECONDS

    def test_rate_limit_backoff_draws_fresh_full_jitter_at_capped_ceilings(self) -> None:
        from rlm.clients.openai import (
            _RATE_LIMIT_BACKOFF_CAP_SECONDS,
            _rate_limit_backoff_seconds,
        )

        attempts_and_ceilings = ((1, 2.0), (2, 4.0), (3, 8.0), (4, 16.0), (6, 60.0))
        draws = [0.25, 1.25, 2.25, 3.25, 4.25]
        with patch("rlm.clients.openai.random.uniform", side_effect=draws) as uniform:
            actual = [
                _rate_limit_backoff_seconds(attempt) for attempt, _ceiling in attempts_and_ceilings
            ]

        assert actual == draws
        assert uniform.call_args_list == [
            call(0, ceiling) for _attempt, ceiling in attempts_and_ceilings
        ]
        assert attempts_and_ceilings[-1][1] == _RATE_LIMIT_BACKOFF_CAP_SECONDS


class TestContextOverflowMapping:
    """A provider 400 for an over-window prompt is a run-level resource
    condition (TokenLimitExceededError -> RESOURCE_TERMINATED run), never an
    experiment-crashing transport error."""

    @staticmethod
    def bad_request(message: str) -> Exception:
        import httpx
        import openai as openai_sdk

        response = httpx.Response(
            400, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        )
        return openai_sdk.BadRequestError(message, response=response, body=None)

    def _client_raising(self, exc: Exception) -> OpenAIClient:
        client = make_client()

        def raise_exc(**kwargs: Any) -> None:
            raise exc

        client.client = SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=raise_exc)),
        )
        return client

    def test_context_overflow_400_maps_to_token_limit(self) -> None:
        from rlm.utils.exceptions import TokenLimitExceededError

        message = (
            "Error code: 400 - This model's maximum context length is 262144 tokens. "
            "However, you requested 4096 output tokens and your prompt contains at "
            "least 258049 input tokens."
        )
        client = self._client_raising(self.bad_request(message))
        with pytest.raises(TokenLimitExceededError) as excinfo:
            client.completion("hi")
        assert "context window" in str(excinfo.value)

    def test_max_seq_len_variant_maps_too(self) -> None:
        from rlm.utils.exceptions import TokenLimitExceededError

        client = self._client_raising(
            self.bad_request("number of input tokens (262841) has exceeded max_seq_len (262144)")
        )
        with pytest.raises(TokenLimitExceededError):
            client.completion("hi")

    def test_unrelated_400_still_raises_bad_request(self) -> None:
        import openai as openai_sdk

        client = self._client_raising(self.bad_request("invalid model slug"))
        with pytest.raises(openai_sdk.BadRequestError):
            client.completion("hi")
