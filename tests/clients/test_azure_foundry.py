"""Tests for the Azure AI Foundry client.

Two tiers live in this file:

* the mocked tier (everything up to ``TestAzureFoundryLive``): zero network,
  always runs;
* the live tier (``TestAzureFoundryLive``): a handful of tiny paid calls
  against the REAL configured deployment of the env-selected config
  (``SHRLM_EXPERIMENT_CONFIG``, else the shipped smoke profile), gated by
  KTD8 (see ``shrlm/experiment/live_gates.py``) -- it runs only when that
  config's runner backend is azure_foundry (this is an azure-specific tier;
  with another backend selected the gate would demand that backend's
  credentials and then run azure code paths) AND both Azure credentials are
  set AND ``SHRLM_RUN_LIVE=1``, and never in CI (live-marked items are
  deselected outright without the opt-in; see ``tests/conftest.py``).
"""

import asyncio
import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shrlm.experiment.live_gates import CONFIG_ENV_KEY, live_config_path, live_skip_reason

VALID_ENDPOINT = "https://my-resource.services.ai.azure.com"
VALID_PRICING = {"input_per_million": 0.60, "output_per_million": 3.00}


def _make_response(
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 500,
    cost: float | None = None,
    finish_reason: str = "stop",
    content: str | None = "hello from foundry",
    usage: Any = "default",
    reasoning_tokens: int | None = None,
) -> SimpleNamespace:
    """Build a fake chat.completions response using SimpleNamespace so that
    hasattr checks behave like real pydantic models (MagicMock would report
    every attribute as present). ``completion_tokens_details`` is emitted only
    when ``reasoning_tokens`` is set, mirroring a deployment that may or may
    not report the detail block."""
    if usage == "default":
        usage_kwargs: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        }
        if cost is not None:
            usage_kwargs["cost"] = cost
        if reasoning_tokens is not None:
            usage_kwargs["completion_tokens_details"] = SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            )
        usage = SimpleNamespace(**usage_kwargs)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_client(monkeypatch, response=None, pricing="default", **kwargs):
    """Construct an AzureFoundryClient with env vars monkeypatched and the
    underlying openai client construction mocked out."""
    from rlm.clients.azure_foundry import AzureFoundryClient

    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", VALID_ENDPOINT)
    if pricing == "default":
        pricing = dict(VALID_PRICING)

    with (
        patch("rlm.clients.openai.openai.OpenAI") as mock_openai,
        patch("rlm.clients.openai.openai.AsyncOpenAI"),
    ):
        mock_sync = MagicMock()
        if response is not None:
            mock_sync.chat.completions.create.return_value = response
        mock_openai.return_value = mock_sync
        if pricing is None:
            client = AzureFoundryClient(model_name="kimi-k2.5", **kwargs)
        else:
            client = AzureFoundryClient(model_name="kimi-k2.5", pricing=pricing, **kwargs)
    return client


def _make_async_client(monkeypatch, response=None, create=None, **kwargs):
    """An ``AzureFoundryClient`` whose async SDK returns ``response`` (or runs
    ``create``); the sync SDK is patched inert. Mirrors ``_make_client``."""
    from rlm.clients.azure_foundry import AzureFoundryClient

    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", VALID_ENDPOINT)
    if create is None:

        async def create(**_kwargs):
            return response

    with (
        patch("rlm.clients.openai.openai.OpenAI"),
        patch("rlm.clients.openai.openai.AsyncOpenAI") as mock_async_openai,
    ):
        mock_async = MagicMock()
        mock_async.chat.completions.create = create
        mock_async_openai.return_value = mock_async
        kwargs.setdefault("model_name", "kimi-k2.5")
        kwargs.setdefault("pricing", dict(VALID_PRICING))
        return AzureFoundryClient(**kwargs)


class TestConstructionValidation:
    def test_missing_api_key_names_variable(self, monkeypatch):
        import rlm.clients.azure_foundry as mod

        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", VALID_ENDPOINT)
        with pytest.raises(ValueError, match="AZURE_API_KEY"):
            mod.AzureFoundryClient(model_name="m", pricing=dict(VALID_PRICING))

    def test_missing_endpoint_names_variable(self, monkeypatch):
        import rlm.clients.azure_foundry as mod

        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="AZURE_FOUNDRY_ENDPOINT"):
            mod.AzureFoundryClient(model_name="m", pricing=dict(VALID_PRICING))

    def test_http_endpoint_rejected_before_client_construction(self, monkeypatch):
        from rlm.clients.azure_foundry import AzureFoundryClient

        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        endpoint = "http://my-resource.services.ai.azure.com"
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", endpoint)
        with patch("rlm.clients.openai.openai.OpenAI") as mock_openai:
            with pytest.raises(ValueError) as exc_info:
                AzureFoundryClient(model_name="m", pricing=dict(VALID_PRICING))
            mock_openai.assert_not_called()
        assert "my-resource.services.ai.azure.com" not in str(exc_info.value)
        assert "AZURE_FOUNDRY_ENDPOINT" in str(exc_info.value)

    def test_wrong_host_rejected_before_client_construction(self, monkeypatch):
        from rlm.clients.azure_foundry import AzureFoundryClient

        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://evil.example.com")
        with patch("rlm.clients.openai.openai.OpenAI") as mock_openai:
            with pytest.raises(ValueError) as exc_info:
                AzureFoundryClient(model_name="m", pricing=dict(VALID_PRICING))
            mock_openai.assert_not_called()
        assert "evil.example.com" not in str(exc_info.value)
        assert "AZURE_FOUNDRY_ENDPOINT" in str(exc_info.value)

    def test_openai_azure_host_accepted(self, monkeypatch):
        client = _make_client(monkeypatch, endpoint="https://my-resource.openai.azure.com/")
        assert client.base_url == "https://my-resource.openai.azure.com/openai/v1/"

    def test_base_url_composition_handles_trailing_slash(self, monkeypatch):
        client = _make_client(monkeypatch, endpoint=VALID_ENDPOINT + "/")
        assert client.base_url == VALID_ENDPOINT + "/openai/v1/"

    def test_missing_env_error_omits_endpoint_host(self, monkeypatch):
        import rlm.clients.azure_foundry as mod

        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", VALID_ENDPOINT)
        with pytest.raises(ValueError) as exc_info:
            mod.AzureFoundryClient(model_name="m", pricing=dict(VALID_PRICING))
        assert "my-resource.services.ai.azure.com" not in str(exc_info.value)


class TestPricingValidation:
    def test_missing_pricing_kwarg(self, monkeypatch):
        with pytest.raises(ValueError, match="pricing"):
            _make_client(monkeypatch, pricing=None)

    @pytest.mark.parametrize(
        "pricing",
        [
            {"input_per_million": 0.60},
            {"output_per_million": 3.00},
            {},
            {"input_per_million": 0.60, "output_per_million": 3.00, "extra": 1.0},
            {"input_per_million": 0.0, "output_per_million": 3.00},
            {"input_per_million": 0.60, "output_per_million": -1.0},
            {"input_per_million": -0.60, "output_per_million": 3.00},
            {"input_per_million": 0.60, "output_per_million": 0},
            {"input_per_million": "0.60", "output_per_million": 3.00},
        ],
    )
    def test_invalid_pricing_rejected(self, monkeypatch, pricing):
        with pytest.raises(ValueError, match="pricing"):
            _make_client(monkeypatch, pricing=pricing)


class TestCostSynthesis:
    def test_happy_path_synthesizes_cost(self, monkeypatch):
        response = _make_response(prompt_tokens=1000, completion_tokens=500)
        client = _make_client(monkeypatch, response=response)
        result = client.completion("hi")
        assert result == "hello from foundry"

        expected = 1000 * 0.60 / 1e6 + 500 * 3.00 / 1e6
        assert client.last_cost == pytest.approx(expected)
        assert client.last_cost_source == "synthesized"

        last = client.get_last_usage()
        assert last.total_cost == pytest.approx(expected)
        assert last.cost_source == "synthesized"

        summary = client.get_usage_summary()
        assert summary.total_cost == pytest.approx(expected)
        assert summary.model_usage_summaries["kimi-k2.5"].cost_source == "synthesized"

    def test_acompletion_synthesizes_cost(self, monkeypatch):
        response = _make_response(prompt_tokens=1000, completion_tokens=500)
        client = _make_async_client(monkeypatch, response=response)

        result = asyncio.run(client.acompletion("hi"))
        assert result == "hello from foundry"
        expected = 1000 * 0.60 / 1e6 + 500 * 3.00 / 1e6
        assert client.last_cost == pytest.approx(expected)
        assert client.last_cost_source == "synthesized"

    def test_provider_cost_takes_precedence(self, monkeypatch):
        response = _make_response(prompt_tokens=1000, completion_tokens=500, cost=0.01)
        client = _make_client(monkeypatch, response=response)
        client.completion("hi")

        assert client.last_cost == pytest.approx(0.01)
        assert client.last_cost_source == "provider"
        assert client.get_last_usage().cost_source == "provider"

        summary = client.get_usage_summary()
        assert summary.total_cost == pytest.approx(0.01)
        assert summary.model_usage_summaries["kimi-k2.5"].cost_source == "provider"

    def test_costs_accumulate_across_calls(self, monkeypatch):
        response = _make_response(prompt_tokens=1000, completion_tokens=500)
        client = _make_client(monkeypatch, response=response)
        client.completion("hi")
        client.completion("hi again")
        expected_per_call = 1000 * 0.60 / 1e6 + 500 * 3.00 / 1e6
        summary = client.get_usage_summary()
        assert summary.total_cost == pytest.approx(2 * expected_per_call)


class TestFailClosedUsage:
    def test_usage_none_raises(self, monkeypatch):
        response = _make_response(usage=None)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(ValueError):
            client.completion("hi")
        assert getattr(client, "last_cost", None) is None

    def test_missing_token_counts_raise(self, monkeypatch):
        usage = SimpleNamespace(total_tokens=0)
        response = _make_response(usage=usage)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(ValueError, match="token"):
            client.completion("hi")

    def test_none_token_counts_raise(self, monkeypatch):
        response = _make_response(prompt_tokens=None, completion_tokens=None)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(ValueError, match="token"):
            client.completion("hi")

    def test_negative_token_counts_raise(self, monkeypatch):
        response = _make_response(prompt_tokens=-5, completion_tokens=500)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(ValueError, match="token"):
            client.completion("hi")

    def test_non_integer_token_counts_raise(self, monkeypatch):
        response = _make_response(prompt_tokens=10.5, completion_tokens=500)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(ValueError, match="token"):
            client.completion("hi")

    def test_negative_provider_cost_raises(self, monkeypatch):
        response = _make_response(prompt_tokens=1000, completion_tokens=500, cost=-0.01)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(ValueError, match="cost"):
            client.completion("hi")

    def test_non_finite_provider_cost_raises(self, monkeypatch):
        response = _make_response(prompt_tokens=1000, completion_tokens=500, cost=float("nan"))
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(ValueError, match="cost"):
            client.completion("hi")


class TestResponseValidation:
    def test_content_filter_raises_and_still_records_the_spend(self, monkeypatch):
        """A content-filtered response is still a billed response: it must
        raise AND its tokens/synthesized cost must already be accumulated, so
        the spend breaker sees the paid spend the sub-call path swallows."""
        response = _make_response(
            prompt_tokens=1000, completion_tokens=500, finish_reason="content_filter", content=None
        )
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="content filter"):
            client.completion("hi")

        expected = 1000 * 0.60 / 1e6 + 500 * 3.00 / 1e6
        summary = client.get_usage_summary()
        assert summary.total_cost == pytest.approx(expected)
        model_summary = summary.model_usage_summaries["kimi-k2.5"]
        assert model_summary.total_input_tokens == 1000
        assert model_summary.total_output_tokens == 500
        assert model_summary.cost_source == "synthesized"

    def test_empty_content_raises_and_still_records_the_spend(self, monkeypatch):
        response = _make_response(content="")
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="empty"):
            client.completion("hi")
        summary = client.get_usage_summary()
        assert summary.total_cost is not None and summary.total_cost > 0

    def test_none_content_raises(self, monkeypatch):
        response = _make_response(content=None)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="empty"):
            client.completion("hi")


class TestReasoningExhaustion:
    """An empty body with ``finish_reason='length'`` whose output budget went
    to reasoning is a deterministic budget exhaustion (R6/KTD3): terminate
    once as ``TokenLimitExceededError`` with the spend already banked, never
    re-send it as a transient empty body."""

    @staticmethod
    def _no_sleep(monkeypatch):
        def _boom(_seconds):  # pragma: no cover - failure reporting
            raise AssertionError("retry backoff must not run for reasoning exhaustion")

        monkeypatch.setattr("rlm.clients.openai.time.sleep", _boom)

    def test_reasoning_dominated_length_terminates_once_with_spend(self, monkeypatch):
        from rlm.utils.exceptions import TokenLimitExceededError

        self._no_sleep(monkeypatch)
        response = _make_response(
            prompt_tokens=1000,
            completion_tokens=500,
            finish_reason="length",
            content="",
            reasoning_tokens=500,
        )
        client = _make_client(monkeypatch, response=response, sampling_args={"max_tokens": 500})
        with pytest.raises(TokenLimitExceededError, match="reasoning") as excinfo:
            client.completion("hi")

        assert client.client.chat.completions.create.call_count == 1
        assert excinfo.value.tokens_used == 500
        assert excinfo.value.token_limit == 500
        expected = 1000 * 0.60 / 1e6 + 500 * 3.00 / 1e6
        summary = client.get_usage_summary()
        assert summary.total_cost == pytest.approx(expected)
        assert summary.model_usage_summaries["kimi-k2.5"].total_output_tokens == 500

    def test_length_without_token_details_terminates_the_same_way(self, monkeypatch):
        from rlm.utils.exceptions import TokenLimitExceededError

        self._no_sleep(monkeypatch)
        response = _make_response(completion_tokens=800, finish_reason="length", content=None)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(TokenLimitExceededError, match="reasoning") as excinfo:
            client.completion("hi")
        assert client.client.chat.completions.create.call_count == 1
        # No max_tokens configured: the limit falls back to what was spent.
        assert excinfo.value.tokens_used == 800
        assert excinfo.value.token_limit == 800
        assert client.get_usage_summary().total_cost > 0

    def test_acompletion_terminates_the_same_way(self, monkeypatch):
        from rlm.utils.exceptions import TokenLimitExceededError

        monkeypatch.setattr(
            "rlm.clients.openai.asyncio.sleep",
            lambda _s: (_ for _ in ()).throw(AssertionError("no backoff")),
            raising=False,
        )
        response = _make_response(
            completion_tokens=500, finish_reason="length", content="", reasoning_tokens=480
        )
        calls = [0]

        async def _create(**kwargs):
            calls[0] += 1
            return response

        client = _make_async_client(monkeypatch, create=_create)

        with pytest.raises(TokenLimitExceededError, match="reasoning"):
            asyncio.run(client.acompletion("hi"))
        assert calls[0] == 1
        assert client.last_cost is not None and client.last_cost > 0

    def test_empty_stop_is_still_retried_then_raises_runtime_error(self, monkeypatch):
        """Kimi's transient empty body (finish_reason='stop') keeps its ladder."""
        from rlm.clients.openai import EMPTY_CONTENT_ATTEMPTS

        monkeypatch.setattr("rlm.clients.openai.time.sleep", lambda _s: None)
        response = _make_response(content="", finish_reason="stop")
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="empty"):
            client.completion("hi")
        assert client.client.chat.completions.create.call_count == EMPTY_CONTENT_ATTEMPTS

    def test_length_with_minor_reasoning_share_is_still_retried(self, monkeypatch):
        """Reasoning did not dominate the budget: not the exhaustion signature."""
        from rlm.clients.openai import EMPTY_CONTENT_ATTEMPTS

        monkeypatch.setattr("rlm.clients.openai.time.sleep", lambda _s: None)
        response = _make_response(
            completion_tokens=500, finish_reason="length", content="", reasoning_tokens=100
        )
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="empty"):
            client.completion("hi")
        assert client.client.chat.completions.create.call_count == EMPTY_CONTENT_ATTEMPTS

    def test_classifier_is_pure(self):
        from rlm.clients.azure_foundry import AzureFoundryClient

        def classify(response):
            return AzureFoundryClient._reasoning_exhausted(
                response.choices[0], getattr(response, "usage", None)
            )

        exhausted = _make_response(
            completion_tokens=500, finish_reason="length", content="", reasoning_tokens=450
        )
        assert classify(exhausted) is True
        with_text = _make_response(
            completion_tokens=500, finish_reason="length", content="x", reasoning_tokens=450
        )
        assert classify(with_text) is False
        stopped = _make_response(completion_tokens=500, finish_reason="stop", content="")
        assert classify(stopped) is False


class TestHarmonyMarkerStripping:
    """gpt-oss harmony control tokens never reach the REPL parser (R6a/KTD4):
    with channel structure only the ``final`` body survives; bare markers are
    stripped; every dropped marker is counted on the client."""

    FINAL_ONLY = "<|channel|>final<|message|>42<|return|>"
    ANALYSIS_THEN_FINAL = (
        "<|channel|>analysis<|message|>think think<|end|>"
        "<|start|>assistant<|channel|>final<|message|>42<|return|>"
    )

    def test_final_channel_body_is_kept(self):
        from rlm.clients.azure_foundry import strip_harmony_markers

        assert strip_harmony_markers(self.FINAL_ONLY) == ("42", 3)

    def test_analysis_body_is_dropped_with_its_markers(self):
        from rlm.clients.azure_foundry import strip_harmony_markers

        text, dropped = strip_harmony_markers(self.ANALYSIS_THEN_FINAL)
        assert text == "42"
        assert "think" not in text
        assert dropped == 7  # channel, message, end, start, channel, message, return

    def test_bare_markers_are_stripped_without_channel_structure(self):
        from rlm.clients.azure_foundry import strip_harmony_markers

        text, dropped = strip_harmony_markers("answer<|return|>")
        assert text == "answer"
        assert dropped == 1
        text, dropped = strip_harmony_markers("<|start|>a<|end|><|constrain|>json<|call|>")
        assert text == "ajson"
        assert dropped == 4

    def test_plain_content_is_byte_identical(self):
        from rlm.clients.azure_foundry import strip_harmony_markers

        plain = "Plan first.\n```repl\nprint('<|not a marker')\n```\n"
        text, dropped = strip_harmony_markers(plain)
        assert text is plain
        assert dropped == 0

    def test_client_completion_returns_the_final_body_and_counts(self, monkeypatch):
        client = _make_client(monkeypatch, response=_make_response(content=self.FINAL_ONLY))
        assert client.completion("hi") == "42"
        assert client.harmony_markers_dropped == 3

    def test_client_completion_drops_the_analysis_channel(self, monkeypatch):
        client = _make_client(
            monkeypatch, response=_make_response(content=self.ANALYSIS_THEN_FINAL)
        )
        assert client.completion("hi") == "42"
        assert client.harmony_markers_dropped == 7

    def test_acompletion_strips_and_counts_too(self, monkeypatch):
        response = _make_response(content=self.FINAL_ONLY)
        client = _make_async_client(monkeypatch, response=response)

        assert asyncio.run(client.acompletion("hi")) == "42"
        assert client.harmony_markers_dropped == 3

    def test_kimi_tool_call_translation_runs_before_harmony_stripping(self, monkeypatch):
        leak = TestNativeToolCallTranslation.LEAK
        content = "<|channel|>final<|message|>" + leak + "<|return|>"
        client = _make_client(monkeypatch, response=_make_response(content=content))
        out = client.completion("hi")
        assert out == "```repl\nprint(type(context))\nprint(len(context))\n```"
        assert client.harmony_markers_dropped == 3

    def test_analysis_only_length_raises_token_limit_with_spend(self, monkeypatch):
        """An analysis-only harmony body cut at the length cap (no ``final``
        channel) has non-empty RAW content but normalizes to "": it must
        terminate once as TokenLimitExceededError with the spend already
        banked, never return a silent empty string."""
        from rlm.utils.exceptions import TokenLimitExceededError

        monkeypatch.setattr(
            "rlm.clients.openai.time.sleep",
            lambda _s: (_ for _ in ()).throw(AssertionError("no backoff for a budget limit")),
        )
        response = _make_response(
            prompt_tokens=1000,
            completion_tokens=500,
            finish_reason="length",
            content="<|channel|>analysis<|message|>think think think",
        )
        client = _make_client(monkeypatch, response=response, sampling_args={"max_tokens": 500})
        with pytest.raises(TokenLimitExceededError, match="no final body") as excinfo:
            client.completion("hi")

        assert client.client.chat.completions.create.call_count == 1
        assert excinfo.value.tokens_used == 500
        assert excinfo.value.token_limit == 500
        expected = 1000 * 0.60 / 1e6 + 500 * 3.00 / 1e6
        summary = client.get_usage_summary()
        assert summary.total_cost == pytest.approx(expected)
        assert summary.model_usage_summaries["kimi-k2.5"].total_output_tokens == 500

    def test_analysis_only_stop_is_retried_then_raises_runtime_error(self, monkeypatch):
        """Same shape with finish_reason='stop': behaves exactly like the raw
        empty body -- retried on the empty-content ladder, then raises."""
        from rlm.clients.openai import EMPTY_CONTENT_ATTEMPTS

        monkeypatch.setattr("rlm.clients.openai.time.sleep", lambda _s: None)
        response = _make_response(
            content="<|channel|>analysis<|message|>only reasoning<|end|>",
            finish_reason="stop",
        )
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="empty"):
            client.completion("hi")
        assert client.client.chat.completions.create.call_count == EMPTY_CONTENT_ATTEMPTS

    def test_plain_content_round_trips_byte_identical_and_counts_nothing(self, monkeypatch):
        plain = "Plan first.\n```repl\nprint(1)\n```"
        client = _make_client(monkeypatch, response=_make_response(content=plain))
        assert client.completion("hi") == plain
        assert client.harmony_markers_dropped == 0

    def test_concurrent_mixed_responses_count_exactly_the_marker_total(self, monkeypatch):
        import threading

        responses = {
            "harmony": _make_response(content=self.ANALYSIS_THEN_FINAL),  # 7 markers
            "plain": _make_response(content="plain answer"),  # 0 markers
            "bare": _make_response(content="answer<|return|>"),  # 1 marker
        }

        def create(**kwargs):
            return responses[kwargs["messages"][0]["content"]]

        client = _make_client(monkeypatch)
        client.client.chat.completions.create = create
        calls_per_thread = 50
        errors: list[BaseException] = []
        outputs: dict[str, set[str]] = {key: set() for key in responses}

        def run(prompt: str) -> None:
            try:
                for _ in range(calls_per_thread):
                    outputs[prompt].add(client.completion(prompt))
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(key,)) for key in responses]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert outputs == {"harmony": {"42"}, "plain": {"plain answer"}, "bare": {"answer"}}
        assert client.harmony_markers_dropped == calls_per_thread * (7 + 0 + 1)


class TestThreadSafety:
    def test_concurrent_completions_accumulate_exactly_the_sum(self, monkeypatch):
        """Two threads on ONE shared client (the lm_handler ThreadingTCPServer
        shape) must accumulate exactly the sum of their own synthesized costs:
        synthesis reads last_prompt_tokens/last_completion_tokens, so without
        the cost lock an interleaving cross-bills one thread's tokens."""
        import threading

        responses = {
            "thread-a": _make_response(prompt_tokens=1000, completion_tokens=500),
            "thread-b": _make_response(prompt_tokens=7000, completion_tokens=900),
        }

        def create(**kwargs):
            return responses[kwargs["messages"][0]["content"]]

        client = _make_client(monkeypatch)
        client.client.chat.completions.create = create

        calls_per_thread = 50
        errors: list[BaseException] = []

        def run(prompt: str) -> None:
            try:
                for _ in range(calls_per_thread):
                    client.completion(prompt)
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [
            threading.Thread(target=run, args=("thread-a",)),
            threading.Thread(target=run, args=("thread-b",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors

        cost_a = 1000 * 0.60 / 1e6 + 500 * 3.00 / 1e6
        cost_b = 7000 * 0.60 / 1e6 + 900 * 3.00 / 1e6
        expected = calls_per_thread * (cost_a + cost_b)
        assert client.model_costs["kimi-k2.5"] == pytest.approx(expected)
        summary = client.get_usage_summary()
        assert summary.total_cost == pytest.approx(expected)
        assert summary.model_usage_summaries["kimi-k2.5"].total_input_tokens == (
            calls_per_thread * (1000 + 7000)
        )


class TestSamplingArgsFidelity:
    def test_sampling_args_forwarded(self, monkeypatch):
        response = _make_response()
        client = _make_client(
            monkeypatch,
            response=response,
            sampling_args={"temperature": 0.7, "max_tokens": 128},
        )
        client.completion("hi")
        call_kwargs = client.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_completion_tokens"] == 128
        assert "max_tokens" not in call_kwargs

    def test_shipped_decoding_reaches_the_wire_per_provider(self, provider, tmp_path, monkeypatch):
        """Provider matrix (KTD5): the shipped ``[decoding]`` table, routed
        through a runner role switched to ``provider``, reaches the mocked
        ``chat.completions.create`` with exactly the row's contract -- the
        expected top-level keys present, the forbidden ones absent, and the
        row's ``extra_body`` byte-for-byte."""
        from shrlm.experiment.config import backend_kwargs_for, load_config
        from tests.experiment.test_config import all_roles_text, write_config

        config = load_config(path=write_config(tmp_path, all_roles_text(provider)))
        backend_kwargs = backend_kwargs_for(config, "runner")
        assert ("pricing" in backend_kwargs) == (provider.pricing is not None)

        client = provider.make_client(
            monkeypatch,
            response=provider.response(),
            sampling_args=backend_kwargs["sampling_args"],
        )
        client.completion("hi")
        call_kwargs = client.client.chat.completions.create.call_args.kwargs
        assert provider.expected_sampling_keys <= set(call_kwargs)
        assert provider.forbidden_sampling_keys.isdisjoint(call_kwargs)
        assert call_kwargs["temperature"] == config.decoding.temperature
        assert call_kwargs["top_p"] == config.decoding.top_p
        assert call_kwargs["max_completion_tokens"] == config.decoding.max_output_tokens
        assert call_kwargs["extra_body"] == provider.expected_extra_body

    def test_row_config_path_decoding_reaches_the_wire(self, provider, monkeypatch):
        """The row's shipped ``config_path`` (not the surgered default) builds
        a client whose ``create`` call carries the row's key contract and the
        file's own output cap as ``max_completion_tokens`` (16384 for gpt-oss);
        ``reasoning_effort`` never hides inside ``extra_body`` and never
        travels with ``chat_template_kwargs``."""
        from shrlm.experiment.config import backend_kwargs_for, load_config

        config = load_config(path=provider.config_path)
        backend_kwargs = backend_kwargs_for(config, "runner")
        client = provider.make_client(
            monkeypatch,
            response=provider.response(),
            sampling_args=backend_kwargs["sampling_args"],
        )
        client.completion("hi")
        call_kwargs = client.client.chat.completions.create.call_args.kwargs
        assert provider.expected_sampling_keys <= set(call_kwargs)
        assert provider.forbidden_sampling_keys.isdisjoint(call_kwargs)
        assert call_kwargs["max_completion_tokens"] == provider.config_max_output_tokens
        extra_body = call_kwargs["extra_body"]
        assert "reasoning_effort" not in extra_body
        # Loader invariant: a reasoning-effort request never carries Kimi's
        # instant-mode chat_template_kwargs (the Foundry route rejects it).
        assert not ("reasoning_effort" in call_kwargs and "chat_template_kwargs" in extra_body)


class TestModelUsageSummaryCostSource:
    def test_to_dict_omits_cost_source_when_unset(self):
        from rlm.core.types import ModelUsageSummary

        summary = ModelUsageSummary(total_calls=1, total_input_tokens=10, total_output_tokens=5)
        assert "cost_source" not in summary.to_dict()

    def test_to_dict_includes_cost_source_when_set(self):
        from rlm.core.types import ModelUsageSummary

        summary = ModelUsageSummary(
            total_calls=1,
            total_input_tokens=10,
            total_output_tokens=5,
            total_cost=0.5,
            cost_source="synthesized",
        )
        data = summary.to_dict()
        assert data["cost_source"] == "synthesized"

    def test_from_dict_round_trip(self):
        from rlm.core.types import ModelUsageSummary

        summary = ModelUsageSummary(
            total_calls=2,
            total_input_tokens=20,
            total_output_tokens=10,
            total_cost=1.5,
            cost_source="provider",
        )
        restored = ModelUsageSummary.from_dict(summary.to_dict())
        assert restored == summary

    def test_from_dict_defaults_cost_source_to_none(self):
        from rlm.core.types import ModelUsageSummary

        restored = ModelUsageSummary.from_dict(
            {"total_calls": 1, "total_input_tokens": 10, "total_output_tokens": 5}
        )
        assert restored.cost_source is None


class TestGetClientRegistration:
    def test_get_client_returns_azure_foundry_client(self, monkeypatch):
        from rlm.clients import get_client
        from rlm.clients.azure_foundry import AzureFoundryClient

        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", VALID_ENDPOINT)
        with (
            patch("rlm.clients.openai.openai.OpenAI"),
            patch("rlm.clients.openai.openai.AsyncOpenAI"),
        ):
            client = get_client(
                "azure_foundry",
                {"model_name": "kimi-k2.5", "pricing": dict(VALID_PRICING)},
            )
        assert isinstance(client, AzureFoundryClient)

    def test_unknown_backend_error_lists_azure_foundry(self):
        from rlm.clients import get_client

        with pytest.raises(ValueError, match="azure_foundry"):
            get_client("nonexistent_backend", {})


# ---------------------------------------------------------------------------
# Live tier (U4): three tiny paid calls against the real deployment (KTD8)
# ---------------------------------------------------------------------------


def _selected_config() -> Any:
    """The env-selected smoke-profile config the live tier runs against
    (KTD7): ``SHRLM_EXPERIMENT_CONFIG`` when set, else the shipped
    ``configs/experiment.toml``."""
    from shrlm.experiment.config import CONFIG_PATH, load_config

    return load_config(profile="smoke", path=live_config_path(CONFIG_PATH))


def _azure_live_skip() -> str | None:
    """Skip reason for the azure-specific live tier, or ``None`` to run it.

    ``live_skip_reason`` gates on the SELECTED config's runner backend, so
    with a non-azure backend selected an open gate would demand that
    backend's credentials and then run this module's azure code paths
    (backend_kwargs without pricing -> constructor error). The azure live
    tier therefore also requires the selected config to run azure_foundry;
    the pricing attestation is checked against that config's rate card.
    """
    try:
        config = _selected_config()
    except Exception as exc:  # bad SHRLM_EXPERIMENT_CONFIG must skip, not error collection
        return f"failed to load config selected via {CONFIG_ENV_KEY}: {exc!r}"
    if config.backends.runner.backend != "azure_foundry":
        return (
            "selected runner backend is not azure_foundry -- azure live tier requires "
            "an azure config (set SHRLM_EXPERIMENT_CONFIG to one)"
        )
    return live_skip_reason(config=config)


_LIVE_SKIP = _azure_live_skip()

# Output caps for the live calls. Trivial prompts under these caps keep the
# whole class in the cents at either configured rate card. Under a reasoning
# config the trivial cap is raised: reasoning tokens bill as completion
# tokens, so 64 would starve the answer by design.
LIVE_TRIVIAL_MAX_TOKENS = 64
LIVE_REASONING_TRIVIAL_MAX_TOKENS = 2048
LIVE_CAP_MAX_TOKENS = 16
# Some gateways report a couple of tokens beyond max_completion_tokens (e.g.
# a stop token); the cap test tolerates that without tolerating a busted cap.
LIVE_CAP_SLACK_TOKENS = 8
# The reasoning-effort comparison needs room for a high-effort chain.
LIVE_EFFORT_MAX_TOKENS = 8192

LIVE_TRIVIAL_PROMPT = "Reply with the single word: ok"
LIVE_EFFORT_PROMPT = (
    "In how many ways can the letters of the word FOUNDRY be arranged so that "
    "no two vowels are adjacent? Reply with just the number."
)


def _probe_expectation() -> Any:
    """What the selected config says a response must look like (KTD6)."""
    from examples.experiment_smoke import probe_expectation

    return probe_expectation(_selected_config())


def _trivial_max_tokens(expectation: Any) -> int:
    return (
        LIVE_REASONING_TRIVIAL_MAX_TOKENS
        if expectation.expects_reasoning
        else LIVE_TRIVIAL_MAX_TOKENS
    )


def _runner_mismatch(provider: Any, config: Any) -> str | None:
    """Why a provider-matrix row must skip under the selected config, or
    ``None`` when the row is the config's runner (KTD7).

    The live class loads ONLY the env-selected config, so a row whose
    backend/model differ from that config's runner has nothing real to run
    against; it skips with a reason naming the mismatch and ``-rs`` shows
    exactly one PASSED row per test. Pure, so the offline tests below
    exercise it without credentials or spend.
    """
    runner = config.backends.runner
    if (provider.backend, provider.model) == (runner.backend, runner.model):
        return None
    return (
        f"provider row {provider.id} ({provider.backend}/{provider.model}) does not "
        f"match the selected config's runner ({runner.backend}/{runner.model})"
    )


def _skip_unmatched_row(provider: Any) -> None:
    reason = _runner_mismatch(provider, _selected_config())
    if reason is not None:
        pytest.skip(reason)


def _live_runner_kwargs(max_tokens: int) -> dict[str, Any]:
    """The REAL configured runner backend_kwargs (the selected config's smoke
    profile) with only the output cap lowered for these tiny calls.

    ``backend_kwargs_for`` builds fresh dicts on every call, and the result is
    deep-copied anyway before the override, so no shared config object is ever
    mutated. Everything else -- temperature, top_p, the extra_body
    (``chat_template_kwargs`` for Kimi), a top-level ``reasoning_effort`` when
    the config sets one, and the nested ``pricing`` -- is exactly what a real
    experiment round sends.
    """
    from shrlm.experiment.config import backend_kwargs_for

    kwargs = copy.deepcopy(backend_kwargs_for(_selected_config(), "runner"))
    kwargs["sampling_args"]["max_tokens"] = max_tokens
    return kwargs


def _raw_live_completion(lm: Any, prompt: str) -> Any:
    """One paid chat completion through the client's own request builders.

    This is ``OpenAIClient.completion`` taken apart into its two halves --
    the exact ``chat.completions.create`` parameter shape, then the client's
    own ``_track_cost`` (the azure_foundry override that validates usage and
    synthesizes the cost) -- so the test can inspect the raw response
    (finish_reason, reasoning fields, usage details) while the cost path
    exercised is byte-for-byte the one every persisted run travels.
    """
    from rlm.clients.openai import _merge_extra_body, _normalize_sampling_args

    response = lm.client.chat.completions.create(
        model=lm.model_name,
        messages=[{"role": "user", "content": prompt}],
        extra_body=_merge_extra_body({}, lm.sampling_args),
        **_normalize_sampling_args(lm.sampling_args),
    )
    lm._track_cost(response, lm.model_name)
    return response


@pytest.mark.live
@pytest.mark.skipif(_LIVE_SKIP is not None, reason=_LIVE_SKIP or "live gates satisfied")
class TestAzureFoundryLive:
    """Provider-parametrized (the ``provider`` fixture from tests/conftest.py),
    but every row loads ONLY the env-selected config: a row whose
    backend/model do not match that config's runner skips with a reason
    naming the mismatch, so ``-rs`` shows exactly one PASSED row per test
    (KTD7). What each test asserts is derived from the config's
    ``probe_expectation``, never from a provider name."""

    def test_trivial_call_synthesizes_cost_and_matches_reasoning_contract(self, provider):
        """One trivial call: non-empty content, positive token counts, a
        positive synthesized cost, and a reasoning signal exactly when the
        config expects one (R10/KTD6 -- the detector is the very one the
        tier-2 smoke probe uses; it raises SmokeError naming each signal)."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        cap = _trivial_max_tokens(expectation)

        lm = get_client("azure_foundry", _live_runner_kwargs(cap))
        response = _raw_live_completion(lm, LIVE_TRIVIAL_PROMPT)
        payload = response.model_dump()

        content = payload["choices"][0]["message"]["content"]
        assert content and content.strip()

        last = lm.get_last_usage()
        assert last.total_input_tokens > 0
        assert last.total_output_tokens > 0
        # A trivial "ok" must come nowhere near the cap; cap-saturating output
        # on this prompt would itself be a runaway-reasoning smell.
        assert last.total_output_tokens < cap
        assert last.total_cost is not None and last.total_cost > 0
        assert last.cost_source == "synthesized"

        from examples.experiment_smoke import check_probe_reasoning

        check_probe_reasoning(payload, expectation)

    def test_output_cap_is_honored_not_merely_tolerated(self, provider):
        """Without reasoning, a verbose prompt under max_tokens=16 must be
        TRUNCATED: finish_reason 'length', completion tokens at the cap --
        the route enforces the cap rather than merely accepting the
        parameter. With reasoning expected, the 16-token budget is consumed
        by reasoning before any content lands, so the client's own
        ``completion()`` must raise TokenLimitExceededError -- while the
        spend-then-validate order still recorded a positive synthesized cost
        for the paid, truncated call."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        lm = get_client("azure_foundry", _live_runner_kwargs(LIVE_CAP_MAX_TOKENS))

        if expectation.expects_reasoning:
            from rlm.utils.exceptions import TokenLimitExceededError

            with pytest.raises(TokenLimitExceededError):
                lm.completion("Count from 1 to 500, one number per line.")
            last = lm.get_last_usage()
            assert last.total_cost is not None and last.total_cost > 0
            assert last.cost_source == "synthesized"
            return

        response = _raw_live_completion(lm, "Count from 1 to 500, one number per line.")
        assert response.choices[0].finish_reason == "length"
        completion_tokens = response.usage.completion_tokens
        assert completion_tokens <= LIVE_CAP_MAX_TOKENS + LIVE_CAP_SLACK_TOKENS

    def test_configured_decoding_args_are_accepted_on_the_client_path(self, provider):
        """The client's own ``completion`` path with the real configured
        sampling args (temperature / top_p / max_completion_tokens /
        extra_body / reasoning_effort all ride this request): a deployment
        that rejects any of them surfaces as an HTTP 400 -> exception here."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        lm = get_client("azure_foundry", _live_runner_kwargs(_trivial_max_tokens(expectation)))
        content = lm.completion(LIVE_TRIVIAL_PROMPT)
        assert content and content.strip()
        assert "<think>" not in content

    def test_reasoning_effort_is_honored_not_merely_accepted(self, provider):
        """R11's stop condition at the client tier: one ``low`` and one
        ``high`` raw call on a reasoning-heavy prompt must be distinguishable
        in completion tokens -- indistinguishable efforts mean the gateway
        ignores the knob. Runs only under a config that sets a real
        reasoning_effort (the trivial and cap tests cover the rest)."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        if not expectation.expects_reasoning:
            pytest.skip(
                "selected config sets no reasoning_effort; the low-vs-high "
                "comparison needs a reasoning config"
            )

        tokens: dict[str, int] = {}
        for effort in ("low", "high"):
            kwargs = _live_runner_kwargs(LIVE_EFFORT_MAX_TOKENS)
            kwargs["sampling_args"]["reasoning_effort"] = effort
            lm = get_client("azure_foundry", kwargs)
            response = _raw_live_completion(lm, LIVE_EFFORT_PROMPT)
            tokens[effort] = int(response.usage.completion_tokens)
        assert tokens["high"] > tokens["low"], (
            f"reasoning_effort indistinguishable: completion tokens {tokens} "
            "(stop condition (a) -- surface before any multi-call tier)"
        )


class TestLiveRowSelectionOffline:
    """KTD7 offline ($0): the live class loads only the env-selected config,
    and a mismatching provider row reports a skip reason naming the runner
    mismatch while the matching row runs."""

    def _gptoss_config(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        from tests.conftest import CONFIG_DIR

        monkeypatch.setenv(
            "SHRLM_EXPERIMENT_CONFIG", str(CONFIG_DIR / "experiment_oolong_gptoss.toml")
        )
        return _selected_config()

    def test_mismatching_rows_name_the_runner_mismatch(self, monkeypatch):
        from tests.conftest import AZURE_KIMI, OPENROUTER_QWEN

        config = self._gptoss_config(monkeypatch)
        for row in (AZURE_KIMI, OPENROUTER_QWEN):
            reason = _runner_mismatch(row, config)
            assert reason is not None
            assert row.id in reason and row.model in reason
            assert "gpt-oss-120b" in reason

    def test_the_selected_configs_own_row_does_not_skip(self, monkeypatch):
        from tests.conftest import AZURE_GPTOSS

        config = self._gptoss_config(monkeypatch)
        assert _runner_mismatch(AZURE_GPTOSS, config) is None

    def test_env_unset_selects_the_shipped_smoke_profile(self, monkeypatch):
        """With the env unset this module resolves the config it loads today:
        the shipped configs/experiment.toml smoke profile (openrouter runner,
        so every azure row mismatches and the module-level gate skips)."""
        from shrlm.experiment.config import CONFIG_PATH

        monkeypatch.delenv("SHRLM_EXPERIMENT_CONFIG", raising=False)
        assert live_config_path(CONFIG_PATH) == CONFIG_PATH


class TestNativeToolCallTranslation:
    """Kimi's leaked ``functions.repl`` calls become ```repl``` blocks; anything
    else is left verbatim so the leak stays visible in the trace."""

    LEAK = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.repl:0"
        '<|tool_call_argument_begin|>{"code": "print(type(context))\\nprint(len(context))"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )

    def test_repl_call_becomes_a_fenced_block(self):
        from rlm.clients.azure_foundry import translate_native_tool_calls

        assert translate_native_tool_calls(self.LEAK) == (
            "```repl\nprint(type(context))\nprint(len(context))\n```"
        )

    def test_plain_text_is_untouched(self):
        from rlm.clients.azure_foundry import translate_native_tool_calls

        text = "Plan first.\n```repl\nprint(1)\n```"
        assert translate_native_tool_calls(text) is text

    def test_non_repl_or_malformed_calls_stay_verbatim(self):
        from rlm.clients.azure_foundry import translate_native_tool_calls

        other = self.LEAK.replace("functions.repl", "functions.search")
        assert translate_native_tool_calls(other) == other
        broken = self.LEAK.replace('"code": ', '"code" ')
        assert translate_native_tool_calls(broken) == broken

    def test_prose_around_the_call_is_kept(self):
        from rlm.clients.azure_foundry import translate_native_tool_calls

        text = "Let me look.\n" + self.LEAK + "\nThen decide."
        out = translate_native_tool_calls(text)
        assert out.startswith("Let me look.\n```repl\n")
        assert out.endswith("```\nThen decide.")

    def test_client_completion_returns_the_translated_block(self, monkeypatch):
        client = _make_client(monkeypatch, response=_make_response(content=self.LEAK))
        assert client.completion("hi").startswith("```repl\n")
