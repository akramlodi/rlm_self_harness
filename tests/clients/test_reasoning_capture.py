"""Provider reasoning is an observation, never the executable response."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlm.core.llm_observation import ObservationPersistenceError, ObservationRecorder
from rlm.utils.exceptions import TokenLimitExceededError
from tests.clients.test_azure_foundry import _make_client, _make_response
from tests.clients.test_openai_transport import good_response, make_client


def response_with_reasoning():
    response = good_response()
    response.choices[0].message.reasoning_content = "Think through the constraint."
    response.choices[0].message.reasoning_details = [
        {"type": "reasoning.summary", "summary": "A summary."},
        {"type": "reasoning.encrypted", "data": "opaque", "signature": "sig"},
    ]
    response.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=3)
    return response


@pytest.mark.parametrize("asynchronous", [False, True])
def test_separate_reasoning_preserves_response_and_usage(asynchronous):
    response = response_with_reasoning()
    client = make_client(response)

    async def create(**kwargs):
        return response

    client.async_client.chat.completions.create = create
    recorder = ObservationRecorder(owner={"stage": "mining", "run_id": "task-1"})
    with recorder.call("root", model="test-model") as observed:
        result = (
            asyncio.run(client.acompletion("task")) if asynchronous else client.completion("task")
        )
    assert result == "ok"
    record = observed.responses[0]
    assert record["reasoning"]["reasoning_content"] == "Think through the constraint."
    assert record["reasoning"]["reasoning_details"] == response.choices[0].message.reasoning_details
    assert record["raw_content"] == "ok"
    assert record["availability"] == "returned"
    assert record["reasoning_tokens"] == 3
    assert record["owner"]["run_id"] == "task-1"
    assert client.get_last_usage().total_output_tokens == 5


def test_capture_precedes_azure_normalization(monkeypatch):
    content = "<|channel|>analysis<|message|>consider this<|end|><|start|>assistant<|channel|>final<|message|>ok<|return|>"
    client = _make_client(monkeypatch, response=_make_response(content=content))
    recorder = ObservationRecorder()
    with recorder.call("root") as observed:
        result = client.completion("task")
    assert "consider this" not in result
    assert observed.responses[0]["raw_content"] == content
    assert observed.responses[0]["reasoning"]["inline_blocks"][0]["text"] == "consider this"


def test_capture_preserves_discarded_empty_response(monkeypatch):
    first = _make_response(content="")
    first.choices[0].message.reasoning = "First attempt."
    second = _make_response(content="ok")
    client = _make_client(monkeypatch)
    client.client.chat.completions.create.side_effect = [first, second]
    recorder = ObservationRecorder()
    with patch("rlm.clients.openai.time.sleep"), recorder.call("root") as observed:
        assert client.completion("task") == "ok"
    assert len(observed.responses) == 2
    assert observed.responses[0]["reasoning"]["reasoning"] == "First attempt."
    assert len({r["attempt_id"] for r in observed.responses}) == 2
    assert client.get_usage_summary().model_usage_summaries[client.model_name].total_calls == 2


def test_reasoning_budget_failure_still_has_response(monkeypatch):
    response = _make_response(content=None, finish_reason="length", reasoning_tokens=500)
    response.choices[0].message.reasoning_content = "Unfinished reasoning."
    client = _make_client(monkeypatch, response=response)
    recorder = ObservationRecorder()
    with pytest.raises(TokenLimitExceededError), recorder.call("proposal") as observed:
        client.completion("task")
    assert observed.responses[0]["reasoning"]["reasoning_content"] == "Unfinished reasoning."


@pytest.mark.parametrize("budget_error", [False, True])
def test_capture_failure_accounts_once_without_retry(monkeypatch, budget_error):
    def fail_response(record):
        if record["event"] == "response":
            raise OSError("disk full")
        return {"call_id": record["call_id"]}

    response = _make_response(
        content=None if budget_error else "ok",
        finish_reason="length" if budget_error else "stop",
        reasoning_tokens=500 if budget_error else 10,
    )
    client = _make_client(monkeypatch, response=response)
    recorder = ObservationRecorder(sink=fail_response)
    with pytest.raises(ObservationPersistenceError, match="disk full"), recorder.call("root"):
        client.completion("task")
    assert client.client.chat.completions.create.call_count == 1
    usage = client.get_usage_summary().model_usage_summaries[client.model_name]
    assert usage.total_calls == 1
    assert usage.total_output_tokens == 500


@pytest.mark.parametrize(
    "value,availability", [(None, "not_returned"), ("", "not_returned"), ("text", "returned")]
)
def test_missing_empty_and_null_reasoning_remain_distinct(value, availability):
    response = good_response()
    response.choices[0].message.reasoning = value
    client = make_client(response)
    with ObservationRecorder().call("root") as observed:
        client.completion("task")
    assert observed.responses[0]["reasoning"] == {"reasoning": value}
    assert observed.responses[0]["availability"] == availability
    assert observed.responses[0]["reasoning_tokens"] is None


def test_opaque_payload_is_not_readable_reasoning():
    response = good_response()
    response.choices[0].message.reasoning_details = [{"type": "reasoning.encrypted", "data": "xyz"}]
    with ObservationRecorder().call("root") as observed:
        make_client(response).completion("task")
    assert observed.responses[0]["availability"] == "opaque_only"


@pytest.mark.parametrize("adapter", ["anthropic", "gemini", "azure_openai", "portkey"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_registered_adapters_capture_native_blocks(adapter, asynchronous):
    response = response_with_reasoning()
    if adapter == "anthropic":
        from rlm.clients.anthropic import AnthropicClient

        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="ok"),
                SimpleNamespace(type="thinking", thinking="reasoning", signature="signature"),
                SimpleNamespace(type="redacted_thinking", data="opaque"),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        with (
            patch("rlm.clients.anthropic.anthropic.Anthropic"),
            patch("rlm.clients.anthropic.anthropic.AsyncAnthropic"),
        ):
            client = AnthropicClient(api_key="test", model_name="test")
        create_owner = client.client.messages
        async_owner = client.async_client.messages
        method = "create"
        expected_field = "thinking_blocks"
    elif adapter == "gemini":
        from rlm.clients.gemini import GeminiClient

        response = SimpleNamespace(
            text="ok",
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text="reasoning", thought=True, thought_signature=b"signature"
                            ),
                            SimpleNamespace(text="ok", thought=False),
                        ]
                    )
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5, thoughts_token_count=3
            ),
        )
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test", model_name="test")
        create_owner = client.client.models
        async_owner = client.client.aio.models
        method = "generate_content"
        expected_field = "thought_parts"
    elif adapter == "azure_openai":
        from rlm.clients.azure_openai import AzureOpenAIClient

        with (
            patch("rlm.clients.azure_openai.openai.AzureOpenAI"),
            patch("rlm.clients.azure_openai.openai.AsyncAzureOpenAI"),
        ):
            client = AzureOpenAIClient(
                api_key="test", model_name="test", azure_endpoint="https://example.openai.azure.com"
            )
        create_owner = client.client.chat.completions
        async_owner = client.async_client.chat.completions
        method = "create"
        expected_field = "reasoning_details"
    else:
        from rlm.clients.portkey import PortkeyClient

        with patch("rlm.clients.portkey.Portkey"), patch("rlm.clients.portkey.AsyncPortkey"):
            client = PortkeyClient(api_key="test", model_name="test")
        create_owner = client.client.chat.completions
        async_owner = client.async_client.chat.completions
        method = "create"
        expected_field = "reasoning_details"

    async def create(**kwargs):
        return response

    getattr(create_owner, method).return_value = response
    setattr(async_owner, method, create)
    with ObservationRecorder().call("root") as observed:
        result = (
            asyncio.run(client.acompletion("task")) if asynchronous else client.completion("task")
        )
    assert result == "ok"
    assert observed.responses[0]["reasoning"][expected_field]
    assert observed.responses[0]["availability"] == "returned"
    assert client.get_last_usage().total_output_tokens == 5
    if adapter == "gemini":
        assert observed.responses[0]["reasoning"][expected_field][0]["thought_signature"] == {
            "encoding": "base64",
            "data": "c2lnbmF0dXJl",
        }
        assert "summary" in observed.responses[0]["reasoning_kinds"]
