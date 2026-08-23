"""Tests for the Azure AI Foundry client (mocked, zero network)."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

VALID_ENDPOINT = "https://my-resource.services.ai.azure.com"
VALID_PRICING = {"input_per_million": 0.60, "output_per_million": 3.00}


def _make_response(
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 500,
    cost: float | None = None,
    finish_reason: str = "stop",
    content: str | None = "hello from foundry",
    usage: Any = "default",
) -> SimpleNamespace:
    """Build a fake chat.completions response using SimpleNamespace so that
    hasattr checks behave like real pydantic models (MagicMock would report
    every attribute as present)."""
    if usage == "default":
        usage_kwargs: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        }
        if cost is not None:
            usage_kwargs["cost"] = cost
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


class TestConstructionValidation:
    def test_missing_api_key_names_variable(self, monkeypatch):
        import rlm.clients.azure_foundry as mod

        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.setattr(mod, "DEFAULT_AZURE_API_KEY", None)
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", VALID_ENDPOINT)
        with pytest.raises(ValueError, match="AZURE_API_KEY"):
            mod.AzureFoundryClient(model_name="m", pricing=dict(VALID_PRICING))

    def test_missing_endpoint_names_variable(self, monkeypatch):
        import rlm.clients.azure_foundry as mod

        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
        monkeypatch.setattr(mod, "DEFAULT_AZURE_FOUNDRY_ENDPOINT", None)
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
        monkeypatch.setattr(mod, "DEFAULT_AZURE_API_KEY", None)
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
        from rlm.clients.azure_foundry import AzureFoundryClient

        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", VALID_ENDPOINT)
        response = _make_response(prompt_tokens=1000, completion_tokens=500)

        with (
            patch("rlm.clients.openai.openai.OpenAI"),
            patch("rlm.clients.openai.openai.AsyncOpenAI") as mock_async_openai,
        ):
            mock_async = MagicMock()

            async def _create(**kwargs):
                return response

            mock_async.chat.completions.create = _create
            mock_async_openai.return_value = mock_async
            client = AzureFoundryClient(model_name="kimi-k2.5", pricing=dict(VALID_PRICING))

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
    def test_content_filter_raises(self, monkeypatch):
        response = _make_response(finish_reason="content_filter", content=None)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="content filter"):
            client.completion("hi")

    def test_empty_content_raises(self, monkeypatch):
        response = _make_response(content="")
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="empty"):
            client.completion("hi")

    def test_none_content_raises(self, monkeypatch):
        response = _make_response(content=None)
        client = _make_client(monkeypatch, response=response)
        with pytest.raises(RuntimeError, match="empty"):
            client.completion("hi")


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
