import math
import os
from typing import Any
from urllib.parse import urlparse

import openai
from dotenv import load_dotenv

from rlm.clients.openai import OpenAIClient
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()

# Load credentials from environment variables
DEFAULT_AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DEFAULT_AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")

# Hosts the Foundry endpoint is allowed to resolve to. Anything else is
# rejected before an API client is ever constructed.
_ALLOWED_HOST_SUFFIXES = (".services.ai.azure.com", ".openai.azure.com")

_PRICING_KEYS = ("input_per_million", "output_per_million")


def _validate_pricing(pricing: Any) -> dict[str, float]:
    """Validate the required pricing dict; a zero or missing rate would let a
    call synthesize a $0 cost and silently disarm the spend breaker."""
    if not isinstance(pricing, dict):
        raise ValueError(
            "Azure Foundry client requires a 'pricing' dict with keys "
            "'input_per_million' and 'output_per_million' (USD per million tokens)."
        )
    if set(pricing.keys()) != set(_PRICING_KEYS):
        raise ValueError(
            "Azure Foundry 'pricing' must contain exactly the keys "
            f"{list(_PRICING_KEYS)}; got {sorted(pricing.keys())}."
        )
    validated: dict[str, float] = {}
    for key in _PRICING_KEYS:
        value = pricing[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Azure Foundry pricing['{key}'] must be a number; got {value!r}.")
        rate = float(value)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(
                f"Azure Foundry pricing['{key}'] must be strictly positive; got {value!r}. "
                "A zero or negative rate would silently disarm the spend breaker."
            )
        validated[key] = rate
    return validated


def _validate_endpoint(endpoint: str) -> None:
    """Reject non-https or non-Azure endpoints. Error messages deliberately
    name the env variable rather than echoing the endpoint value."""
    parsed = urlparse(endpoint)
    if parsed.scheme != "https":
        raise ValueError(
            "AZURE_FOUNDRY_ENDPOINT must be an https:// URL; refusing to construct a client."
        )
    hostname = parsed.hostname or ""
    if not hostname.endswith(_ALLOWED_HOST_SUFFIXES):
        raise ValueError(
            "AZURE_FOUNDRY_ENDPOINT host must end with '.services.ai.azure.com' or "
            "'.openai.azure.com'; refusing to construct a client."
        )


def _extract_provider_cost(usage: Any) -> float | None:
    """Mirror OpenAIClient._track_cost's provider-cost extraction so the value
    can be validated before the parent silently drops a bad one."""
    cost = None
    if hasattr(usage, "cost") and usage.cost:
        cost = usage.cost
    elif hasattr(usage, "model_extra") and usage.model_extra:
        extra = usage.model_extra
        if extra.get("cost"):
            cost = extra["cost"]
        elif extra.get("cost_details", {}).get("upstream_inference_cost"):
            cost = extra["cost_details"]["upstream_inference_cost"]
    return cost


class AzureFoundryClient(OpenAIClient):
    """
    LM Client for models served from Azure AI Foundry over the
    OpenAI-compatible ``/openai/v1`` route.

    Inherits ``completion``/``acompletion`` from :class:`OpenAIClient` so the
    full ``sampling_args`` fidelity is preserved (identical sampling args at
    root and sub-calls). Every call reports a cost: provider-reported when
    present, otherwise synthesized client-side from token counts multiplied by
    the required ``pricing`` (USD per million tokens).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        endpoint: str | None = None,
        sampling_args: dict[str, Any] | None = None,
        pricing: dict[str, float] | None = None,
        **kwargs,
    ):
        if api_key is None:
            api_key = os.getenv("AZURE_API_KEY", DEFAULT_AZURE_API_KEY)
        if api_key is None:
            raise ValueError(
                "AZURE_API_KEY is not set. The azure_foundry backend requires it "
                "(or pass api_key explicitly)."
            )

        if endpoint is None:
            endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT", DEFAULT_AZURE_FOUNDRY_ENDPOINT)
        if endpoint is None:
            raise ValueError(
                "AZURE_FOUNDRY_ENDPOINT is not set. The azure_foundry backend requires it "
                "(or pass endpoint explicitly)."
            )

        # Validate before any API client is constructed.
        _validate_endpoint(endpoint)
        self.pricing: dict[str, float] = _validate_pricing(pricing)

        base_url = endpoint.rstrip("/") + "/openai/v1/"
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            sampling_args=sampling_args,
            **kwargs,
        )

        # Where the last call's cost came from ("provider" or "synthesized"),
        # and whether any call so far had a provider-reported cost.
        self.last_cost_source: str | None = None
        self._provider_cost_seen: bool = False

    def _validate_response(self, response: openai.ChatCompletion) -> None:
        """Fail loud on filtered or empty responses instead of returning junk."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("Azure Foundry returned a response with no choices.")
        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise RuntimeError(
                "Azure Foundry blocked this response via content filter "
                "(finish_reason='content_filter')."
            )
        content = getattr(getattr(choice, "message", None), "content", None)
        if content is None or content == "":
            raise RuntimeError("Azure Foundry returned empty content for this response.")

    def _validate_usage(self, usage: Any) -> None:
        """A paid call must never count as free: token counts must be present,
        integral, and non-negative; any provider-reported cost must be a
        finite, non-negative number."""
        for field in ("prompt_tokens", "completion_tokens"):
            value = getattr(usage, field, None)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Azure Foundry usage is missing a valid '{field}' token count "
                    f"(got {value!r}). Refusing to record this call as free."
                )
        cost = _extract_provider_cost(usage)
        if cost is not None:
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                raise ValueError(
                    f"Azure Foundry provider-reported cost is not a number (got {cost!r})."
                )
            if not math.isfinite(float(cost)) or float(cost) < 0:
                raise ValueError(
                    f"Azure Foundry provider-reported cost is non-finite or negative "
                    f"(got {cost!r}). Refusing to record this call."
                )

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        self._validate_response(response)

        usage = getattr(response, "usage", None)
        if usage is not None:
            # Validate before the parent accumulates anything.
            self._validate_usage(usage)

        # Parent raises when usage is None, accumulates token counts, and sets
        # last_cost only when the provider reported a positive cost.
        super()._track_cost(response, model)

        if self.last_cost is not None:
            self.last_cost_source = "provider"
            self._provider_cost_seen = True
        else:
            synthesized = (
                self.last_prompt_tokens * self.pricing["input_per_million"]
                + self.last_completion_tokens * self.pricing["output_per_million"]
            ) / 1e6
            self.last_cost = synthesized
            self.model_costs[model] += synthesized
            self.last_cost_source = "synthesized"

    def get_usage_summary(self) -> UsageSummary:
        summary = super().get_usage_summary()
        if summary.model_usage_summaries:
            source = "provider" if self._provider_cost_seen else "synthesized"
            for model_summary in summary.model_usage_summaries.values():
                model_summary.cost_source = source
        return summary

    def get_last_usage(self) -> ModelUsageSummary:
        usage = super().get_last_usage()
        usage.cost_source = self.last_cost_source
        return usage
