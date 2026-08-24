import math
import os
import threading
from typing import Any
from urllib.parse import urlparse

import openai
from dotenv import load_dotenv

from rlm.clients.openai import OpenAIClient, extract_provider_cost
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()

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


class AzureFoundryClient(OpenAIClient):
    """
    LM Client for models served from Azure AI Foundry over the
    OpenAI-compatible ``/openai/v1`` route.

    Inherits ``completion``/``acompletion`` from :class:`OpenAIClient` so the
    full ``sampling_args`` fidelity is preserved (identical sampling args at
    root and sub-calls). Every call reports a cost: provider-reported when
    present, otherwise synthesized client-side from token counts multiplied by
    the required ``pricing`` (USD per million tokens).

    Cost tracking is ordered so paid spend is never lost: usage integrity is
    validated first (malformed token counts mean there is nothing valid to
    record), then tokens and cost are accumulated, and only THEN is the
    response content validated -- a billed response that was content-filtered
    or came back empty still raises, but its spend has already been recorded,
    so the spend breaker sees it. The whole sequence runs under a lock because
    one client instance is shared across sub-call handler threads.
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
        api_key = api_key or os.getenv("AZURE_API_KEY")
        if api_key is None:
            raise ValueError(
                "AZURE_API_KEY is not set. The azure_foundry backend requires it "
                "(or pass api_key explicitly)."
            )

        endpoint = endpoint or os.getenv("AZURE_FOUNDRY_ENDPOINT")
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
        # and each model's accumulated source ("synthesized" wins once any of
        # that model's calls was synthesized, matching UsageSummary.cost_source).
        self.last_cost_source: str | None = None
        self.model_cost_sources: dict[str, str] = {}

        # One client instance is shared across sub-call handler threads (the
        # ThreadingTCPServer in rlm/core/lm_handler.py); synthesis reads the
        # last_* fields the parent just set, so the whole track-cost sequence
        # must be atomic per call.
        self._cost_lock = threading.Lock()

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
        cost = extract_provider_cost(usage)
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
        with self._cost_lock:
            usage = getattr(response, "usage", None)
            if usage is not None:
                # Malformed or missing token counts leave nothing valid to
                # record, so this raises before any accumulation.
                self._validate_usage(usage)

            # Parent raises when usage is None, accumulates token counts, and
            # sets last_cost whenever the provider reported a non-negative
            # cost (including an explicit zero, for free-tier backends).
            super()._track_cost(response, model)

            # Azure Foundry is a paid backend: a provider-reported zero here
            # is not credible, so anything non-positive falls through to the
            # synthesized [pricing.list_price] cost -- a paid call must never
            # count as free. The parent accumulated the bogus zero into
            # model_costs, which adds nothing to the total.
            if self.last_cost is not None and self.last_cost > 0:
                self.last_cost_source = "provider"
                self.model_cost_sources.setdefault(model, "provider")
            else:
                synthesized = (
                    self.last_prompt_tokens * self.pricing["input_per_million"]
                    + self.last_completion_tokens * self.pricing["output_per_million"]
                ) / 1e6
                self.last_cost = synthesized
                self.model_costs[model] += synthesized
                self.last_cost_source = "synthesized"
                self.model_cost_sources[model] = "synthesized"

            # LAST, after the spend is on the books: a billed response that was
            # content-filtered or empty still raises, but the spend breaker has
            # already seen its cost.
            self._validate_response(response)

    def get_usage_summary(self) -> UsageSummary:
        summary = super().get_usage_summary()
        for model, model_summary in summary.model_usage_summaries.items():
            model_summary.cost_source = self.model_cost_sources.get(model)
        return summary

    def get_last_usage(self) -> ModelUsageSummary:
        usage = super().get_last_usage()
        usage.cost_source = self.last_cost_source
        return usage
