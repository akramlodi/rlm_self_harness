import asyncio
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any

import openai
from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()

# Load API keys from environment variables
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_VERCEL_API_KEY = os.getenv("AI_GATEWAY_API_KEY")
DEFAULT_PRIME_API_KEY = os.getenv("PRIME_API_KEY")
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"


def _normalize_sampling_args(sampling_args: dict[str, Any]) -> dict[str, Any]:
    """Match the rename done by verifiers' OpenAIChatCompletionsClient so the
    same sampling_args dict produces byte-equivalent chat.completions.create
    calls in both harnesses. Pops ``extra_body`` so the caller can merge it
    with its own ``extra_body`` rather than passing it twice (TypeError).
    """
    args = dict(sampling_args or {})
    if "max_tokens" in args:
        args["max_completion_tokens"] = args.pop("max_tokens")
    args.pop("extra_body", None)
    return {k: v for k, v in args.items() if v is not None}


# A provider can return HTTP 200 with a deficient body -- no usage block, no
# choices, sometimes an embedded error payload (seen intermittently on
# OpenRouter when an upstream provider fails). The OpenAI SDK retries only
# HTTP-level failures, so the client retries these itself with exponential
# backoff and full jitter. Retries stay BOUNDED because each deficient
# response may still be a paid call of unknown cost that the spend breaker
# cannot see -- unbounded retries would be unbounded invisible spend. A call
# whose cost is unknowable is never recorded; after the retries run out the
# error is raised loudly with whatever the provider embedded.
TRANSPORT_ATTEMPTS = 6
_TRANSPORT_BACKOFF_BASE_SECONDS = 1.0
_TRANSPORT_BACKOFF_CAP_SECONDS = 30.0


def _transport_backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff: uniform in (0, min(cap, base * 2^(n-1)))."""
    ceiling = min(
        _TRANSPORT_BACKOFF_CAP_SECONDS, _TRANSPORT_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
    )
    return random.uniform(0, ceiling)


def _response_deficiency(response: Any) -> str | None:
    """Why this 200 response cannot be used, or None when it is usable."""
    problems = []
    if getattr(response, "usage", None) is None:
        problems.append("no usage data received")
    if not getattr(response, "choices", None):
        problems.append("no choices")
    if not problems:
        return None
    error = getattr(response, "error", None)
    if error is None and getattr(response, "model_extra", None):
        error = response.model_extra.get("error")
    if error:
        problems.append(f"provider error payload: {str(error)[:300]}")
    return "; ".join(problems)


def extract_provider_cost(usage: Any) -> Any:
    """The provider-reported cost on a usage block, or None.

    OpenRouter surfaces cost either as a direct ``usage.cost`` attribute or
    inside ``usage.model_extra`` (``cost``, then
    ``cost_details.upstream_inference_cost`` for BYOK routes). Shared with
    subclasses that validate the value before recording it.

    An explicit ``0`` is a report, not an absence: free-tier OpenRouter models
    bill $0 per call, and a breaker fed ``None`` for those runs refuses to
    count them at all (``costs.breaker_run_cost``). Only a missing field
    returns None.
    """
    if getattr(usage, "cost", None) is not None:
        return usage.cost
    if hasattr(usage, "model_extra") and usage.model_extra:
        extra = usage.model_extra
        if extra.get("cost") is not None:
            return extra["cost"]
        if extra.get("cost_details", {}).get("upstream_inference_cost") is not None:
            return extra["cost_details"]["upstream_inference_cost"]
    return None


def _merge_extra_body(
    hardcoded: dict[str, Any], sampling_args: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge an ``extra_body`` from sampling_args into the hardcoded extra_body."""
    merged = dict(hardcoded or {})
    user = (sampling_args or {}).get("extra_body")
    if user:
        merged.update(user)
    return merged


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM as well.

    Any additional keyword arguments (e.g. default_headers, default_query, max_retries)
    are passed through to the underlying openai.OpenAI and openai.AsyncOpenAI constructors.
    Only model_name is excluded, since it is not a client constructor argument.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        sampling_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, sampling_args=sampling_args, **kwargs)

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key = DEFAULT_OPENAI_API_KEY
            elif base_url == "https://openrouter.ai/api/v1":
                api_key = DEFAULT_OPENROUTER_API_KEY
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key = DEFAULT_VERCEL_API_KEY
            elif base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
                api_key = DEFAULT_PRIME_API_KEY

        # Pass through arbitrary kwargs to the OpenAI client (e.g. default_headers, default_query, max_retries).
        # Exclude model_name since it is not an OpenAI client constructor argument.
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            **{k: v for k, v in self.kwargs.items() if k != "model_name"},
        }
        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url  # Track for cost extraction

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_costs: dict[str, float] = defaultdict(float)  # Cost in USD

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body: dict[str, Any] = {}
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}
        extra_body = _merge_extra_body(extra_body, self.sampling_args)

        for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                **_normalize_sampling_args(self.sampling_args),
            )
            deficiency = _response_deficiency(response)
            if deficiency is None:
                break
            if attempt == TRANSPORT_ATTEMPTS:
                raise ValueError(
                    f"Deficient completion response after {TRANSPORT_ATTEMPTS} attempts "
                    f"({deficiency}). Tracking tokens not possible."
                )
            print(
                f"Deficient completion response ({deficiency}); "
                f"retrying ({attempt}/{TRANSPORT_ATTEMPTS})...",
                file=sys.stderr,
            )
            time.sleep(_transport_backoff_seconds(attempt))
        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body: dict[str, Any] = {}
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}
        extra_body = _merge_extra_body(extra_body, self.sampling_args)

        for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                **_normalize_sampling_args(self.sampling_args),
            )
            deficiency = _response_deficiency(response)
            if deficiency is None:
                break
            if attempt == TRANSPORT_ATTEMPTS:
                raise ValueError(
                    f"Deficient completion response after {TRANSPORT_ATTEMPTS} attempts "
                    f"({deficiency}). Tracking tokens not possible."
                )
            print(
                f"Deficient completion response ({deficiency}); "
                f"retrying ({attempt}/{TRANSPORT_ATTEMPTS})...",
                file=sys.stderr,
            )
            await asyncio.sleep(_transport_backoff_seconds(attempt))
        self._track_cost(response, model)
        return response.choices[0].message.content

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")

        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens

        # Extract cost from OpenRouter responses (cost is in USD). A reported
        # zero (free-tier model) is recorded as 0.0 so the spend breaker sees
        # a cost-reporting backend; negative values are provider junk and are
        # dropped like an absent field.
        self.last_cost: float | None = None
        cost = extract_provider_cost(usage)

        if cost is not None and cost >= 0:
            self.last_cost = float(cost)
            self.model_costs[model] += self.last_cost

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            # .get() distinguishes "never reported" (absent key -> None) from
            # an accumulated $0 total on a free-tier model (0.0 stays 0.0).
            cost = self.model_costs.get(model)
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=cost,
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=getattr(self, "last_cost", None),
        )
