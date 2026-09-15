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
from rlm.utils.exceptions import TokenLimitExceededError

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

# Transient refusals -- 429s, 5xx provider errors, connection drops -- are
# different from a deficient 200: the provider did not complete the call, so
# nothing was executed and (unlike a deficient body) nothing of unknown cost
# was plausibly spent -- retrying is safe. The budget here is therefore more
# generous than the transport retries above, sized for shared-pool upstream
# throttling and flapping providers (observed on OpenRouter stealth models:
# both 429 storms and intermittent 502s, where the SDK's own 2 internal
# retries are nowhere near enough). Still bounded: a client embedded in a
# host with no deadline of its own must not spin forever against a provider
# that is down.
RATE_LIMIT_ATTEMPTS = 20
_RATE_LIMIT_BACKOFF_BASE_SECONDS = 2.0
_RATE_LIMIT_BACKOFF_CAP_SECONDS = 60.0
# openai.APIConnectionError also covers APITimeoutError (its subclass).
_TRANSIENT_API_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)

# An intermittent empty completion body is a provider glitch, not a model
# answer. Verified against Azure Foundry / Kimi-K2.5 on 2026-08-25: replaying a
# prompt that had failed this way returned content on three consecutive
# attempts with finish_reason='stop', using 1,679-2,914 of 8,192 available
# tokens -- so the empty body was neither truncation nor a property of the
# prompt. Subclasses whose _track_cost treats an empty body as fatal opt in by
# overriding _empty_content_retry_reason.
EMPTY_CONTENT_ATTEMPTS = 6

# A content filter fires on the SAMPLED RESPONSE, so it is probabilistic, not a
# property of the prompt. Verified against Azure Foundry / Kimi-K2.5 on
# 2026-08-26: one mining instance was blocked by label 'Jailbreak' on two
# consecutive attempts (different request ids, 930 and 954 prompt tokens) and
# then completed normally on the third. Treating the first block as fatal cost
# the whole experiment a crash and two restart cycles for a run that was always
# going to succeed. A prompt that trips the filter DETERMINISTICALLY still
# exhausts these attempts and raises, and the caller decides what that means.
CONTENT_FILTER_ATTEMPTS = 6
_CONTENT_FILTER_MARKERS = ("content_filter", "responsibleai", "content management policy")


def _transport_backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff: uniform in (0, min(cap, base * 2^(n-1)))."""
    ceiling = min(
        _TRANSPORT_BACKOFF_CAP_SECONDS, _TRANSPORT_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
    )
    return random.uniform(0, ceiling)


def _rate_limit_backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff for 429s, on the rate-limit constants."""
    ceiling = min(
        _RATE_LIMIT_BACKOFF_CAP_SECONDS, _RATE_LIMIT_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
    )
    return random.uniform(0, ceiling)


# Provider phrasings for "this prompt does not fit the model's context
# window" inside an HTTP 400 body (OpenRouter relays the upstream text
# verbatim; observed from Nebius/CoreWeave, SiliconFlow, StreamLake, Alibaba).
# A prompt that outgrew the window is a run-level resource condition, not a
# transport fault: it is mapped to TokenLimitExceededError so the driver
# records the run as RESOURCE_TERMINATED instead of crashing the experiment.
_CONTEXT_OVERFLOW_MARKERS = (
    "maximum context length",
    "max input length",
    "max_seq_len",
    "range of input length",
    "input tokens",
    "context length",
    "context_length_exceeded",
)


def _context_overflow_error(exc: openai.BadRequestError) -> TokenLimitExceededError | None:
    """A TokenLimitExceededError when this 400 is a context-window overflow."""
    text = str(exc).lower()
    if not any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS):
        return None
    return TokenLimitExceededError(
        tokens_used=0,
        token_limit=0,
        message=f"Prompt exceeded the model's context window (provider 400): {str(exc)[:600]}",
    )


def is_content_filter_error(exc: openai.BadRequestError) -> bool:
    """Whether this 400 is a provider content-filter block rather than a bad request."""
    return any(marker in str(exc).lower() for marker in _CONTENT_FILTER_MARKERS)


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

        transport_attempt = 0
        rate_limit_attempt = 0
        empty_content_attempt = 0
        content_filter_attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body=extra_body,
                    **_normalize_sampling_args(self.sampling_args),
                )
            except openai.BadRequestError as exc:
                overflow = _context_overflow_error(exc)
                if overflow is not None:
                    raise overflow from exc
                if is_content_filter_error(exc):
                    content_filter_attempt += 1
                    if content_filter_attempt >= CONTENT_FILTER_ATTEMPTS:
                        raise
                    print(
                        f"Content filter blocked the response; "
                        f"retrying ({content_filter_attempt}/{CONTENT_FILTER_ATTEMPTS})...",
                        file=sys.stderr,
                    )
                    time.sleep(_transport_backoff_seconds(content_filter_attempt))
                    continue
                raise
            except _TRANSIENT_API_ERRORS as exc:
                rate_limit_attempt += 1
                if rate_limit_attempt >= RATE_LIMIT_ATTEMPTS:
                    raise
                print(
                    f"Transient API error ({type(exc).__name__}); "
                    f"retrying ({rate_limit_attempt}/{RATE_LIMIT_ATTEMPTS})...",
                    file=sys.stderr,
                )
                time.sleep(_rate_limit_backoff_seconds(rate_limit_attempt))
                continue
            deficiency = _response_deficiency(response)
            if deficiency is None:
                empty_reason = self._empty_content_retry_reason(response)
                if empty_reason is None:
                    break
                empty_content_attempt += 1
                if empty_content_attempt >= EMPTY_CONTENT_ATTEMPTS:
                    # Exhausted: fall through WITHOUT banking here, so the
                    # post-loop _track_cost records this attempt exactly once
                    # and raises, preserving the pre-retry behavior.
                    break
                # A discarded attempt was still billed: bank it before retrying.
                self._record_spend(response, model)
                print(
                    f"Empty completion content ({empty_reason}); "
                    f"retrying ({empty_content_attempt}/{EMPTY_CONTENT_ATTEMPTS})...",
                    file=sys.stderr,
                )
                time.sleep(_transport_backoff_seconds(empty_content_attempt))
                continue
            transport_attempt += 1
            if transport_attempt >= TRANSPORT_ATTEMPTS:
                raise ValueError(
                    f"Deficient completion response after {TRANSPORT_ATTEMPTS} attempts "
                    f"({deficiency}). Tracking tokens not possible."
                )
            print(
                f"Deficient completion response ({deficiency}); "
                f"retrying ({transport_attempt}/{TRANSPORT_ATTEMPTS})...",
                file=sys.stderr,
            )
            time.sleep(_transport_backoff_seconds(transport_attempt))
        self._track_cost(response, model)
        # content is None when the model emitted no visible text -- reasoning
        # models can spend the whole max_tokens budget on hidden reasoning
        # (observed on OpenRouter stealth/ox-alpha). The declared return type
        # is str, and callers regex/parse it, so an absent text is "".
        return self._normalize_content(response.choices[0].message.content or "")

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

        transport_attempt = 0
        rate_limit_attempt = 0
        empty_content_attempt = 0
        content_filter_attempt = 0
        while True:
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body=extra_body,
                    **_normalize_sampling_args(self.sampling_args),
                )
            except openai.BadRequestError as exc:
                overflow = _context_overflow_error(exc)
                if overflow is not None:
                    raise overflow from exc
                if is_content_filter_error(exc):
                    content_filter_attempt += 1
                    if content_filter_attempt >= CONTENT_FILTER_ATTEMPTS:
                        raise
                    print(
                        f"Content filter blocked the response; "
                        f"retrying ({content_filter_attempt}/{CONTENT_FILTER_ATTEMPTS})...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(_transport_backoff_seconds(content_filter_attempt))
                    continue
                raise
            except _TRANSIENT_API_ERRORS as exc:
                rate_limit_attempt += 1
                if rate_limit_attempt >= RATE_LIMIT_ATTEMPTS:
                    raise
                print(
                    f"Transient API error ({type(exc).__name__}); "
                    f"retrying ({rate_limit_attempt}/{RATE_LIMIT_ATTEMPTS})...",
                    file=sys.stderr,
                )
                await asyncio.sleep(_rate_limit_backoff_seconds(rate_limit_attempt))
                continue
            deficiency = _response_deficiency(response)
            if deficiency is None:
                empty_reason = self._empty_content_retry_reason(response)
                if empty_reason is None:
                    break
                empty_content_attempt += 1
                if empty_content_attempt >= EMPTY_CONTENT_ATTEMPTS:
                    # Exhausted: fall through WITHOUT banking here, so the
                    # post-loop _track_cost records this attempt exactly once
                    # and raises, preserving the pre-retry behavior.
                    break
                # A discarded attempt was still billed: bank it before retrying.
                self._record_spend(response, model)
                print(
                    f"Empty completion content ({empty_reason}); "
                    f"retrying ({empty_content_attempt}/{EMPTY_CONTENT_ATTEMPTS})...",
                    file=sys.stderr,
                )
                await asyncio.sleep(_transport_backoff_seconds(empty_content_attempt))
                continue
            transport_attempt += 1
            if transport_attempt >= TRANSPORT_ATTEMPTS:
                raise ValueError(
                    f"Deficient completion response after {TRANSPORT_ATTEMPTS} attempts "
                    f"({deficiency}). Tracking tokens not possible."
                )
            print(
                f"Deficient completion response ({deficiency}); "
                f"retrying ({transport_attempt}/{TRANSPORT_ATTEMPTS})...",
                file=sys.stderr,
            )
            await asyncio.sleep(_transport_backoff_seconds(transport_attempt))
        self._track_cost(response, model)
        # Same None-content coercion as the sync path above.
        return self._normalize_content(response.choices[0].message.content or "")

    def _normalize_content(self, content: str) -> str:
        """Provider-specific repair of the returned text; identity here.

        The one sanctioned place to rewrite what the model said before the
        harness parses it. A subclass overrides it for a provider whose model
        leaks a non-text serialization into ``content`` (see
        ``AzureFoundryClient``); everything else returns the text unchanged.
        """
        return content

    def _empty_content_retry_reason(self, response: Any) -> str | None:
        """Why this 200 response should be retried for an empty body, or None.

        The base client coerces an absent text to "" -- a reasoning model may
        legitimately spend its whole budget on hidden reasoning -- so nothing is
        retried here. Subclasses whose _track_cost RAISES on an empty body
        override this, which moves the retry inside the loop where the discarded
        attempt's spend can still be recorded.
        """
        return None

    def _record_spend(self, response: openai.ChatCompletion, model: str) -> None:
        """Bank one attempt's spend without validating its content.

        Called for a billed attempt that is about to be discarded and retried:
        the money left the account, so the breaker must see it.
        """
        self._track_cost(response, model)

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
