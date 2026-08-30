import json
import math
import os
import re
import threading
from typing import Any
from urllib.parse import urlparse

import openai
from dotenv import load_dotenv

from rlm.clients.openai import OpenAIClient, extract_provider_cost
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.utils.exceptions import TokenLimitExceededError

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


# Kimi-K2.5's native function-calling serialization. When the model decides to
# "call" the REPL as a tool instead of writing a ```repl``` block, the Foundry
# route returns these tokens as plain text in ``message.content`` -- no tools
# were declared, so nothing parses them, the block never executes, and the
# model repeats the same call until the iteration budget is gone (14 of 240
# mining runs in experiment_kimi, 0% pass; POST_MORTEM.md section 9.1). The
# intent is unambiguous -- ``functions.repl`` with a ``code`` argument IS a
# REPL block -- so it is translated back into one rather than dropped.
_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<name>[\w.]+?)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>(?P<args>.*?)<\|tool_call_end\|>",
    re.DOTALL,
)
_TOOL_SECTION_RE = re.compile(r"<\|tool_calls_section_(?:begin|end)\|>")
NATIVE_TOOL_CALL_MARKER = "<|tool_call"


def translate_native_tool_calls(content: str) -> str:
    """Rewrite leaked ``functions.repl`` tool calls as ```repl``` blocks.

    Only a call named ``repl`` (any ``functions.`` prefix) whose JSON argument
    object carries a string ``code`` is translated; any other call, or one
    whose arguments do not parse, is left byte-for-byte so the leak stays
    visible in the trace rather than being silently swallowed. Text without
    the marker is returned unchanged.
    """
    if NATIVE_TOOL_CALL_MARKER not in content:
        return content

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name").rsplit(".", 1)[-1]
        if name != "repl":
            return match.group(0)
        try:
            arguments = json.loads(match.group("args"))
        except json.JSONDecodeError:
            return match.group(0)
        code = arguments.get("code") if isinstance(arguments, dict) else None
        if not isinstance(code, str):
            return match.group(0)
        return f"```repl\n{code}\n```"

    translated = _TOOL_CALL_RE.sub(_replace, content)
    if translated == content:
        return content
    return _TOOL_SECTION_RE.sub("", translated).strip()


# gpt-oss's harmony response format. The Foundry chat route is expected to
# unwrap it, but a leak into ``message.content`` would hand the REPL parser
# reasoning text and control tokens (KTD4). With channel structure present
# only the ``final`` channel's message body is the answer; ``analysis`` and
# ``commentary`` bodies are reasoning and are dropped with their markers.
# Bare control tokens without channel structure are simply stripped. Every
# dropped marker is counted on the client, never raised: raising would escape
# ``execute_run`` as a non-limit exception and take the experiment down.
HARMONY_CHANNEL_MARKER = "<|channel|>"
_HARMONY_CONTROL_TOKENS = (
    HARMONY_CHANNEL_MARKER,
    "<|message|>",
    "<|call|>",
    "<|return|>",
    "<|start|>",
    "<|end|>",
    "<|constrain|>",
)
_HARMONY_CONTROL_RE = re.compile("|".join(re.escape(token) for token in _HARMONY_CONTROL_TOKENS))
# One channel: ``<|channel|>NAME[<|constrain|>TYPE]<|message|>BODY`` terminated
# by ``<|end|>``, ``<|return|>``, ``<|call|>``, the next ``<|start|>``, or the end
# of the text. Any ``<|start|>ROLE`` header preceding the channel is part of it.
_HARMONY_CHANNEL_RE = re.compile(
    r"(?:<\|start\|>[^<]*)?<\|channel\|>(?P<name>[^<]*?)"
    r"(?:<\|constrain\|>[^<]*)?<\|message\|>"
    r"(?P<body>.*?)"
    r"(?:<\|end\|>|<\|return\|>|<\|call\|>|(?=<\|start\|>)|\Z)",
    re.DOTALL,
)
HARMONY_FINAL_CHANNEL = "final"


def _count_control_tokens(text: str) -> int:
    return len(_HARMONY_CONTROL_RE.findall(text))


def strip_harmony_markers(content: str) -> tuple[str, int]:
    """Drop harmony control tokens from ``content``; return the text and the
    number of markers dropped.

    With channel structure (``<|channel|>`` present) only the ``final``
    channel's message body is kept; every other channel's body is dropped
    together with its markers, so reasoning never reaches the parser. Without
    channel structure the bare control tokens are stripped and the text
    around them is kept. Text carrying no marker is returned as-is (the same
    object) with a count of zero.
    """
    if HARMONY_CHANNEL_MARKER not in content:
        if _HARMONY_CONTROL_RE.search(content) is None:
            return content, 0
        stripped, dropped = _HARMONY_CONTROL_RE.subn("", content)
        return stripped, dropped

    total = _count_control_tokens(content)
    kept: list[str] = []
    cursor = 0
    for match in _HARMONY_CHANNEL_RE.finditer(content):
        # Text between channels is not a message body; keep it minus markers.
        between = content[cursor : match.start()]
        kept.append(_HARMONY_CONTROL_RE.sub("", between))
        if match.group("name").strip() == HARMONY_FINAL_CHANNEL:
            kept.append(_HARMONY_CONTROL_RE.sub("", match.group("body")))
        cursor = match.end()
    kept.append(_HARMONY_CONTROL_RE.sub("", content[cursor:]))
    return "".join(kept).strip(), total


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

        # Harmony control tokens dropped from ``message.content`` over this
        # client's lifetime (see strip_harmony_markers). Process-local, not
        # persisted; incremented under the cost lock.
        self.harmony_markers_dropped: int = 0

    @staticmethod
    def _reasoning_exhausted(choice: Any, usage: Any) -> bool:
        """Whether an empty choice spent its whole output budget on reasoning.

        True when ``finish_reason == "length"``, the content is empty, and
        either ``usage.completion_tokens_details.reasoning_tokens`` accounts
        for at least 90% of ``completion_tokens`` or the detail block is not
        reported at all (Azure does not document it for gpt-oss chat
        completions; an empty body cut at the length cap is then the only
        signal). Takes the caller's already-extracted first choice and usage.
        Pure: the client is shared across threads, so no per-call state lives
        on the instance.
        """
        if getattr(choice, "finish_reason", None) != "length":
            return False
        content = getattr(getattr(choice, "message", None), "content", None)
        if content is not None and content != "":
            return False
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if reasoning_tokens is None:
            return True
        completion_tokens = getattr(usage, "completion_tokens", None)
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            return True
        return reasoning_tokens >= 0.9 * completion_tokens

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
        usage = getattr(response, "usage", None)
        if self._reasoning_exhausted(choice, usage):
            # A limit, not a glitch: surfaces as the same class the RLM's own
            # iteration cap raises, so the root run persists as
            # resource_terminated with the spend banked just above.
            completion_tokens = getattr(usage, "completion_tokens", None)
            tokens_used = completion_tokens if isinstance(completion_tokens, int) else 0
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            token_limit = self.sampling_args.get("max_tokens") or tokens_used
            raise TokenLimitExceededError(
                tokens_used=tokens_used,
                token_limit=token_limit,
                message=(
                    "Azure Foundry output budget exhausted by reasoning: empty content "
                    f"with finish_reason='length' ({tokens_used:,} completion tokens, "
                    f"reasoning_tokens={reasoning_tokens!r}, max_tokens={token_limit:,})."
                ),
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

    def _record_spend(self, response: openai.ChatCompletion, model: str) -> None:
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

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        # Spend first, validation LAST: a billed response that was
        # content-filtered or empty still raises, but the spend breaker has
        # already seen its cost.
        self._record_spend(response, model)
        self._validate_response(response)

    def _empty_content_retry_reason(self, response: Any) -> str | None:
        """Azure Foundry intermittently returns a 200 with an empty body.

        Measured 2026-08-25 against the live Kimi-K2.5 deployment: 9 of ~1,100
        runs died here, and replaying the smallest failing prompt returned
        content on three consecutive attempts (finish_reason='stop',
        1,679-2,914 completion tokens against an 8,192 budget). So the empty
        body is transient, and raising on the first one turns a provider glitch
        into a dead run -- while an ordinary 429 gets 20 attempts.

        A content filter is a verdict rather than a glitch, so it is left to
        _validate_response to raise; so is an empty body whose output budget
        went to reasoning (finish_reason='length'), which is deterministic for
        the prompt and would only be billed again on every re-send.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return None  # _response_deficiency already owns this case
        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            return None  # a deliberate refusal; retrying would only re-spend
        if self._reasoning_exhausted(choice, getattr(response, "usage", None)):
            return None  # a budget limit; _validate_response raises it
        content = getattr(getattr(choice, "message", None), "content", None)
        if content is None or content == "":
            return f"finish_reason={getattr(choice, 'finish_reason', None)!r}"
        return None

    def _normalize_content(self, content: str) -> str:
        """Kimi tool-call translation first, then harmony marker stripping, so
        a ``final`` channel wrapping a leaked tool call still yields the fenced
        block. Runs for ``completion`` and ``acompletion`` alike."""
        translated = translate_native_tool_calls(content)
        stripped, dropped = strip_harmony_markers(translated)
        if dropped:
            with self._cost_lock:
                self.harmony_markers_dropped += dropped
        return stripped

    def get_usage_summary(self) -> UsageSummary:
        summary = super().get_usage_summary()
        for model, model_summary in summary.model_usage_summaries.items():
            model_summary.cost_source = self.model_cost_sources.get(model)
        return summary

    def get_last_usage(self) -> ModelUsageSummary:
        usage = super().get_last_usage()
        usage.cost_source = self.last_cost_source
        return usage
