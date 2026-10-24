import os
import re
from typing import Any

from rlm.clients.openai import OpenAIClient

# A self-hosted base_url is arbitrary (user-configured port/host), unlike
# OpenAIClient's fixed set of well-known hosted endpoints, so it cannot be
# matched by exact base_url the way DEFAULT_OPENROUTER_API_KEY etc. are.
# Read once at import time, matching the module-level DEFAULT_* pattern in
# rlm.clients.openai.
DEFAULT_VLLM_API_KEY = os.getenv("VLLM_API_KEY")

# Matches a ```python or ```py fence opener (not ```pythonic or other
# look-alikes, via the \b boundary) and rewrites it to ```repl so
# rlm.utils.parsing.find_code_blocks recognizes the block. Local/self-hosted
# models served behind an OpenAI-compatible endpoint commonly default to the
# conventional ```python fence instead of this harness's ```repl convention --
# confirmed against Qwen3.6-35B-A3B, where 0 of 10 turns' code blocks executed
# with the raw response until this rewrite was applied.
_PYTHON_FENCE_RE = re.compile(r"```py(?:thon)?\b[ \t]*\n")


def rewrite_python_fence_to_repl(content: str) -> str:
    """Rewrite a leaked ```python / ```py fence opener to ```repl.

    A cheap substring check avoids the regex entirely on the common case
    (fence already correct, or no code block at all).
    """
    if "```py" not in content:
        return content
    return _PYTHON_FENCE_RE.sub("```repl\n", content)


class VLLMClient(OpenAIClient):
    """OpenAIClient for self-hosted OpenAI-compatible servers (vLLM, Ollama, LM Studio, ...).

    Identical to OpenAIClient except it rewrites a leaked ```python/```py
    fence opener to ```repl before find_code_blocks ever sees the response --
    see rewrite_python_fence_to_repl -- and defaults ``api_key`` from
    ``VLLM_API_KEY`` when the caller omits it. The governed shrlm pipeline
    never passes ``api_key`` as a kwarg (``reject_sensitive_backend_kwargs``
    refuses anything with "key" in its name, since kwargs are persisted
    verbatim into every trace), so this is the only way shrlm's vllm role
    ever supplies a credential; a direct ``RLM(backend="vllm", ...)`` caller
    may still pass ``api_key`` explicitly and this default is skipped.
    """

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        super().__init__(api_key=api_key if api_key is not None else DEFAULT_VLLM_API_KEY, **kwargs)

    def _normalize_content(self, content: str) -> str:
        return rewrite_python_fence_to_repl(content)
