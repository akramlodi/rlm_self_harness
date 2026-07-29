import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import (
    RESERVED_TOOL_NAMES,
    NonIsolatedEnv,
    extract_tool_value,
    validate_custom_tools,
)
from rlm.utils.parsing import has_syntax_error

# =============================================================================
# S6 runtime-policy vocabulary
# =============================================================================

# Both batched sub-call helpers return ``list[str]``, so an over-budget call is
# refused in-band with a parseable prefix the root model can read and branch on.
SUBCALL_REFUSAL_PREFIX = "SUBCALL_REFUSED:"
# A sub-output rejected by the configured validator is marked, not dropped.
SUBCALL_INVALID_PREFIX = "SUBCALL_INVALID:"


class _AnswerDict(dict):
    """REPL-visible dict where ``answer["ready"] = True`` signals completion.

    Behaves exactly like ``dict`` for the model, but invokes ``on_ready`` the
    first time ``ready`` flips truthy. The callback receives the current
    ``content``, lets the env capture it (in-process attr, broker push, etc.),
    and the next ``execute_code`` will surface it as ``REPLResult.final_answer``.
    """

    def __init__(self, on_ready=None):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            try:
                self._on_ready(self.get("content", ""))
            except Exception:
                pass


# =============================================================================
# Safe Builtins
# =============================================================================

# Safe builtins - blocks dangerous operations like eval/exec/input
_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.
    Executes code in a sandboxed namespace with access to context data.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        subcall_fn: Callable[[str, str | None], RLMChatCompletion] | None = None,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        max_concurrent_subcalls: int = 4,
        runtime_policy: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            persistent=persistent,
            depth=depth,
            max_concurrent_subcalls=max_concurrent_subcalls,
            **kwargs,
        )

        # S6 runtime policy. ``None`` or ``{"enabled": False}`` means every seam
        # below is a pass-through and the environment behaves exactly as shipped.
        self.runtime_policy: dict[str, Any] = runtime_policy or {}
        self.lm_handler_address = lm_handler_address
        self.subcall_fn = subcall_fn  # Callback for recursive RLM calls (depth > 1 support)
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0
        self.compaction = compaction

        # Custom tools: functions available in the REPL
        self.custom_tools = custom_tools or {}
        # Sub-tools: inherited from custom_tools if not specified
        self.custom_sub_tools = (
            custom_sub_tools if custom_sub_tools is not None else self.custom_tools
        )

        # Validate custom tools don't override reserved names
        validate_custom_tools(self.custom_tools)

        # Setup globals, locals, and modules in environment.
        self.setup()

        if compaction:
            self._compaction_history: list[Any] = []
            self.locals["history"] = self._compaction_history

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Setup the environment."""
        # Create sandboxed globals
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[RLMChatCompletion] = []
        # Captured the first time the model sets ``answer["ready"] = True``.
        self._last_final_answer: str | None = None

        # Add helper functions
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched

        # The model marks completion via ``answer["ready"] = True``; the
        # custom dict captures the content as soon as that happens so we
        # don't have to probe the namespace after every cell.
        self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)

        # Add custom tools to globals
        # Tools can be either plain values or (value, description) tuples
        for name, entry in self.custom_tools.items():
            value = extract_tool_value(entry)
            if callable(value):
                self.globals[name] = value
            else:
                # For non-callable values (constants, data), add to locals
                self.locals[name] = value

    def _capture_answer(self, content: Any) -> None:
        self._last_final_answer = str(content)

    def reset_answer(self) -> None:
        """Reset the REPL-visible answer dict to ``{"content": "", "ready": False}``.

        Used by the S9 redirect branch. Clearing only ``_last_final_answer`` is
        not enough: the ``_AnswerDict`` in ``self.locals`` persists across turns
        and ``_restore_scaffold`` leaves it untouched, so the model's namespace
        would still read ``ready: True`` and it would conclude it had already
        submitted an answer.
        """
        self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)
        self._last_final_answer = None

    def _show_vars(self) -> str:
        """Show all available variables in the REPL environment."""
        available = {
            k: type(v).__name__
            for k, v in self.locals.items()
            if not k.startswith("_") and k != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    # -------------------------------------------------------------------
    # S6 runtime policy (retry, sub-output validation, hard caps)
    # -------------------------------------------------------------------

    @property
    def _policy_enabled(self) -> bool:
        """True when an S6 policy is configured and switched on."""
        return bool(self.runtime_policy.get("enabled"))

    def _retry_budget(self) -> int:
        """Number of extra attempts allowed for a syntax-classified sub-call error."""
        if not self._policy_enabled or not self.runtime_policy.get("retry_on_syntax_error"):
            return 0
        return int(self.runtime_policy.get("max_retries", 1))

    @staticmethod
    def _is_syntax_error(text: str) -> bool:
        """Classify a sub-call result string as a syntax error."""
        return text.startswith("Error:") and has_syntax_error(text)

    def _validate_sub_output(self, text: str) -> str:
        """Mark a sub-output invalid when the configured validator rejects it."""
        validator = self.runtime_policy.get("validate_sub_output") if self._policy_enabled else None
        if validator is None or validator(text):
            return text
        return f"{SUBCALL_INVALID_PREFIX} sub-output rejected by the configured validator"

    def _batch_width_refusal(self, prompts: list[str]) -> list[str] | None:
        """Whole-call refusal when a batch is wider than the configured cap."""
        if not self._policy_enabled:
            return None
        max_width = self.runtime_policy.get("max_batch_width")
        if max_width is None or len(prompts) <= max_width:
            return None
        refusal = (
            f"{SUBCALL_REFUSAL_PREFIX} batch width {len(prompts)} "
            f"exceeds max_batch_width {max_width}"
        )
        return [refusal] * len(prompts)

    def _prompt_chars_refusal(self, prompt: str) -> str | None:
        """Per-element refusal when one prompt is over the configured char ceiling."""
        if not self._policy_enabled:
            return None
        max_chars = self.runtime_policy.get("max_prompt_chars")
        if max_chars is None or len(prompt) <= max_chars:
            return None
        return (
            f"{SUBCALL_REFUSAL_PREFIX} prompt of {len(prompt)} chars "
            f"exceeds max_prompt_chars {max_chars}"
        )

    def _partition_by_prompt_cap(self, prompts: list[str]) -> tuple[list[int], list[str]]:
        """Split a batch into runnable indices and a pre-filled results list of refusals."""
        results: list[str] = [""] * len(prompts)
        allowed: list[int] = []
        for i, prompt in enumerate(prompts):
            refusal = self._prompt_chars_refusal(prompt)
            if refusal is None:
                allowed.append(i)
            else:
                results[i] = refusal
        return allowed, results

    def _record_call(
        self,
        completion: RLMChatCompletion,
        retries: int = 0,
        syntax_error: bool = False,
    ) -> None:
        """Attach per-call trace metrics and queue the completion for the turn record."""
        completion.trace_metrics = {
            "cost": completion.usage_summary.total_cost if completion.usage_summary else None,
            "retries": retries,
            "syntax_error": syntax_error,
        }
        self._pending_llm_calls.append(completion)

    def _call_with_retry(
        self,
        fn: Callable[[str, str | None], tuple[str, RLMChatCompletion | None]],
        prompt: str,
        model: str | None,
    ) -> tuple[str, RLMChatCompletion | None, int, bool]:
        """Run ``fn`` once, then retry while its result classifies as a syntax error."""
        text, completion = fn(prompt, model)
        saw_syntax_error = self._is_syntax_error(text)
        retries = 0
        budget = self._retry_budget()
        while self._is_syntax_error(text) and retries < budget:
            retries += 1
            text, completion = fn(prompt, model)
        return text, completion, retries, saw_syntax_error

    def _retry_batch_element(
        self,
        fn: Callable[[str, str | None], tuple[str, RLMChatCompletion | None]],
        prompt: str,
        model: str | None,
        current: str,
    ) -> tuple[str, RLMChatCompletion | None, int]:
        """Retry one already-attempted batch element that came back a syntax error."""
        text = current
        completion: RLMChatCompletion | None = None
        retries = 0
        budget = self._retry_budget()
        while self._is_syntax_error(text) and retries < budget:
            retries += 1
            text, completion = fn(prompt, model)
        return text, completion, retries

    # -------------------------------------------------------------------
    # Sub-call helpers exposed in the REPL
    # -------------------------------------------------------------------

    def _llm_query_once(
        self, prompt: str, model: str | None = None
    ) -> tuple[str, RLMChatCompletion | None]:
        """One plain LM call. Returns (result_text, completion_or_None)."""
        if not self.lm_handler_address:
            return "Error: No LM handler configured", None

        try:
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}", None

            return response.chat_completion.response, response.chat_completion
        except Exception as e:
            return f"Error: LM query failed - {e}", None

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """Query the LM with a single plain completion (no REPL, no recursion).

        This always makes a direct LM call via the handler, regardless of depth.
        Under a configured S6 policy, a syntax-classified error is retried and
        the result is passed through the sub-output validator.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
        """
        text, completion, retries, saw_syntax_error = self._call_with_retry(
            self._llm_query_once, prompt, model
        )
        if completion is not None:
            self._record_call(completion, retries=retries, syntax_error=saw_syntax_error)
        return self._validate_sub_output(text)

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Query the LM with multiple prompts concurrently (no REPL, no recursion).

        This always makes direct LM calls via the handler, regardless of depth.
        Under a configured S6 policy, an over-wide batch is refused in full and
        an over-long prompt is refused per element; both refusals are in-band
        strings prefixed with ``SUBCALL_REFUSED:``.

        Args:
            prompts: List of prompts to send to the LM.
            model: Optional model name to use (if handler has multiple clients).

        Returns:
            List of responses in the same order as input prompts.
        """
        width_refusal = self._batch_width_refusal(prompts)
        if width_refusal is not None:
            return width_refusal

        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)

        allowed, results = self._partition_by_prompt_cap(prompts)
        try:
            responses = send_lm_request_batched(
                self.lm_handler_address,
                [prompts[i] for i in allowed],
                model=model,
                depth=self.depth,
            )

            for index, response in zip(allowed, responses, strict=True):
                if not response.success:
                    results[index] = f"Error: {response.error}"
                else:
                    self._record_call(response.chat_completion)
                    results[index] = response.chat_completion.response
        except Exception as e:
            for index in allowed:
                results[index] = f"Error: LM query failed - {e}"
            return results

        return self._finalize_batch(self._llm_query_once, prompts, model, allowed, results)

    def _finalize_batch(
        self,
        fn: Callable[[str, str | None], tuple[str, RLMChatCompletion | None]],
        prompts: list[str],
        model: str | None,
        allowed: list[int],
        results: list[str],
    ) -> list[str]:
        """Apply the S6 per-element retry and sub-output validation to a batch."""
        if self._retry_budget():
            for index in allowed:
                if not self._is_syntax_error(results[index]):
                    continue
                text, completion, retries = self._retry_batch_element(
                    fn, prompts[index], model, results[index]
                )
                if completion is not None:
                    self._record_call(completion, retries=retries, syntax_error=True)
                results[index] = text

        for index in allowed:
            results[index] = self._validate_sub_output(results[index])
        return results

    def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        """Spawn a recursive RLM sub-call for deeper thinking on a subtask.

        When a subcall callback is available (max_depth > 1), this spawns a child
        RLM with its own REPL that can reason over the prompt iteratively.
        Falls back to a plain llm_query if no recursive capability is configured.

        Args:
            prompt: The prompt to send to the child RLM.
            model: Optional model name override for the child.
        """
        if self.subcall_fn is not None:
            text, completion, retries, saw_syntax_error = self._call_with_retry(
                self._rlm_query_once, prompt, model
            )
            if completion is not None:
                self._record_call(completion, retries=retries, syntax_error=saw_syntax_error)
            return self._validate_sub_output(text)

        # Fall back to plain LM call if no recursive capability
        return self._llm_query(prompt, model)

    def _rlm_query_once(
        self, prompt: str, model: str | None = None
    ) -> tuple[str, RLMChatCompletion | None]:
        """One recursive sub-call. Returns (result_text, completion_or_None)."""
        try:
            completion = self.subcall_fn(prompt, model)
            return completion.response, completion
        except Exception as e:
            return f"Error: RLM query failed - {e}", None

    def _rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Spawn recursive RLM sub-calls for multiple prompts in parallel.

        Each prompt gets its own child RLM for deeper thinking. When multiple
        prompts are provided, subcalls run concurrently using a thread pool
        (bounded by max_concurrent_subcalls) since they are independent and
        I/O-bound. Results are returned in the same order as input prompts.

        Falls back to llm_query_batched if no recursive capability is configured.

        Args:
            prompts: List of prompts for child RLMs.
            model: Optional model name override for the children.

        Returns:
            List of responses in the same order as input prompts.
        """
        width_refusal = self._batch_width_refusal(prompts)
        if width_refusal is not None:
            return width_refusal

        if self.subcall_fn is not None:
            allowed, results = self._partition_by_prompt_cap(prompts)

            # For 0 or 1 runnable prompts, no need for thread pool overhead
            if len(allowed) <= 1:
                for index in allowed:
                    text, completion = self._rlm_query_once(prompts[index], model)
                    if completion is not None:
                        self._record_call(completion)
                    results[index] = text
                return self._finalize_batch(self._rlm_query_once, prompts, model, allowed, results)

            # Parallel execution for multiple prompts
            max_workers = min(self.max_concurrent_subcalls, len(allowed))
            completions: list[tuple[int, RLMChatCompletion]] = []
            lock = threading.Lock()

            def _run_subcall(index: int, prompt: str) -> None:
                text, completion = self._rlm_query_once(prompt, model)
                if completion is not None:
                    with lock:
                        completions.append((index, completion))
                results[index] = text

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_subcall, index, prompts[index]) for index in allowed
                ]
                # Wait for all futures to complete; exceptions are captured inside _run_subcall
                for future in as_completed(futures):
                    future.result()  # Re-raises unexpected executor errors

            # Append completions in original prompt order for deterministic metadata
            completions.sort(key=lambda x: x[0])
            for _, completion in completions:
                self._record_call(completion)

            return self._finalize_batch(self._rlm_query_once, prompts, model, allowed, results)

        # Fall back to plain batched LM call if no recursive capability
        return self._llm_query_batched(prompts, model)

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues with nested dicts
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    def append_compaction_entry(self, entry: list[dict[str, Any]] | dict[str, Any]) -> None:
        """
        Append a trajectory segment or a summary to the compaction history.

        Entry is either a list of message dicts (trajectory segment) or
        a dict with "type": "summary" and "content": str.
        """
        if not self.compaction:
            return
        self._compaction_history.append(copy.deepcopy(entry))

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def _restore_scaffold(self) -> None:
        """Restore scaffold names after execution so overwrites (e.g. context = 'x') don't persist."""
        for name in RESERVED_TOOL_NAMES:
            if name == "llm_query":
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched":
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query":
                self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched":
                self.globals["rlm_query_batched"] = self._rlm_query_batched
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "answer":
                current = self.locals.get("answer")
                # If the model rebound ``answer`` to a plain dict, the
                # _AnswerDict callback never fired; capture content here if
                # ``ready=True``, then re-wrap so the next cell signals.
                if not isinstance(current, _AnswerDict):
                    replacement = _AnswerDict(on_ready=self._capture_answer)
                    if isinstance(current, dict):
                        for k, v in current.items():
                            dict.__setitem__(replacement, k, v)
                        if current.get("ready") and self._last_final_answer is None:
                            self._last_final_answer = str(current.get("content", ""))
                    self.locals["answer"] = replacement
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]
            elif name == "history" and "history_0" in self.locals and not self.compaction:
                self.locals["history"] = self.locals["history_0"]
            elif name == "history" and self.compaction:
                self.locals["history"] = self._compaction_history

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls from previous execution
        self._pending_llm_calls = []

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                # Restore scaffold so model overwrites (context = ..., llm_query = ...) don't persist
                self._restore_scaffold()

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        final_answer = self._last_final_answer
        self._last_final_answer = None

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            final_answer=final_answer,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __del__(self):
        self.cleanup()
