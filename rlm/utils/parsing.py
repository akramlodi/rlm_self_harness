"""
Parsing utilities for RLM trjaectories.
"""

import re
from collections.abc import Callable, Sized
from typing import Any

from rlm.core.types import REPLResult, RLMIteration

# S7: the metadata builder sees the formatted execution result and a redacted
# inventory of the REPL namespace — ``{name: (type_name, length)}`` — never the
# raw locals. The redaction is what keeps the prompt (which lives in
# ``locals["context_0"]``) out of the root context.
MetadataBuilder = Callable[[str, dict[str, tuple[str, int]]], str]

DEFAULT_MAX_CHARACTER_LENGTH = 20000

# Substrings that classify an execution result or a sub-call error as a syntax error.
SYNTAX_ERROR_MARKERS = ("SyntaxError", "IndentationError", "invalid syntax")


def has_syntax_error(text: str) -> bool:
    """True when ``text`` carries a syntax-error signature (stderr or sub-call error)."""
    return any(marker in text for marker in SYNTAX_ERROR_MARKERS)


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def build_repl_inventory(repl_locals: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """
    Build the redacted inventory of a REPL namespace: ``{name: (type_name, length)}``.

    Values are never carried through. ``length`` is ``len(value)`` for sized
    objects and ``0`` otherwise, so a caller can see that ``context_0`` is a
    400,000-character ``str`` without being able to read a byte of it.

    Args:
        repl_locals: The REPL locals mapping (e.g. ``REPLResult.locals``).

    Returns:
        A mapping from public variable name to ``(type_name, length)``.
    """
    inventory: dict[str, tuple[str, int]] = {}
    for name, value in repl_locals.items():
        if name.startswith("_"):
            continue
        # The namespace holds arbitrary model-generated objects, so ``__len__``
        # is not trustworthy: it may raise or return a non-int. A hostile or
        # merely buggy object must not break history formatting on the default
        # path, where nothing previously touched these values at all. Compare
        # ``format_execution_result`` below, which restricts itself to builtin
        # types for the same reason.
        length = 0
        if isinstance(value, Sized):
            try:
                measured = len(value)
            except Exception:
                measured = 0
            length = measured if isinstance(measured, int) and measured >= 0 else 0
        inventory[name] = (type(value).__name__, length)
    return inventory


def default_metadata_builder(
    stdout: str,
    repl_inventory: dict[str, tuple[str, int]],
    max_character_length: int = DEFAULT_MAX_CHARACTER_LENGTH,
) -> str:
    """
    The shipped S7 default: head-truncate the formatted execution result at 20K chars.

    Args:
        stdout: The formatted execution result (stdout, stderr, and the REPL
            variable summary) for one code block.
        repl_inventory: The redacted namespace inventory. Unused by the default
            builder; present so injected builders share one signature.
        max_character_length: Per-block cap on the formatted execution result.

    Returns:
        The (possibly truncated) execution result string.
    """
    if len(stdout) <= max_character_length:
        return stdout
    return stdout[:max_character_length] + f"... + [{len(stdout) - max_character_length} chars...]"


def format_iteration(
    iteration: RLMIteration,
    max_character_length: int = DEFAULT_MAX_CHARACTER_LENGTH,
    metadata_builder: MetadataBuilder | None = None,
    metrics: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Each iteration produces exactly two messages in history: one assistant
    turn containing the model's response (with any ```repl``` blocks
    embedded), followed by a single user message that concatenates the
    outputs of all executed code blocks in that turn. This keeps the
    per-turn shape assistant-then-user even when the model emits several
    blocks in one response, and avoids redundantly echoing the code
    (which is already in the assistant message) back in the user reply.
    Each block's output is still individually truncated at
    ``max_character_length``.

    Args:
        iteration: The iteration to format
        max_character_length: Per-block cap on the formatted execution
            result. Longer outputs are tail-trimmed.
        metadata_builder: Optional S7 seam. Receives the formatted execution
            result and the redacted REPL inventory, and returns whatever should
            carry into the next turn. Defaults to the shipped 20K
            head-truncation, byte for byte.
        metrics: Optional out-parameter. When given, ``truncation_event`` is set
            to True if the metadata step shortened any block's output.

    Returns:
        A list of messages to add to the next prompt — always length 1
        (just the assistant) when no code was run, or length 2 (assistant
        + one combined user reply) otherwise.
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    parts = []
    truncated = False
    multi = len(iteration.code_blocks) > 1
    for i, code_block in enumerate(iteration.code_blocks):
        formatted = format_execution_result(code_block.result)
        inventory = build_repl_inventory(code_block.result.locals)
        if metadata_builder is None:
            result = default_metadata_builder(formatted, inventory, max_character_length)
        else:
            result = metadata_builder(formatted, inventory)
        truncated = truncated or len(result) < len(formatted)
        header = f"REPL output (block {i + 1}):" if multi else "REPL output:"
        parts.append(f"{header}\n{result}")

    if metrics is not None:
        metrics["truncation_event"] = truncated

    if parts:
        messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
