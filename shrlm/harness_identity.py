"""Serialization and identity for a ``Harness``: what exactly did this run run?

A mining round is only evidence if the harness it measured can be reconstructed
byte for byte. This module turns a ``shrlm.rlm_harness.Harness`` into a plain
JSON-serializable dict covering all ten surfaces plus the ``orchestrator``
scalar, and hashes the canonical form of that dict so two harnesses compare by
content, not by object identity.

What is hashed and what is not:
    - Hashed: the five prompt strings (S1-S5), the runtime policy (S6), the S7
      metadata builder's source and its ``declared_bound``, every S8 helper
      entry (callable source plus non-callable metadata), the S9 middleware's
      source, every S10 skill entry (name, description, body), and the
      ``orchestrator`` flag -- the flag changes the effective system prompt, so
      it is part of identity.
    - Not hashed: ``name``. It is recorded in the serialization as informational
      metadata; renaming a harness must not change its hash.

Callables are serialized by ``inspect.getsource`` (dedented). A callable whose
source cannot be recovered -- a builtin, a C-level callable, a ``functools.
partial`` -- raises ``HarnessSerializationError`` naming the surface rather
than silently falling back, per the fail-fast convention in AGENTS.md.

Known stage-2 concern: source text identifies a callable only when the callable
is defined statically. A stage-2 edit loop that generates callables dynamically
(``exec``-built functions, closures over run-time values) will need to carry its
own provenance for them; this module deliberately refuses to guess.
"""

import dataclasses
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any

from rlm.environments.base_env import parse_tool_entry
from shrlm.rlm_harness import Harness, SkillEntry
from shrlm.runner import declared_metadata_bound

# The harness identity envelope's format tag. Bumped from v1 when S10 joined the
# surface set (KTD3): the serialized key set *is* the contract, so a nine-surface
# document must be rejected as the wrong version, not as a malformed shape. The
# same tag is declared as ``HARNESS_FORMAT`` in ``shrlm.optimization.candidates``
# (where the loader checks it) and ``shrlm.optimization.proposal`` (where the
# proposal writer stamps it); all three must agree.
HARNESS_FORMAT = "shrlm-harness/v2"


class HarnessSerializationError(RuntimeError):
    """A harness surface could not be serialized to reconstructible form."""


def callable_source(fn: Any, surface: str) -> str:
    """Recover the dedented source text of ``fn``, failing loudly on ``surface``.

    Args:
        fn: The callable whose source identifies it.
        surface: The surface label (e.g. ``"S9 answer_middleware"``) used in the
            error message when the source is unavailable.

    Returns:
        The dedented source of ``fn``.

    Raises:
        HarnessSerializationError: If ``inspect.getsource`` cannot recover the
            source (builtins, C-level callables, dynamically built functions).
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as error:
        raise HarnessSerializationError(
            f"cannot serialize {surface}: source unavailable for {fn!r} "
            f"({error}); dynamically generated callables need explicit provenance"
        ) from error
    return textwrap.dedent(source)


def json_safe_value(value: Any, surface: str) -> Any:
    """Pass ``value`` through only if JSON can represent it exactly.

    Args:
        value: A non-callable surface value (policy field, S8 data entry).
        surface: The surface label used in the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        HarnessSerializationError: If ``value`` is not JSON-serializable, since
            a lossy representation would break reconstructibility.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError) as error:
        raise HarnessSerializationError(
            f"cannot serialize {surface}: value {value!r} is not JSON-serializable"
        ) from error
    return value


def serialize_helper_entry(name: str, entry: Any, surface: str) -> Any:
    """Serialize one S8 helper entry to a reconstructible form.

    The entry is interpreted through ``rlm.environments.base_env.
    parse_tool_entry`` -- the runtime's own reading -- so serialization can
    never diverge from what the REPL actually installs (e.g. a non-string
    ``description`` is dropped to None, exactly as the runtime drops it).
    Callables become ``{"kind": "callable", "source": ...}``; plain data
    becomes ``{"kind": "value", "value": ...}``; tool dicts keep the description
    alongside the serialized tool.

    Args:
        name: The helper's REPL name.
        entry: The raw dict value from ``repl_helpers`` / ``sub_repl_helpers``.
        surface: The surface label (``"S8 repl_helpers"`` or the sub variant).

    Returns:
        A JSON-serializable representation of the entry.
    """
    label = f"{surface}[{name!r}]"
    info = parse_tool_entry(name, entry)
    if info.is_callable:
        serialized = {"kind": "callable", "source": callable_source(info.value, label)}
    else:
        serialized = {"kind": "value", "value": json_safe_value(info.value, label)}
    if isinstance(entry, dict) and "tool" in entry:
        return {"description": info.description, "tool": serialized}
    return serialized


def serialize_skill_entry(index: int, entry: Any, surface: str) -> dict[str, Any]:
    """Serialize one S10 skill entry to its ``name``/``description``/``body`` mapping.

    Args:
        index: The entry's position in ``Harness.skills`` (used in the label).
        entry: The raw list element; must be a ``SkillEntry``.
        surface: The surface label (``"S10 skills"``).

    Returns:
        A JSON-serializable dict with keys ``name``, ``description``, ``body``.

    Raises:
        HarnessSerializationError: If ``entry`` is not a ``SkillEntry`` or one
            of its fields is not JSON-serializable, naming the surface and the
            offending position.
    """
    label = f"{surface}[{index}]"
    if not isinstance(entry, SkillEntry):
        raise HarnessSerializationError(
            f"cannot serialize {label}: expected a SkillEntry, got {entry!r}"
        )
    return json_safe_value(dataclasses.asdict(entry), label)


def serialize_harness(harness: Harness) -> dict[str, Any]:
    """Serialize all ten surfaces plus ``orchestrator`` and the unhashed ``name``.

    Args:
        harness: The harness to serialize.

    Returns:
        A JSON-serializable dict with top-level keys ``name``, ``orchestrator``,
        and ``surfaces`` (one entry per surface, ``S1_...`` through ``S10_...``).
        ``name`` is informational only and excluded from ``harness_hash``.

    Raises:
        HarnessSerializationError: If any callable surface has unrecoverable
            source, if S7's builder lacks a valid ``declared_bound`` (a
            positive int, per ``shrlm.runner.declared_metadata_bound``), if a
            data value is not JSON-serializable, or if an S10 entry is not a
            ``SkillEntry``.
    """
    metadata = harness.metadata
    try:
        declared_bound = declared_metadata_bound(metadata)
    except ValueError as error:
        raise HarnessSerializationError(f"cannot serialize S7 metadata: {error}") from error
    return {
        "name": harness.name,
        "orchestrator": harness.orchestrator,
        "surfaces": {
            "S1_repl_contract": harness.repl_contract,
            "S2_decomposition_instruction": harness.decomposition_instruction,
            "S3_execution_instruction": harness.execution_instruction,
            "S4_verification_instruction": harness.verification_instruction,
            "S5_recovery_instruction": harness.recovery_instruction,
            "S6_runtime_policy": {
                key: json_safe_value(value, f"S6 runtime_policy[{key!r}]")
                for key, value in sorted(harness.runtime_policy.items())
            },
            "S7_metadata": {
                "declared_bound": declared_bound,
                "source": callable_source(metadata, "S7 metadata"),
            },
            "S8_repl_helpers": {
                name: serialize_helper_entry(name, entry, "S8 repl_helpers")
                for name, entry in sorted(harness.repl_helpers.items())
            },
            "S8_sub_repl_helpers": {
                name: serialize_helper_entry(name, entry, "S8 sub_repl_helpers")
                for name, entry in sorted(harness.sub_repl_helpers.items())
            },
            "S9_answer_middleware": {
                "source": callable_source(harness.answer_middleware, "S9 answer_middleware"),
            },
            "S10_skills": [
                serialize_skill_entry(index, entry, "S10 skills")
                for index, entry in enumerate(harness.skills)
            ],
        },
    }


def canonical_json(material: Any) -> str:
    """The canonical JSON rendering every comparison and hash runs on.

    Canonical means sorted keys, compact separators, ASCII, so the text is
    stable across processes and dict insertion orders.
    """
    return json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_sha256(material: dict[str, Any]) -> str:
    """Sha256 over the canonical JSON rendering of ``material``."""
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def hash_of_serialization(serialization: dict[str, Any]) -> str:
    """The content hash of an already-serialized harness dict.

    The hashed material is the serialization minus the informational ``name``
    field (renaming a harness must not change its hash).
    """
    material = dict(serialization)
    material.pop("name", None)
    return canonical_json_sha256(material)


def harness_hash(harness: Harness) -> str:
    """Content hash of the harness: sha256 over the canonical hashed material.

    Args:
        harness: The harness to hash.

    Returns:
        The sha256 hex digest identifying the harness's content.
    """
    return hash_of_serialization(serialize_harness(harness))


def write_harness_json(harness: Harness, path: str | Path) -> dict[str, Any]:
    """Write the harness identity envelope to ``path`` and return it.

    The envelope carries a format tag, the informational ``name``, the content
    ``hash``, and the full ``harness`` serialization -- everything a later stage
    needs to attribute a round to an exact harness.

    Args:
        harness: The harness to record.
        path: Destination file path for the JSON document.

    Returns:
        The envelope dict that was written.
    """
    serialization = serialize_harness(harness)
    envelope = {
        "format": HARNESS_FORMAT,
        "name": harness.name,
        "hash": hash_of_serialization(serialization),
        "harness": serialization,
    }
    Path(path).write_text(json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    return envelope
