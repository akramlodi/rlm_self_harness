"""Tests for ``shrlm.harness_identity``: serialization and hashing of a harness.

The contract under test: every one of the nine surfaces plus the ``orchestrator``
scalar is hashed, the ``name`` field is recorded but never hashed, serialization
is deterministic across construction order and across processes, and a callable
whose source cannot be recovered fails loudly naming its surface.
"""

import dataclasses
import json
import subprocess
import sys
from typing import Any

import pytest

from rlm.core.types import AnswerDecision
from rlm.utils.parsing import DEFAULT_MAX_CHARACTER_LENGTH
from shrlm.harness_identity import (
    HarnessSerializationError,
    harness_hash,
    serialize_harness,
    write_harness_json,
)
from shrlm.rlm_harness import (
    H0,
    H0_STAR,
    Harness,
    build_metadata,
    build_runtime_policy,
)

# ---------------------------------------------------------------------------
# Alternate surface values used to probe hash sensitivity. All are module-level
# so ``inspect.getsource`` can recover them.
# ---------------------------------------------------------------------------


def alt_metadata(
    stdout: str,
    repl_inventory: dict[str, tuple[str, int]],
    max_character_length: int = DEFAULT_MAX_CHARACTER_LENGTH,
) -> str:
    """An S7 builder with a different bound and a different body than the floor."""
    return stdout[:max_character_length]


alt_metadata.declared_bound = 12345


def alt_metadata_same_bound(
    stdout: str,
    repl_inventory: dict[str, tuple[str, int]],
    max_character_length: int = DEFAULT_MAX_CHARACTER_LENGTH,
) -> str:
    """An S7 builder with the floor's bound but a different source."""
    return stdout[:max_character_length] + ""


alt_metadata_same_bound.declared_bound = DEFAULT_MAX_CHARACTER_LENGTH


def alt_middleware(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
    """An S9 middleware whose source differs from the floor's ``accept_answer``."""
    return AnswerDecision.accept(answer.strip())


def helper_word_count(text: str) -> int:
    """Count whitespace-separated words in ``text``."""
    return len(text.split())


def helper_word_count_redoc(text: str) -> int:
    """Count the words in ``text`` (same behavior, different docstring)."""
    return len(text.split())


def enabled_runtime_policy() -> dict[str, Any]:
    """H0's runtime policy with exactly one field flipped."""
    policy = build_runtime_policy()
    policy["enabled"] = True
    return policy


VARIANTS: dict[str, dict[str, Any]] = {
    "S1_repl_contract": {"repl_contract": H0.repl_contract + "\nExtra clause."},
    "S2_decomposition_instruction": {"decomposition_instruction": "Probe differently."},
    "S3_execution_instruction": {"execution_instruction": "Execute differently."},
    "S4_verification_instruction": {"verification_instruction": "Verify differently."},
    "S5_recovery_instruction": {"recovery_instruction": "Recover differently."},
    "S6_runtime_policy": {"runtime_policy": enabled_runtime_policy()},
    "S7_metadata_bound_and_source": {"metadata": alt_metadata},
    "S7_metadata_source_only": {"metadata": alt_metadata_same_bound},
    "S8_repl_helpers": {"repl_helpers": {"word_count": helper_word_count}},
    "S8_sub_repl_helpers": {"sub_repl_helpers": {"word_count": helper_word_count}},
    "S9_answer_middleware": {"answer_middleware": alt_middleware},
}


def rebuild_h0(shuffle: bool) -> Harness:
    """Construct a harness equal to ``H0``, optionally with reversed dict insertion order.

    Args:
        shuffle: When true, ``runtime_policy`` is rebuilt with reversed key order.

    Returns:
        A fresh ``Harness`` whose nine surfaces match ``H0``'s exactly.
    """
    policy = build_runtime_policy()
    if shuffle:
        policy = dict(reversed(list(policy.items())))
    return dataclasses.replace(H0, runtime_policy=policy)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_serialization_is_byte_identical_across_constructions() -> None:
    first = rebuild_h0(shuffle=False)
    second = rebuild_h0(shuffle=True)
    canonical_first = json.dumps(serialize_harness(first), sort_keys=True)
    canonical_second = json.dumps(serialize_harness(second), sort_keys=True)
    assert canonical_first == canonical_second
    assert harness_hash(first) == harness_hash(second)


def test_hash_is_stable_across_processes() -> None:
    code = "from shrlm.harness_identity import harness_hash\n"
    code += "from shrlm.rlm_harness import H0\n"
    code += "print(harness_hash(H0))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == harness_hash(H0)


# ---------------------------------------------------------------------------
# Sensitivity: every surface and the orchestrator scalar move the hash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant_id", sorted(VARIANTS))
def test_changing_any_surface_changes_the_hash(variant_id: str) -> None:
    variant = dataclasses.replace(H0, **VARIANTS[variant_id])
    assert harness_hash(variant) != harness_hash(H0)


def test_orchestrator_flag_changes_the_hash() -> None:
    flipped = dataclasses.replace(H0, orchestrator=True)
    assert harness_hash(flipped) != harness_hash(H0)


def test_name_alone_does_not_change_the_hash() -> None:
    renamed = dataclasses.replace(H0, name="H0-renamed")
    assert harness_hash(renamed) == harness_hash(H0)
    assert serialize_harness(renamed)["name"] == "H0-renamed"


# ---------------------------------------------------------------------------
# The two starting harnesses
# ---------------------------------------------------------------------------


def test_h0_and_h0_star_hash_differently() -> None:
    assert harness_hash(H0) != harness_hash(H0_STAR)


def test_h0_serialization_names_every_surface_and_round_trips() -> None:
    serialized = serialize_harness(H0)
    assert serialized["name"] == "H0"
    assert serialized["orchestrator"] is False
    surfaces = serialized["surfaces"]
    assert sorted(surfaces) == [
        "S1_repl_contract",
        "S2_decomposition_instruction",
        "S3_execution_instruction",
        "S4_verification_instruction",
        "S5_recovery_instruction",
        "S6_runtime_policy",
        "S7_metadata",
        "S8_repl_helpers",
        "S8_sub_repl_helpers",
        "S9_answer_middleware",
    ]
    assert surfaces["S7_metadata"]["declared_bound"] == build_metadata.declared_bound
    assert json.loads(json.dumps(serialized)) == serialized


# ---------------------------------------------------------------------------
# S8 entry shapes
# ---------------------------------------------------------------------------


def test_s8_tool_entry_serializes_source_and_description() -> None:
    harness = dataclasses.replace(
        H0,
        repl_helpers={
            "word_count": {"tool": helper_word_count, "description": "Count words."},
            "corpus_label": "arxiv",
        },
    )
    entries = serialize_harness(harness)["surfaces"]["S8_repl_helpers"]
    tool_entry = entries["word_count"]
    assert tool_entry["description"] == "Count words."
    assert "def helper_word_count" in tool_entry["tool"]["source"]
    assert entries["corpus_label"] == {"kind": "value", "value": "arxiv"}


def test_s8_non_string_description_serializes_as_none_like_the_runtime() -> None:
    """The runtime (``parse_tool_entry``) drops a non-string description; the
    serialization must record the same interpretation, not the raw entry."""
    harness = dataclasses.replace(
        H0,
        repl_helpers={"word_count": {"tool": helper_word_count, "description": 42}},
    )
    entries = serialize_harness(harness)["surfaces"]["S8_repl_helpers"]
    assert entries["word_count"]["description"] is None
    assert "def helper_word_count" in entries["word_count"]["tool"]["source"]


def test_s8_docstring_change_changes_the_hash() -> None:
    base = dataclasses.replace(H0, repl_helpers={"word_count": helper_word_count})
    redoc = dataclasses.replace(H0, repl_helpers={"word_count": helper_word_count_redoc})
    assert harness_hash(base) != harness_hash(redoc)


# ---------------------------------------------------------------------------
# Fail-fast on sourceless callables
# ---------------------------------------------------------------------------


def test_sourceless_s9_callable_fails_naming_the_surface() -> None:
    broken = dataclasses.replace(H0, answer_middleware=len)
    with pytest.raises(HarnessSerializationError, match="S9"):
        serialize_harness(broken)


def test_sourceless_s8_callable_fails_naming_the_surface() -> None:
    broken = dataclasses.replace(H0, repl_helpers={"length": len})
    with pytest.raises(HarnessSerializationError, match="S8"):
        serialize_harness(broken)


# ---------------------------------------------------------------------------
# Fail-fast on invalid S7 declared_bound (the runtime's own validator)
# ---------------------------------------------------------------------------


def bool_bound_metadata(
    stdout: str,
    repl_inventory: dict[str, tuple[str, int]],
    max_character_length: int = DEFAULT_MAX_CHARACTER_LENGTH,
) -> str:
    """An S7 builder whose declared_bound is a bool, not an int."""
    return stdout[:max_character_length]


bool_bound_metadata.declared_bound = True


def zero_bound_metadata(
    stdout: str,
    repl_inventory: dict[str, tuple[str, int]],
    max_character_length: int = DEFAULT_MAX_CHARACTER_LENGTH,
) -> str:
    """An S7 builder whose declared_bound is non-positive."""
    return stdout[:max_character_length]


zero_bound_metadata.declared_bound = 0


@pytest.mark.parametrize("builder", [bool_bound_metadata, zero_bound_metadata])
def test_invalid_s7_declared_bound_fails_naming_the_surface(builder) -> None:
    broken = dataclasses.replace(H0, metadata=builder)
    with pytest.raises(HarnessSerializationError, match="S7"):
        serialize_harness(broken)


# ---------------------------------------------------------------------------
# The written envelope
# ---------------------------------------------------------------------------


def test_write_harness_json_envelope(tmp_path) -> None:
    path = tmp_path / "harness.json"
    envelope = write_harness_json(H0, path)
    on_disk = json.loads(path.read_text())
    assert on_disk == envelope
    assert on_disk["name"] == "H0"
    assert on_disk["hash"] == harness_hash(H0)
    assert on_disk["harness"] == serialize_harness(H0)
