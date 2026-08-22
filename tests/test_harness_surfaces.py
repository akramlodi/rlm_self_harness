"""Tests for the U2 harness module: three invariants, ten surfaces, two harnesses.

The module is meant to read like a figure — a flat list of builders — so these
tests pin the shape of that figure: the registry matches the module's public
builders exactly, H0* is byte-identical to the shipped reference, and H0 stays
sparse so the optimization loop has something left to rediscover.
"""

import dataclasses
from typing import Any

import pytest

from rlm.core.types import QueryMetadata
from rlm.environments.base_env import RESERVED_TOOL_NAMES, validate_custom_tools
from rlm.utils.parsing import (
    DEFAULT_MAX_CHARACTER_LENGTH,
    default_metadata_builder,
    format_iteration,
)
from rlm.utils.prompts import (
    ORCHESTRATOR_ADDENDUM,
    RLM_SYSTEM_PROMPT,
    build_rlm_system_prompt,
)
from shrlm import rlm_harness
from shrlm.rlm_harness import (
    H0,
    H0_STAR,
    INVARIANTS,
    SKILL_INDEX_PREAMBLE,
    SKILL_LOADER_NAME,
    SURFACES,
    Harness,
    SkillEntry,
    assemble_system_prompt,
    build_skills,
    escape_braces,
)

# Clauses that ORCHESTRATOR_ADDENDUM carries and that H0 must not contain: the
# star's hand-tuned orchestration prior, which the loop is supposed to rediscover.
STAR_CLAUSES = [
    "act as an orchestrator, not a solver",
    "state explicitly how the task decomposes",
    "~100K characters per prompt",
    "~20 prompts per batch",
    'only flip `answer["ready"] = True` once you have actually printed the candidate answer',
]


def system_prompt_for(harness: Harness) -> str:
    """Assemble ``harness`` through the shipped prompt builder and return the system text."""
    messages = build_rlm_system_prompt(
        system_prompt=assemble_system_prompt(harness),
        query_metadata=QueryMetadata("hello"),
        orchestrator=harness.orchestrator,
    )
    return messages[0]["content"]


def make_iteration(stdout: str):
    from rlm.core.types import CodeBlock, REPLResult, RLMIteration

    result = REPLResult(stdout=stdout, stderr="", locals={"x": 1}, execution_time=0.01)
    return RLMIteration(
        prompt="p", response="resp", code_blocks=[CodeBlock(code="p", result=result)]
    )


# ---------------------------------------------------------------------------
# The figure: invariants and the surface registry
# ---------------------------------------------------------------------------


class TestSurfaceRegistry:
    def test_ten_surfaces_no_more_no_fewer(self):
        assert len(SURFACES) == 10
        assert list(SURFACES) == [f"S{i}" for i in range(1, 11)]

    def test_registry_matches_module_public_builders_exactly(self):
        module_builders = {
            name
            for name, obj in vars(rlm_harness).items()
            if name.startswith("build_") and callable(obj)
        }
        registry_builders = {
            builder.__name__ for surface in SURFACES.values() for builder in surface.builders
        }
        assert registry_builders == module_builders

    def test_every_surface_name_resolves_to_a_module_level_callable(self):
        for surface in SURFACES.values():
            for builder in surface.builders:
                resolved = getattr(rlm_harness, builder.__name__)
                assert resolved is builder
                assert callable(resolved)

    def test_three_invariants_are_declared_and_frozen(self):
        assert isinstance(INVARIANTS, tuple)
        assert len(INVARIANTS) == 3
        assert all(isinstance(text, str) and text for text in INVARIANTS)

    def test_each_builder_docstring_names_its_loop_phase(self):
        for sid, surface in SURFACES.items():
            for builder in surface.builders:
                doc = builder.__doc__ or ""
                assert surface.phase in doc, f"{sid} builder {builder.__name__} omits its phase"

    def test_harness_fields_follow_the_builder_naming_convention(self):
        field_names = {f.name for f in dataclasses.fields(Harness)}
        for surface in SURFACES.values():
            for builder in surface.builders:
                assert builder.__name__.removeprefix("build_") in field_names


# ---------------------------------------------------------------------------
# The two harnesses
# ---------------------------------------------------------------------------


class TestHarnesses:
    def test_h0_star_system_prompt_is_byte_identical_to_shipped(self):
        shipped = build_rlm_system_prompt(
            system_prompt=RLM_SYSTEM_PROMPT,
            query_metadata=QueryMetadata("hello"),
            orchestrator=True,
        )
        harnessed = build_rlm_system_prompt(
            system_prompt=assemble_system_prompt(H0_STAR),
            query_metadata=QueryMetadata("hello"),
            orchestrator=H0_STAR.orchestrator,
        )
        assert harnessed == shipped
        assert ORCHESTRATOR_ADDENDUM in harnessed[0]["content"]

    def test_h0_star_is_the_orchestrator_and_h0_is_not(self):
        assert H0_STAR.orchestrator is True
        assert H0.orchestrator is False

    def test_h0_prompt_contains_none_of_the_star_clauses(self):
        prompt = system_prompt_for(H0)
        for clause in STAR_CLAUSES:
            assert clause not in prompt, f"H0 pre-populated with star clause: {clause!r}"
        assert ORCHESTRATOR_ADDENDUM not in prompt

    def test_h0_and_h0_star_differ_only_in_s1_to_s5_and_orchestrator(self):
        prompt_fields = {
            "repl_contract",
            "decomposition_instruction",
            "execution_instruction",
            "verification_instruction",
            "recovery_instruction",
        }
        # S6-S10 are equal for both harnesses.
        assert H0.runtime_policy == H0_STAR.runtime_policy
        assert H0.metadata is H0_STAR.metadata
        assert H0.repl_helpers == H0_STAR.repl_helpers
        assert H0.sub_repl_helpers == H0_STAR.sub_repl_helpers
        assert H0.answer_middleware is H0_STAR.answer_middleware
        assert H0.skills == H0_STAR.skills == []
        # Nothing else varies apart from the prompt surfaces, the orchestrator
        # scalar, and the harness's own name.
        differing = {
            f.name
            for f in dataclasses.fields(Harness)
            if getattr(H0, f.name) != getattr(H0_STAR, f.name)
        }
        assert differing <= prompt_fields | {"orchestrator", "name"}

    def test_eight_of_ten_h0_defaults_are_empty_disabled_or_one_line(self):
        """The floor's sparseness is the experimental design, not an oversight."""
        sparse: list[str] = []
        one_liners = {
            "S2": H0.decomposition_instruction,
            "S3": H0.execution_instruction,
            "S4": H0.verification_instruction,
            "S5": H0.recovery_instruction,
        }
        for sid, text in one_liners.items():
            assert text.strip(), f"{sid} must carry one generic line, not nothing"
            assert "\n" not in text.strip(), f"{sid} is more than one line"
            sparse.append(sid)

        assert H0.runtime_policy["enabled"] is False
        sparse.append("S6")
        assert H0.repl_helpers == {} and H0.sub_repl_helpers == {}
        sparse.append("S8")
        assert H0.answer_middleware("done", {}).answer == "done"
        sparse.append("S9")
        assert H0.skills == []
        sparse.append("S10")

        assert len(sparse) == 8
        # The two non-sparse surfaces are the factual contract and the memory default.
        assert len(H0.repl_contract.splitlines()) > 1
        assert H0.metadata("x" * 50, {}) == "x" * 50

    def test_harness_is_frozen_but_replaceable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            H0.orchestrator = True  # type: ignore[misc]
        variant = dataclasses.replace(H0, name="H0-variant")
        assert variant.name == "H0-variant"
        assert H0.name != "H0-variant"


# ---------------------------------------------------------------------------
# The brace constraint on the prompt surfaces
# ---------------------------------------------------------------------------


class TestBraceConstraint:
    def test_both_harness_prompts_survive_the_format_slot(self):
        for harness in (H0, H0_STAR):
            prompt = system_prompt_for(harness)
            assert "{custom_tools_section}" not in prompt

    def test_escaped_literal_brace_round_trips_without_raising(self):
        example = escape_braces('Return a dict like {"id": 1, "score": 0.5}.')
        harness = dataclasses.replace(H0, decomposition_instruction=example)
        prompt = system_prompt_for(harness)
        assert '{"id": 1, "score": 0.5}' in prompt

    def test_escape_braces_preserves_the_custom_tools_slot(self):
        assert escape_braces("a {custom_tools_section} b") == "a {custom_tools_section} b"
        assert escape_braces("{x}") == "{{x}}"

    def test_unescaped_brace_is_what_the_helper_exists_to_prevent(self):
        harness = dataclasses.replace(H0, decomposition_instruction='Return {"id": 1}.')
        with pytest.raises((KeyError, IndexError, ValueError)):
            system_prompt_for(harness)


# ---------------------------------------------------------------------------
# S10 — skills
# ---------------------------------------------------------------------------

# Two brace-free entries. Bodies carry a marker that appears nowhere else in any
# prompt, so "the body never reaches the prompt" is a real assertion.
TWO_SKILLS = [
    SkillEntry(
        name="chunk_context",
        description="when `context` is larger than one sub-call can take",
        body="BODY-ONE 1. Measure len(context).\n2. Split into slices.\n3. One sub-call per slice.",
    ),
    SkillEntry(
        name="recheck_quotes",
        description="when the candidate answer quotes `context` verbatim",
        body="BODY-TWO 1. Re-find each quote in context.\n2. Drop any quote not found.",
    ),
]


def five_part_join(harness: Harness) -> str:
    """The pre-S10 assembly formula: S1-S5, empty surfaces dropped, joined by a blank line."""
    parts = [
        harness.repl_contract,
        harness.decomposition_instruction,
        harness.execution_instruction,
        harness.verification_instruction,
        harness.recovery_instruction,
    ]
    return "\n\n".join(part for part in parts if part)


class TestSkillsSurface:
    def test_build_skills_is_empty_and_fresh_per_call(self):
        first = build_skills()
        assert first == []
        first.append(TWO_SKILLS[0])
        assert build_skills() == []

    def test_both_starting_harnesses_carry_an_empty_s10(self):
        assert H0.skills == []
        assert H0_STAR.skills == []

    def test_skill_entry_is_a_frozen_record_of_name_description_body(self):
        entry = TWO_SKILLS[0]
        assert (entry.name, entry.description, entry.body) == (
            "chunk_context",
            "when `context` is larger than one sub-call can take",
            TWO_SKILLS[0].body,
        )
        assert {f.name for f in dataclasses.fields(SkillEntry)} == {"name", "description", "body"}
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.name = "other"  # type: ignore[misc]

    def test_h0_prompt_is_byte_identical_to_the_pre_s10_assembly(self):
        assert assemble_system_prompt(H0) == five_part_join(H0)
        assert assemble_system_prompt(H0_STAR) == five_part_join(H0_STAR) == RLM_SYSTEM_PROMPT

    def test_empty_s10_renders_neither_wrapper_nor_index(self):
        harness = dataclasses.replace(H0, skills=[])
        prompt = assemble_system_prompt(harness)
        assert prompt == assemble_system_prompt(H0)
        for text in (prompt, assemble_system_prompt(H0_STAR)):
            assert SKILL_INDEX_PREAMBLE not in text
            assert SKILL_LOADER_NAME not in text

    def test_two_entries_render_wrapper_then_index_lines_after_s5_and_no_bodies(self):
        harness = dataclasses.replace(H0, skills=TWO_SKILLS)
        prompt = assemble_system_prompt(harness)
        # S1-S5 are untouched: the index is appended after them, never interleaved.
        assert prompt.startswith(assemble_system_prompt(H0))
        s5 = prompt.index(H0.recovery_instruction)
        wrapper = prompt.index(SKILL_INDEX_PREAMBLE)
        first = prompt.index(TWO_SKILLS[0].name)
        second = prompt.index(TWO_SKILLS[1].name)
        assert s5 < wrapper < first < second
        assert prompt.count(SKILL_INDEX_PREAMBLE) == 1
        for entry in TWO_SKILLS:
            line = next(ln for ln in prompt.splitlines() if entry.name in ln)
            assert entry.description in line
            assert entry.body not in prompt
            assert "BODY-" not in prompt

    def test_wrapper_names_the_loader_and_says_what_it_returns_only(self):
        assert SKILL_LOADER_NAME in SKILL_INDEX_PREAMBLE
        assert "procedure" in SKILL_INDEX_PREAMBLE
        # Purely declarative: no per-turn "when to call it" guidance (R14/KTD1).
        for cue in ("before", "whenever", "always", "should", "must", "each turn"):
            assert cue not in SKILL_INDEX_PREAMBLE.lower(), cue

    def test_loader_name_is_an_identifier_outside_the_runtime_reserved_set(self):
        assert SKILL_LOADER_NAME == "load_skill"
        assert SKILL_LOADER_NAME.isidentifier()
        assert SKILL_LOADER_NAME not in RESERVED_TOOL_NAMES

    def test_index_rendering_survives_the_format_slot(self):
        harness = dataclasses.replace(H0, skills=TWO_SKILLS)
        prompt = system_prompt_for(harness)
        assert "{custom_tools_section}" not in prompt
        assert TWO_SKILLS[1].description in prompt

    def test_s10_governs_carries_the_content_contract(self):
        surface = SURFACES["S10"]
        assert surface.builders == (build_skills,)
        assert "across turns" in surface.phase
        governs = surface.governs
        for phrase in (
            "one named",
            "procedure",
            "`name`",
            "`description`",
            "`body`",
            "ordered steps",
        ):
            assert phrase in governs, phrase
        assert "per-turn execution" in governs and "decomposition" in governs

    def test_s8_governs_names_the_loader_exclusion(self):
        governs = SURFACES["S8"].governs
        assert "proposer-written" in governs
        assert "skill loader" in governs
        assert "S10" in governs


# ---------------------------------------------------------------------------
# S6 — runtime policy
# ---------------------------------------------------------------------------


class TestRuntimePolicy:
    def test_default_policy_is_disabled_with_every_field_none(self):
        policy = rlm_harness.build_runtime_policy()
        assert policy["enabled"] is False
        assert set(policy) >= {
            "enabled",
            "max_prompt_chars",
            "max_batch_width",
            "max_depth",
            "retry_on_syntax_error",
            "max_retries",
            "validate_sub_output",
        }
        for key, value in policy.items():
            if key == "enabled":
                continue
            assert value is None, f"policy field {key} is pre-populated with {value!r}"

    def test_policy_is_a_fresh_dict_per_call(self):
        first = rlm_harness.build_runtime_policy()
        first["enabled"] = True
        assert rlm_harness.build_runtime_policy()["enabled"] is False


# ---------------------------------------------------------------------------
# S7 — metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_default_reproduces_the_shipped_20k_truncation(self):
        stdout = "x" * (DEFAULT_MAX_CHARACTER_LENGTH + 5000)
        assert H0.metadata(stdout, {}) == default_metadata_builder(stdout, {})
        assert len(H0.metadata(stdout, {})) < len(stdout)
        iteration = make_iteration("y" * 50000)
        assert format_iteration(iteration, metadata_builder=H0.metadata) == format_iteration(
            iteration
        )

    def test_smaller_metadata_edit_is_bounded_and_size_invariant(self):
        def tiny(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            return f"HEAD:{stdout[:16]} LEN:{len(stdout)}"

        harness = dataclasses.replace(H0, metadata=tiny)
        small = format_iteration(make_iteration("a" * 1000), metadata_builder=harness.metadata)[1]
        large = format_iteration(make_iteration("a" * 400000), metadata_builder=harness.metadata)[1]
        assert len(large["content"]) < 200
        assert len(large["content"]) - len(small["content"]) <= 4


# ---------------------------------------------------------------------------
# S8 — REPL helpers
# ---------------------------------------------------------------------------


class TestReplHelpers:
    def test_helpers_are_empty_dicts_by_default(self):
        assert rlm_harness.build_repl_helpers() == {}
        assert rlm_harness.build_sub_repl_helpers() == {}

    def test_helper_keys_never_collide_with_reserved_repl_names(self):
        for helpers in (H0.repl_helpers, H0.sub_repl_helpers, H0_STAR.repl_helpers):
            assert isinstance(helpers, dict)
            assert set(helpers) & RESERVED_TOOL_NAMES == set()
            # The environment backstop must never be what saves the harness.
            validate_custom_tools(helpers)

    def test_helpers_are_fresh_dicts_per_call(self):
        first: dict[str, Any] = rlm_harness.build_repl_helpers()
        first["chunk"] = lambda: None
        assert rlm_harness.build_repl_helpers() == {}


# ---------------------------------------------------------------------------
# S9 — answer middleware
# ---------------------------------------------------------------------------


class TestAnswerMiddleware:
    def test_default_is_identity(self):
        middleware = rlm_harness.build_answer_middleware()
        decision = middleware("the final answer", {"buf": ("str", 10)})
        assert decision.accepted is True
        assert decision.answer == "the final answer"
        assert decision.nudge is None


if __name__ == "__main__":
    pytest.main([__file__])
