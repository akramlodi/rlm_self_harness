"""Tests for the U3 runner: two-tier invariant protection around a harnessed RLM.

The tiers are not interchangeable and the tests are grouped to keep them apart.

*Structural* checks run at construction and raise. They cover what a machine can
actually decide: that S7's output does not grow with the prompt (I1 as a growth
property), that the prompt's stated limits equal the limits the runtime enforces,
that the sub-call and answer-from-variable plumbing is present and unshadowable,
that S9 can only accept or redirect, and that no harness reaches the incidental
hazards the experiment owns.

*Behavioral* checks run a harness against a stubbed client and only observe. I2
and I3 in practice are prose-level properties -- an S2 line saying "just answer
directly" violates neither wiring nor type -- so the runner reports them through
the U1 trace metrics and never raises on them. Those tests assert the metrics are
present and carry the right values; they must not assert a refusal.
"""

import dataclasses
import re
from typing import Any
from unittest.mock import patch

import pytest

import rlm.core.rlm as rlm_module
from rlm.core.types import AnswerDecision, ModelUsageSummary, UsageSummary
from rlm.utils.parsing import DEFAULT_MAX_CHARACTER_LENGTH
from rlm.utils.prompts import DEFAULT_CAPACITY_SENTENCE, ORCHESTRATOR_ADDENDUM, RLM_SYSTEM_PROMPT
from shrlm.rlm_harness import H0, H0_STAR, Harness, build_runtime_policy
from shrlm.runner import (
    PROBE_GROWTH_FACTOR,
    PROBE_PROMPT_CHARS,
    REQUIRED_REPL_PLUMBING,
    acceptance_inputs,
    build_harnessed_rlm,
    check_metadata_boundedness,
    declared_metadata_bound,
    derive_capacity_sentence,
    derive_truncation_sentence,
    effective_system_prompt,
    format_char_bound,
)
from tests.mock_lm import MockLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ScriptedLM(MockLM):
    """A ``MockLM`` with a scripted queue and a per-call cost, so runs yield a cost total.

    Exhausting the script raises rather than improvising: a test whose scripted
    turns run out is a broken test, not a passing run.
    """

    def __init__(self, responses: list[str], cost_per_call: float = 0.001):
        super().__init__(model_name="mock-model", responses=list(responses))
        self.cost_per_call = cost_per_call
        self.prompts: list[Any] = []

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self.prompts.append(prompt)
        self._call_count += 1
        if not self._responses:
            raise IndexError(f"ScriptedLM: script exhausted after {self._call_count} calls")
        return self._responses.pop(0)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._call_count,
                    total_input_tokens=self._call_count * 10,
                    total_output_tokens=self._call_count * 10,
                    total_cost=self.cost_per_call * self._call_count,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=10,
            total_output_tokens=10,
            total_cost=self.cost_per_call,
        )


def final(content: str) -> str:
    """A ```repl``` block that returns the answer from the REPL variable (I3's path)."""
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


def run_offline(harness: Harness, responses: list[str], prompt: str = "ctx", **kwargs: Any):
    """Build ``harness`` and run one completion against a scripted client."""
    with patch.object(rlm_module, "get_client") as get_client:
        client = ScriptedLM(responses)
        get_client.return_value = client
        harnessed = build_harnessed_rlm(
            harness, backend="openai", backend_kwargs={"model_name": "test"}, **kwargs
        )
        run = harnessed.completion(prompt)
    return run, client


def bounded_builder(bound: int):
    """An S7 builder that emits a fixed-size prefix plus a length, declaring ``bound``."""

    def prefix_and_length(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
        return f"HEAD:{stdout[:200]} LEN:{len(stdout)}"

    prefix_and_length.declared_bound = bound
    return prefix_and_length


def restate_truncation(harness: Harness, bound: int) -> Harness:
    """Rewrite the harness's S1 truncation sentence to state ``bound``."""
    contract = harness.repl_contract.replace(
        "REPL outputs over ~20K characters are truncated",
        f"REPL outputs over {format_char_bound(bound)} characters are truncated",
    )
    return dataclasses.replace(harness, repl_contract=contract)


def policy(**overrides: Any) -> dict[str, Any]:
    """An S6 policy dict at the floor, with ``overrides`` applied."""
    base = build_runtime_policy()
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Structural: I1 as a growth property
# ---------------------------------------------------------------------------


class TestBoundedness:
    def test_growing_builder_fails_the_probe(self):
        def echo(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            return stdout

        echo.declared_bound = DEFAULT_MAX_CHARACTER_LENGTH
        with pytest.raises(ValueError, match="grew"):
            check_metadata_boundedness(echo)

    def test_prefix_plus_length_builder_passes_the_probe(self):
        check_metadata_boundedness(bounded_builder(5000))

    def test_shipped_default_passes_the_probe(self):
        check_metadata_boundedness(H0.metadata)

    def test_builder_without_a_declared_bound_is_rejected(self):
        def undeclared(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            return stdout[:10]

        with pytest.raises(ValueError, match="declared_bound"):
            declared_metadata_bound(undeclared)

    def test_builder_exceeding_its_own_declared_bound_is_rejected(self):
        def fat(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            return "x" * 50_000

        fat.declared_bound = 1000
        with pytest.raises(ValueError, match="declared bound"):
            check_metadata_boundedness(fat)

    def test_probe_scales_the_synthetic_prompt_a_hundredfold(self):
        assert PROBE_GROWTH_FACTOR == 100
        assert PROBE_PROMPT_CHARS > 0

    def test_construction_runs_the_probe(self):
        def echo(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            return stdout

        echo.declared_bound = DEFAULT_MAX_CHARACTER_LENGTH
        harness = dataclasses.replace(H0, metadata=echo)
        with pytest.raises(ValueError, match="grew"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})


# ---------------------------------------------------------------------------
# Structural: stated limits equal enforced limits
# ---------------------------------------------------------------------------


class TestStatedLimitsMatchRuntime:
    def test_both_starting_harnesses_state_the_default_bound(self):
        for harness in (H0, H0_STAR):
            assert declared_metadata_bound(harness.metadata) == DEFAULT_MAX_CHARACTER_LENGTH
            assert derive_truncation_sentence(harness) in effective_system_prompt(harness)

    def test_both_starting_harnesses_construct(self):
        for harness in (H0, H0_STAR):
            harnessed = build_harnessed_rlm(
                harness, backend="openai", backend_kwargs={"model_name": "t"}
            )
            assert harnessed.metadata_bound == DEFAULT_MAX_CHARACTER_LENGTH

    def test_metadata_bound_moved_without_the_sentence_raises(self):
        harness = dataclasses.replace(H0, metadata=bounded_builder(5000))
        with pytest.raises(ValueError, match="truncation"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_sentence_moved_without_the_metadata_bound_raises(self):
        harness = restate_truncation(H0, 5000)
        with pytest.raises(ValueError, match="truncation"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_bound_and_sentence_moved_together_construct(self):
        harness = restate_truncation(dataclasses.replace(H0, metadata=bounded_builder(5000)), 5000)
        harnessed = build_harnessed_rlm(
            harness, backend="openai", backend_kwargs={"model_name": "t"}
        )
        assert harnessed.metadata_bound == 5000
        assert "~5K characters are truncated" in harnessed.system_prompt

    def test_a_prompt_that_states_no_bound_raises(self):
        harness = dataclasses.replace(H0, repl_contract="Answer the question.")
        with pytest.raises(ValueError, match="truncation"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_capacity_sentence_is_the_shipped_default_when_the_policy_is_off(self):
        assert derive_capacity_sentence(H0.runtime_policy) == DEFAULT_CAPACITY_SENTENCE
        harnessed = build_harnessed_rlm(
            H0_STAR, backend="openai", backend_kwargs={"model_name": "t"}
        )
        assert harnessed.capacity_sentence == DEFAULT_CAPACITY_SENTENCE
        assert harnessed.rlm.capacity_sentence == DEFAULT_CAPACITY_SENTENCE

    def test_capacity_sentence_is_derived_from_an_enabled_policy(self):
        sentence = derive_capacity_sentence(policy(enabled=True, max_prompt_chars=50_000))
        assert "~50K characters" in sentence
        assert sentence != DEFAULT_CAPACITY_SENTENCE

    def test_policy_cap_disagreeing_with_the_stated_capacity_raises(self):
        # H0*'s addendum states "~100K characters per prompt"; the policy says 50K.
        harness = dataclasses.replace(
            H0_STAR, runtime_policy=policy(enabled=True, max_prompt_chars=50_000)
        )
        with pytest.raises(ValueError, match="capacity"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_policy_cap_agreeing_with_the_stated_capacity_constructs(self):
        harness = dataclasses.replace(
            H0_STAR, runtime_policy=policy(enabled=True, max_prompt_chars=100_000)
        )
        harnessed = build_harnessed_rlm(
            harness, backend="openai", backend_kwargs={"model_name": "t"}
        )
        assert "~100K characters" in harnessed.capacity_sentence

    def test_policy_batch_width_disagreeing_with_the_stated_fanout_raises(self):
        # H0*'s addendum states "~20 prompts per batch"; the policy says 4.
        harness = dataclasses.replace(
            H0_STAR, runtime_policy=policy(enabled=True, max_batch_width=4)
        )
        with pytest.raises(ValueError, match="batch"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_policy_batch_width_agreeing_with_the_stated_fanout_constructs(self):
        harness = dataclasses.replace(
            H0_STAR, runtime_policy=policy(enabled=True, max_batch_width=20)
        )
        build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_a_floor_prompt_stating_no_capacity_is_free_to_take_any_cap(self):
        # H0 states no per-prompt capacity, so nothing can contradict the policy;
        # the derived sentence is what tells the model.
        harness = dataclasses.replace(
            H0, runtime_policy=policy(enabled=True, max_prompt_chars=30_000)
        )
        harnessed = build_harnessed_rlm(
            harness, backend="openai", backend_kwargs={"model_name": "t"}
        )
        assert "~30K characters" in harnessed.capacity_sentence


# ---------------------------------------------------------------------------
# Structural: incidental hazards stay experiment-owned
# ---------------------------------------------------------------------------


class TestIncidentalHazards:
    def test_harness_declaring_other_backends_raises(self):
        harness = dataclasses.replace(H0, runtime_policy=policy(other_backends=["openrouter"]))
        with pytest.raises(ValueError, match="depth routing"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_other_backends_passed_as_a_runner_kwarg_raises(self):
        with pytest.raises(ValueError, match="depth routing"):
            build_harnessed_rlm(
                H0,
                backend="openai",
                backend_kwargs={"model_name": "t"},
                other_backends=["openrouter"],
            )

    def test_harness_supplying_a_tracing_callback_raises(self):
        harness = dataclasses.replace(
            H0, runtime_policy=policy(on_iteration_start=lambda depth, i: None)
        )
        with pytest.raises(ValueError, match="experiment"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_harness_supplying_a_hard_budget_cap_raises(self):
        harness = dataclasses.replace(H0, runtime_policy=policy(max_budget=1.0))
        with pytest.raises(ValueError, match="experiment"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_harness_supplying_a_token_or_timeout_cap_raises(self):
        for field, value in (("max_tokens", 100), ("max_timeout", 5.0), ("max_errors", 2)):
            harness = dataclasses.replace(H0, runtime_policy=policy(**{field: value}))
            with pytest.raises(ValueError, match="experiment"):
                build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_unknown_policy_key_raises(self):
        harness = dataclasses.replace(H0, runtime_policy=policy(rewrite_program=True))
        with pytest.raises(ValueError, match="unknown"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_experiment_may_still_own_its_caps(self):
        harnessed = build_harnessed_rlm(
            H0,
            backend="openai",
            backend_kwargs={"model_name": "t"},
            max_budget=1.0,
            max_iterations=3,
        )
        assert harnessed.rlm.max_budget == 1.0
        assert harnessed.rlm.max_iterations == 3

    def test_harness_owned_kwarg_from_the_caller_raises(self):
        for kwarg, value in (
            ("custom_system_prompt", "hi"),
            ("orchestrator", True),
            ("metadata_builder", H0.metadata),
            ("runtime_policy", {}),
            ("custom_tools", {}),
            ("logger", None),
        ):
            with pytest.raises(ValueError, match="harness"):
                build_harnessed_rlm(
                    H0,
                    backend="openai",
                    backend_kwargs={"model_name": "t"},
                    **{kwarg: value},
                )

    def test_max_depth_declared_twice_raises(self):
        harness = dataclasses.replace(H0, runtime_policy=policy(enabled=True, max_depth=2))
        with pytest.raises(ValueError, match="max_depth"):
            build_harnessed_rlm(
                harness, backend="openai", backend_kwargs={"model_name": "t"}, max_depth=3
            )

    def test_policy_max_depth_reaches_the_runtime(self):
        harness = dataclasses.replace(H0, runtime_policy=policy(enabled=True, max_depth=2))
        harnessed = build_harnessed_rlm(
            harness, backend="openai", backend_kwargs={"model_name": "t"}
        )
        assert harnessed.rlm.max_depth == 2


# ---------------------------------------------------------------------------
# Structural: the plumbing I2 and I3 ride on
# ---------------------------------------------------------------------------


class TestPlumbing:
    def test_required_plumbing_covers_both_paths(self):
        assert "llm_query" in REQUIRED_REPL_PLUMBING
        assert "rlm_query" in REQUIRED_REPL_PLUMBING
        assert "answer" in REQUIRED_REPL_PLUMBING

    def test_helpers_shadowing_the_sub_call_path_raise(self):
        harness = dataclasses.replace(H0, repl_helpers={"llm_query": lambda p: "nope"})
        with pytest.raises(ValueError, match="may not shadow"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_helpers_shadowing_the_answer_variable_raise(self):
        harness = dataclasses.replace(H0, repl_helpers={"answer": {}})
        with pytest.raises(ValueError, match="may not shadow"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_sub_helpers_shadowing_the_sub_call_path_raise(self):
        harness = dataclasses.replace(H0, sub_repl_helpers={"rlm_query": lambda p: "nope"})
        with pytest.raises(ValueError, match="may not shadow"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_plumbing_is_live_in_the_repl_of_a_real_run(self):
        probe = (
            "```repl\n"
            "print(' '.join(str(callable(f)) for f in "
            "[llm_query, llm_query_batched, rlm_query, rlm_query_batched]))\n"
            "print('ANSWER_DICT', isinstance(answer, dict))\n"
            "```"
        )
        run, client = run_offline(H0, [probe, final("done")])
        stdout = str(client.prompts[-1])
        assert "True True True True" in stdout
        assert "ANSWER_DICT True" in stdout
        assert run.completion.response == "done"


# ---------------------------------------------------------------------------
# Structural: S9 cannot rewrite the root's program (KTD1 condition 1)
# ---------------------------------------------------------------------------


class TestNonProgramRewriting:
    def test_middleware_taking_more_than_the_two_declared_inputs_raises(self):
        def greedy(answer: str, repl_inventory: dict[str, Any], environment: Any) -> AnswerDecision:
            return AnswerDecision.accept(answer)

        harness = dataclasses.replace(H0, answer_middleware=greedy)
        with pytest.raises(ValueError, match="two"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_middleware_taking_varargs_raises(self):
        def sneaky(*args: Any, **kwargs: Any) -> AnswerDecision:
            return AnswerDecision.accept("x")

        harness = dataclasses.replace(H0, answer_middleware=sneaky)
        with pytest.raises(ValueError, match="two"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_middleware_returning_something_other_than_a_decision_raises(self):
        def rewriter(answer: str, repl_inventory: dict[str, Any]) -> Any:
            return "```repl\nanswer['content'] = 'mine'\n```"

        harness = dataclasses.replace(H0, answer_middleware=rewriter)
        with pytest.raises(TypeError, match="AnswerDecision"):
            build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_redirect_is_the_only_alternative_to_accept(self):
        def redirector(answer: str, repl_inventory: dict[str, Any]) -> AnswerDecision:
            return AnswerDecision.redirect("check the buffer first")

        harness = dataclasses.replace(H0, answer_middleware=redirector)
        build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})

    def test_middleware_sees_only_the_answer_and_a_redacted_inventory(self):
        seen: list[tuple[str, dict[str, tuple[str, int]]]] = []

        def recorder(answer: str, repl_inventory: dict[str, Any]) -> AnswerDecision:
            seen.append((answer, dict(repl_inventory)))
            return AnswerDecision.accept(answer)

        harness = dataclasses.replace(H0, answer_middleware=recorder)
        run, _ = run_offline(harness, [final("done")], prompt="SECRETCONTEXT" * 500)

        assert run.completion.response == "done"
        # The construction-time probe plus the live call.
        answer, inventory = seen[-1]
        assert answer == "done"
        assert "SECRETCONTEXT" not in repr(inventory)
        assert inventory["context_0"][0] == "str"
        assert inventory["context_0"][1] == len("SECRETCONTEXT" * 500)


# ---------------------------------------------------------------------------
# Structural: prompt assembly order
# ---------------------------------------------------------------------------


class TestPromptAssembly:
    def test_s1_to_s5_appear_in_phase_order(self):
        prompt = effective_system_prompt(H0)
        positions = [
            prompt.index(H0.repl_contract),
            prompt.index(H0.decomposition_instruction),
            prompt.index(H0.execution_instruction),
            prompt.index(H0.verification_instruction),
            prompt.index(H0.recovery_instruction),
        ]
        assert positions == sorted(positions)
        assert len(set(positions)) == 5

    def test_an_empty_surface_contributes_nothing_not_a_blank_section(self):
        harness = dataclasses.replace(H0, execution_instruction="")
        prompt = effective_system_prompt(harness)
        assert H0.execution_instruction not in prompt
        assert "\n\n\n" not in prompt
        assert H0.decomposition_instruction in prompt
        assert H0.verification_instruction in prompt

    def test_h0_star_assembles_to_the_shipped_reference(self):
        assert effective_system_prompt(H0_STAR) == f"{RLM_SYSTEM_PROMPT}\n\n{ORCHESTRATOR_ADDENDUM}"

    def test_the_assembled_prompt_is_what_the_runtime_gets(self):
        harnessed = build_harnessed_rlm(H0, backend="openai", backend_kwargs={"model_name": "t"})
        assert harnessed.rlm.orchestrator is False
        assert harnessed.rlm.system_prompt == harnessed.system_prompt
        assert H0.recovery_instruction in harnessed.rlm.system_prompt


# ---------------------------------------------------------------------------
# Behavioral: monitored, never prevented
# ---------------------------------------------------------------------------

RECURSION_HARNESS = dataclasses.replace(
    H0, runtime_policy=build_runtime_policy() | {"enabled": True, "max_depth": 2}
)


class TestBehavioralMonitoring:
    def test_an_s2_that_discourages_recursion_still_builds(self):
        harness = dataclasses.replace(
            RECURSION_HARNESS,
            decomposition_instruction="Just answer directly; never use a sub-call.",
        )
        run, _ = run_offline(harness, [final("42")])
        assert run.metrics["sub_call_count"] == 0
        assert run.completion.response == "42"

    def test_the_sub_call_count_is_retrievable_for_the_lineage_record(self):
        harness = dataclasses.replace(
            RECURSION_HARNESS,
            decomposition_instruction="Just answer directly; never use a sub-call.",
        )
        run, _ = run_offline(harness, [final("42")])
        logged = run.completion.metadata["iterations"]
        assert [turn["trace_metrics"]["sub_call_count"] for turn in logged] == [0]

    def test_a_recursing_run_shows_a_non_zero_sub_call_count(self):
        responses = [
            "```repl\nout = rlm_query('what is 2+2?')\nprint(out)\n```",
            final("4"),
            final("done"),
        ]
        run, _ = run_offline(RECURSION_HARNESS, responses)
        assert run.metrics["sub_call_count"] == 1
        assert run.completion.response == "done"

    def test_h0_run_completes_returns_from_a_variable_and_populates_metrics(self):
        run, _ = run_offline(H0, [final("h0 answer")])
        assert run.completion.response == "h0 answer"
        assert run.metrics["answer_from_variable"] is True
        assert run.metrics["turns"] == 1
        assert run.metrics["cost"] is not None and run.metrics["cost"] > 0
        assert run.metrics["syntax_error_turns"] == 0

    def test_h0_star_run_completes_returns_from_a_variable_and_populates_metrics(self):
        run, _ = run_offline(H0_STAR, [final("h0 star answer")])
        assert run.completion.response == "h0 star answer"
        assert run.metrics["answer_from_variable"] is True
        assert run.metrics["cost"] is not None and run.metrics["cost"] > 0

    def test_syntax_errors_and_truncation_are_reported_not_prevented(self):
        responses = [
            "```repl\nx = (\n```",
            "```repl\nprint('y' * 40000)\n```",
            final("done"),
        ]
        run, _ = run_offline(H0, responses)
        assert run.metrics["syntax_error_turns"] == 1
        assert run.metrics["truncation_events"] == 1
        assert run.metrics["turns"] == 3

    def test_acceptance_inputs_are_computable_from_two_logged_runs(self):
        baseline, _ = run_offline(H0, [final("a")])
        candidate, _ = run_offline(
            RECURSION_HARNESS,
            [
                "```repl\nout = rlm_query('probe')\nprint(out)\n```",
                final("sub"),
                final("b"),
            ],
        )
        inputs = acceptance_inputs(baseline, candidate)
        assert inputs["baseline_cost"] > 0
        assert inputs["candidate_cost"] > 0
        assert inputs["baseline_sub_calls"] == 0
        assert inputs["candidate_sub_calls"] == 1
        assert inputs["sub_call_delta"] == 1
        assert inputs["cost_delta"] == pytest.approx(
            inputs["candidate_cost"] - inputs["baseline_cost"]
        )
        # Both runs produced an answer, the pass-rate side of the gate's input.
        assert inputs["baseline_answered"] is True
        assert inputs["candidate_answered"] is True

    def test_acceptance_inputs_fail_loudly_without_cost_data(self):
        run, _ = run_offline(H0, [final("a")])
        stripped = dataclasses.replace(run, metrics=dict(run.metrics) | {"cost": None})
        with pytest.raises(ValueError, match="cost"):
            acceptance_inputs(stripped, run)


# ---------------------------------------------------------------------------
# The derived-sentence helpers
# ---------------------------------------------------------------------------


class TestDerivedSentences:
    def test_char_bound_formatting(self):
        assert format_char_bound(20_000) == "~20K"
        assert format_char_bound(100_000) == "~100K"
        assert format_char_bound(1234) == "~1234"

    def test_truncation_sentence_matches_the_shipped_wording(self):
        sentence = derive_truncation_sentence(H0)
        assert re.search(r"REPL outputs over ~20K characters are truncated", sentence)


if __name__ == "__main__":
    pytest.main([__file__])
