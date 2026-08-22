"""Tests for the U1 runtime seams: S7 metadata, S9 answer middleware, S6 policy, trace metrics.

Every seam must default to current behavior, so each test comes in a pair:
the configured behavior fires, and the unconfigured default is unchanged.

The last section covers the S10 construction checks (U7): an unusable skill
list, a brace in the formatted prompt, or an S8 helper binding the loader's
reserved name is refused by ``check_harness`` before any instance is spent.
"""

from dataclasses import replace
from typing import Any
from unittest.mock import Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.types import (
    AnswerDecision,
    ModelUsageSummary,
    QueryMetadata,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    UsageSummary,
)
from rlm.core.types import CodeBlock as CodeBlockType
from rlm.environments.base_env import SkillLoader
from rlm.environments.local_repl import (
    SUBCALL_INVALID_PREFIX,
    SUBCALL_REFUSAL_PREFIX,
    LocalREPL,
)
from rlm.logger import RLMLogger
from rlm.utils.parsing import (
    DEFAULT_MAX_CHARACTER_LENGTH,
    build_repl_inventory,
    format_execution_result,
    format_iteration,
)
from rlm.utils.prompts import DEFAULT_CAPACITY_SENTENCE, build_rlm_system_prompt
from shrlm.rlm_harness import (
    H0,
    H0_STAR,
    SKILL_BODY_MAX_CHARS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_LOADER_NAME,
    SKILL_MAX_ENTRIES,
    SKILL_NAME_MAX_CHARS,
    SKILL_TOTAL_MAX_CHARS,
    Harness,
    SkillEntry,
    build_runtime_policy,
    escape_braces,
)
from shrlm.runner import (
    _install_skill_loader,
    build_harnessed_rlm,
    build_skill_loader,
    check_harness,
    check_plumbing,
    check_stated_limits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def prepatch_format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """Verbatim copy of the pre-patch ``format_iteration`` body (the byte-identity oracle)."""
    messages = [{"role": "assistant", "content": iteration.response}]

    parts = []
    multi = len(iteration.code_blocks) > 1
    for i, code_block in enumerate(iteration.code_blocks):
        result = format_execution_result(code_block.result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )
        header = f"REPL output (block {i + 1}):" if multi else "REPL output:"
        parts.append(f"{header}\n{result}")

    if parts:
        messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages


def make_iteration(
    stdout: str = "hello", locals_dict: dict[str, Any] | None = None, blocks: int = 1
) -> RLMIteration:
    code_blocks = []
    for i in range(blocks):
        result = REPLResult(
            stdout=stdout,
            stderr="",
            locals=dict(locals_dict or {"x": 1}),
            execution_time=0.01,
        )
        code_blocks.append(CodeBlockType(code=f"print({i})", result=result))
    return RLMIteration(prompt="p", response="resp", code_blocks=code_blocks)


def patch_repl_global(monkeypatch, name: str, value: Any) -> None:
    """Patch a module global in the namespace ``LocalREPL``'s methods resolve against.

    ``tests/test_imports.py`` deletes ``rlm.*`` from ``sys.modules`` and re-imports,
    so the module object registered under ``rlm.environments.local_repl`` is not
    necessarily the one this module's ``LocalREPL`` was defined in. Patching the
    function globals directly is order-independent.
    """
    monkeypatch.setitem(LocalREPL._llm_query_once.__globals__, name, value)


def create_mock_lm(responses: list[str]) -> Mock:
    mock = Mock()
    mock.completion.side_effect = list(responses)
    mock.get_usage_summary.return_value = UsageSummary(
        model_usage_summaries={
            "mock": ModelUsageSummary(total_calls=1, total_input_tokens=10, total_output_tokens=5)
        }
    )
    mock.get_last_usage.return_value = mock.get_usage_summary.return_value
    return mock


def final(content: str) -> str:
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


def mock_completion(usage_cost: float | None = None, response: str = "sub-ok"):
    usage = UsageSummary(
        model_usage_summaries={
            "mock": ModelUsageSummary(
                total_calls=1,
                total_input_tokens=1,
                total_output_tokens=1,
                total_cost=usage_cost,
            )
        }
    )
    return RLMChatCompletion(
        root_model="mock",
        prompt="p",
        response=response,
        usage_summary=usage,
        execution_time=0.0,
    )


# ---------------------------------------------------------------------------
# S7 — metadata function
# ---------------------------------------------------------------------------


class TestS7Metadata:
    def test_unconfigured_entry_is_byte_identical(self):
        iteration = make_iteration(stdout="x" * 50000, locals_dict={"buf": "y" * 10})
        assert format_iteration(iteration) == prepatch_format_iteration(iteration)

    def test_unconfigured_multi_block_entry_is_byte_identical(self):
        iteration = make_iteration(stdout="z" * 30, blocks=3)
        assert format_iteration(iteration) == prepatch_format_iteration(iteration)

    def test_configured_builder_shrinks_entry_and_is_size_invariant(self):
        def builder(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            return f"HEAD:{stdout[:16]} LEN:{len(stdout)}"

        small = make_iteration(stdout="a" * 1000)
        large = make_iteration(stdout="a" * 400000)

        small_entry = format_iteration(small, metadata_builder=builder)[1]["content"]
        large_entry = format_iteration(large, metadata_builder=builder)[1]["content"]

        assert len(large_entry) < 200
        # Entry size does not grow with prompt length (only the digit count of LEN).
        assert len(large_entry) - len(small_entry) <= 4

    def test_builder_receives_only_redacted_inventory(self):
        secret = "SECRET-PAYLOAD-9F3A"
        seen: list[dict[str, tuple[str, int]]] = []

        def builder(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            seen.append(repl_inventory)
            return "ok"

        iteration = make_iteration(
            stdout="no secret here",
            locals_dict={"context_0": secret, "context": secret, "n": 5},
        )
        format_iteration(iteration, metadata_builder=builder)

        assert len(seen) == 1
        inventory = seen[0]
        assert inventory["context_0"] == ("str", len(secret))
        assert inventory["n"] == ("int", 0)
        # No value of any variable is reachable from what the builder is handed.
        assert secret not in repr(inventory)

    def test_build_repl_inventory_redacts_values(self):
        inventory = build_repl_inventory({"s": "abc", "d": {"k": "v"}, "_hidden": "h", "i": 7})
        assert inventory == {"s": ("str", 3), "d": ("dict", 1), "i": ("int", 0)}

    def test_build_repl_inventory_survives_untrustworthy_dunder_len(self):
        """The namespace holds model-generated objects; ``__len__`` may misbehave.

        The default path did not touch these values before the S7 seam existed,
        so a raising or non-int ``__len__`` must not become a new way to break
        history formatting mid-run.
        """

        class Raising:
            def __len__(self):
                raise ValueError("boom")

        class Negative:
            def __len__(self):
                return -1

        inventory = build_repl_inventory(
            {"ctx": "abc", "raiser": Raising(), "negative": Negative()}
        )
        assert inventory == {
            "ctx": ("str", 3),
            "raiser": ("Raising", 0),
            "negative": ("Negative", 0),
        }

    def test_rlm_threads_metadata_builder_to_history(self):
        calls: list[dict[str, tuple[str, int]]] = []

        def builder(stdout: str, repl_inventory: dict[str, tuple[str, int]]) -> str:
            calls.append(repl_inventory)
            return "TINY"

        responses = [
            "```repl\nbuf = 'q' * 50000\nprint(buf)\n```",
            final("done"),
        ]
        with patch.object(rlm_module, "get_client") as get_client:
            get_client.return_value = create_mock_lm(responses)
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                metadata_builder=builder,
            )
            result = model.completion("ctx")

        assert result.response == "done"
        assert len(calls) == 1
        assert calls[0]["buf"] == ("str", 50000)


# ---------------------------------------------------------------------------
# S9 — answer middleware
# ---------------------------------------------------------------------------


class TestS9AnswerMiddleware:
    def test_unconfigured_terminates_as_before(self):
        with patch.object(rlm_module, "get_client") as get_client:
            mock = create_mock_lm([final("first")])
            get_client.return_value = mock
            model = RLM(backend="openai", backend_kwargs={"model_name": "test"})
            result = model.completion("ctx")

        assert result.response == "first"
        assert mock.completion.call_count == 1

    def test_identity_middleware_terminates_as_before(self):
        def middleware(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
            return AnswerDecision.accept(answer)

        with patch.object(rlm_module, "get_client") as get_client:
            mock = create_mock_lm([final("first")])
            get_client.return_value = mock
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                answer_middleware=middleware,
            )
            result = model.completion("ctx")

        assert result.response == "first"
        assert mock.completion.call_count == 1

    def test_redirect_suppresses_answer_and_continues_loop(self):
        seen: list[str] = []

        def middleware(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
            seen.append(answer)
            if len(seen) == 1:
                return AnswerDecision.redirect("Not so fast: verify against the buffer first.")
            return AnswerDecision.accept(answer)

        prompts: list[Any] = []

        def response_fn(prompt):
            prompts.append(prompt)
            return [final("first"), final("second")][len(prompts) - 1]

        with patch.object(rlm_module, "get_client") as get_client:
            mock = create_mock_lm([])
            mock.completion.side_effect = response_fn
            get_client.return_value = mock
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                answer_middleware=middleware,
            )
            result = model.completion("ctx")

        assert seen == ["first", "second"]
        assert result.response == "second"
        # The nudge reached the model.
        second_prompt = prompts[1]
        assert any("Not so fast" in str(m.get("content", "")) for m in second_prompt)

    def test_redirect_resets_namespace_ready_and_recaptures(self):
        def middleware(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
            if answer == "first":
                return AnswerDecision.redirect("try again")
            return AnswerDecision.accept(answer)

        prompts: list[Any] = []
        scripted = [
            final("first"),
            "```repl\nprint('READY=' + str(answer['ready']))\n```",
            final("second"),
        ]

        def response_fn(prompt):
            prompts.append(prompt)
            return scripted[len(prompts) - 1]

        with patch.object(rlm_module, "get_client") as get_client:
            mock = create_mock_lm([])
            mock.completion.side_effect = response_fn
            get_client.return_value = mock
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                answer_middleware=middleware,
            )
            result = model.completion("ctx")

        third_prompt = prompts[2]
        rendered = "\n".join(str(m.get("content", "")) for m in third_prompt)
        assert "READY=False" in rendered
        # A later answer["ready"] = True still fires capture.
        assert result.response == "second"

    def test_middleware_sees_redacted_inventory(self):
        seen: list[dict[str, tuple[str, int]]] = []

        def middleware(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
            seen.append(repl_inventory)
            return AnswerDecision.accept(answer)

        responses = [
            "```repl\nverified = ['a', 'b', 'c']\nprint('ok')\n```",
            final("done"),
        ]
        with patch.object(rlm_module, "get_client") as get_client:
            get_client.return_value = create_mock_lm(responses)
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                answer_middleware=middleware,
            )
            model.completion("ctx")

        assert seen[0]["verified"] == ("list", 3)

    def test_local_repl_reset_answer(self):
        env = LocalREPL()
        env.execute_code("answer['content'] = 'a'\nanswer['ready'] = True")
        env.reset_answer()
        assert env.locals["answer"]["ready"] is False
        assert env.locals["answer"]["content"] == ""
        result = env.execute_code("answer['content'] = 'b'\nanswer['ready'] = True")
        assert result.final_answer == "b"
        env.cleanup()


# ---------------------------------------------------------------------------
# S6 — retry / sub-output validation
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, chat_completion=None, error=None):
        self.chat_completion = chat_completion
        self.error = error

    @property
    def success(self) -> bool:
        return self.error is None


class TestS6Retry:
    def test_llm_query_retries_syntax_error_when_configured(self, monkeypatch):
        calls = {"n": 0}

        def fake_send(address, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise SyntaxError("invalid syntax")
            return _FakeResponse(chat_completion=mock_completion(response="recovered"))

        patch_repl_global(monkeypatch, "send_lm_request", fake_send)
        env = LocalREPL(
            lm_handler_address=("127.0.0.1", 1),
            runtime_policy={"enabled": True, "retry_on_syntax_error": True, "max_retries": 1},
        )
        assert env.globals["llm_query"]("hi") == "recovered"
        assert calls["n"] == 2
        env.cleanup()

    def test_llm_query_unchanged_by_default(self, monkeypatch):
        calls = {"n": 0}

        def fake_send(address, request):
            calls["n"] += 1
            raise SyntaxError("invalid syntax")

        patch_repl_global(monkeypatch, "send_lm_request", fake_send)
        env = LocalREPL(lm_handler_address=("127.0.0.1", 1))
        out = env.globals["llm_query"]("hi")
        assert out == "Error: LM query failed - invalid syntax"
        assert calls["n"] == 1
        env.cleanup()

    def test_rlm_query_retries_syntax_error_when_configured(self):
        calls = {"n": 0}

        def subcall_fn(prompt, model=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise SyntaxError("invalid syntax")
            return mock_completion(response="recovered")

        env = LocalREPL(
            subcall_fn=subcall_fn,
            runtime_policy={"enabled": True, "retry_on_syntax_error": True, "max_retries": 1},
        )
        assert env.globals["rlm_query"]("hi") == "recovered"
        assert calls["n"] == 2
        env.cleanup()

    def test_rlm_query_unchanged_by_default(self):
        calls = {"n": 0}

        def subcall_fn(prompt, model=None):
            calls["n"] += 1
            raise SyntaxError("invalid syntax")

        env = LocalREPL(subcall_fn=subcall_fn)
        assert env.globals["rlm_query"]("hi") == "Error: RLM query failed - invalid syntax"
        assert calls["n"] == 1
        env.cleanup()

    def test_rlm_query_batched_retries_per_element(self):
        attempts: dict[str, int] = {}

        def subcall_fn(prompt, model=None):
            attempts[prompt] = attempts.get(prompt, 0) + 1
            if prompt == "bad" and attempts[prompt] == 1:
                raise SyntaxError("invalid syntax")
            return mock_completion(response=f"{prompt}-ok")

        env = LocalREPL(
            subcall_fn=subcall_fn,
            runtime_policy={"enabled": True, "retry_on_syntax_error": True, "max_retries": 1},
        )
        out = env.globals["rlm_query_batched"](["good", "bad"])
        assert out == ["good-ok", "bad-ok"]
        assert attempts["bad"] == 2
        env.cleanup()

    def test_rlm_query_batched_unchanged_by_default(self):
        attempts: dict[str, int] = {}

        def subcall_fn(prompt, model=None):
            attempts[prompt] = attempts.get(prompt, 0) + 1
            if prompt == "bad":
                raise SyntaxError("invalid syntax")
            return mock_completion(response=f"{prompt}-ok")

        env = LocalREPL(subcall_fn=subcall_fn)
        out = env.globals["rlm_query_batched"](["good", "bad"])
        assert out == ["good-ok", "Error: RLM query failed - invalid syntax"]
        assert attempts["bad"] == 1
        env.cleanup()

    def test_sub_output_validation_marks_invalid(self):
        def subcall_fn(prompt, model=None):
            return mock_completion(response="garbage")

        env = LocalREPL(
            subcall_fn=subcall_fn,
            runtime_policy={
                "enabled": True,
                "validate_sub_output": lambda text: text.startswith("OK"),
            },
        )
        out = env.globals["rlm_query"]("hi")
        assert out.startswith(SUBCALL_INVALID_PREFIX)
        env.cleanup()


# ---------------------------------------------------------------------------
# S6 — hard caps
# ---------------------------------------------------------------------------


class TestS6Caps:
    def _batched_env(self, policy: dict[str, Any] | None):
        def subcall_fn(prompt, model=None):
            return mock_completion(response=f"{prompt}-ok")

        return LocalREPL(subcall_fn=subcall_fn, runtime_policy=policy)

    def test_rlm_query_batched_refuses_over_width(self):
        env = self._batched_env({"enabled": True, "max_batch_width": 2})
        out = env.globals["rlm_query_batched"](["a", "b", "c"])
        assert len(out) == 3
        assert all(o.startswith(SUBCALL_REFUSAL_PREFIX) for o in out)
        env.cleanup()

    def test_rlm_query_batched_refuses_over_char_cap_per_element(self):
        env = self._batched_env({"enabled": True, "max_prompt_chars": 10})
        out = env.globals["rlm_query_batched"](["a", "b" * 50])
        assert out[0] == "a-ok"
        assert out[1].startswith(SUBCALL_REFUSAL_PREFIX)
        env.cleanup()

    def test_llm_query_batched_refuses_over_width(self, monkeypatch):
        def fake_send_batched(address, prompts, model=None, depth=0):
            return [_FakeResponse(chat_completion=mock_completion(response="ok")) for _ in prompts]

        patch_repl_global(monkeypatch, "send_lm_request_batched", fake_send_batched)
        env = LocalREPL(
            lm_handler_address=("127.0.0.1", 1),
            runtime_policy={"enabled": True, "max_batch_width": 2},
        )
        out = env.globals["llm_query_batched"](["a", "b", "c"])
        assert all(o.startswith(SUBCALL_REFUSAL_PREFIX) for o in out)
        env.cleanup()

    def test_llm_query_batched_refuses_over_char_cap_per_element(self, monkeypatch):
        def fake_send_batched(address, prompts, model=None, depth=0):
            return [
                _FakeResponse(chat_completion=mock_completion(response=f"{p}-ok")) for p in prompts
            ]

        patch_repl_global(monkeypatch, "send_lm_request_batched", fake_send_batched)
        env = LocalREPL(
            lm_handler_address=("127.0.0.1", 1),
            runtime_policy={"enabled": True, "max_prompt_chars": 10},
        )
        out = env.globals["llm_query_batched"](["a", "b" * 50])
        assert out[0] == "a-ok"
        assert out[1].startswith(SUBCALL_REFUSAL_PREFIX)
        env.cleanup()

    def test_both_batched_paths_unbounded_by_default(self, monkeypatch):
        def fake_send_batched(address, prompts, model=None, depth=0):
            return [
                _FakeResponse(chat_completion=mock_completion(response=f"{p}-ok")) for p in prompts
            ]

        patch_repl_global(monkeypatch, "send_lm_request_batched", fake_send_batched)
        env = LocalREPL(lm_handler_address=("127.0.0.1", 1))
        wide = [str(i) for i in range(50)] + ["z" * 100000]
        out = env.globals["llm_query_batched"](wide)
        assert len(out) == 51
        assert not any(o.startswith(SUBCALL_REFUSAL_PREFIX) for o in out)
        env.cleanup()

        env2 = self._batched_env(None)
        out2 = env2.globals["rlm_query_batched"](wide)
        assert len(out2) == 51
        assert not any(o.startswith(SUBCALL_REFUSAL_PREFIX) for o in out2)
        env2.cleanup()


# ---------------------------------------------------------------------------
# S6 — capacity sentence
# ---------------------------------------------------------------------------


class TestS6CapacitySentence:
    def test_default_capacity_sentence(self):
        messages = build_rlm_system_prompt(
            system_prompt="SYS {custom_tools_section}",
            query_metadata=QueryMetadata("hello"),
        )
        assert (
            "Each sub-LLM call can handle roughly ~100k tokens at once." in messages[1]["content"]
        )
        assert DEFAULT_CAPACITY_SENTENCE in messages[1]["content"]

    def test_injected_capacity_sentence(self):
        messages = build_rlm_system_prompt(
            system_prompt="SYS {custom_tools_section}",
            query_metadata=QueryMetadata("hello"),
            capacity_sentence="Each sub-LLM call is capped at 40k characters.",
        )
        assert "Each sub-LLM call is capped at 40k characters." in messages[1]["content"]
        assert "~100k tokens" not in messages[1]["content"]

    def test_rlm_threads_capacity_sentence(self):
        with patch.object(rlm_module, "get_client") as get_client:
            get_client.return_value = create_mock_lm([final("done")])
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                capacity_sentence="CAP-SENTENCE-XYZ",
            )
            history = model._setup_prompt("ctx")
        assert any("CAP-SENTENCE-XYZ" in m["content"] for m in history)


# ---------------------------------------------------------------------------
# Trace metrics
# ---------------------------------------------------------------------------


class TestTraceMetrics:
    def test_chat_completion_trace_metrics_round_trip(self):
        completion = mock_completion(usage_cost=0.25)
        completion.trace_metrics = {"cost": 0.25, "syntax_error": False, "retries": 0}
        data = completion.to_dict()
        assert data["trace_metrics"]["cost"] == 0.25
        restored = RLMChatCompletion.from_dict(data)
        assert restored.trace_metrics == {"cost": 0.25, "syntax_error": False, "retries": 0}

    def test_prepatch_record_still_deserializes(self):
        prepatch = {
            "root_model": "gpt-4",
            "prompt": "hi",
            "response": "hello",
            "usage_summary": {"model_usage_summaries": {}},
            "execution_time": 1.0,
        }
        restored = RLMChatCompletion.from_dict(prepatch)
        assert restored.trace_metrics is None
        assert "trace_metrics" not in restored.to_dict()

    def test_socket_round_trip_preserves_trace_metrics(self):
        from rlm.core.comms_utils import LMResponse

        completion = mock_completion(usage_cost=0.5)
        completion.trace_metrics = {"cost": 0.5, "syntax_error": True, "retries": 1}
        response = LMResponse(chat_completion=completion)
        restored = LMResponse.from_dict(response.to_dict())
        assert restored.chat_completion.trace_metrics == {
            "cost": 0.5,
            "syntax_error": True,
            "retries": 1,
        }

    def test_iteration_trace_metrics_recorded_for_a_run(self):
        responses = [
            "```repl\nx = (\n```",
            "```repl\nprint('y' * 40000)\n```",
            final("done"),
        ]
        with patch.object(rlm_module, "get_client") as get_client:
            get_client.return_value = create_mock_lm(responses)
            logger = _RecordingLogger()
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                logger=logger,
            )
            model.completion("ctx")

        metrics = [it.trace_metrics for it in logger.iterations]
        assert metrics[0]["syntax_error"] is True
        assert metrics[0]["sub_call_count"] == 0
        assert metrics[1]["syntax_error"] is False
        assert metrics[1]["truncation_event"] is True
        assert metrics[2]["answer_event"] == "answer_submitted"

    def test_sub_call_count_and_cost_recorded(self):
        def subcall_fn(prompt, model=None):
            return mock_completion(usage_cost=0.125, response="sub")

        env = LocalREPL(subcall_fn=subcall_fn)
        result = env.execute_code("out = rlm_query('hi')\nprint(out)")
        assert len(result.rlm_calls) == 1
        assert result.rlm_calls[0].trace_metrics["cost"] == 0.125
        env.cleanup()


# ---------------------------------------------------------------------------
# Skill-load events: recorded by the environment beside its sub-call events
# ---------------------------------------------------------------------------

SKILL_BODIES = {
    "chunk_context": "BODY-ONE {not a format field} 1. Measure len(context).",
    "recheck_quotes": "BODY-TWO 1. Re-find each quote.",
}
SKILL_INDEX = [
    {"name": "chunk_context", "description": "when `context` is too large"},
    {"name": "recheck_quotes", "description": "when the answer quotes `context`"},
]


def make_skill_loader() -> SkillLoader:
    def load(name: str) -> str:
        if name not in SKILL_BODIES:
            raise LookupError(f"unknown skill {name!r}")
        return SKILL_BODIES[name]

    return SkillLoader(load=load, index=SKILL_INDEX)


class TestSkillLoadRecording:
    """Mirror of the sub-call recording tests: a loader call is a per-turn trace fact."""

    def test_loader_call_is_recorded_with_the_environment_depth(self):
        env = LocalREPL(custom_tools={"load_skill": make_skill_loader()}, depth=2)
        result = env.execute_code("body = load_skill('chunk_context')\nprint(body)")
        assert result.stderr == ""
        assert SKILL_BODIES["chunk_context"] in result.stdout
        assert result.skill_loads == [{"skill": "chunk_context", "depth": 2}]
        env.cleanup()

    def test_loader_returns_the_body_verbatim_through_the_repl(self):
        env = LocalREPL(custom_tools={"load_skill": make_skill_loader()})
        env.execute_code("body = load_skill('chunk_context')")
        assert env.locals["body"] == SKILL_BODIES["chunk_context"]
        env.cleanup()

    def test_pending_skill_loads_reset_every_turn_like_sub_calls(self):
        env = LocalREPL(custom_tools={"load_skill": make_skill_loader()})
        first = env.execute_code(
            "a = load_skill('chunk_context')\nb = load_skill('recheck_quotes')"
        )
        second = env.execute_code("print(len(a) + len(b))")
        assert [event["skill"] for event in first.skill_loads] == [
            "chunk_context",
            "recheck_quotes",
        ]
        assert second.skill_loads == []
        assert second.rlm_calls == []
        env.cleanup()

    def test_a_failed_load_is_not_recorded_and_surfaces_in_stderr(self):
        env = LocalREPL(custom_tools={"load_skill": make_skill_loader()})
        result = env.execute_code("load_skill('nope')")
        assert result.skill_loads == []
        assert "LookupError" in result.stderr and "nope" in result.stderr
        env.cleanup()

    def test_plain_callable_tools_are_not_recorded_as_skill_loads(self):
        env = LocalREPL(custom_tools={"helper": lambda name: f"h:{name}"})
        result = env.execute_code("print(helper('x'))")
        assert "h:x" in result.stdout
        assert result.skill_loads == []
        env.cleanup()

    def test_tool_dict_form_of_the_loader_is_still_recorded(self):
        env = LocalREPL(
            custom_tools={"load_skill": {"tool": make_skill_loader(), "description": "d"}}
        )
        result = env.execute_code("load_skill('recheck_quotes')")
        assert result.skill_loads == [{"skill": "recheck_quotes", "depth": 1}]
        env.cleanup()

    def test_repl_result_to_dict_carries_skill_loads_beside_rlm_calls(self):
        result = REPLResult(
            stdout="",
            stderr="",
            locals={},
            execution_time=0.0,
            skill_loads=[{"skill": "chunk_context", "depth": 1}],
        )
        data = result.to_dict()
        assert data["rlm_calls"] == []
        assert data["skill_loads"] == [{"skill": "chunk_context", "depth": 1}]
        # A result built without the field persists an empty list, like rlm_calls.
        assert REPLResult(stdout="", stderr="", locals={}).to_dict()["skill_loads"] == []

    def test_turn_metrics_and_run_metadata_carry_loads_and_index(self):
        responses = [
            "```repl\nbody = load_skill('chunk_context')\nprint(body[:8])\n```",
            final("done"),
        ]
        with patch.object(rlm_module, "get_client") as get_client:
            get_client.return_value = create_mock_lm(responses)
            logger = RLMLogger()
            model = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                custom_tools={"load_skill": make_skill_loader()},
                logger=logger,
            )
            result = model.completion("ctx")

        turns = result.metadata["iterations"]
        assert turns[0]["trace_metrics"]["skill_load_count"] == 1
        assert turns[0]["trace_metrics"]["sub_call_count"] == 0
        assert turns[1]["trace_metrics"]["skill_load_count"] == 0
        assert turns[0]["code_blocks"][0]["result"]["skill_loads"] == [
            {"skill": "chunk_context", "depth": 1}
        ]
        # The run-start record names what was available, once, beside the run metadata.
        assert result.metadata["run_metadata"]["skill_index"] == SKILL_INDEX

    def test_run_metadata_omits_the_index_without_a_loader(self):
        with patch.object(rlm_module, "get_client") as get_client:
            get_client.return_value = create_mock_lm([final("done")])
            logger = RLMLogger()
            model = RLM(backend="openai", backend_kwargs={"model_name": "test"}, logger=logger)
            result = model.completion("ctx")
        assert "skill_index" not in result.metadata["run_metadata"]
        assert result.metadata["iterations"][0]["trace_metrics"]["skill_load_count"] == 0


class _RecordingLogger:
    """Minimal logger stand-in that keeps the RLMIteration objects themselves."""

    def __init__(self):
        self.iterations: list[RLMIteration] = []

    def clear_iterations(self) -> None:
        self.iterations = []

    def log(self, iteration: RLMIteration) -> None:
        self.iterations.append(iteration)

    def log_metadata(self, metadata) -> None:
        pass

    def get_trajectory(self) -> dict:
        return {"iterations": [it.to_dict() for it in self.iterations]}


if __name__ == "__main__":
    pytest.main([__file__])


class TestReviewRegressions:
    """Regressions found in code review of the seam work."""

    def test_retry_enabled_on_the_floor_policy_does_not_crash(self):
        """Turning retry on without naming a count is the most natural first edit.

        The shipped floor policy carries every field as an explicit ``None``, so a
        ``get(..., 1)`` default never fires and the budget computation used to raise
        ``TypeError`` before any sub-call was made.
        """
        from shrlm.rlm_harness import build_runtime_policy

        policy = build_runtime_policy()
        policy["enabled"] = True
        policy["retry_on_syntax_error"] = True

        env = LocalREPL(runtime_policy=policy)
        try:
            assert env._retry_budget() == 1
        finally:
            env.cleanup()

    def test_truncation_event_fires_for_a_block_just_over_the_threshold(self):
        """A just-over-threshold block comes back longer than it went in.

        The default builder replaces the tail with a ``... + [N chars...]`` marker,
        so a shrinkage test reports no truncation for exactly the blocks that sit
        closest to the boundary — dropping the event from the failure signature.
        """
        oversize = "x" * (DEFAULT_MAX_CHARACTER_LENGTH + 5)
        iteration = make_iteration(stdout=oversize)
        metrics: dict[str, object] = {}

        format_iteration(iteration, metrics=metrics)

        assert metrics["truncation_event"] is True

    def test_untruncated_block_reports_no_truncation_event(self):
        iteration = make_iteration(stdout="short output")
        metrics: dict[str, object] = {}

        format_iteration(iteration, metrics=metrics)

        assert metrics["truncation_event"] is False


# ---------------------------------------------------------------------------
# S10 construction checks (U7): refused harnesses, and the ones that pass
# ---------------------------------------------------------------------------


def skill(
    name: str = "chunk_context",
    description: str = "when `context` is larger than one sub-call can take",
    body: str = "1. Measure len(context).\n2. Split it into sub-call-sized slices.",
) -> SkillEntry:
    """A legal S10 entry with ``overrides`` applied."""
    return SkillEntry(name=name, description=description, body=body)


def skilled(*entries: SkillEntry, base: Harness = H0, **overrides: Any) -> Harness:
    """``base`` carrying ``entries`` as its S10, plus any other field overrides."""
    return replace(base, skills=list(entries), **overrides)


def build(harness: Harness):
    return build_harnessed_rlm(harness, backend="openai", backend_kwargs={"model_name": "t"})


def enabled_policy(**overrides: Any) -> dict[str, Any]:
    return build_runtime_policy() | {"enabled": True} | overrides


class TestS10IndexFormatSafety:
    """R5 / KTD8: the assembled prompt is formatted once, at construction."""

    def test_brace_in_a_description_is_refused_naming_s10(self):
        harness = skilled(skill(description="when `answer` looks like {partial}"))
        with pytest.raises(ValueError, match="S10"):
            check_harness(harness)
        with pytest.raises(ValueError, match="S10"):
            build(harness)

    def test_literal_braces_in_a_body_are_accepted_and_returned_verbatim(self):
        body = '1. Build {"ready": False}.\n2. Fill answer["content"] from {slices}.'
        harness = skilled(skill(body=body))
        check_harness(harness)
        harnessed = build(harness)
        assert SKILL_LOADER_NAME in harnessed.rlm.custom_tools
        assert harnessed.rlm.custom_tools[SKILL_LOADER_NAME]["tool"]("chunk_context") == body

    def test_a_stray_brace_in_s2_is_refused_naming_s2(self):
        # The gap ``escape_braces`` was exported to close: before U7 this died
        # with KeyError at completion time, as an infrastructure error.
        harness = replace(H0, decomposition_instruction="Probe {context} first.")
        with pytest.raises(ValueError, match="S2"):
            check_harness(harness)

    def test_an_unbalanced_brace_in_s4_is_refused_naming_s4(self):
        harness = replace(H0, verification_instruction="Check answer[ready} before flipping.")
        with pytest.raises(ValueError, match="S4"):
            check_harness(harness)

    def test_escaped_braces_in_a_prompt_surface_are_accepted(self):
        text = escape_braces('Keep answer as {"ready": False, "content": ""}.')
        harness = replace(H0, execution_instruction=text)
        check_harness(harness)
        harnessed = build(harness)
        assert '{"ready": False, "content": ""}' in harnessed.rlm._setup_prompt("q")[0]["content"]

    def test_the_placeholder_slot_stays_the_one_live_field(self):
        # S1 carries the slot once; formatting consumed it and nothing else, so
        # the prompt the runtime produces carries the rendered index verbatim
        # and no unconsumed field.
        harness = skilled(skill(), skill(name="recheck_quotes"))
        system = build(harness).rlm._setup_prompt("q")[0]["content"]
        assert "{custom_tools_section}" not in system
        assert "- `chunk_context`: when `context` is larger than one sub-call can take" in system


class TestS10StatedLimits:
    """R6: each body and description is scanned with the prompt-level patterns."""

    def test_body_stating_a_different_truncation_bound_is_refused_naming_s10(self):
        body = "1. Note REPL outputs over ~5K characters are truncated.\n2. Print less."
        with pytest.raises(ValueError, match=r"S10.*truncation") as caught:
            check_harness(skilled(skill(body=body)))
        assert "~5K" in str(caught.value) and "~20K" in str(caught.value)

    def test_body_stating_the_active_truncation_bound_is_accepted(self):
        body = "1. Note REPL outputs over ~20K characters are truncated.\n2. Print less."
        check_harness(skilled(skill(body=body)))

    def test_description_stating_a_different_truncation_bound_is_refused_naming_s10(self):
        description = "when REPL outputs over ~5K characters are truncated"
        with pytest.raises(ValueError, match=r"S10.*truncation"):
            check_harness(skilled(skill(description=description)))

    def test_body_stating_a_capacity_the_policy_contradicts_is_refused_naming_s10(self):
        body = "1. Send ~100K characters per prompt.\n2. Collect the replies."
        harness = skilled(skill(body=body), runtime_policy=enabled_policy(max_prompt_chars=50_000))
        with pytest.raises(ValueError, match=r"S10.*capacity"):
            check_harness(harness)
        # The same body under a policy that agrees, or states no cap, is fine.
        check_harness(
            skilled(skill(body=body), runtime_policy=enabled_policy(max_prompt_chars=100_000))
        )
        check_harness(skilled(skill(body=body)))

    def test_body_stating_a_fan_out_the_policy_contradicts_is_refused_naming_s10(self):
        body = "1. Batch ~20 prompts per batch.\n2. Collect the replies."
        harness = skilled(skill(body=body), runtime_policy=enabled_policy(max_batch_width=4))
        with pytest.raises(ValueError, match=r"S10.*batch"):
            check_harness(harness)
        check_harness(skilled(skill(body=body), runtime_policy=enabled_policy(max_batch_width=20)))

    def test_prompt_level_scan_is_unaffected_by_legal_index_text(self):
        # The index joins the assembled prompt; with no field stating a limit,
        # the expected-set equality the prompt-level scan asserts still holds.
        harness = skilled(
            skill(), skill(name="recheck_quotes", description="when quotes are reused")
        )
        check_stated_limits(harness)
        check_harness(harness)


class TestS10EntryShape:
    """R1 / R14 at construction: each entry is a well-formed ``SkillEntry``."""

    @pytest.mark.parametrize(
        "skills, needle",
        [
            ([{"name": "a", "description": "b", "body": "1. x\n2. y"}], "SkillEntry"),
            ([SkillEntry(name=1, description="b", body="1. x\n2. y")], "name"),
            ([SkillEntry(name="ok", description=None, body="1. x\n2. y")], "description"),
            ([SkillEntry(name="ok", description="b", body=["1. x", "2. y"])], "body"),
            ([skill(name="")], "name"),
            ([skill(name="not an identifier")], "identifier"),
            ([skill(name="class")], "identifier"),
            ([skill(name="café")], "identifier"),
            ([skill(description="   ")], "description"),
            ([skill(description="two\nlines")], "single line"),
            ([skill(body="")], "body"),
            ([skill(body="Just prose, no steps at all.")], "ordered steps"),
            ([skill(body="1. only one step")], "ordered steps"),
            ([skill(), skill(description="a second entry, same name")], "unique"),
        ],
    )
    def test_malformed_entries_are_refused_naming_s10(self, skills, needle):
        harness = replace(H0, skills=skills)
        with pytest.raises(ValueError, match="S10") as caught:
            check_harness(harness)
        assert needle in str(caught.value)

    def test_skills_that_is_not_a_list_is_refused_naming_s10(self):
        with pytest.raises(ValueError, match="S10"):
            check_harness(replace(H0, skills=(skill(),)))

    def test_a_legal_multi_entry_s10_passes_and_is_constructed(self):
        harness = skilled(
            skill(),
            skill(name="recheck_quotes", description="when the answer quotes `context`"),
            skill(name="split_by_heading", description="when `context` has markdown headings"),
        )
        check_harness(harness)
        harnessed = build(harness)
        for name in ("chunk_context", "recheck_quotes", "split_by_heading"):
            assert f"- `{name}`:" in harnessed.system_prompt

    def test_h0_and_h0_star_pass_unchanged(self):
        for harness in (H0, H0_STAR):
            check_harness(harness)
            assert harness.skills == []


class TestS10Caps:
    """R7: the proposal-time numbers hold at construction too."""

    def test_name_over_the_cap_is_refused(self):
        name = "n" * (SKILL_NAME_MAX_CHARS + 1)
        with pytest.raises(ValueError, match=rf"S10.*name.*{SKILL_NAME_MAX_CHARS}"):
            check_harness(skilled(skill(name=name)))
        check_harness(skilled(skill(name="n" * SKILL_NAME_MAX_CHARS)))

    def test_description_over_the_cap_is_refused(self):
        description = "d" * (SKILL_DESCRIPTION_MAX_CHARS + 1)
        with pytest.raises(ValueError, match=rf"S10.*description.*{SKILL_DESCRIPTION_MAX_CHARS}"):
            check_harness(skilled(skill(description=description)))
        check_harness(skilled(skill(description="d" * SKILL_DESCRIPTION_MAX_CHARS)))

    def test_body_over_the_cap_is_refused(self):
        body = "1. a\n2. " + "b" * SKILL_BODY_MAX_CHARS
        with pytest.raises(ValueError, match=rf"S10.*body.*{SKILL_BODY_MAX_CHARS}"):
            check_harness(skilled(skill(body=body)))
        at_cap = "1. a\n2. " + "b" * (SKILL_BODY_MAX_CHARS - len("1. a\n2. "))
        assert len(at_cap) == SKILL_BODY_MAX_CHARS
        check_harness(skilled(skill(body=at_cap)))

    def test_too_many_entries_is_refused(self):
        names = [f"skill_{i}" for i in range(SKILL_MAX_ENTRIES + 1)]
        with pytest.raises(ValueError, match=rf"S10.*{SKILL_MAX_ENTRIES}"):
            check_harness(skilled(*(skill(name=n) for n in names)))
        check_harness(skilled(*(skill(name=n) for n in names[:SKILL_MAX_ENTRIES])))

    def test_total_over_the_cap_is_refused(self):
        per_body = SKILL_BODY_MAX_CHARS - 100
        count = SKILL_TOTAL_MAX_CHARS // per_body + 1
        assert count <= SKILL_MAX_ENTRIES
        body = "1. a\n2. " + "b" * (per_body - len("1. a\n2. "))
        entries = [skill(name=f"skill_{i}", body=body) for i in range(count)]
        with pytest.raises(ValueError, match=rf"S10.*total.*{SKILL_TOTAL_MAX_CHARS}"):
            check_harness(skilled(*entries))


class TestS10LoaderInstallAndReservedName:
    """KTD9 / R15 at construction: install only when non-empty; the name is reserved."""

    def test_h0_and_h0_star_install_no_loader(self):
        for harness in (H0, H0_STAR):
            harnessed = build(harness)
            assert SKILL_LOADER_NAME not in (harnessed.rlm.custom_tools or {})
            assert SKILL_LOADER_NAME not in (harnessed.rlm.custom_sub_tools or {})

    def test_non_empty_s10_installs_the_loader_in_both_dicts(self):
        harnessed = build(skilled(skill(), skill(name="recheck_quotes")))
        for tools in (harnessed.rlm.custom_tools, harnessed.rlm.custom_sub_tools):
            entry = tools[SKILL_LOADER_NAME]
            assert entry["tool"]("recheck_quotes") == skill().body
        # The harness's own S8 dicts were never mutated.
        assert H0.repl_helpers == {} and H0.sub_repl_helpers == {}

    @pytest.mark.parametrize("field", ["repl_helpers", "sub_repl_helpers"])
    def test_an_s8_helper_binding_the_reserved_name_is_refused_naming_s10(self, field):
        def load_skill(name: str) -> str:
            return "not the harness loader"

        harness = replace(H0, **{field: {SKILL_LOADER_NAME: load_skill}})
        with pytest.raises(ValueError, match=rf"S8 {field}.*S10") as caught:
            check_plumbing(harness)
        assert SKILL_LOADER_NAME in str(caught.value)
        with pytest.raises(ValueError, match="S10"):
            build(harness)
        # The same refusal with S10 populated, so the collision is never a
        # silent overwrite at install time either.
        with pytest.raises(ValueError, match="S10"):
            build(skilled(skill(), **{field: {SKILL_LOADER_NAME: load_skill}}))

    def test_installing_over_a_residual_binding_is_a_refusal_not_an_overwrite(self):
        loader = build_skill_loader([skill()])
        with pytest.raises(ValueError, match="S10"):
            _install_skill_loader({SKILL_LOADER_NAME: lambda name: "shadow"}, loader)
        merged = _install_skill_loader({"helper": lambda: 1}, loader)
        assert list(merged) == ["helper", SKILL_LOADER_NAME]
        assert merged[SKILL_LOADER_NAME]["tool"] is loader
