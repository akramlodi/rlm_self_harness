"""Tests for the U1 runtime seams: S7 metadata, S9 answer middleware, S6 policy, trace metrics.

Every seam must default to current behavior, so each test comes in a pair:
the configured behavior fires, and the unconfigured default is unchanged.
"""

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
from rlm.environments.local_repl import (
    SUBCALL_INVALID_PREFIX,
    SUBCALL_REFUSAL_PREFIX,
    LocalREPL,
)
from rlm.utils.parsing import (
    DEFAULT_MAX_CHARACTER_LENGTH,
    build_repl_inventory,
    format_execution_result,
    format_iteration,
)
from rlm.utils.prompts import DEFAULT_CAPACITY_SENTENCE, build_rlm_system_prompt

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
