"""Synthetic trajectory fixtures for the weakness-mining tests.

These builders double as documentation of the logged trajectory format the
walker consumes. The shape mirrors ``rlm.core.types.RLMChatCompletion.to_dict``
exactly: a completion dict whose ``metadata`` holds ``run_metadata`` (the
``RLMMetadata.to_dict`` payload) and ``iterations`` (one ``RLMLogger.log``
entry per root turn, each carrying ``code_blocks[].result`` with ``stdout``,
``stderr``, ``locals``, ``rlm_calls`` and ``final_answer``). Nested recursive
children are ``rlm_calls`` entries that carry a complete ``metadata`` of their
own; plain ``llm_query`` calls carry none.

Four canonical runs are provided:

* ``shallow_run`` -- a root that answered directly, no sub-calls of any kind.
* ``nested_run`` -- a depth-2 tree: root -> recursive child -> grandchild,
  plus one plain LM leaf. ``run_metadata.max_depth`` is deliberately the same
  constant in root and child, because the runtime copies it unchanged; depth
  must come from nesting.
* ``swallowed_error_run`` -- one sub-call raised and survives only as the
  ``Error: RLM query failed - `` marker printed to stdout, plus one errored
  completion that did make it into ``rlm_calls``.
* ``fallback_run`` -- the iteration budget ran out, so the trailing iteration
  has ``code_blocks=[]`` and a synthesized ``final_answer``.
"""

from typing import Any

from rlm.core.types import RLMChatCompletion, UsageSummary
from shrlm.optimization.taxonomy import (
    AgentMechanism,
    CausalStatus,
    FailingLevel,
    VerifierCause,
)
from shrlm.optimization.types import (
    AttributionDetail,
    FailureRecord,
    FailureSignature,
    MiningConfig,
    TraceIntegrity,
    TreeStats,
    Verdict,
)

ROOT_MODEL = "mock-model"

# A distinctive value planted in REPL locals. The digest must never leak it:
# locals are lossy once serialized and are deliberately not carried on the tree.
SECRET_LOCAL = "XYZZY-SECRET-LOCAL-VALUE-DO-NOT-LEAK"

# Sized so the collapse ratio of the nested run is exactly 300 / 400 = 0.75.
NESTED_ROOT_CONTEXT = "C" * 400
NESTED_CHILD_PROMPT = "c" * 300


def usage(calls: int = 1) -> dict[str, Any]:
    """A ``UsageSummary.to_dict`` payload for a call that really ran."""
    return {
        "model_usage_summaries": {
            ROOT_MODEL: {
                "total_calls": calls,
                "total_input_tokens": 100,
                "total_output_tokens": 20,
            }
        }
    }


def empty_usage() -> dict[str, Any]:
    """The empty usage every error completion carries (see rlm/core/rlm.py)."""
    return {"model_usage_summaries": {}}


def run_metadata(max_depth: int, environment_type: str = "local") -> dict[str, Any]:
    """An ``RLMMetadata.to_dict`` payload, the ``run_metadata`` of a trajectory."""
    return {
        "root_model": ROOT_MODEL,
        "max_depth": max_depth,
        "max_iterations": 5,
        "backend": "openai",
        "backend_kwargs": {},
        "environment_type": environment_type,
        "environment_kwargs": {},
        "other_backends": None,
    }


def code_block(
    code: str,
    stdout: str = "",
    stderr: str = "",
    locals_: dict[str, Any] | None = None,
    rlm_calls: list[dict[str, Any]] | None = None,
    final_answer: str | None = None,
) -> dict[str, Any]:
    """One executed ```repl block: ``CodeBlock.to_dict`` with its ``REPLResult``."""
    return {
        "code": code,
        "result": {
            "stdout": stdout,
            "stderr": stderr,
            "locals": locals_ or {},
            "execution_time": 0.1,
            "rlm_calls": rlm_calls or [],
            "final_answer": final_answer,
        },
    }


def iteration_entry(
    index: int,
    response: str,
    code_blocks: list[dict[str, Any]],
    final_answer: str | None = None,
) -> dict[str, Any]:
    """One ``RLMLogger.log`` entry: the iteration payload plus the logger's envelope."""
    return {
        "type": "iteration",
        "iteration": index,
        "timestamp": "2026-01-01T00:00:00",
        "prompt": "(prompt elided)",
        "response": response,
        "code_blocks": code_blocks,
        "final_answer": final_answer,
        "iteration_time": 0.5,
    }


def completion_dict(
    prompt: str | dict[str, Any],
    response: str,
    iterations: list[dict[str, Any]],
    max_depth: int,
    environment_type: str = "local",
) -> dict[str, Any]:
    """A serialized root ``RLMChatCompletion`` carrying a full trajectory."""
    return {
        "root_model": ROOT_MODEL,
        "prompt": prompt,
        "response": response,
        "usage_summary": usage(),
        "execution_time": 1.0,
        "metadata": {
            "run_metadata": run_metadata(max_depth, environment_type),
            "iterations": iterations,
        },
    }


def as_completion(data: dict[str, Any]) -> RLMChatCompletion:
    """Rehydrate a fixture dict through the real ``from_dict`` path."""
    return RLMChatCompletion.from_dict(data)


def no_metadata_completion() -> RLMChatCompletion:
    """A completion from an RLM constructed without a logger: no trajectory at all."""
    return RLMChatCompletion(
        root_model=ROOT_MODEL,
        prompt="what is 2 + 2?",
        response="4",
        usage_summary=UsageSummary.from_dict(usage()),
        execution_time=0.2,
    )


def shallow_run() -> dict[str, Any]:
    """Root only: max_depth=1, so recursion was never available, and no sub-calls."""
    block = code_block(
        code="result = 2 + 2\nanswer['content'] = str(result)\nanswer['ready'] = True",
        stdout="",
        locals_={"result": 4, "scratch": SECRET_LOCAL},
        final_answer="4",
    )
    return completion_dict(
        prompt="what is 2 + 2?",
        response="4",
        iterations=[
            iteration_entry(
                index=1,
                response="Trivial.\n```repl\nanswer['content'] = '4'\n```",
                code_blocks=[block],
                final_answer="4",
            )
        ],
        max_depth=1,
    )


def nested_run() -> dict[str, Any]:
    """Depth-2 tree: root -> recursive child -> grandchild, plus one plain LM leaf.

    ``max_depth`` is 2 everywhere (the runtime copies it unchanged into the
    child), so observed depth 2 is derivable only from nesting. The grandchild
    sits at depth 2 == max_depth with no metadata, which is exactly the
    indeterminate rlm-vs-llm case the walker refuses to invent an answer for.
    """
    grandchild = {
        "root_model": ROOT_MODEL,
        "prompt": "summarize the middle slice",
        "response": "the middle slice says B",
        "usage_summary": usage(),
        "execution_time": 0.2,
    }
    child_block = code_block(
        code="part = rlm_query('summarize the middle slice')\nprint(part)",
        stdout="the middle slice says B\n",
        rlm_calls=[grandchild],
    )
    child = {
        "root_model": ROOT_MODEL,
        "prompt": NESTED_CHILD_PROMPT,
        "response": "slice summary: B",
        "usage_summary": usage(2),
        "execution_time": 0.6,
        "metadata": {
            "run_metadata": run_metadata(max_depth=2),
            "iterations": [
                iteration_entry(
                    index=1,
                    response="Splitting further.\n```repl\npart = rlm_query(...)\n```",
                    code_blocks=[child_block],
                    final_answer="slice summary: B",
                )
            ],
        },
    }
    leaf = {
        "root_model": ROOT_MODEL,
        "prompt": "classify the header line",
        "response": "header: report",
        "usage_summary": usage(),
        "execution_time": 0.1,
    }
    root_block = code_block(
        code=(
            "slice_summary = rlm_query(context[:300])\n"
            "header_kind = llm_query('classify the header line')\n"
            "print(slice_summary)\n"
            "print(header_kind)\n"
            "# keep intermediate results in variables for the final merge step\n"
        ),
        stdout="slice summary: B\nheader: report\n",
        locals_={"slice_summary": "slice summary: B", "scratch": SECRET_LOCAL},
        rlm_calls=[child, leaf],
    )
    commit_block = code_block(
        code="answer['content'] = 'B'\nanswer['ready'] = True",
        final_answer="B",
    )
    return completion_dict(
        prompt=NESTED_ROOT_CONTEXT,
        response="B",
        iterations=[
            iteration_entry(
                index=1,
                response="Delegating the slice.\n```repl\n...\n```",
                code_blocks=[root_block],
            ),
            iteration_entry(
                index=2,
                response="Committing.\n```repl\nanswer['ready'] = True\n```",
                code_blocks=[commit_block],
                final_answer="B",
            ),
        ],
        max_depth=2,
    )


def swallowed_error_run() -> dict[str, Any]:
    """One sub-call raised and vanished; another errored but stayed in ``rlm_calls``.

    The raised sub-call survives only as the ``Error: RLM query failed - ``
    string it returned into the REPL, which the model printed to stdout. The
    errored completion that did get recorded has an error-string response and
    the empty usage every error completion carries, so the walker classifies
    it ERRORED rather than mistaking it for a leaf.
    """
    errored_call = {
        "root_model": ROOT_MODEL,
        "prompt": "count the entries in part two",
        "response": "Error: LM query failed - connection reset",
        "usage_summary": empty_usage(),
        "execution_time": 0.0,
    }
    block = code_block(
        code=(
            "first = rlm_query('count the entries in part one')\n"
            "second = llm_query('count the entries in part two')\n"
            "print(first)\nprint(second)"
        ),
        stdout=(
            "Error: RLM query failed - ZeroDivisionError: division by zero\n"
            "Error: LM query failed - connection reset\n"
        ),
        rlm_calls=[errored_call],
    )
    commit_block = code_block(
        code="answer['content'] = '17'\nanswer['ready'] = True",
        final_answer="17",
    )
    return completion_dict(
        prompt="how many entries are there in total?",
        response="17",
        iterations=[
            iteration_entry(
                index=1,
                response="Counting both parts.\n```repl\n...\n```",
                code_blocks=[block],
            ),
            iteration_entry(
                index=2,
                response="Committing anyway.\n```repl\n...\n```",
                code_blocks=[commit_block],
                final_answer="17",
            ),
        ],
        max_depth=2,
    )


def fallback_run() -> dict[str, Any]:
    """The iteration budget ran out: RLM._default_answer logs a trailing
    iteration with no code blocks and a synthesized final answer."""
    block = code_block(
        code="notes = context.splitlines()\nprint(len(notes))",
        stdout="12\n",
    )
    return completion_dict(
        prompt="summarize the log",
        response="(fallback) the log describes twelve events",
        iterations=[
            iteration_entry(
                index=1,
                response="Let me look at the log first.\n```repl\n...\n```",
                code_blocks=[block],
            ),
            iteration_entry(
                index=2,
                response="(fallback) the log describes twelve events",
                code_blocks=[],
                final_answer="(fallback) the log describes twelve events",
            ),
        ],
        max_depth=2,
    )


def make_stats(**overrides: Any) -> TreeStats:
    """A sane ``TreeStats`` matching the nested run, with per-test overrides."""
    values: dict[str, Any] = {
        "n_nodes": 4,
        "n_rlm_children": 1,
        "n_llm_leaves": 1,
        "n_errored": 0,
        "n_indeterminate": 1,
        "max_observed_depth": 2,
        "n_iterations": 2,
        "root_context_chars": 400,
        "max_child_prompt_chars": 300,
        "collapse_ratio": 0.75,
        "terminated_by_fallback": False,
        "recursion_available": True,
        "block_attribution_reliable": True,
        "suspected_lost_subcalls": 0,
        "trace_integrity": TraceIntegrity.COMPLETE,
    }
    values.update(overrides)
    return TreeStats(**values)


def make_verdict(
    cause: VerifierCause = VerifierCause.WRONG_VALUE,
    gold: str = "B",
    produced: str = "C",
) -> Verdict:
    return Verdict(passed=False, cause=cause, gold=gold, produced=produced)


def make_signature(
    cause: VerifierCause = VerifierCause.WRONG_VALUE,
    level: FailingLevel = FailingLevel.CHILD,
    status: CausalStatus = CausalStatus.CAUSAL,
    mechanism: AgentMechanism = AgentMechanism.LOSSY_AGGREGATION,
) -> FailureSignature:
    return FailureSignature(
        verifier_cause=cause,
        failing_level=level,
        causal_status=status,
        agent_mechanism=mechanism,
    )


def make_record(
    instance_id: str,
    signature: FailureSignature | None = None,
    detail: AttributionDetail | None = None,
    stats: TreeStats | None = None,
    level_grounded: bool = True,
    attribution_failed: bool = False,
    verdict: Verdict | None = None,
) -> FailureRecord:
    if signature is None and not attribution_failed:
        signature = make_signature()
    if detail is None and signature is not None:
        detail = AttributionDetail(
            symptom_summary="the merge step dropped one sub-result",
            evidence_node_ids=["r/i0/b0/c0"],
            agent_mechanism_detail="merge dropped a sub-result",
        )
    return FailureRecord(
        instance_id=instance_id,
        verdict=verdict or make_verdict(),
        stats=stats or make_stats(),
        signature=signature,
        detail=detail,
        level_grounded=level_grounded,
        digest_sha256="0" * 64,
        attribution_failed=attribution_failed,
        attribution_error="no valid attribution after 3 attempts" if attribution_failed else "",
    )


def make_config(**overrides: Any) -> MiningConfig:
    values: dict[str, Any] = {
        "round_index": 3,
        "harness_version": "H0",
        "split_id": "held_in_v1",
        "taxonomy_version": "1.0.0",
        "prompt_version": "1.0.0",
        "digest_version": "1.0.0",
        "prompt_sha256": "a" * 64,
        "attributor_model": ROOT_MODEL,
        "attributor_sampling_args": {"temperature": 0.0},
        "digest_char_budget": 12000,
        "digest_focus_k": 4,
        "max_attempts": 3,
        "sub_verifier_enabled": True,
        "min_support": 2,
        "actionability_weights": {
            "causal": 0.4,
            "grounded": 0.3,
            "surface": 0.2,
            "homogeneity": 0.1,
        },
        "verifier_config": {},
        "sampling_seed": None,
        "validator_version": "1.0.0",
        "attribution_cache_path": None,
        "harness_hash": "",
    }
    values.update(overrides)
    return MiningConfig(**values)
