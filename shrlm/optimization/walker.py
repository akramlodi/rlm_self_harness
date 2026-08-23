"""
Reconstruct a recursive call tree from a logged RLM trajectory.

The runtime records a run as nested dictionaries: a trajectory holds
iterations, an iteration holds code blocks, a code block's REPL result holds
the model calls issued from inside it, and any call that was itself a
recursive RLM carries a complete trajectory of its own. This module turns that
into an addressable tree and reports what the trace could not tell us.

Three properties of the logged format drive the design and are handled
explicitly rather than assumed away:

* ``rlm_calls`` commingles plain ``llm_query`` calls with recursive
  ``rlm_query`` children. The runtime's own discriminator is the presence of
  ``metadata`` (see rlm/logger/verbose.py), which has known false negatives.
* Depth is not recorded anywhere. ``run_metadata.max_depth`` is a constant that
  is copied unchanged into every child, so depth must come from nesting.
* A sub-call that raises is swallowed into a bare string and appends nothing to
  ``rlm_calls`` (rlm/environments/local_repl.py), so the node disappears. We
  cannot recover it, but we can count the evidence it left behind.
"""

import json
from dataclasses import dataclass
from typing import Any

from rlm.core.types import RLMChatCompletion
from shrlm.optimization.types import (
    CallNode,
    CodeBlockNode,
    IterationNode,
    NodeKind,
    TraceIntegrity,
    TreeStats,
    iter_nodes,
)

ERROR_PREFIX = "Error: "

# Emitted by LocalREPL._rlm_query and its batched variants when a sub-call
# raises. The string is *returned into the REPL* and never recorded as a
# completion, so every occurrence in captured output marks a child that is
# missing from the tree.
LOST_SUBCALL_MARKER = "Error: RLM query failed - "

# Exact prefixes constructed in rlm/core/rlm.py and rlm/environments/local_repl.py.
# Ordered longest-first so a specific prefix wins over a generic one.
ERROR_KIND_PREFIXES: tuple[tuple[str, str], ...] = (
    ("Error: Child RLM completion failed - ", "child_completion_failed"),
    ("Error: Child RLM budget exceeded - ", "child_budget_exceeded"),
    ("Error: LM query failed at max depth - ", "max_depth_lm_failed"),
    ("Error: No LM handler configured", "no_lm_handler"),
    ("Error: RLM query failed - ", "rlm_query_failed"),
    ("Error: LM query failed - ", "lm_query_failed"),
    ("Error: Budget exhausted ", "budget_exhausted"),
    ("Error: Timeout exhausted ", "timeout_exhausted"),
)

UNKNOWN_ERROR_KIND = "unknown"

# Code-block association of calls is unreliable in this environment: calls can
# be attributed to a later block than the one that issued them.
UNRELIABLE_BLOCK_ATTRIBUTION_ENVS = frozenset({"ipython"})


@dataclass(frozen=True)
class WalkContext:
    """Root-level facts that govern classification of every node below."""

    root_max_depth: int
    recursion_available: bool
    block_attribution_reliable: bool


def text_length(value: Any) -> int:
    """Character length of a prompt, which may be a string, dict, or message list."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, default=str))


def classify_error_kind(message: str) -> str:
    for prefix, kind in ERROR_KIND_PREFIXES:
        if message.startswith(prefix):
            return kind
    return UNKNOWN_ERROR_KIND


def classify_node(
    call: dict[str, Any], depth: int, ctx: WalkContext
) -> tuple[NodeKind, str | None]:
    """
    Decide what a logged call was.

    Order matters. Errors are tested first because an error completion carries
    neither metadata nor usage and would otherwise be mistaken for a leaf.
    """
    response = call.get("response") or ""
    usage = (call.get("usage_summary") or {}).get("model_usage_summaries") or {}

    error_field = call.get("error")
    if error_field is not None:
        return NodeKind.ERRORED, classify_error_kind(str(error_field))

    # The empty-usage conjunct matters: every error completion built in
    # rlm/core/rlm.py has an empty UsageSummary, so a genuine answer that
    # happens to begin "Error: " is not misclassified.
    if response.startswith(ERROR_PREFIX) and not usage:
        return NodeKind.ERRORED, classify_error_kind(response)

    if call.get("metadata") is not None:
        return NodeKind.RLM_CHILD, None

    # max_depth == 1 leaves subcall_fn unset, so rlm_query silently degrades to
    # llm_query. Every call really was a plain LM call; the tree-level
    # recursion_available flag records that recursion was never possible.
    if not ctx.recursion_available:
        return NodeKind.LLM_LEAF, None

    # At or beyond max_depth an rlm_query falls back to a plain LM completion,
    # producing a record byte-identical to an llm_query. The distinction is
    # genuinely absent from the trace, so we decline to invent it.
    if depth >= ctx.root_max_depth:
        return NodeKind.INDETERMINATE, None

    return NodeKind.LLM_LEAF, None


def build_call_tree(completion: RLMChatCompletion) -> CallNode:
    """
    Reconstruct the call tree of a completed run.

    Raises if the trajectory is absent, which means the RLM was constructed
    without an RLMLogger. That is unrecoverable rather than degraded: children
    inherit the parent's logger, so the entire tree is missing, not just the
    root.
    """
    if completion.metadata is None:
        raise ValueError(
            "Completion has no trajectory metadata. Construct the RLM with "
            "logger=RLMLogger() -- children inherit the parent's logger, so without one "
            "no part of the call tree is recorded."
        )
    return build_call_tree_from_dict(completion.to_dict())


def build_call_tree_from_dict(data: dict[str, Any]) -> CallNode:
    """Reconstruct the call tree from a serialized RLMChatCompletion."""
    metadata = data.get("metadata")
    if metadata is None:
        raise ValueError("Serialized completion has no trajectory metadata")

    run_metadata = metadata.get("run_metadata") or {}
    max_depth = int(run_metadata.get("max_depth", 1))
    environment_type = str(run_metadata.get("environment_type", "local"))
    ctx = WalkContext(
        root_max_depth=max_depth,
        recursion_available=max_depth > 1,
        block_attribution_reliable=environment_type not in UNRELIABLE_BLOCK_ATTRIBUTION_ENVS,
    )

    return build_node(
        call=data,
        node_id="r",
        parent_id=None,
        depth=0,
        kind=NodeKind.ROOT,
        error_kind=None,
        ctx=ctx,
    )


def build_node(
    call: dict[str, Any],
    node_id: str,
    parent_id: str | None,
    depth: int,
    kind: NodeKind,
    error_kind: str | None,
    ctx: WalkContext,
) -> CallNode:
    """Build one node and, when it carries a trajectory, everything beneath it."""
    node = CallNode(
        node_id=node_id,
        parent_id=parent_id,
        kind=kind,
        depth=depth,
        model=str(call.get("root_model", "unknown")),
        prompt=call.get("prompt", ""),
        response=str(call.get("response", "")),
        prompt_chars=text_length(call.get("prompt")),
        response_chars=text_length(call.get("response")),
        execution_time=call.get("execution_time"),
        error_kind=error_kind,
        ambiguous=kind is NodeKind.INDETERMINATE,
    )

    metadata = call.get("metadata")
    if metadata is None:
        return node

    # The run-start record of what skills this node's REPL could load. Absent
    # (None) when no loader was installed; an rlm_query child carries its own,
    # because the loader is installed in the child namespace too.
    skill_index = (metadata.get("run_metadata") or {}).get("skill_index")
    if skill_index is not None:
        node.skill_index = [dict(entry) for entry in skill_index]

    for iteration_index, entry in enumerate(metadata.get("iterations") or []):
        node.iterations.append(build_iteration(entry, iteration_index, node, ctx))

    return node


def build_iteration(
    entry: dict[str, Any], iteration_index: int, parent: CallNode, ctx: WalkContext
) -> IterationNode:
    """Build one iteration, appending any calls it made to the parent's children."""
    code_blocks_raw = entry.get("code_blocks") or []
    final_answer = entry.get("final_answer")

    blocks: list[CodeBlockNode] = []
    for block_index, block_entry in enumerate(code_blocks_raw):
        result = block_entry.get("result") or {}
        block = CodeBlockNode(
            code=str(block_entry.get("code", "")),
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            final_answer=result.get("final_answer"),
            # Loader invocations recorded beside rlm_calls; a pre-S10 trace
            # has no key at all, which reads the same as no loads.
            skill_loads=[dict(event) for event in (result.get("skill_loads") or [])],
        )

        for call_index, call in enumerate(result.get("rlm_calls") or []):
            child_id = f"{parent.node_id}/i{iteration_index}/b{block_index}/c{call_index}"
            child_depth = parent.depth + 1
            child_kind, child_error_kind = classify_node(call, child_depth, ctx)
            child = build_node(
                call=call,
                node_id=child_id,
                parent_id=parent.node_id,
                depth=child_depth,
                kind=child_kind,
                error_kind=child_error_kind,
                ctx=ctx,
            )
            block.calls.append(child)
            parent.children.append(child)

        blocks.append(block)

    return IterationNode(
        index=int(entry.get("iteration", iteration_index + 1)),
        response=str(entry.get("response", "")),
        code_blocks=blocks,
        final_answer=final_answer,
        # RLM._default_answer logs a trailing iteration with no code blocks when
        # the iteration budget runs out. The success path always logs the
        # committing iteration together with the code that committed it.
        terminated_by_fallback=not code_blocks_raw and final_answer is not None,
    )


def iter_code_blocks(root: CallNode):
    """Every code block in the tree, in trajectory order."""
    for node in iter_nodes(root):
        for iteration in node.iterations:
            yield from iteration.code_blocks


def iter_skill_loads(root: CallNode):
    """Every recorded skill load in the tree as ``(skill, depth)``, in trajectory order.

    Walks the root's blocks first and then each child's, in the same order
    ``iter_nodes`` visits them, so the digest's loaded-skills line is a
    deterministic function of the tree. Depth is the recording environment's
    own depth (a child's load is listed at the child's depth), which is what
    separates "the root consulted it" from "a child did".
    """
    for block in iter_code_blocks(root):
        for event in block.skill_loads:
            yield str(event.get("skill", "")), int(event.get("depth", 0))


def count_lost_subcalls(root: CallNode) -> int:
    """
    Lower bound on sub-calls missing from the tree.

    A sub-call that raises returns its error as a *value* inside the REPL and is
    never recorded as a completion. When the model printed that value we can see
    it; when it did not, the sub-call is invisible and this count understates.
    """
    total = 0
    for block in iter_code_blocks(root):
        total += block.stdout.count(LOST_SUBCALL_MARKER)
        total += block.stderr.count(LOST_SUBCALL_MARKER)
    return total


def compute_tree_stats(root: CallNode, ctx: WalkContext | None = None) -> TreeStats:
    """
    Mechanical statistics over a reconstructed tree.

    Nothing here consults a language model, which is what allows cluster-level
    shared symptoms to remain a deterministic function of the records.
    """
    if ctx is None:
        run_max_depth = max((node.depth for node in iter_nodes(root)), default=0)
        ctx = WalkContext(
            root_max_depth=run_max_depth,
            recursion_available=run_max_depth > 0,
            block_attribution_reliable=True,
        )

    nodes = list(iter_nodes(root))
    descendants = nodes[1:]

    kinds = [node.kind for node in descendants]
    child_prompt_chars = [node.prompt_chars for node in descendants]
    max_child_prompt_chars = max(child_prompt_chars, default=0)
    root_context_chars = root.prompt_chars

    lost = count_lost_subcalls(root)

    return TreeStats(
        n_nodes=len(nodes),
        n_rlm_children=kinds.count(NodeKind.RLM_CHILD),
        n_llm_leaves=kinds.count(NodeKind.LLM_LEAF),
        n_errored=kinds.count(NodeKind.ERRORED),
        n_indeterminate=kinds.count(NodeKind.INDETERMINATE),
        max_observed_depth=max((node.depth for node in nodes), default=0),
        n_iterations=len(root.iterations),
        root_context_chars=root_context_chars,
        max_child_prompt_chars=max_child_prompt_chars,
        collapse_ratio=(max_child_prompt_chars / root_context_chars if root_context_chars else 0.0),
        terminated_by_fallback=(
            root.iterations[-1].terminated_by_fallback if root.iterations else False
        ),
        recursion_available=ctx.recursion_available,
        block_attribution_reliable=ctx.block_attribution_reliable,
        suspected_lost_subcalls=lost,
        trace_integrity=TraceIntegrity.DEGRADED if lost else TraceIntegrity.COMPLETE,
    )


def walk(completion: RLMChatCompletion) -> tuple[CallNode, TreeStats]:
    """Reconstruct a tree and compute its statistics in one step."""
    metadata = completion.metadata
    if metadata is None:
        raise ValueError(
            "Completion has no trajectory metadata. Construct the RLM with logger=RLMLogger()."
        )
    run_metadata = metadata.get("run_metadata") or {}
    max_depth = int(run_metadata.get("max_depth", 1))
    environment_type = str(run_metadata.get("environment_type", "local"))
    ctx = WalkContext(
        root_max_depth=max_depth,
        recursion_available=max_depth > 1,
        block_attribution_reliable=environment_type not in UNRELIABLE_BLOCK_ATTRIBUTION_ENVS,
    )
    root = build_call_tree(completion)
    return root, compute_tree_stats(root, ctx)
