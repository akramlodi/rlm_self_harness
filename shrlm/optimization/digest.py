"""
Compress a call tree into a bounded, deterministic textual view.

An RLM trace can exceed a context window -- that is the premise of the whole
paradigm -- so the attributor cannot be shown the raw trajectory. The obvious
alternative, summarizing it with a language model, would insert a second
uncontrolled sampling step upstream of every mined pattern and destroy any
claim that the mining round is reproducible from saved configuration. So this
compression is entirely mechanical.

Two consequences are worth stating plainly. Truncation is always announced in
the text, so the attributor knows it is looking at a partial view rather than a
complete one. And ``TraceDigest.coverage`` records how much of the available
material survived, because the truncation policy is a hidden hyperparameter of
every attribution and belongs in the results.
"""

import hashlib
from dataclasses import dataclass

from shrlm.optimization.types import CallNode, NodeKind, TreeStats, Verdict, iter_nodes

DIGEST_VERSION = "1.0.0"

DEFAULT_CHAR_BUDGET = 12000
DEFAULT_FOCUS_K = 4

# Above this many sub-calls the per-call table is replaced by a per-depth
# aggregate, so a wide decomposition cannot crowd out the focused excerpts.
DEFAULT_CHILD_TABLE_THRESHOLD = 40

# Fractions of the budget. The header is exempt: it is small, bounded, and
# carries the verifier outcome the rest of the digest is evidence for. The
# sub-call table is exempt too -- it is bounded by construction at
# child_table_threshold rows of two PREVIEW_CHARS previews each.
ROOT_SKELETON_SHARE = 0.50
EXCERPT_SHARE = 0.50

PREVIEW_CHARS = 200
QUESTION_CHARS = 600
ANSWER_CHARS = 600


@dataclass(frozen=True)
class DigestConfig:
    char_budget: int = DEFAULT_CHAR_BUDGET
    focus_k: int = DEFAULT_FOCUS_K
    child_table_threshold: int = DEFAULT_CHILD_TABLE_THRESHOLD


@dataclass
class TraceDigest:
    """
    The rendered digest plus the metadata attribution needs about it.

    ``aggregated`` records that the sub-call table was collapsed to a per-depth
    aggregate (the wide-tree case), which means the table carries no node ids.
    The attributor reads this flag to soften its evidence-citation demand, so
    it must travel with the text rather than being re-derived downstream.
    """

    text: str
    sha256: str
    coverage: float
    chars_available: int
    chars_kept: int
    aggregated: bool = False


def head_tail(text: str, limit: int) -> str:
    """
    Keep the head and tail of ``text``, announcing what was dropped.

    Both ends matter: the head of a sub-call prompt shows what was asked, and
    the tail of a response shows what was concluded.
    """
    if limit <= 0:
        return f"...[{len(text)} chars omitted]..."
    if len(text) <= limit:
        return text
    head_len = (limit * 2) // 3
    tail_len = limit - head_len
    omitted = len(text) - limit
    return f"{text[:head_len]}\n...[{omitted} chars omitted]...\n{text[-tail_len:]}"


def flatten_prompt(prompt: object) -> str:
    if isinstance(prompt, str):
        return prompt
    return str(prompt)


def render_header(instance_id: str, question: str, verdict: Verdict, stats: TreeStats) -> str:
    lines = [
        "## Run",
        f"instance_id: {instance_id}",
        f"question: {head_tail(question, QUESTION_CHARS)}",
        f"gold_answer: {head_tail(verdict.gold, ANSWER_CHARS)}",
        f"produced_answer: {head_tail(verdict.produced, ANSWER_CHARS)}",
        f"verifier_cause: {verdict.cause.value if verdict.cause else 'none'}",
        "",
        "## Tree statistics",
        f"iterations: {stats.n_iterations}",
        f"sub_calls_total: {stats.n_nodes - 1}",
        f"recursive_children: {stats.n_rlm_children}",
        f"plain_lm_leaves: {stats.n_llm_leaves}",
        f"errored_calls: {stats.n_errored}",
        f"indeterminate_calls: {stats.n_indeterminate}",
        f"max_observed_depth: {stats.max_observed_depth}",
        f"root_context_chars: {stats.root_context_chars}",
        f"largest_subcall_prompt_chars: {stats.max_child_prompt_chars}",
        f"collapse_ratio: {stats.collapse_ratio:.3f}",
        f"terminated_by_fallback: {stats.terminated_by_fallback}",
        f"recursion_available: {stats.recursion_available}",
        f"block_attribution_reliable: {stats.block_attribution_reliable}",
        f"suspected_lost_subcalls: {stats.suspected_lost_subcalls}",
        f"trace_integrity: {stats.trace_integrity.value}",
    ]
    return "\n".join(lines)


def render_root_skeleton(root: CallNode, budget: int) -> tuple[str, int]:
    """
    Render what the root actually did, turn by turn.

    The code is carried close to verbatim because in this paradigm the code
    *is* the decomposition: it is the artifact that determines which sub-calls
    were made over which slices. stderr is never truncated -- tracebacks are
    short and are the highest-signal content in the trace.
    """
    if not root.iterations:
        return "## Root iterations\n(none)", 0

    per_iteration = max(budget // max(len(root.iterations), 1), 200)
    available = 0
    lines = ["## Root iterations"]

    for iteration in root.iterations:
        lines.append(f"### iteration {iteration.index}")
        if iteration.terminated_by_fallback:
            lines.append("(no code executed; answer synthesized by the fallback)")
        for block_index, block in enumerate(iteration.code_blocks):
            available += len(block.code) + len(block.stdout) + len(block.stderr)
            lines.append(f"code[{block_index}]:")
            lines.append(head_tail(block.code, per_iteration // 2))
            if block.stdout:
                lines.append(f"stdout[{block_index}]:")
                lines.append(head_tail(block.stdout, per_iteration // 2))
            if block.stderr:
                lines.append(f"stderr[{block_index}]:")
                lines.append(block.stderr)
        if iteration.final_answer is not None:
            lines.append(f"final_answer_set: {head_tail(iteration.final_answer, ANSWER_CHARS)}")

    return "\n".join(lines), available


def render_child_table(root: CallNode, cfg: DigestConfig) -> tuple[str, int, bool]:
    """One row per sub-call, or a per-depth aggregate when there are too many.

    Returns the rendered table, the character count it drew on, and whether the
    per-depth aggregate replaced the per-call rows -- the flag the digest
    carries so the attributor knows the table names no node ids.
    """
    children = [node for node in iter_nodes(root) if node.node_id != root.node_id]
    available = sum(node.prompt_chars + node.response_chars for node in children)

    if not children:
        return "## Sub-calls\n(none)", 0, False

    if len(children) > cfg.child_table_threshold:
        by_depth: dict[int, list[CallNode]] = {}
        for node in children:
            by_depth.setdefault(node.depth, []).append(node)
        lines = [f"## Sub-calls ({len(children)} total, aggregated by depth)"]
        for depth in sorted(by_depth):
            group = by_depth[depth]
            failed = sum(1 for node in group if node.sub_verdict is False)
            lines.append(
                f"depth {depth}: n={len(group)} "
                f"kinds={sorted({node.kind.value for node in group})} "
                f"mean_prompt_chars={sum(n.prompt_chars for n in group) // len(group)} "
                f"sub_verifier_failed={failed}"
            )
        return "\n".join(lines), available, True

    lines = [
        "## Sub-calls",
        "node_id | depth | kind | prompt_chars | sub_verdict | prompt -> response",
    ]
    for node in children:
        verdict = "n/a" if node.sub_verdict is None else str(node.sub_verdict)
        prompt_preview = head_tail(flatten_prompt(node.prompt), PREVIEW_CHARS).replace("\n", " ")
        response_preview = head_tail(node.response, PREVIEW_CHARS).replace("\n", " ")
        lines.append(
            f"{node.node_id} | {node.depth} | {node.kind.value} | {node.prompt_chars} | "
            f"{verdict} | {prompt_preview} -> {response_preview}"
        )
    return "\n".join(lines), available, False


def select_focus_nodes(root: CallNode, focus_k: int) -> list[CallNode]:
    """
    Choose which sub-calls to show in full.

    A fixed priority, so the selection is reproducible: sub-calls the
    sub-verifier rejected, then errored ones, then the largest by prompt size
    (the natural place a whole-input collapse shows up).
    """
    children = [node for node in iter_nodes(root) if node.node_id != root.node_id]
    failed = sorted(
        (node for node in children if node.sub_verdict is False), key=lambda n: n.node_id
    )
    errored = sorted(
        (node for node in children if node.kind is NodeKind.ERRORED), key=lambda n: n.node_id
    )
    largest = sorted(children, key=lambda n: (-n.prompt_chars, n.node_id))

    selected: list[CallNode] = []
    for node in [*failed, *errored, *largest]:
        if node not in selected:
            selected.append(node)
        if len(selected) == focus_k:
            break
    return selected


def render_excerpts(root: CallNode, cfg: DigestConfig, budget: int) -> tuple[str, int]:
    nodes = select_focus_nodes(root, cfg.focus_k)
    if not nodes:
        return "## Focused sub-call excerpts\n(none)", 0

    per_node = max(budget // len(nodes), 200)
    available = sum(node.prompt_chars + node.response_chars for node in nodes)

    lines = ["## Focused sub-call excerpts"]
    for node in nodes:
        verdict = "n/a" if node.sub_verdict is None else str(node.sub_verdict)
        lines.append(
            f"### {node.node_id} (depth {node.depth}, {node.kind.value}, sub_verdict={verdict})"
        )
        lines.append("prompt:")
        lines.append(head_tail(flatten_prompt(node.prompt), per_node // 2))
        lines.append("response:")
        lines.append(head_tail(node.response, per_node // 2))
    return "\n".join(lines), available


def build_digest(
    instance_id: str,
    question: str,
    root: CallNode,
    stats: TreeStats,
    verdict: Verdict,
    cfg: DigestConfig | None = None,
) -> TraceDigest:
    """
    Render a bounded view of one failed run.

    Deterministic: the same tree and config always produce byte-identical text,
    which is what makes the attribution cache and the reproducibility check
    meaningful.
    """
    cfg = cfg or DigestConfig()

    header = render_header(instance_id, question, verdict, stats)
    skeleton, skeleton_available = render_root_skeleton(
        root, int(cfg.char_budget * ROOT_SKELETON_SHARE)
    )
    table, table_available, aggregated = render_child_table(root, cfg)
    excerpts, _excerpt_available = render_excerpts(root, cfg, int(cfg.char_budget * EXCERPT_SHARE))

    text = "\n\n".join([header, skeleton, table, excerpts])

    # Coverage is measured over the trace material the digest draws on -- root
    # code and output, plus sub-call prompts and responses -- not over the
    # header, which is a summary rather than an excerpt. Excerpted nodes also
    # appear in the table, so their budget is counted once.
    chars_available = skeleton_available + table_available
    chars_kept = min(len(text), chars_available) if chars_available else 0

    return TraceDigest(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        coverage=(chars_kept / chars_available) if chars_available else 1.0,
        chars_available=chars_available,
        chars_kept=chars_kept,
        aggregated=aggregated,
    )
