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

import ast
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind, TreeStats, Verdict, iter_nodes
from shrlm.optimization.walker import iter_skill_loads

# Version of the digest rendering scheme. This constant reaches bundle ids
# via ``MiningConfig.digest_version`` only -- it is deliberately NOT part of
# the attribution cache key. When a rendering change alters a digest's bytes,
# that digest's sha256 changes and with it the affected records' cache keys:
# invalidation rides on the bytes themselves, so records whose rendering is
# untouched keep their cached attributions across a bump.
#
# 1.2.0: the Run header carries an available_skills / loaded_skills pair when
# the trace's run-start record names a skill index (a loader was installed,
# i.e. S10 was non-empty). A trace without one -- every pre-S10 trace, and
# every trace under an empty S10 -- renders byte-identically to 1.1.0.
DIGEST_VERSION = "1.6.0"

DEFAULT_CHAR_BUDGET = 12000
DEFAULT_FOCUS_K = 4

# Above this many sub-calls the per-call table is replaced by a per-depth
# aggregate, so a wide decomposition cannot crowd out the focused excerpts.
DEFAULT_CHILD_TABLE_THRESHOLD = 40

PREVIEW_CHARS = 200
QUESTION_CHARS = 600
ANSWER_CHARS = 600

RETRY_NOTICE = re.compile(
    r"(?:Content filter blocked the response|"
    r"(?:Transient API error|Empty completion content|Deficient completion response) "
    r"\([^\n]*\)); retrying \(\d+/\d+\)\.\.\."
)


def split_retry_notices(stderr: str) -> tuple[str, int]:
    """Recognize only complete notice lines emitted by our client; keep other errors."""
    lines = stderr.splitlines(keepends=True)
    notices = [line for line in lines if RETRY_NOTICE.fullmatch(line.strip())]
    return "".join(line for line in lines if not RETRY_NOTICE.fullmatch(line.strip())), len(notices)


def payload_structure(value: str) -> dict[str, Any]:
    """Describe a bounded complete JSON payload, never infer task coverage."""
    unknown = {"status": "not_assessed"}
    if len(value) > 100_000:
        return unknown

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate keys")
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    try:
        parsed = json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return unknown
    if not isinstance(parsed, (dict, list)):
        return unknown
    return {
        "status": "complete_json",
        "type": "object" if isinstance(parsed, dict) else "array",
        "item_count": len(parsed),
        "semantic_coverage": "not_assessed",
    }


def code_names(code: str) -> tuple[set[str], set[str], bool] | None:
    """Bounded static names and computation flag; no alias or semantic inference."""
    if len(code) > 100_000:
        return None
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError):
        return None
    reads, writes = set(), set()
    computational = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (writes if isinstance(node.ctx, ast.Store) else reads).add(node.id)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            computational = True
        # Method calls can mutate their receiver. This is only a selection hint.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            computational = True
            if isinstance(node.func.value, ast.Name):
                writes.add(node.func.value.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            not in {
                "print",
                "len",
                "str",
                "repr",
                "type",
                "list",
                "dict",
                "set",
                "sorted",
            }
        ):
            computational = True
    return reads, writes, computational


def following_operations(codes: Sequence[str], position: int) -> tuple[list[tuple[int, str]], str]:
    """Select two computational hops and a later observation within this node only."""
    origin = code_names(codes[position])
    if origin is None:
        return [], "unestablished: cited code could not be parsed"
    names = origin[1]
    if not names:
        return [], "unestablished: no named produced value"
    selected = []
    scanned = [
        (i, code_names(codes[i])) for i in range(position + 1, min(len(codes), position + 129))
    ]
    for i, flow in scanned:
        if flow is not None and flow[2] and names.intersection(flow[0] | flow[1]):
            selected.append(
                (i, "computational consumer; static name relation, semantics unverified")
            )
            names |= flow[1]
            if len(selected) == 2:
                break
    last = selected[-1][0] if selected else position
    observations = [i for i, flow in scanned if i > last and flow is not None and names & flow[0]]
    if observations:
        selected.append((observations[-1], "later related observation; recovery not established"))
    return (
        selected,
        "static name relation only" if selected else "unestablished: no related later operation",
    )


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
    ``n_descendants`` records how many nodes beyond the root the tree holds; a
    zero-descendant run also lists no citable node ids. The attributor reads
    both to soften its evidence-citation demand, so they must travel with the
    text rather than being re-derived downstream.
    """

    text: str
    sha256: str
    coverage: float
    chars_available: int
    chars_kept: int
    aggregated: bool = False
    n_descendants: int = 0

    @property
    def no_subcalls(self) -> bool:
        """Whether the run made no sub-calls at all (and so cites no node ids)."""
        return self.n_descendants == 0


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


def render_skill_lines(
    skill_index: Sequence[dict[str, str]] | None, skill_loads: Sequence[tuple[str, int]]
) -> list[str]:
    """The available_skills / loaded_skills pair, or nothing at all.

    Rendered only when the run-start record names a skill index -- a loader
    was installed, so S10 was non-empty. Under an empty S10 there is nothing
    to report and the digest must not change by a byte, which is what keeps
    pre-S10 bundles and attribution caches (keyed on digest bytes) intact.

    ``loaded_skills`` lists each distinct (name, depth) pair once, in
    first-load order, so the attributor can tell "the root consulted it" from
    "a child did" and an unloaded available skill is a visible absence.
    """
    if skill_index is None:
        return []
    available = [str(entry.get("name", "")) for entry in skill_index]
    loaded = list(dict.fromkeys(skill_loads))
    return [
        f"available_skills: {', '.join(available) if available else '(none)'}",
        "loaded_skills: "
        + (", ".join(f"{name} (depth {depth})" for name, depth in loaded) if loaded else "(none)"),
    ]


def render_header(
    instance_id: str,
    question: str,
    verdict: Verdict,
    stats: TreeStats,
    skill_lines: Sequence[str] = (),
    verifier_environment: str | None = None,
) -> str:
    from shrlm.environments.oolong_pairs import recorded_pair_metrics

    pair_metrics = (
        recorded_pair_metrics(verdict) if verifier_environment == "oolong_pairs" else None
    )
    lines = [
        "## Run",
        f"instance_id: {instance_id}",
        f"question: {head_tail(question, QUESTION_CHARS)}",
        *(
            [f"pair_diagnostics: {pair_metrics or 'unavailable (unscored or legacy verdict)'}"]
            if verifier_environment == "oolong_pairs"
            else []
        ),
        f"gold_answer: {head_tail(verdict.gold, ANSWER_CHARS)}",
        f"produced_answer: {head_tail(verdict.produced, ANSWER_CHARS)}",
        f"verifier_cause: {verdict.cause.value if verdict.cause else 'none'}",
        *(
            [f"verifier_detail: {head_tail(verdict.detail, ANSWER_CHARS)}"]
            if verdict.detail
            and (verifier_environment != "oolong_pairs" or pair_metrics is None)
            and verdict.cause is not VerifierCause.RUNTIME_ERROR
            else []
        ),
        *(
            [f"execution_error: {head_tail(verdict.detail, ANSWER_CHARS)}"]
            if verdict.cause is VerifierCause.RUNTIME_ERROR
            else []
        ),
        *skill_lines,
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
            # Per-depth rule: a depth group whose children ALL carry
            # sub_verdict None had no sub-verifier statistic computed over
            # it, so its aggregate must say so (n/a) rather than assert a 0
            # indistinguishable from "ran and all passed". Any verdict at
            # the depth -- even all-passing -- keeps that depth's count
            # numeric. A tree with no verdicts anywhere renders n/a at every
            # depth, which subsumes the old tree-level rule.
            any_verdicts = any(node.sub_verdict is not None for node in group)
            failed = sum(1 for node in group if node.sub_verdict is False)
            lines.append(
                f"depth {depth}: n={len(group)} "
                f"kinds={sorted({node.kind.value for node in group})} "
                f"mean_prompt_chars={sum(n.prompt_chars for n in group) // len(group)} "
                f"sub_verifier_failed={failed if any_verdicts else 'n/a'}"
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
            f"{verdict} | {prompt_preview} -> {response_preview} "
            f"structure={json.dumps(payload_structure(node.response), sort_keys=True)}"
        )
    return "\n".join(lines), available, False


def select_focus_nodes(root: CallNode, focus_k: int) -> list[CallNode]:
    """
    Choose which sub-calls to show in full.

    A fixed priority, so the selection is reproducible: sub-calls the
    sub-verifier rejected, then errored ones, then the largest by prompt size
    (the natural place a whole-input collapse shows up).

    The verdict-aware first tier is a DESIGNED mode-visible signal, not a
    leak: it is part of the sub-verdict evidence the digest deliberately
    surfaces to the attributor (alongside the per-call verdict column), so a
    grounded and an ablated digest of the same tree may excerpt different
    nodes. Neutralizing it would hide exactly the grounding signal the
    sub-verification ablation measures.
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


def build_digest(
    instance_id: str,
    question: str,
    root: CallNode,
    stats: TreeStats,
    verdict: Verdict,
    cfg: DigestConfig | None = None,
    verifier_environment: str | None = None,
) -> TraceDigest:
    """
    Render a bounded view of one failed run.

    Deterministic: the same tree and config always produce byte-identical text,
    which is what makes the attribution cache and the reproducibility check
    meaningful.
    """
    cfg = cfg or DigestConfig()

    # The skill facts come from the trace itself (U12's run-start index and
    # per-block loader events), never from a live harness or the round
    # envelope: the digest stays a function of the persisted tree alone.
    header = render_header(
        instance_id,
        question,
        verdict,
        stats,
        skill_lines=render_skill_lines(root.skill_index, list(iter_skill_loads(root))),
        verifier_environment=verifier_environment,
    )
    if cfg.char_budget < 128:
        raise ValueError("digest char_budget must be at least 128")
    nodes = list(iter_nodes(root))
    focus = select_focus_nodes(root, cfg.focus_k)
    focus_ids = {node.node_id for node in focus}
    blocks = [
        (node, iteration, index, block)
        for node in nodes
        for iteration in node.iterations
        for index, block in enumerate(iteration.code_blocks)
    ]
    chars_available = sum(
        len(b.code) + len(b.stdout) + len(b.stderr) for _, _, _, b in blocks
    ) + sum(n.prompt_chars + n.response_chars for n in nodes[1:])
    chars_kept = 0
    # Admit complete operations by structural priority before filling spare
    # space with omission coordinates. A wide prefix must not evict a late fault.
    skeleton: dict[int, str] = {}
    fallback_lines = [
        f"{node.node_id} iteration {iteration.index}: no code executed; answer synthesized by the fallback"
        for node in nodes
        for iteration in node.iterations
        if iteration.terminated_by_fallback
    ]
    sections = [
        header,
        "",
        "## Sub-calls\n(table omitted: budget)",
        "## Focused sub-call excerpts\n(none)",
    ]

    def render_operations() -> None:
        missing = len(blocks) - len(skeleton)
        rows = [skeleton[position] for position in sorted(skeleton)]
        if missing:
            rows.append(f"[{missing} additional operation coordinates omitted: budget]")
        sections[1] = "## Root iterations / operation skeleton\n" + (
            "\n".join([*rows, *fallback_lines]) or "(none)"
        )

    def size() -> int:
        return len("\n\n".join(sections))

    render_operations()
    if size() > cfg.char_budget:
        # Extremely small budgets still identify the omission explicitly.
        fallback_lines = []
        render_operations()
        reserve = size() - len(header)
        if reserve + 40 > cfg.char_budget:
            sections[2:] = ["", ""]
            reserve = size() - len(header)
        available = max(0, cfg.char_budget - reserve - 40)
        sections[0] = header[:available] + "\n...[header chars omitted]..."

    residuals = {
        position: split_retry_notices(block.stderr)
        for position, (_, _, _, block) in enumerate(blocks)
    }
    notices = [
        (f"{node.node_id}/i{iteration.index}/b{index}", residuals[position][1])
        for position, (node, iteration, index, _) in enumerate(blocks)
        if residuals[position][1]
    ]
    if notices:
        summary = (
            "\nClient retry notices (recovery not established): "
            + json.dumps(notices[:16])
            + (f"; {len(notices) - 16} further locations" if len(notices) > 16 else "")
        )
        if size() + len(summary) <= cfg.char_budget:
            sections[0] += summary
    consumers = set()
    for node in nodes:
        local = [(i, block) for i, (owner, _, _, block) in enumerate(blocks) if owner is node]
        codes = [block.code for _, block in local]
        for position, (global_index, block) in enumerate(local):
            if block.calls or residuals[global_index][0].strip():
                chain, _ = following_operations(codes, position)
                consumers.update(local[index][0] for index, _ in chain)
    ranked = sorted(
        enumerate(blocks),
        key=lambda pair: (
            0
            if residuals[pair[0]][0].strip()
            or any(c.kind is NodeKind.ERRORED for c in pair[1][3].calls)
            else 1
            if pair[0] in consumers
            else 2
            if any(c.node_id in focus_ids for c in pair[1][3].calls)
            else 3,
            pair[1][0].node_id != root.node_id,
            -pair[1][1].index,
            pair[0],
        ),
    )
    payload_kept: dict[str, int] = {}
    for position, (node, iteration, index, block) in ranked:
        skeleton[position] = (
            f"{node.node_id} iteration {iteration.index} code[{index}] (complete):\n{block.code}"
        )
        observed = {}
        for stream, value in (("stdout", block.stdout), ("stderr", residuals[position][0])):
            if not value:
                continue
            label = f"{node.node_id} iteration {iteration.index} {stream}[{index}]:\n"
            structure = payload_structure(value)
            if structure["status"] != "not_assessed":
                label += f"structure={json.dumps(structure, sort_keys=True)}\n"
            limit = max(0, 500 - len(label))
            excerpt = value
            kept = len(value)
            if len(value) > limit:
                marker = "\n...[output truncated]...\n"
                kept = max(0, limit - len(marker))
                head = kept * 2 // 3
                excerpt = value[:head] + marker + (value[-(kept - head) :] if kept > head else "")
            skeleton[position] += "\n" + label + excerpt
            observed[label] = kept
        render_operations()
        if size() <= cfg.char_budget:
            chars_kept += len(block.code)
            payload_kept.update(observed)
        else:
            del skeleton[position]
            render_operations()
    for position, (node, iteration, index, block) in enumerate(blocks):
        if position in skeleton:
            continue
        skeleton[position] = (
            f"{node.node_id} iteration {iteration.index} code[{index}] omitted ({len(block.code)} chars)"
        )
        render_operations()
        if size() > cfg.char_budget:
            del skeleton[position]
            render_operations()

    # Payloads are bounded separately and never count marker/header bytes as
    # surviving trace content. Each source is counted at most once.
    payloads = [
        (f"{node.node_id} {label}", value)
        for node in focus
        for label, value in (("prompt", flatten_prompt(node.prompt)), ("response", node.response))
    ]
    excerpt_lines = []
    for offset, (label, value) in enumerate(payloads):
        remaining = cfg.char_budget - size()
        structure = (
            "\nstructure=" + json.dumps(payload_structure(value), sort_keys=True)
            if label.endswith("response")
            else ""
        )
        allowance = max(
            0, remaining // max(1, len(payloads) - offset) - len(label) - len(structure) - 50
        )
        if allowance <= 0:
            continue
        excerpt = head_tail(value, allowance)
        excerpt_lines.append(f"{label}:{structure}\n{excerpt}")
        sections[3] = "## Focused sub-call excerpts\n" + "\n".join(excerpt_lines)
        payload_kept[label] = min(len(value), allowance)

    table, _, aggregated = render_child_table(root, cfg)
    if size() - len(sections[2]) + len(table) <= cfg.char_budget:
        sections[2] = table
        if not aggregated:
            for node in nodes[1:]:
                for label, length in (
                    ("prompt", node.prompt_chars),
                    ("response", node.response_chars),
                ):
                    key = f"{node.node_id} {label}"
                    payload_kept[key] = max(payload_kept.get(key, 0), min(length, PREVIEW_CHARS))
    else:
        aggregated = bool(nodes[1:])
    text = "\n\n".join(sections)
    chars_kept += sum(payload_kept.values())

    return TraceDigest(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        coverage=(chars_kept / chars_available) if chars_available else 1.0,
        chars_available=chars_available,
        chars_kept=chars_kept,
        aggregated=aggregated,
        n_descendants=stats.n_nodes - 1,
    )
