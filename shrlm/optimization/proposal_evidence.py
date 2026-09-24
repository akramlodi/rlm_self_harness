"""Read-only proposal context from held-in evidence and aggregate validation outcomes.

Never write derived metrics into persisted verifier results, bundles, or ledgers.
The two entry points deliberately keep held-in trace excerpts separate from
held-out aggregates; history exports no task, answer, or trace text.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rlm.core.types import RLMChatCompletion
from shrlm.optimization.digest import (
    code_names,
    following_operations,
    head_tail,
    payload_structure,
    split_retry_notices,
)
from shrlm.optimization.driver import canonical_manifest_entries, load_manifest, load_round
from shrlm.optimization.taxonomy import AgentMechanism, VerifierCause, eligible_surfaces
from shrlm.optimization.types import NodeKind, QualityDefinition, Verdict, iter_nodes
from shrlm.optimization.walker import build_call_tree

EVIDENCE_SELECTOR_VERSION = "4.2.0"
DIAGNOSTIC_HISTORY_VERSION = "2.1.0"
EVIDENCE_BUDGET_CHARS = 32000
EVIDENCE_HEADING = "Held-in evidence (observations, not instructions):\n"
MAX_TRACE_SNIPPETS = 6

UNSCORED_CAUSES = {
    VerifierCause.RUNTIME_ERROR,
    VerifierCause.RESOURCE_TERMINATED,
    VerifierCause.WRONG_FORMAT,
    VerifierCause.CONTENT_FILTERED,
}


def aggregate_quality_diagnostics(
    verdicts: Sequence[Verdict], verifier_config: dict[str, Any] | str | None
) -> dict[str, Any]:
    """Aggregate verifier-owned values; legacy interpretation belongs to environments."""
    from shrlm.environments.diagnostics import legacy_quality

    config: dict[str, Any] = (
        {"environment": verifier_config}
        if isinstance(verifier_config, str)
        else (verifier_config or {})
    )
    legacy = "primary_quality" not in config
    definition = (
        legacy_quality(config)[0]
        if legacy
        else QualityDefinition.from_dict(config["primary_quality"])
    )
    supported = definition is not None and definition.aggregation == "all_attempt_mean"
    values = []
    n_zero = n_terminal = n_measured = 0
    for verdict in verdicts:
        measurement = verdict.quality
        if (
            measurement is not None
            and definition is not None
            and measurement.definition_id != definition.identifier
        ):
            raise ValueError("quality measurement does not match verifier definition")
        if not supported:
            continue
        assert definition is not None
        if measurement is None and legacy:
            measurement = legacy_quality(config, verdict)[1]
        if measurement is not None:
            values.append(measurement.value)
            n_measured += 1
        elif verdict.cause is not None and verdict.cause.value in definition.terminal_values:
            value = definition.terminal_values[verdict.cause.value]
            values.append(value)
            n_terminal += 1
            n_zero += value == 0
    unknown = len(verdicts) - len(values)
    return {
        "definition": definition.to_dict() if definition is not None and supported else None,
        "n_attempts": len(verdicts),
        "n_measured": n_measured,
        "n_known_zero": n_zero,
        "n_declared_terminal": n_terminal,
        "n_unknown": unknown,
        "mean": math.fsum(values) / len(verdicts) if verdicts and not unknown else None,
    }


def compare_quality_diagnostics(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    definition = baseline.get("definition")
    old, new = baseline.get("mean"), candidate.get("mean")
    if (
        not definition
        or definition != candidate.get("definition")
        or old is None
        or new is None
        or not math.isfinite(old)
        or not math.isfinite(new)
        or definition.get("direction") not in {"higher", "lower"}
    ):
        return "not_assessed"
    improved = new > old if definition["direction"] == "higher" else new < old
    return "potentially_promising" if improved else "no_measured_improvement"


def pair_diagnostics(verdict: Verdict) -> dict[str, Any]:
    from shrlm.environments.oolong_pairs import recorded_pair_metrics

    metrics = recorded_pair_metrics(verdict)
    return {"scored": metrics is not None, "metrics": metrics or "unavailable"}


def aggregate_pair_diagnostics(verdicts: Sequence[Verdict]) -> dict[str, Any]:
    from shrlm.environments.oolong_pairs import recorded_pair_metrics

    measured = [recorded_pair_metrics(verdict) for verdict in verdicts]
    known = [metrics for metrics in measured if metrics is not None]
    unaccounted = sum(
        metrics is None and verdict.cause not in UNSCORED_CAUSES
        for metrics, verdict in zip(measured, verdicts, strict=True)
    )
    return {
        "n_attempts": len(verdicts),
        "n_scored": len(known),
        "n_unscored": len(verdicts) - len(known),
        "n_missing_legacy_metrics": unaccounted,
        "mean_f1_all_attempts": (
            math.fsum(float(m["f1"]) for m in known) / len(verdicts)
            if verdicts and not unaccounted
            else None
        ),
        "missing_total_measured": sum(int(m["missing"]) for m in known) if known else None,
        "extra_total_measured": sum(int(m["extra"]) for m in known) if known else None,
        "counts_denominator": len(known),
    }


def read_persisted_round(
    path: Path,
) -> tuple[
    list[tuple[dict[str, Any], RLMChatCompletion]],
    list[Verdict],
    dict[str, Any],
    list[dict[str, Any]],
]:
    match = re.fullmatch(r"round_(\d+)", path.name)
    if match is None:
        raise ValueError(f"not a persisted round directory: {path}")
    return load_round(path.parent, int(match[1]))


def bounded_excerpt(value: str, limit: int) -> str:
    """Include truncation markers in the budget, unlike the legacy head_tail helper."""
    if len(value) <= limit:
        return value
    marker = "\n...[truncated]...\n"
    kept = max(0, limit - len(marker))
    head = kept * 2 // 3
    return (value[:head] + marker + (value[-(kept - head) :] if kept > head else ""))[:limit]


def trace_excerpt(
    completion: RLMChatCompletion,
    evidence_node_ids: Sequence[str] = (),
    operation_evidence: Sequence[dict[str, Any]] = (),
    *,
    related_operations: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if completion.metadata is None:
        return {"observation": "trace excerpt unavailable"}
    root = build_call_tree(completion)
    nodes = {node.node_id: node for node in iter_nodes(root)}
    by_node: dict[str, list[dict[str, Any]]] = {}
    callers: dict[str, tuple[str, int]] = {}
    retry_notices = []
    relationships = []
    for node in nodes.values():
        blocks = []
        for iteration in node.iterations:
            for block_index, block in enumerate(iteration.code_blocks):
                for child in block.calls:
                    callers[child.node_id] = (node.node_id, len(blocks))
                residual, retry_count = split_retry_notices(block.stderr)
                if retry_count:
                    retry_notices.append(
                        {
                            "node_id": node.node_id,
                            "iteration_index": iteration.index,
                            "code_block_index": block_index,
                            "count": retry_count,
                        }
                    )
                blocks.append(
                    {
                        "node_id": node.node_id,
                        "iteration_index": iteration.index,
                        "code_block_index": block_index,
                        "code": block.code,
                        "stdout": block.stdout,
                        "stderr": residual,
                        "error_observed": bool(residual.strip())
                        or any(child.kind is NodeKind.ERRORED for child in block.calls),
                    }
                )
        by_node[node.node_id] = blocks

    selected: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(snippet: dict[str, Any], reason: str, required: bool = True) -> None:
        key = (snippet["node_id"], snippet.get("iteration_index"), snippet.get("code_block_index"))
        if key not in selected or reason == "cited operation":
            selected[key] = {**snippet, "reason": reason, "required": required}
        elif required:
            selected[key]["required"] = True

    def following(node_id: str, position: int, error: bool) -> None:
        blocks = by_node[node_id]
        chain, status = following_operations([b["code"] for b in blocks], position)
        relationships.append({"node_id": node_id, "position": position, "status": status})
        for index, reason in chain:
            add({**blocks[index], "follows_error": error}, reason)

    cited = list(dict.fromkeys(evidence_node_ids))
    unresolved = False
    for operation in operation_evidence:
        node_id = operation.get("node_id")
        if node_id not in nodes:
            unresolved = True
            continue
        if node_id not in cited:
            cited.append(node_id)
        if operation.get("iteration_index") is None:
            continue
        matching = [
            b
            for b in by_node[node_id]
            if b["iteration_index"] == operation.get("iteration_index")
            and b["code_block_index"] == operation.get("code_block_index")
        ]
        if matching:
            add(matching[0], "cited operation")
            position = by_node[node_id].index(matching[0])
            following(node_id, position, matching[0]["error_observed"])
            # Resolve the nearest prior child producer for a cited consumer.
            producer_positions = [
                pos
                for child, (parent, pos) in callers.items()
                if parent == node_id
                and pos <= position
                and (
                    pos == position
                    or (
                        (consumer_flow := code_names(matching[0]["code"])) is not None
                        and (producer_flow := code_names(by_node[node_id][pos]["code"])) is not None
                        and consumer_flow[0] & producer_flow[1]
                    )
                )
            ]
            if producer_positions:
                producer = max(producer_positions)
                add(by_node[node_id][producer], "linked child producer")
                producer_children = [
                    child
                    for child, (parent, pos) in callers.items()
                    if parent == node_id and pos == producer
                ]
                for offset, child in enumerate(producer_children):
                    if child in nodes:
                        node = nodes[child]
                        add(
                            {
                                "node_id": child,
                                "prompt": str(node.prompt),
                                "response": node.response,
                                "error_observed": node.kind is NodeKind.ERRORED,
                            },
                            "linked child prompt/return",
                            required=offset == 0,
                        )
        else:
            unresolved = True
    for node_id in cited:
        if node_id not in nodes:
            unresolved = True
            continue
        if node_id in callers:
            parent_id, position = callers[node_id]
            block = by_node[parent_id][position]
            add(block, "cited call's caller")
            following(parent_id, position, block["error_observed"])
            node = nodes[node_id]
            add(
                {
                    "node_id": node_id,
                    "prompt": str(node.prompt),
                    "response": node.response,
                    "error_observed": node.kind is NodeKind.ERRORED,
                },
                "cited child prompt/return",
            )
    selection = "cited operations and caller context"
    if not selected and related_operations:
        for blocks in by_node.values():
            for position, block in enumerate(blocks):
                if operation_names(block["code"]) & related_operations:
                    add(
                        block,
                        "passing run with shared operation names; semantic equivalence unverified",
                    )
                    following(node_id=block["node_id"], position=position, error=False)
        selection = "passing contrast selected by shared operation names"
    if not selected:
        selection = (
            "structural fallback: no resolvable operation or relevant contrast; code unavailable"
        )
    if unresolved:
        selection += "; unresolved citations"
    ordered = sorted(
        selected.values(),
        key=lambda op: (
            not op["required"],
            op.get("reason") != "cited operation",
            op["node_id"],
            op.get("iteration_index", -1),
            op.get("code_block_index", -1),
        ),
    )
    core_complete = sum(op["required"] for op in ordered) <= MAX_TRACE_SNIPPETS and not unresolved
    snippets = ordered[:MAX_TRACE_SNIPPETS]
    for snippet in snippets:
        # Preserve whole code. Only payloads are excerpted; final packing counts
        # the serialized size of every field and can omit an oversized operation.
        for key in ("stdout", "response"):
            if snippet.get(key):
                snippet[key + "_structure"] = payload_structure(snippet[key])
        for key in ("stdout", "stderr", "prompt", "response"):
            if snippet.get(key):
                snippet[key] = bounded_excerpt(snippet[key], 1200)
        if "code" in snippet:
            snippet["code_complete"] = True
    return {
        "observation": "partial observed behavior; labels and causal effectiveness are not verified",
        "selection": selection,
        "snippets": snippets,
        "core_complete": core_complete,
        "relationships": relationships,
        "retry_notices": retry_notices,
        "retry_status": "notices only; recovery not established",
        "omitted_operations": [
            {key: op.get(key) for key in ("node_id", "iteration_index", "code_block_index")}
            for op in ordered[MAX_TRACE_SNIPPETS:]
        ],
        "observed_child_calls": len(root.children),
    }


def operation_names(code: str) -> frozenset[str]:
    """A conservative operation-name overlap, never a claim of equal semantics."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset()
    generic = {
        "print",
        "len",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "range",
        "enumerate",
        "zip",
        "sorted",
    }
    return frozenset(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id not in generic
    )


class EvidenceBudgetExceeded(ValueError):
    """Even the compact inventory exceeds the local evidence budget."""


def context_operations(context: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        name
        for snippet in context.get("trace", {}).get("snippets", [])
        for name in operation_names(snippet.get("code", ""))
    )


def operation_support(mechanism: str, operations: dict[str, Any]) -> dict[str, list[str]]:
    """Structural opportunity only; semantic relevance remains a model judgment."""
    complete = {ref: op for ref, op in operations.items() if op.get("code_complete")}
    cited = [ref for ref, op in complete.items() if op.get("reason") == "cited operation"]
    children = [ref for ref, op in operations.items() if "response" in op]
    if mechanism == AgentMechanism.LOSSY_AGGREGATION and cited:
        return {"S8": cited}
    if mechanism == AgentMechanism.UNPARSED_CHILD_OUTPUT and cited and children:
        return {"S8": [*cited, *children]}
    if mechanism in {
        AgentMechanism.ITERATION_BUDGET_EXHAUSTION,
        AgentMechanism.REPL_EXECUTION_FAULT,
    }:
        errors = [ref for ref, op in operations.items() if op.get("error_observed")]
        recovery = [ref for ref, op in complete.items() if op.get("follows_error")]
        if errors and recovery:
            return {"S5": list(dict.fromkeys([*errors, *recovery]))}
    return {}


def pack_evidence(
    inventory: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    k: int,
    budget: int = EVIDENCE_BUDGET_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Pack whole examples deterministically, measuring actual rendered characters."""
    section: dict[str, Any] = {
        "inventory": copy.deepcopy(inventory),
        "expanded": {},
        "operations": {},
        "passing": [],
        "passing_ids": evidence.get("passing_ids", []),
        "passing_status": evidence.get("passing_status", "no passing runs observed"),
        "omitted": {
            str(row["index"]): "not expanded: mechanism limit or lower priority"
            for row in inventory
        },
        "contrast_status": "no relevant passing contrast available; following operations may show recovery, but proximity does not establish it",
    }

    def render(value: dict[str, Any]) -> str:
        for row in value["inventory"]:
            context = value["expanded"].get(str(row["index"]), {})
            refs = context.get("trace", {}).get("snippets", [])
            operations = {r["operation_ref"]: value["operations"][r["operation_ref"]] for r in refs}
            mechanism = row["signature"]["agent_mechanism"]
            support = operation_support(mechanism, operations)
            row["eligible_surfaces"] = (
                [
                    s.value
                    for s in eligible_surfaces(
                        AgentMechanism(mechanism),
                        support,
                        causal_status=row["signature"].get("causal_status"),
                    )
                ]
                if refs
                else []
            )
            row["route_support"] = support
            row["admitted_refs"] = [ref["operation_ref"] for ref in refs]
            row["selectable"] = bool(row["eligible_surfaces"])
            row["nonselectable_reason"] = (
                "" if refs else value["omitted"].get(str(row["index"]), "no complete evidence")
            )
            row["predecessors"] = {
                surface: prior
                for surface, prior in predecessors.get(row["index"], {}).items()
                if surface in row["eligible_surfaces"]
            }
        return EVIDENCE_HEADING + json.dumps(value, sort_keys=True)

    predecessors = {row["index"]: row.get("predecessors", {}) for row in inventory}
    if len(render(section)) > budget:
        raise EvidenceBudgetExceeded("compact evidence inventory exceeds rendered evidence budget")

    def options(index: int) -> list[dict[str, Any]]:
        return evidence.get("alternatives", {}).get(index) or [
            evidence.get("patterns", {}).get(
                index, {"diagnosis": "representative evidence unavailable"}
            )
        ]

    def core_snippets(context: dict[str, Any]) -> list[dict[str, Any]]:
        if context.get("trace", {}).get("core_complete") is False:
            return []
        return [
            op
            for op in context.get("trace", {}).get("snippets", [])
            if op.get("node_id")
            and (
                op.get("code")
                and op.get("code_complete") is not False
                or op.get("prompt")
                and (op.get("response") or op.get("error_observed"))
            )
            and op.get(
                "required",
                not op.get("reason", "").startswith("following") or op.get("follows_error", False),
            )
        ]

    def grounded(row: dict[str, Any]) -> bool:
        return any(core_snippets(context) for context in options(row["index"]))

    ranked = sorted(
        inventory,
        key=lambda row: (
            -int(row.get("instance_support") or len(row.get("instance_ids") or [])),
            -float(row.get("actionability") or 0),
            not grounded(row),
            row["index"],
        ),
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in ranked:
        reason = None
        if row["signature"].get("causal_status") == "unattributed":
            reason = "unattributed: no eligible intervention"
        elif row.get("actionability") is not None and row["actionability"] <= 0:
            reason = "explicitly non-actionable"
        elif not grounded(row):
            reason = "no resolvable operation or child-call observation"
        if reason:
            section["omitted"][str(row["index"])] = reason
        else:
            groups.setdefault(row["signature"]["agent_mechanism"], []).append(row)

    def with_context(
        value: dict[str, Any], context: dict[str, Any], identity: str
    ) -> dict[str, Any]:
        context = copy.deepcopy(context)
        trace = context.get("trace", {})
        snippets = trace.get("snippets", [])
        refs = []
        for snippet in snippets:
            source = [
                context.get("run_id", identity),
                snippet.get("node_id"),
                snippet.get("iteration_index"),
                snippet.get("code_block_index"),
                "code" if "code" in snippet else "child",
            ]
            key = hashlib.sha256(json.dumps(source).encode()).hexdigest()[:16]
            value["operations"].setdefault(key, {"source": source, **snippet})
            refs.append({"operation_ref": key})
        if snippets:
            trace["snippets"] = refs
        return context

    expanded_indices: list[int] = []
    admitted: dict[int, dict[str, Any]] = {}
    representative_sizes: dict[str, int] = {}
    limit = min(k, 4)

    def admit(rows: list[dict[str, Any]]) -> None:
        nonlocal section
        # Compare complete packets across signatures, measuring their actual
        # incremental serialized size after operation deduplication.
        best = None
        current_size = len(render(section))
        for priority, row in enumerate(rows):
            index = row["index"]
            if index in admitted:
                continue
            for context in options(index):
                core = copy.deepcopy(context)
                trace = core.get("trace", {})
                trace["snippets"] = core_snippets(context)
                if not trace["snippets"]:
                    continue
                trial = copy.deepcopy(section)
                trial["expanded"][str(index)] = with_context(trial, core, f"pattern-{index}")
                del trial["omitted"][str(index)]
                size = len(render(trial))
                key = (size - current_size, priority, str(context.get("run_id", "")))
                if best is None or key < best[0]:
                    best = (key, index, context, trial)
        if best is not None:
            (delta, _, _), index, context, trial = best
            if current_size + delta <= budget:
                section = trial
                expanded_indices.append(index)
                admitted[index] = context
                representative_sizes[str(index)] = delta
                return
        for row in rows:
            if row["index"] not in admitted:
                section["omitted"][str(row["index"])] = (
                    "complete core packet exceeds remaining budget"
                )

    for rows in groups.values():
        if len(expanded_indices) >= limit:
            break
        admit(rows)
    eligible_indices = {row["index"] for rows in groups.values() for row in rows}
    for row in ranked:
        if len(expanded_indices) >= limit:
            break
        if row["index"] in eligible_indices and row["index"] not in admitted:
            admit([row])

    core_count = len(section["operations"])
    for index, context in admitted.items():
        for op in context.get("trace", {}).get("snippets", []):
            trial = copy.deepcopy(section)
            extra = with_context(
                trial,
                {"trace": {"snippets": [op]}, "run_id": context.get("run_id", f"pattern-{index}")},
                f"pattern-{index}",
            )
            refs = trial["expanded"][str(index)].get("trace", {}).get("snippets", [])
            for ref in extra["trace"]["snippets"]:
                if ref not in refs:
                    refs.append(ref)
            if len(render(trial)) <= budget:
                section = trial
    names = frozenset(name for context in admitted.values() for name in context_operations(context))
    for passing in evidence.get("passing", []):
        if not names.intersection(context_operations(passing)):
            continue
        trial = copy.deepcopy(section)
        trial["passing"].append(with_context(trial, passing, "passing"))
        trial["contrast_status"] = (
            "passing held-in run shares operation names; equivalence and intermediate correctness are unverified"
        )
        if len(render(trial)) <= budget:
            section = trial
            break
    rendered = render(section)
    return rendered, {
        "evidence_selector_version": EVIDENCE_SELECTOR_VERSION,
        "evidence_budget_chars": budget,
        "evidence_chars": len(rendered),
        "expanded_patterns": expanded_indices,
        "distinct_actionable_mechanisms": len(groups),
        "representative_incremental_chars": representative_sizes,
        "omitted_patterns": section["omitted"],
        "operation_count": len(section["operations"]),
        "core_operation_count": core_count,
        "optional_operation_count": len(section["operations"]) - core_count,
        "expanded_mechanisms": [
            inventory_row["signature"]["agent_mechanism"]
            for index in expanded_indices
            for inventory_row in inventory
            if inventory_row["index"] == index
        ],
        "route_support": {str(row["index"]): row["route_support"] for row in section["inventory"]},
        "admitted_refs": {
            index: [ref["operation_ref"] for ref in context.get("trace", {}).get("snippets", [])]
            for index, context in section["expanded"].items()
        },
        "choices": {
            str(row["index"]): {
                key: row[key]
                for key in (
                    "selectable",
                    "eligible_surfaces",
                    "admitted_refs",
                    "route_support",
                    "predecessors",
                )
            }
            for row in section["inventory"]
            if row["selectable"]
        },
    }


def withhold_choices(rendered: str, audit: dict[str, Any], indices: Sequence[str]) -> str:
    """Finalize once after history packing; never reallocate the released capacity."""
    section = json.loads(rendered[len(EVIDENCE_HEADING) :])
    for row in section["inventory"]:
        index = str(row["index"])
        if index in indices:
            row.update(
                selectable=False,
                eligible_surfaces=[],
                predecessors={},
                nonselectable_reason="required history unavailable within budget",
            )
            audit["choices"].pop(index, None)
    result = EVIDENCE_HEADING + json.dumps(section, sort_keys=True)
    if len(result) > audit["evidence_budget_chars"]:
        raise EvidenceBudgetExceeded("final choice metadata exceeds evidence budget")
    audit["evidence_chars"] = len(result)
    audit["history_withheld_patterns"] = list(indices)
    return result


def load_proposal_evidence(
    mining_round_path: Path, bundle: dict[str, Any], *, bundle_path: Path | None = None
) -> dict[str, Any]:
    """Join one selected bundle's records to SHA-verified held-in runs only."""
    bundle_path = bundle_path or mining_round_path / "bundle.json"
    if bundle_path.exists() and json.loads(bundle_path.read_text()) != bundle:
        raise ValueError("selected proposal bundle differs from its persisted evidence")
    records_path = bundle_path.parent / "records.jsonl"
    records = (
        [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        if records_path.exists()
        else []
    )
    instances_path = mining_round_path / "instances.jsonl"
    instances = (
        [json.loads(line) for line in instances_path.read_text().splitlines() if line.strip()]
        if instances_path.exists()
        else []
    )
    by_instance = {str(instance["id"]): instance for instance in instances}
    runs, verdicts, entries = [], [], []
    if (mining_round_path / "runs.jsonl").exists():
        runs, verdicts, _, entries = read_persisted_round(mining_round_path)
    contexts = {}
    alternatives = {}
    verifier_config = bundle.get("config", {}).get("verifier_config") or {}
    environment = verifier_config.get("environment")
    is_pairs = environment == "oolong_pairs"
    for index, pattern in enumerate(bundle.get("patterns", [])):
        representative_ids = dict.fromkeys(
            [*pattern.get("representatives", []), *pattern.get("instance_ids", [])]
        )
        candidates = [
            record
            for instance_id in representative_ids
            for record in records
            if str(record["instance_id"]) == str(instance_id)
            and record.get("signature") == pattern.get("signature")
        ]
        if not candidates:
            contexts[index] = {"diagnosis": "representative evidence unavailable"}
            continue
        alternatives[index] = []
        for record in sorted(
            candidates, key=lambda r: (str(r.get("run_id", "")), str(r["instance_id"]))
        ):
            detail = record.get("detail") or {}
            instance = by_instance.get(str(record["instance_id"]), {})
            verdict = Verdict.from_dict(record["verdict"])
            context: dict[str, Any] = {
                "symptom_summary": head_tail(detail.get("symptom_summary", "unavailable"), 2000),
                "evidence_node_ids": detail.get("evidence_node_ids", []),
                "operation_evidence": detail.get("operation_evidence", []),
                "verification_limits": head_tail(
                    detail.get("verification_limits", "unavailable"), 500
                ),
                "level_grounded": record.get("level_grounded", False),
                "task_question": str(instance.get("question", "unavailable")),
                "verifier_outcome": {
                    "passed": verdict.passed,
                    "cause": verdict.cause.value if verdict.cause else None,
                },
                "verifier_detail": "unparsed: " + bounded_excerpt(verdict.detail, 2000),
            }
            if (
                detail.get("coverage_basis") is not None
                or pattern["signature"].get("agent_mechanism") == "incomplete_coverage"
            ):
                context["coverage_basis"] = (
                    detail.get("coverage_basis") or "coverage basis not assessed"
                )
            quality = aggregate_quality_diagnostics([verdict], verifier_config)
            if quality["mean"] is not None:
                context["quality_diagnostics"] = quality
            if quality["n_measured"]:
                del context["verifier_detail"]
            if environment in {"oolong_pairs", "graphwalks"}:
                context["pair_diagnostics"] = pair_diagnostics(verdict)
            if is_pairs:
                if context["pair_diagnostics"]["scored"]:
                    from shrlm.environments.oolong_pairs import extract_answer_pairs

                    gold = set(extract_answer_pairs(verdict.gold) or [])
                    produced = set(extract_answer_pairs(verdict.produced) or [])
                    context["missing_examples"] = sorted(gold - produced)[:3]
                    context["extra_examples"] = sorted(produced - gold)[:3]
            if record.get("run_id"):
                matches = [
                    i for i, entry in enumerate(entries) if entry["run_id"] == record["run_id"]
                ]
            else:
                matches = [
                    i
                    for i, entry in enumerate(entries)
                    if str(entry["instance_id"]) == str(record["instance_id"])
                    and entry["verdict"] == record["verdict"]
                ]
            context["trace"] = {
                "observation": "trace unavailable: missing or ambiguous run linkage"
            }
            if len(matches) == 1:
                selected = matches[0]
                entry = entries[selected]
                if str(entry["instance_id"]) != str(record["instance_id"]):
                    raise ValueError("record instance_id disagrees with linked run")
                for key in ("trace_path", "trace_sha256"):
                    if record.get(key) and record[key] != entry.get(key):
                        raise ValueError(f"record {key} disagrees with manifest")
                if verdict.to_dict() != verdicts[selected].to_dict():
                    raise ValueError("record verdict disagrees with linked run")
                context["run_id"] = entry["run_id"]
                context["trace"] = trace_excerpt(
                    runs[selected][1],
                    detail.get("evidence_node_ids", []),
                    detail.get("operation_evidence", []),
                )
            alternatives[index].append(context)
        alternatives[index].sort(
            key=lambda c: (
                not bool(c.get("trace", {}).get("snippets")),
                not c.get("level_grounded", False),
                str(c.get("run_id", "")),
            )
        )
        contexts[index] = alternatives[index][0]

    passing = []
    seen = set()
    related = frozenset(
        name for context in contexts.values() for name in context_operations(context)
    )
    for (instance, completion), verdict, entry in zip(runs, verdicts, entries, strict=True):
        if verdict.passed and str(instance["id"]) not in seen:
            seen.add(str(instance["id"]))
            passing.append(
                {
                    "instance_id": str(instance["id"]),
                    "task_question": str(instance.get("question", "unavailable")),
                    "run_id": entry["run_id"],
                    "trace": trace_excerpt(completion, related_operations=related),
                }
            )
    return {
        "patterns": contexts,
        "alternatives": alternatives,
        "passing": passing,
        "passing_status": "observed held-in examples"
        if passing
        else "no passing trace evidence available",
    }


def structural_execution_error(verdict: Verdict) -> dict[str, str] | None:
    if verdict.cause is not VerifierCause.RUNTIME_ERROR:
        return None
    exception_type, _, message = verdict.detail.partition(": ")
    # Fixed vocabulary only: even the suffix of a familiar error can contain
    # a provider response or task payload. Never echo arbitrary exception text.
    messages = {
        "AnswerDecision.accept() missing 1 required positional argument: 'answer'",
        "type object 'AnswerDecision' has no attribute 'reject'",
        "expected string or bytes-like object, got 'tuple'",
        "'tuple' object has no attribute 'strip'",
        "'tuple' object has no attribute 'splitlines'",
    }
    return {
        "exception_type": (
            exception_type
            if exception_type
            in {
                "TypeError",
                "AttributeError",
                "ValueError",
                "RuntimeError",
                "TimeoutError",
                "KeyError",
                "IndexError",
                "NameError",
            }
            else "other exception"
        ),
        "message": message if message in messages else "message omitted: may contain task data",
    }


def validation_history_data(
    validation_path: Path, record: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Read internal comparison inputs; identifiers and contract data stay here."""
    links = record.get("links") or {}
    split = (links.get("splits") or {}).get("heldout") or {}
    if not split.get("round_dir"):
        return None
    path = (validation_path / split["round_dir"]).resolve()
    if not path.is_relative_to(validation_path.resolve()):
        raise ValueError("validation history link escapes its round")
    if not (path / "runs.jsonl").exists():
        return None
    # Durable verdicts contain all exported fields, including structural errors.
    # History must not open large/transient trace bodies just to report numbers.
    instances = [
        json.loads(line)
        for line in (path / "instances.jsonl").read_text().splitlines()
        if line.strip()
    ]
    entries = canonical_manifest_entries(
        load_manifest(path.parent, int(path.name.removeprefix("round_"))), instances
    )
    contract_path = (validation_path / str(record["subject_id"]) / "evaluation.json").resolve()
    if not contract_path.is_relative_to(validation_path.resolve()):
        raise ValueError("validation subject escapes its round")
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    if (
        record.get("harness_hash")
        and contract.get("harness_hash")
        and contract["harness_hash"] != record["harness_hash"]
    ):
        raise ValueError("validation subject disagrees with its saved harness identity")
    return entries, contract


def history_diagnostics_from_data(
    entries: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    verdicts = [Verdict.from_dict(entry["verdict"]) for entry in entries]
    costs = [float(e["cost"]) for e in entries if e.get("cost") is not None]
    diagnostics = {
        "n_attempts": len(verdicts),
        "exact_passes": sum(v.passed for v in verdicts),
        "n_runtime_errors": sum(v.cause is VerifierCause.RUNTIME_ERROR for v in verdicts),
        "terminal_failures": {
            cause.value: sum(v.cause is cause for v in verdicts)
            for cause in sorted(UNSCORED_CAUSES, key=lambda cause: cause.value)
        },
        "total_cost": math.fsum(costs)
        if len(costs) == len(entries) and all(math.isfinite(cost) for cost in costs)
        else None,
        "n_cost_measurements": len(costs),
    }
    # Environment identity comes from the persisted subject contract, not task data.
    environment = (contract.get("verifier_config") or {}).get("environment")
    diagnostics["quality"] = aggregate_quality_diagnostics(
        verdicts, contract.get("verifier_config")
    )
    if environment in {"oolong_pairs", "graphwalks"}:
        diagnostics["pairs" if environment == "oolong_pairs" else "sets"] = (
            aggregate_pair_diagnostics(verdicts)
        )
    diagnostics["first_runtime_error"] = next(
        (
            structural_execution_error(verdict)
            for verdict in verdicts
            if verdict.cause is VerifierCause.RUNTIME_ERROR
        ),
        None,
    )
    return diagnostics


def validation_history_diagnostics(validation_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Export only aggregate numbers and a sanitized structural runtime error."""
    data = validation_history_data(validation_path, record)
    return (
        history_diagnostics_from_data(*data)
        if data
        else {"status": "validation diagnostics unavailable"}
    )


def validation_history_behavior(
    validation_path: Path, record: dict[str, Any]
) -> dict[str, Any] | None:
    """Read only persisted aggregate telemetry, never held-out trace bodies."""
    from shrlm.optimization.behavior import BEHAVIOR_SCHEMA
    from shrlm.optimization.validation import load_summary

    link = (record.get("links") or {}).get("summary")
    if not link:
        return None
    path = (validation_path / link).resolve()
    if not path.is_relative_to(validation_path.resolve()):
        raise ValueError("behavior summary link escapes its validation round")
    if not path.exists():
        return None
    summary = load_summary(path.parent)
    behavior = summary.get("splits", {}).get("heldout", {}).get("behavior")
    if not behavior or behavior.get("schema") != BEHAVIOR_SCHEMA:
        return None
    # Readers export only detector counts through activation_for_surface.
    return behavior


def validation_history_progress(
    validation_path: Path, record: dict[str, Any], baseline: dict[str, Any] | None
) -> dict[str, Any]:
    """Advisory comparison owned by the evaluated subject, never its constituents."""
    result = {
        "version": DIAGNOSTIC_HISTORY_VERSION,
        "status": "not_assessed",
        "reason": "no comparable rejected evaluation",
        "interpretation": "Descriptive only; v=1 provides no reliable causal estimate. Promotion is unchanged.",
    }
    if record.get("decision") != "rejected" or not baseline:
        return result
    candidate_data = validation_history_data(validation_path, record)
    baseline_data = validation_history_data(validation_path, baseline)
    if candidate_data is None or baseline_data is None:
        return result
    candidate_entries, candidate_contract = candidate_data
    baseline_entries, baseline_contract = baseline_data
    identity_keys = ("verifier_config", "verifier_type", "repetitions", "validation_protocol")
    if any(
        key not in baseline_contract
        or key not in candidate_contract
        or baseline_contract[key] != candidate_contract[key]
        for key in identity_keys
    ):
        result["reason"] = "verifier or evaluation contracts are missing or incompatible"
        return result
    candidate_set = sorted((entry["instance_id"], entry["attempt"]) for entry in candidate_entries)
    baseline_set = sorted((entry["instance_id"], entry["attempt"]) for entry in baseline_entries)
    if candidate_set != baseline_set or len(set(candidate_set)) != len(candidate_set):
        result["reason"] = "evaluated instance/attempt sets differ or contain duplicates"
        return result
    old = history_diagnostics_from_data(*baseline_data)
    new = history_diagnostics_from_data(*candidate_data)
    result["status"] = compare_quality_diagnostics(old["quality"], new["quality"])
    result["reason"] = {
        "potentially_promising": "primary recorded quality improved despite rejection; consider a materially different refinement",
        "no_measured_improvement": "primary recorded quality did not improve",
        "not_assessed": "primary recorded quality is unavailable or incompatible",
    }[result["status"]]
    result["baseline"] = old
    result["candidate"] = new
    return result
