"""Read-only proposal context from held-in evidence and aggregate validation outcomes.

Never write derived metrics into persisted verifier results, bundles, or ledgers.
The two entry points deliberately keep held-in trace excerpts separate from
held-out aggregates; history exports no task, answer, or trace text.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rlm.core.types import RLMChatCompletion
from shrlm.optimization.digest import head_tail
from shrlm.optimization.driver import canonical_manifest_entries, load_manifest, load_round
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict, iter_nodes
from shrlm.optimization.walker import build_call_tree

EVIDENCE_SELECTOR_VERSION = "2.0.0"
DIAGNOSTIC_HISTORY_VERSION = "1.0.0"
TRACE_EXCERPT_CHARS = 4800
MAX_TRACE_SNIPPETS = 6

UNSCORED_CAUSES = {
    VerifierCause.RUNTIME_ERROR,
    VerifierCause.RESOURCE_TERMINATED,
    VerifierCause.WRONG_FORMAT,
    VerifierCause.CONTENT_FILTERED,
}

OOLONG_SCORE_RE = re.compile(
    r"score=(0\.\d+|1\.0+) exact=(True|False) "
    r"kind=(numeric|comparison|date|month_year|list|label|user|string)"
)


def aggregate_quality_diagnostics(
    verdicts: Sequence[Verdict], environment: str | None
) -> dict[str, Any]:
    """Interpret only known verifier formats, never model diagnoses or answer text."""
    from shrlm.environments.oolong_pairs import recorded_pair_metrics

    supported = environment in {"oolong_pairs", "graphwalks", "oolong"}
    definition = (
        {
            "name": "score" if environment == "oolong" else "f1",
            "direction": "higher",
            "aggregation": "all_attempt_mean",
            "missing_policy": "known_terminal_zero_unknown_unavailable/v1",
        }
        if supported
        else None
    )
    values = []
    n_zero = 0
    for verdict in verdicts:
        if not supported:
            continue
        if verdict.cause in UNSCORED_CAUSES or (
            environment == "oolong"
            and verdict.cause is VerifierCause.NO_ANSWER
            and verdict.detail == "final line carried an explicit empty marker"
        ):
            n_zero += 1
        elif environment == "oolong":
            match = OOLONG_SCORE_RE.fullmatch(verdict.detail)
            if match:
                values.append(float(match[1]))
        else:
            # GraphWalks emits the identical strict set-metric format.
            metrics = recorded_pair_metrics(verdict)
            if metrics is not None:
                values.append(float(metrics["f1"]))
    unknown = len(verdicts) - len(values) - n_zero
    return {
        "definition": definition,
        "n_attempts": len(verdicts),
        "n_measured": len(values),
        "n_known_zero": n_zero,
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
) -> dict[str, Any]:
    if completion.metadata is None:
        return {"observation": "trace excerpt unavailable"}
    root = build_call_tree(completion)
    nodes = {node.node_id: node for node in iter_nodes(root)}
    by_node: dict[str, list[dict[str, Any]]] = {}
    callers: dict[str, tuple[str, int]] = {}
    for node in nodes.values():
        blocks = []
        for iteration in node.iterations:
            for block_index, block in enumerate(iteration.code_blocks):
                for child in block.calls:
                    callers[child.node_id] = (node.node_id, len(blocks))
                blocks.append(
                    {
                        "node_id": node.node_id,
                        "iteration_index": iteration.index,
                        "code_block_index": block_index,
                        "code": block.code,
                        "stdout": block.stdout,
                        "stderr": block.stderr,
                    }
                )
        by_node[node.node_id] = blocks

    selected: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(snippet: dict[str, Any], reason: str) -> None:
        key = (snippet["node_id"], snippet.get("iteration_index"), snippet.get("code_block_index"))
        if key not in selected and len(selected) < MAX_TRACE_SNIPPETS:
            selected[key] = {**snippet, "reason": reason}

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
        else:
            unresolved = True
    for node_id in cited:
        if node_id not in nodes:
            unresolved = True
            continue
        if node_id in callers:
            parent_id, position = callers[node_id]
            window = [b for b in by_node[parent_id][position:] if b["code"].strip()][:3]
            for offset, block in enumerate(window):
                add(block, "cited call's caller" if offset == 0 else "following consumer context")
            node = nodes[node_id]
            add(
                {"node_id": node_id, "prompt": str(node.prompt), "response": node.response},
                "cited child prompt/return",
            )
    selection = "cited operations and caller context"
    if not selected:
        selection = "structural fallback: no resolvable operation or child citation"
        blocks = by_node[root.node_id]
        if blocks:
            for block in (blocks[0], blocks[-1]):
                add(block, "structural fallback")
    if unresolved:
        selection += "; unresolved citations"
    snippets = list(selected.values())
    per_snippet = TRACE_EXCERPT_CHARS // max(len(snippets), 1)
    for snippet in snippets:
        fields = [
            key for key in ("code", "stdout", "stderr", "prompt", "response") if snippet.get(key)
        ]
        per_field = per_snippet // max(len(fields), 1)
        for key in fields:
            snippet[key] = bounded_excerpt(snippet[key], per_field)
        snippet["node_id"] = bounded_excerpt(snippet["node_id"], 200)
    return {
        "observation": "partial observed behavior; labels and causal effectiveness are not verified",
        "selection": selection,
        "snippets": snippets,
        "observed_child_calls": len(root.children),
    }


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
    is_pairs = (bundle.get("config", {}).get("verifier_config") or {}).get(
        "environment"
    ) == "oolong_pairs"
    for index, pattern in enumerate(bundle.get("patterns", [])):
        representative_ids = pattern.get("representatives") or []
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
        record = candidates[0]
        detail = record.get("detail") or {}
        instance = by_instance.get(str(record["instance_id"]), {})
        verdict = Verdict.from_dict(record["verdict"])
        context = {
            "symptom_summary": head_tail(detail.get("symptom_summary", "unavailable"), 2000),
            "evidence_node_ids": detail.get("evidence_node_ids", []),
            "operation_evidence": detail.get("operation_evidence", []),
            "verification_limits": head_tail(detail.get("verification_limits", "unavailable"), 500),
            "level_grounded": record.get("level_grounded", False),
            "task_question": head_tail(str(instance.get("question", "unavailable")), 8000),
            "verifier_detail": bounded_excerpt(verdict.detail, 2000),
        }
        if is_pairs:
            context["pair_diagnostics"] = pair_diagnostics(verdict)
            if context["pair_diagnostics"]["scored"]:
                from shrlm.environments.oolong_pairs import extract_answer_pairs

                gold = set(extract_answer_pairs(verdict.gold) or [])
                produced = set(extract_answer_pairs(verdict.produced) or [])
                context["missing_examples"] = sorted(gold - produced)[:3]
                context["extra_examples"] = sorted(produced - gold)[:3]
        if record.get("run_id"):
            matches = [i for i, entry in enumerate(entries) if entry["run_id"] == record["run_id"]]
        else:
            matches = [
                i
                for i, entry in enumerate(entries)
                if str(entry["instance_id"]) == str(record["instance_id"])
                and entry["verdict"] == record["verdict"]
            ]
        context["trace"] = {"observation": "trace unavailable: missing or ambiguous run linkage"}
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
            context["trace"] = trace_excerpt(
                runs[selected][1],
                detail.get("evidence_node_ids", []),
                detail.get("operation_evidence", []),
            )
        contexts[index] = context
    passing = []
    seen = set()
    for (instance, completion), verdict in zip(runs, verdicts, strict=True):
        if verdict.passed and str(instance["id"]) not in seen:
            seen.add(str(instance["id"]))
            passing.append(
                {
                    "instance_id": str(instance["id"]),
                    "task_question": head_tail(str(instance.get("question", "unavailable")), 8000),
                    "trace": trace_excerpt(completion),
                }
            )
            if len(passing) == 2:
                break
    return {
        "patterns": contexts,
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
    diagnostics["quality"] = aggregate_quality_diagnostics(verdicts, environment)
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
