"""Factual prior-attempt identities, independent of the prompt's history budget."""

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from shrlm.harness_identity import canonical_json_sha256
from shrlm.optimization.candidates import SURFACE_SERIALIZATION_KEYS

HISTORY_BUDGET_CHARS = 12000
HISTORY_SCHEMA = "proposal-history/v2"


def surface_fingerprint(serialization: dict[str, Any], surface: str) -> str:
    content = {key: serialization["surfaces"][key] for key in SURFACE_SERIALIZATION_KEYS[surface]}
    return canonical_json_sha256(content)


def prior_attempts(
    history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "round": decision.get("round", position),
            "incumbent_hash": record.get("incumbent_hash", decision.get("incumbent_hash")),
        }
        for position, (records, decision) in enumerate(history)
        for record in records
    ]


def prior_evaluations(
    history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Index evaluated subjects, never assign a batch's rejection to constituents."""
    return [
        {key: record.get(key) for key in ("round", "subject_id", "incumbent_hash", "harness_hash")}
        for record in prior_attempts(history)
        if record.get("decision") == "rejected"
        and record.get("links")
        and record.get("harness_hash")
        and record.get("incumbent_hash")
    ]


def select_predecessor(
    *,
    surface: str,
    mechanism: str,
    attempts: Sequence[dict[str, Any]],
    fingerprint: str | None = None,
) -> dict[str, Any] | None:
    related = [
        record
        for record in attempts
        if record.get("surface") == surface
        and (
            record.get("mechanism") == mechanism
            or (fingerprint is not None and record.get("effective_edit_fingerprint") == fingerprint)
        )
    ]
    exact = [
        record
        for record in related
        if fingerprint is not None and record.get("effective_edit_fingerprint") == fingerprint
    ]
    measured = [
        record
        for record in related
        if record.get("decision") == "bundled"
        or (record.get("decision") in {"promoted", "rejected"} and record.get("links"))
    ]
    ranked = exact or measured or related
    return ranked[-1] if ranked else None


def revision_violation(
    revision: dict[str, Any] | None,
    *,
    surface: str,
    mechanism: str,
    fingerprint: str,
    attempts: Sequence[dict[str, Any]],
) -> str | None:
    prior = select_predecessor(
        surface=surface, mechanism=mechanism, fingerprint=fingerprint, attempts=attempts
    )
    if revision is None:
        if prior:
            brief = {
                key: str(prior[key])[:240]
                for key in (
                    "surface",
                    "mechanism",
                    "behavioral_change",
                    "observed_failure",
                    "decision",
                )
                if prior.get(key) is not None
            }
            return (
                f"revisited intervention requires revision referencing round {prior['round']} "
                f"subject {prior['subject_id']} and a changed operation or joint hypothesis; "
                f"predecessor: {json.dumps(brief, ensure_ascii=True)}"
            )
        return None
    matches = [
        record
        for record in attempts
        if record.get("round") == revision["round"]
        and record.get("subject_id") == revision["subject_id"]
    ]
    if len(matches) != 1:
        return "revision reference does not resolve to one prior attempt"
    if matches[0].get("surface") != surface and matches[0].get("mechanism") != mechanism:
        return "revision reference is unrelated to this surface and mechanism"
    return None


def compact_history(
    history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]],
    *,
    choices: dict[str, dict[str, Any]] | None = None,
    mechanisms: set[str] | None = None,
    incumbent_hash: str | None = None,
    budget: int = HISTORY_BUDGET_CHARS,
) -> tuple[str, list[str]]:
    """Pack recent facts first, then whole mandatory identities per offered choice."""
    attempts = prior_attempts(history)
    definitions: dict[str, Any] = {}

    def diagnostics(value: dict[str, Any]) -> dict[str, Any]:
        quality = dict(value.get("quality") or {})
        definition = quality.pop("definition", None)
        if definition:
            key = canonical_json_sha256(definition)[:16]
            definitions[key] = definition
            quality["definition_ref"] = key
        return {
            "exact_passes": value.get("exact_passes"),
            "n_attempts": value.get("n_attempts"),
            "quality": quality or {"status": "not_assessed"},
        }

    def row(record: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: record[key]
            for key in (
                "round",
                "subject_id",
                "surface",
                "mechanism",
                "decision",
                "effective_edit_fingerprint",
                "owner",
                "batch_subject_id",
            )
            if key in record
        }
        result["behavioral_change"] = str(record.get("behavioral_change", "not_assessed"))[:240]
        result["predicted_effect"] = str(record.get("predicted_effect", "not_assessed"))[:240]
        if record.get("merge"):
            result["constituent_ids"] = record["merge"].get("constituent_ids", [])
        result["observed_failure"] = str(record.get("observed_failure", "not_assessed"))[:240]
        activation = record.get("activation") or {"status": "not_assessed"}
        result["activation"] = {
            key: activation[key]
            for key in ("status", "reason", "n_observed", "n_eligible")
            if key in activation
        }
        if record.get("decision") in {"preflight_rejected", "not_materialized"}:
            result["owner_relation"] = record.get("owner", "unknown (legacy or no recorded owner)")
        result["reasons"] = [str(reason)[:240] for reason in record.get("reasons", [])[:2]]
        # A combined measurement belongs only to its evaluated subject.
        if record.get("decision") != "bundled":
            progress = record.get("diagnostic_progress") or {}
            result["quality_signal"] = progress.get("status", "not_assessed")
            result["baseline"] = diagnostics(progress.get("baseline") or {})
            result["measured"] = diagnostics(
                record.get("diagnostics") or progress.get("candidate") or {}
            )
        else:
            result["measurement"] = "combined outcome only; no individual score"
        return result

    rows = {(record["round"], record.get("subject_id")): row(record) for record in attempts}
    summaries = {}
    for position, (records, decision) in enumerate(history):
        label = decision.get("round", position)
        summaries[position] = {
            "round": label,
            "promoted": decision.get("promoted"),
            "promoted_harness_hash": decision.get("promoted_harness_hash"),
            "attempts": len(records),
            "baseline": diagnostics(decision.get("baseline_diagnostics") or {}),
            "outcomes": [
                {
                    "subject_id": r.get("subject_id"),
                    "decision": r.get("decision"),
                    "measured": diagnostics(r.get("diagnostics") or {}),
                }
                for r in records
                if r.get("links")
            ],
        }
    selected_rounds = set(range(max(0, len(history) - 3), len(history)))
    promotions = [
        i
        for i, (_, decision) in enumerate(history)
        if decision.get("promoted")
        and (incumbent_hash is None or decision.get("promoted_harness_hash") == incumbent_hash)
    ]
    if promotions:
        selected_rounds.add(promotions[-1])
    promising = [
        i
        for i, (records, _) in enumerate(history)
        if i not in selected_rounds
        and any(
            r.get("diagnostic_progress", {}).get("status") == "potentially_promising"
            and (not mechanisms or r.get("mechanism") in mechanisms)
            for r in records
        )
    ]
    if promising:
        selected_rounds.add(promising[-1])
    selected: dict[tuple[Any, Any], dict[str, Any]] = {}
    omitted_choices: list[str] = []

    def render() -> str:
        # Only definitions used by included rows consume the prompt budget.
        content = {
            "rounds": [summaries[i] for i in sorted(selected_rounds)],
            "attempts": list(selected.values()),
        }
        serialized = json.dumps(content, sort_keys=True)
        header = {
            "history_schema": HISTORY_SCHEMA,
            "rounds_total": len(history),
            "attempts_total": len(attempts),
            "attempts_omitted": len(attempts) - len(selected),
            "status_totals": dict(Counter(str(r.get("decision", "unknown")) for r in attempts)),
            "omission_reason": "rendered budget; full archive and reference index retained",
            "withheld_choices": omitted_choices,
            "metric_definitions": {k: v for k, v in definitions.items() if k in serialized},
        }
        return json.dumps(header, sort_keys=True) + "\n" + serialized

    if len(render()) > budget:
        raise ValueError("compact recent history exceeds history budget")
    # These complete identity rows must be visible if their choices are offered.
    for choice_id, choice in (choices or {}).items():
        before = dict(selected)
        for prior in choice.get("predecessors", {}).values():
            key = (prior["round"], prior["subject_id"])
            selected[key] = rows[key]
        if len(render()) > budget:
            selected = before
            omitted_choices.append(choice_id)
    # Recent/promotion/qualified-positive detail precedes optional old context.
    priority = sorted(selected_rounds, reverse=True) + [
        i for i in reversed(range(len(history))) if i not in selected_rounds
    ]
    for index in priority:
        label = history[index][1].get("round", index)
        for key, value in rows.items():
            if key[0] != label or key in selected:
                continue
            selected[key] = value
            if len(render()) > budget:
                del selected[key]
    return render(), omitted_choices
