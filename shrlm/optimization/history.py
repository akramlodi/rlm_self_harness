"""Factual prior-attempt identities, independent of the prompt's history budget."""

import json
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


def revision_violation(
    revision: dict[str, Any] | None,
    *,
    surface: str,
    mechanism: str,
    fingerprint: str,
    attempts: Sequence[dict[str, Any]],
) -> str | None:
    related = [
        record
        for record in attempts
        if record.get("surface") == surface
        and (
            record.get("effective_edit_fingerprint") == fingerprint
            or record.get("mechanism") == mechanism
        )
    ]
    if revision is None:
        if related:
            exact = [r for r in related if r.get("effective_edit_fingerprint") == fingerprint]
            prior = (exact or related)[-1]
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
