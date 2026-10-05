"""Aggregate existing hook events, without inferring instruction compliance."""

from collections.abc import Sequence
from typing import Any

from rlm.core.types import ANSWER_REDIRECTED, RLMChatCompletion

BEHAVIOR_SCHEMA = "behavior-observations/v1"


def observation(count: int, measured: int, total: int) -> dict[str, Any]:
    return {
        "count": count,
        "n_measured_runs": measured,
        "n_unknown_runs": total - measured,
        "status": "observed"
        if count
        else "not_observed"
        if measured == total and total
        else "not_assessed",
    }


def summarize_behavior(runs: Sequence[tuple[RLMChatCompletion, bool]]) -> dict[str, Any]:
    """Follow canonical block call edges once; S6/S9 observations are root only."""
    totals = {"syntax_retries": 0, "answer_redirects": 0}
    measured = dict.fromkeys(totals, 0)
    skills: dict[str, int] = {}
    skill_measured = 0
    for completion, terminated in runs:
        metadata = completion.metadata or {}
        turns = metadata.get("iterations", [])
        complete = bool(turns) and not terminated
        retries_complete = redirects_complete = complete
        for turn in turns:
            telemetry = turn.get("trace_metrics") or {}
            redirects_complete &= "answer_event" in telemetry
            totals["answer_redirects"] += telemetry.get("answer_event") == ANSWER_REDIRECTED
            retries_complete &= bool(telemetry)
            for block in turn.get("code_blocks", []):
                result = block.get("result", {})
                retries_complete &= "rlm_calls" in result
                for child in result.get("rlm_calls", []):
                    value = (child.get("trace_metrics") or {}).get("retries")
                    valid = type(value) is int and value >= 0
                    retries_complete &= valid
                    if valid:
                        assert isinstance(value, int)
                        totals["syntax_retries"] += value
        measured["syntax_retries"] += retries_complete
        measured["answer_redirects"] += redirects_complete
        names = {
            entry["name"] for entry in (metadata.get("run_metadata") or {}).get("skill_index", [])
        }
        counts = dict.fromkeys(names, 0)
        skill_complete = complete

        def visit(node_metadata: dict[str, Any], counts: dict[str, int]) -> None:
            nonlocal skill_complete
            for turn in node_metadata.get("iterations", []):
                for block in turn.get("code_blocks", []):
                    result = block.get("result", {})
                    skill_complete &= "skill_loads" in result
                    for event in result.get("skill_loads", []):
                        name = event.get("skill")
                        if name in counts:
                            counts[name] += 1
                    for child in result.get("rlm_calls", []):
                        child_metadata = child.get("metadata")
                        if child_metadata is not None:
                            visit(child_metadata, counts)

        visit(metadata, counts)
        skill_measured += skill_complete
        for name, count in counts.items():
            skills[name] = skills.get(name, 0) + count
    return {
        "schema": BEHAVIOR_SCHEMA,
        "n_runs": len(runs),
        **{key: observation(count, measured[key], len(runs)) for key, count in totals.items()},
        "skill_loads": {
            **observation(sum(skills.values()), skill_measured, len(runs)),
            "by_skill": skills,
        },
    }


def activation_for_surface(
    surface: str | None, summary: dict[str, Any] | None, *, skill_name: str | None = None
) -> dict[str, Any]:
    """Only explicit hooks have detectors; quality never substitutes for activation."""
    if not summary or summary.get("schema") != BEHAVIOR_SCHEMA:
        return {"status": "not_assessed"}
    detector = {"S6": "syntax_retries", "S9": "answer_redirects", "S10": "skill_loads"}.get(surface)
    if detector is None or (surface == "S10" and skill_name is None):
        return {"status": "not_assessed"}
    source = summary[detector]
    if surface == "S10" and skill_name not in source.get("by_skill", {}):
        return {"status": "not_assessed"}
    count = source["count"] if surface != "S10" else source.get("by_skill", {}).get(skill_name, 0)
    result = observation(count, source["n_measured_runs"], summary["n_runs"])
    return {
        "detector": detector,
        "scope": "tree" if surface == "S10" else "root",
        "intended_behavior": "not_assessed",
        **result,
    }
