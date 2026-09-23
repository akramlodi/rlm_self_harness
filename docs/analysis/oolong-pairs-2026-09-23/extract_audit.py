"""Rebuild the proposal inventory and diagnostic tables without model calls."""

import csv
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from shrlm.environments.oolong_pairs import recorded_pair_metrics
from shrlm.optimization.proposal import extract_proposal_response
from shrlm.optimization.types import Verdict

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiment_oolong_pairs_dsv4f_20260916_131837"
OUT = Path(__file__).resolve().parent
APPENDIX_MARKER = "<!-- GENERATED APPENDICES -->"


def fenced(value: str, language: str = "text") -> str:
    fence = "````"
    while fence in value:
        fence += "`"
    return f"{fence}{language}\n{value}" + ("" if value.endswith("\n") else "\n") + f"{fence}\n"


def exact_edit(edit: dict[str, Any]) -> str:
    kind = edit["kind"]
    if kind == "text":
        return "Full replacement instruction:\n\n" + fenced(edit["new_text"])
    if kind == "code":
        metadata = {key: value for key, value in edit.items() if key != "source"}
        return (
            "Edit metadata:\n\n"
            + fenced(json.dumps(metadata, indent=2), "json")
            + "\nComplete function source:\n\n"
            + fenced(edit["source"], "python")
        )
    if kind == "policy":
        return "Complete policy edit, including unset fields:\n\n" + fenced(
            json.dumps(edit, indent=2), "json"
        )
    if kind == "skills":
        metadata = {key: value for key, value in edit.items() if key != "body"}
        return (
            "Complete skill metadata:\n\n"
            + fenced(json.dumps(metadata, indent=2), "json")
            + "\nComplete skill body:\n\n"
            + fenced(edit["body"], "markdown")
        )
    raise ValueError(f"Unsupported edit kind: {kind}")


def write_appendices(audit: dict[str, Any]) -> None:
    report_path = OUT / "report.md"
    narrative, marker, _ = report_path.read_text().partition(APPENDIX_MARKER)
    assert marker, "The report must contain the appendix insertion marker."
    edit_dir = OUT / "edits"
    edit_dir.mkdir(exist_ok=True)
    parts = [
        "\n## Exact-edit appendices\n\n"
        "The replacements below are complete, including unchanged material retained by the "
        "proposer. A proposal replaces its named surface on that round's incumbent; it is not "
        "an instruction to append the entire block. Each item links to its exact edit JSON "
        "and, for materialized candidates, a unified before/after diff of the stored surface. "
        "Those files and `audit.json` preserve whitespace and terminal newlines that may be "
        "hard to distinguish in Markdown.\n\n"
        "Text edits use the recorded `literal-text/v1` contract: the displayed instruction "
        "is literal model-facing text. The host escapes braces once for template storage; "
        "stored-surface diffs therefore show doubled literal braces where needed. Code and "
        "policy payloads are shown separately according to their edit kind. No proposed "
        "instruction below has been corrected or rewritten for this report.\n\n"
        "- [Appendix A: five promoted edits](#appendix-a-promoted-edits)\n"
        "- [Appendix B: eleven tested, unpromoted edits](#appendix-b-tested-unpromoted-edits)\n"
        "- [Appendix C: eleven unmaterialized drafts](#appendix-c-unmaterialized-drafts)\n"
    ]
    for letter, promoted, title in (
        ("A", True, "Promoted edits"),
        ("B", False, "Tested, unpromoted edits"),
    ):
        parts.append(f"\n## Appendix {letter}: {title}\n")
        selected = [p for p in audit["proposals"] if p["promoted"] == promoted]
        for index, proposal in enumerate(selected, 1):
            candidate_id = proposal["id"]
            edit_name = f"{candidate_id}.edit.json"
            diff_name = f"{candidate_id}.patch"
            (edit_dir / edit_name).write_text(json.dumps(proposal["edit"], indent=2) + "\n")
            (edit_dir / diff_name).write_text(proposal["diff"])
            parts.append(
                f"\n### {letter}{index}. {candidate_id} — {proposal['surface']}\n\n"
                f"Round {proposal['round']}; stored field `{proposal['field']}`. "
                f"Validation subject: `{proposal['batch_subject']}`. "
                f"Outcome: {'promoted' if promoted else 'not promoted'}. "
                "The validation score belongs to this subject, including its siblings when "
                "bundled.\n\n"
                f"Base harness: `{proposal['base_hash']}`.\n\n"
                f"[Saved proposal](../../../{proposal['source']}) · "
                f"[Exact edit JSON](edits/{edit_name}) · "
                f"[Stored-surface diff](edits/{diff_name})\n\n" + exact_edit(proposal["edit"])
            )
    parts.append(
        "\n## Appendix C: Unmaterialized drafts\n\n"
        "Each distinct edit payload is listed once per round. Repeated submissions of the "
        "same payload are listed together, even if the explanation changed. Violations below "
        "are the saved response-level messages and can mention sibling candidates. These "
        "drafts were never run through held-out validation.\n"
    )
    for index, draft in enumerate(audit["unmaterialized_drafts"], 1):
        edit_name = f"C{index:02d}-r{draft['round']:02d}-{draft['surface'].lower()}.edit.json"
        (edit_dir / edit_name).write_text(json.dumps(draft["edit"], indent=2) + "\n")
        source = f"{audit['experiment']}/opt/round_{draft['round']:02d}/proposals_complete.json"
        parts.append(
            f"\n### C{index}. Round {draft['round']} — {draft['surface']}\n\n"
            f"[Saved responses](../../../{source}) · [Exact edit JSON](edits/{edit_name})\n\n"
        )
        for occurrence in draft["occurrences"]:
            parts.append(
                f"Submission attempt {occurrence['attempt']}, candidate position "
                f"{occurrence['position']}. Saved response violation:\n\n"
                + fenced(occurrence["attempt_violation"])
                + "\n"
            )
        parts.append(exact_edit(draft["edit"]))
    report_path.write_text(narrative + APPENDIX_MARKER + "\n" + "".join(parts))


def read_json(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(edit):
    # Code validation adds the function name inferred from its unchanged source.
    canonical = {key: value for key, value in edit.items() if key != "def_name"}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def metrics(rows):
    scored = [recorded_pair_metrics(Verdict.from_dict(row["verdict"])) for row in rows]
    available = [m for m in scored if m is not None]
    return {
        "n": len(rows),
        "passes": sum(row["passed"] for row in rows),
        "f1_all": sum(m["f1"] for m in available) / len(rows),
        "f1_scored": sum(m["f1"] for m in available) / len(available) if available else None,
        "n_scored": len(available),
        "causes": dict(Counter(row.get("cause") or "passed" for row in rows)),
        "cost": sum(row.get("cost") or 0 for row in rows),
        "cost_lower_bound": any(row.get("usage_lower_bound") for row in rows),
        "mean_seconds": sum(row["execution_time"] for row in rows) / len(rows),
        "mean_missing_scored": sum(m["missing"] for m in available) / len(available)
        if available
        else None,
        "mean_extra_scored": sum(m["extra"] for m in available) / len(available)
        if available
        else None,
    }


def small_run(row):
    return {
        k: row.get(k) for k in ("run_id", "passed", "cause", "cost", "execution_time", "trace_path")
    } | {"metrics": recorded_pair_metrics(Verdict.from_dict(row["verdict"]))}


def surface_text(value):
    return value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True)


def main():
    rounds, proposals, drafts, attempts = [], [], [], []
    initial = read_json(EXP / "opt/round_01/mining/round_01/harness.json")
    expected_incumbent_hash = initial["hash"]
    for rd in sorted((EXP / "opt").glob("round_*")):
        number = int(rd.name.split("_")[1])
        validation = rd / "validation" / rd.name
        decision = read_json(validation / "decision.json")
        ledger = jsonl(validation / "promotions.jsonl")
        measured = [row for row in ledger if row["decision"] != "bundled"]
        assert len(measured) == 1
        subject = measured[0]["subject_id"]
        baseline_path = validation / "baseline/heldout/round_00"
        candidate_path = validation / subject / "heldout/round_00"
        baseline_rows = jsonl(baseline_path / "runs.jsonl")
        candidate_rows = jsonl(candidate_path / "runs.jsonl")
        baseline_harness = read_json(baseline_path / "harness.json")
        candidate_harness = read_json(candidate_path / "harness.json")
        assert baseline_harness["hash"] == expected_incumbent_hash
        assert candidate_harness["hash"] == measured[0]["harness_hash"]
        rule = measured[0]["rule"]["heldout"]
        assert rule["baseline_pass_count"] == sum(row["passed"] for row in baseline_rows)
        assert rule["candidate_pass_count"] == sum(row["passed"] for row in candidate_rows)
        if decision["promoted"]:
            expected_incumbent_hash = candidate_harness["hash"]
        assert (baseline_path / "instances.jsonl").read_bytes() == (
            candidate_path / "instances.jsonl"
        ).read_bytes()
        bmap = {row["run_id"]: row for row in baseline_rows}
        cmap = {row["run_id"]: row for row in candidate_rows}
        assert bmap.keys() == cmap.keys() and len(bmap) == 10
        pairs = [
            {"run_id": key, "baseline": small_run(bmap[key]), "candidate": small_run(cmap[key])}
            for key in sorted(bmap)
        ]
        deltas = [
            (pair["candidate"]["metrics"] or {"f1": 0})["f1"]
            - (pair["baseline"]["metrics"] or {"f1": 0})["f1"]
            for pair in pairs
        ]
        rounds.append(
            {
                "round": number,
                "promoted": decision["promoted"],
                "subject": subject,
                "constituents": decision["constituent_ids"],
                "baseline": metrics(baseline_rows),
                "candidate": metrics(candidate_rows),
                "decision": measured[0],
                "paired_runs": pairs,
                "f1_win_tie_loss": [
                    sum(d > 0 for d in deltas),
                    sum(d == 0 for d in deltas),
                    sum(d < 0 for d in deltas),
                ],
                "exact_wins": [
                    key for key in bmap if not bmap[key]["passed"] and cmap[key]["passed"]
                ],
                "exact_losses": [
                    key for key in bmap if bmap[key]["passed"] and not cmap[key]["passed"]
                ],
                "validation_path": str(validation.relative_to(REPO)),
            }
        )
        incumbent = baseline_harness
        survivors = read_json(rd / "work/proposal_result.json")["result"]["survivors"]
        materialized = {fingerprint(s["spec"]["edit"]): s["candidate_id"] for s in survivors}
        for survivor in survivors:
            proposal_path = rd / "proposals" / survivor["candidate_id"] / "proposal.json"
            proposal = read_json(proposal_path)
            before = incumbent["harness"]["surfaces"]
            after = proposal["harness"]["harness"]["surfaces"]
            changed = [key for key in before if before[key] != after[key]]
            assert len(changed) == 1 and changed[0].startswith(proposal["surface"] + "_")
            field = changed[0]
            diff = (
                "\n".join(
                    difflib.unified_diff(
                        surface_text(before[field]).splitlines(),
                        surface_text(after[field]).splitlines(),
                        fromfile=f"round_{number:02d}_incumbent/{field}",
                        tofile=f"{proposal['candidate_id']}/{field}",
                        lineterm="",
                    )
                )
                + "\n"
            )
            proposals.append(
                {
                    "round": number,
                    "id": proposal["candidate_id"],
                    "surface": proposal["surface"],
                    "field": field,
                    "promoted": decision["promoted"],
                    "batch_subject": subject,
                    "source": str(proposal_path.relative_to(REPO)),
                    "base_hash": proposal["base_harness_hash"],
                    "candidate_hash": proposal["harness"]["hash"],
                    "edit": survivor["spec"]["edit"],
                    "spec": survivor["spec"],
                    "before": before[field],
                    "after": after[field],
                    "diff": diff,
                }
            )
        pc = read_json(rd / "proposals_complete.json")
        draft_map = {}
        submitted = set()
        for attempt in pc["attempts"]:
            parsed = extract_proposal_response(attempt["raw_response"])
            attempts.append(
                {
                    "round": number,
                    "attempt": attempt["attempt"],
                    "accepted": attempt["accepted"],
                    "violation": attempt["violation"],
                    "n_submitted_candidates": len(parsed["candidates"]),
                }
            )
            for position, candidate in enumerate(parsed["candidates"], 1):
                key = fingerprint(candidate["edit"])
                submitted.add(key)
                if key in materialized:
                    continue
                entry = draft_map.setdefault(
                    key,
                    {
                        "round": number,
                        "surface": candidate["surface"],
                        "edit": candidate["edit"],
                        "spec": candidate,
                        "occurrences": [],
                        "measured": False,
                    },
                )
                entry["occurrences"].append(
                    {
                        "attempt": attempt["attempt"],
                        "position": position,
                        "attempt_violation": attempt["violation"],
                    }
                )
        drafts.extend(draft_map.values())
        assert materialized.keys() <= submitted
    assert len(proposals) == 16 and sum(p["promoted"] for p in proposals) == 5
    assert len(drafts) == 11 and len(attempts) == 14
    final = read_json(EXP / "sh_rlm/harness.json")
    assert final["hash"] == expected_incumbent_hash
    changed_final = [
        key
        for key in initial["harness"]["surfaces"]
        if initial["harness"]["surfaces"][key] != final["harness"]["surfaces"][key]
    ]
    audit = {
        "experiment": str(EXP.relative_to(REPO)),
        "rounds": rounds,
        "proposals": proposals,
        "unmaterialized_drafts": drafts,
        "proposal_attempts": attempts,
        "final_changed_surfaces": changed_final,
        "initial_hash": initial["hash"],
        "final_hash": final["hash"],
        "f1_method": "Mean recorded three-decimal F1 over all 10 held-out attempts, zero for failures with no scored answer; scored-only mean also reported.",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_appendices(audit)
    with (OUT / "rounds.csv").open("w") as stream:
        fields = [
            "round",
            "promoted",
            "constituents",
            "baseline_pass",
            "candidate_pass",
            "baseline_f1",
            "candidate_f1",
            "f1_delta",
            "baseline_cost",
            "candidate_cost",
            "f1_wins",
            "f1_ties",
            "f1_losses",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rounds:
            b, c = row["baseline"], row["candidate"]
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [
                            row["round"],
                            row["promoted"],
                            ";".join(row["constituents"]),
                            b["passes"],
                            c["passes"],
                            b["f1_all"],
                            c["f1_all"],
                            c["f1_all"] - b["f1_all"],
                            b["cost"],
                            c["cost"],
                            *row["f1_win_tie_loss"],
                        ],
                        strict=True,
                    )
                )
            )
    print(
        json.dumps(
            {
                "proposals": len(proposals),
                "promoted": sum(p["promoted"] for p in proposals),
                "unmaterialized_drafts": len(drafts),
                "attempts": len(attempts),
                "final_changed_surfaces": changed_final,
            }
        )
    )


if __name__ == "__main__":
    main()
