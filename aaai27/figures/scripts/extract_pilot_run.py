#!/usr/bin/env python3
"""Extract the pilot-run tables used by make_figures.py from a SH-RLM run dir.

Usage:
    python3 extract_pilot_run.py <run_dir> <out_dir>

<run_dir> is an optimization output directory with opt/round_XX/... and
stage_usage.jsonl (here: rerun_mining_round2_test, DeepSeek-V4-Flash).
Writes pilot_validation.csv, pilot_mining.csv and pilot_stage_cost.csv.
Only aggregates the run already stored; it makes no model calls.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

LENGTHS = ("8192", "16384", "32768")


def subject_rows(run_dir: str) -> list[dict]:
    """One row per (round, stage, subject) from screening/validation summaries."""
    rows = []
    pattern = os.path.join(run_dir, "opt", "round_*", "*", "round_*", "*", "summary.json")
    for path in sorted(glob.glob(pattern)):
        parts = path.split(os.sep)
        subject, stage, round_dir = parts[-2], parts[-4], parts[-5]
        if stage not in ("screening", "validation"):
            continue
        with open(path) as fh:
            summary = json.load(fh)
        h = summary["splits"]["heldout"]
        pq = h["primary_quality"]
        causes = h.get("cause_counts", {})
        row = {
            "round": int(round_dir.split("_")[1]),
            "stage": stage,
            "subject": subject,
            "surface": "" if subject == "baseline" else subject.rsplit("-", 1)[1].upper(),
            "n_instances": h["n_instances"],
            "n_runs": h["n_runs"],
        }
        for length in LENGTHS:
            row[f"f1_{length}"] = round(pq["by_context_length"][length]["mean"], 4)
        row["f1_macro"] = round(pq["macro_by_context_length"], 4)
        row["f1_conditional"] = round(pq["conditional_on_measurement"]["mean"], 4)
        row["pass_count"] = h["pass_count"]
        row["mean_cost_usd"] = round(h["mean_cost"], 4)
        row["mean_sub_calls"] = round(h["mean_sub_calls"], 1)
        for cause in ("wrong_format", "resource_terminated", "incomplete",
                      "mixed_set_error", "spurious"):
            row[f"n_{cause}"] = causes.get(cause, 0)
        rows.append(row)
    # Attach the gate decision to each validated candidate.
    for path in sorted(glob.glob(os.path.join(run_dir, "opt", "round_*", "validation",
                                              "round_*", "promotions.jsonl"))):
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                for row in rows:
                    if row["stage"] == "validation" and row["subject"] == rec["subject_id"]:
                        row["decision"] = rec["decision"]
                        for length in LENGTHS:
                            row[f"tau_{length}"] = rec["tau_regression"][length]
                        row["tau_improvement"] = rec["tau_improvement"]
    return rows


def mining_rows(run_dir: str) -> list[dict]:
    rows = []
    for mdir in sorted(glob.glob(os.path.join(run_dir, "opt", "round_*", "mining", "round_*"))):
        rnd = int(os.path.basename(mdir).split("_")[1])
        ctx = {}
        with open(os.path.join(mdir, "instances.jsonl")) as fh:
            for line in fh:
                inst = json.loads(line)
                ctx[inst["id"]] = inst.get("context_len")
        with open(os.path.join(mdir, "runs.jsonl")) as fh:
            for line in fh:
                run = json.loads(line)
                detail = (run.get("verdict") or {}).get("detail") or ""
                m = re.search(r"f1=([0-9.]+)", detail)
                rows.append({
                    "round": rnd,
                    "instance_id": run["instance_id"],
                    "context_len": ctx.get(run["instance_id"]),
                    "attempt": run["attempt"],
                    "f1": float(m.group(1)) if m else 0.0,
                    "passed": run["passed"],
                    "cause": run["cause"] or "",
                    "cost_usd": round(run["cost"], 5),
                    "execution_time_s": round(run["execution_time"], 1),
                })
    return rows


def stage_cost_rows(run_dir: str) -> list[dict]:
    totals: dict[tuple[int, str], dict] = {}
    with open(os.path.join(run_dir, "stage_usage.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            key = (rec["round_index"], rec["stage"])
            t = totals.setdefault(key, {"round": key[0], "stage": key[1], "cost_usd": 0.0,
                                        "wall_hours": 0.0, "lower_bound": False})
            t["cost_usd"] += rec["cost"]
            t["wall_hours"] += rec["wall_seconds"] / 3600
            t["lower_bound"] = t["lower_bound"] or rec["lower_bound"]
    out = []
    for t in sorted(totals.values(), key=lambda r: (r["round"], r["stage"])):
        t["cost_usd"] = round(t["cost_usd"], 4)
        t["wall_hours"] = round(t["wall_hours"], 3)
        out.append(t)
    return out


def write(path: str, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    run_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    write(os.path.join(out_dir, "pilot_validation.csv"), subject_rows(run_dir))
    write(os.path.join(out_dir, "pilot_mining.csv"), mining_rows(run_dir))
    write(os.path.join(out_dir, "pilot_stage_cost.csv"), stage_cost_rows(run_dir))


if __name__ == "__main__":
    main()
