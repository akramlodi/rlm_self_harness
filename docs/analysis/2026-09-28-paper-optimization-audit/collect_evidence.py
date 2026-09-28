"""Recompute the paper audit inventory from saved local artifacts, without model calls."""

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/analysis/2026-09-28-paper-optimization-audit"


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return (
        [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if p.exists()
        else []
    )


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return str(p.relative_to(ROOT))


def score(run):
    v = run.get("verdict") or {}
    q = v.get("quality") or {}
    if q.get("definition_id") == "f1/v1":
        return q.get("value")
    m = re.fullmatch(
        r"precision=(\d\.\d+) recall=(\d\.\d+) f1=(\d\.\d+) missing=(\d+) extra=(\d+)",
        v.get("detail", ""),
    )
    if m:
        return float(m[3])
    if v.get("cause") in {
        "runtime_error",
        "resource_terminated",
        "wrong_format",
        "content_filtered",
    }:
        return 0.0
    return None


experiments = []
for p in sorted(ROOT.glob("experiment_oolong_pairs_dsv4f*")):
    if not p.is_dir():
        continue
    launch = p / ".launch/launch.json"
    cfg = p / ".launch/source/configs/experiment_oolong_pairs_DeepSeekV4Flash.toml"
    entry = {
        "directory": p.name,
        "launch_commit": read(launch).get("base_commit") if launch.exists() else None,
        "config_source": rel(cfg) if cfg.exists() else None,
    }
    if cfg.exists():
        conf = tomllib.loads(cfg.read_text())
        entry["config"] = {k: conf[k] for k in ["splits", "loop", "promotion"]}
        entry["config_sha256"] = sha(cfg)
    rounds = []
    for rp in sorted(p.glob("opt/round_*")):
        if not rp.is_dir():
            continue
        done = rp / "round.json"
        decisions = list(rp.glob("validation/round_*/decision.json"))
        x = {
            "round": int(rp.name.split("_")[-1]),
            "completed": done.exists(),
            "promoted": read(done).get("promoted") if done.exists() else None,
        }
        if decisions:
            dec = read(decisions[0])
            x["protocol"] = dec.get("validation_protocol", "legacy/unmarked")
            x["promoted"] = dec.get("promoted")
            x["constituent_ids"] = dec.get("constituent_ids", [])
            ledger = rows(decisions[0].parent / "promotions.jsonl")
            x["ledger_decisions"] = [
                {k: item.get(k) for k in ["subject_id", "surface", "decision", "batch_subject_id"]}
                for item in ledger
            ]
            x["subjects"] = []
            for sp in sorted(decisions[0].parent.glob("*/summary.json")):
                s = read(sp)
                sx = {
                    k: s.get(k)
                    for k in [
                        "subject_id",
                        "repetitions",
                        "outcome",
                        "harness_hash",
                        "validation_protocol",
                    ]
                }
                sx["splits"] = {}
                for split, ss in s.get("splits", {}).items():
                    ssx = {
                        k: ss.get(k)
                        for k in [
                            "n_instances",
                            "n_runs",
                            "pass_count",
                            "n_resource_terminated",
                            "n_runtime_errors",
                            "total_cost",
                        ]
                    }
                    manifests = sp.parent / ss.get("round_path", f"{split}/round_00") / "runs.jsonl"
                    rs = rows(manifests)
                    vals = [score(r) for r in rs]
                    ssx["quality_known"] = sum(v is not None for v in vals)
                    ssx["quality_mean"] = (
                        sum(vals) / len(vals) if vals and all(v is not None for v in vals) else None
                    )
                    ssx["manifest"] = rel(manifests) if manifests.exists() else None
                    sx["splits"][split] = ssx
                x["subjects"].append(sx)
        rounds.append(x)
    entry["rounds"] = rounds
    entry["completed_rounds"] = sum(r["completed"] for r in rounds)
    entry["promoted_batches"] = sum(r["completed"] and bool(r["promoted"]) for r in rounds)
    experiments.append(entry)
latest = ROOT / "experiment_oolong_pairs_dsv4f_20260924_161551_3rounds"
splits = {}
for p in sorted((latest / "splits").glob("*.jsonl")):
    rs = rows(p)
    groups = {}
    for r in rs:
        start = r["prompt"].rfind(r["question"])
        assert start >= 0
        corpus = r["prompt"][:start]
        key = hashlib.sha256(corpus.encode()).hexdigest()
        groups.setdefault(
            key,
            {
                "window_id": r["context_window_id"],
                "tasks": [],
                "characters": len(corpus),
                "num_entries": r["num_entries"],
            },
        )["tasks"].append(r["task_id"])
    splits[p.stem] = {
        "n": len(rs),
        "unique_ids": len({r["id"] for r in rs}),
        "sha256": sha(p),
        "corpora": groups,
    }
tex = []
for p in sorted((ROOT / "paper").glob("*.tex")):
    text = p.read_text()
    active = "\n".join(re.split(r"(?<!\\)%", line)[0] for line in text.splitlines())
    includes = re.findall(r"\\(?:input|include)\{([^}]+)\}", active)
    tex.append(
        {
            "file": rel(p),
            "lines": len(text.splitlines()),
            "sha256": sha(p),
            "includes": includes,
            "missing_includes": [
                i
                for i in includes
                if not (p.parent / (i if i.endswith(".tex") else i + ".tex")).exists()
            ],
            "todo_answers": active.count("\\answerTODO{}"),
        }
    )
long_evaluation = {}
long_root = ROOT / "experiment_oolong_pairs_dsv4f_20260916_131837/eval"
long_paths = {
    "initial": long_root / "initial/oolong_pairs_long/round_00/runs.jsonl",
    "edited_partial": long_root
    / "paired_20260923T140326Z/sh_rlm/oolong_pairs_long/round_00/runs.jsonl",
}
long_rows = {name: rows(path) for name, path in long_paths.items()}
matched_keys = {(run["instance_id"], run["attempt"]) for run in long_rows["edited_partial"]}
long_rows["initial_matched"] = [
    run for run in long_rows["initial"] if (run["instance_id"], run["attempt"]) in matched_keys
]
for name, runs in long_rows.items():
    values = [score(run) for run in runs]
    causes = {}
    for run in runs:
        cause = run.get("cause")
        causes[cause] = causes.get(cause, 0) + 1
    source = long_paths.get(name, long_paths["initial"])
    long_evaluation[name] = {
        "source": rel(source),
        "source_sha256": sha(source),
        "completed_attempts": len(runs),
        "unique_tasks": len({run["instance_id"] for run in runs}),
        "exact_passes": sum(bool(run["passed"]) for run in runs),
        "causes": causes,
        "quality_known": sum(value is not None for value in values),
        "quality_mean": sum(values) / len(values)
        if values and all(value is not None for value in values)
        else None,
    }

result = {
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "date": "2026-09-28",
    "method": "Read-only manifest, configuration, verdict and LaTeX inventory; F1 uses recorded f1/v1 or exact legacy grammar plus declared terminal zeros; no model calls. Context hash is prefix before the exact task question.",
    "tex": tex,
    "latest_split_corpora": splits,
    "experiments": experiments,
    "long_evaluation": long_evaluation,
}
(OUT / "evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("Evidence:", rel(OUT / "evidence.json"))
for e in experiments:
    c = e.get("config", {})
    print(
        e["directory"],
        e["completed_rounds"],
        "rounds",
        e["promoted_batches"],
        "promotions",
        c.get("loop", {}),
        "protocols",
        sorted({r.get("protocol", "none") for r in e["rounds"]}),
    )
for k, v in splits.items():
    print(k, v["n"], [(c["window_id"], len(c["tasks"]), h[:12]) for h, c in v["corpora"].items()])
for r in experiments[-1]["rounds"]:
    print("latest", r["round"], [(s["subject_id"], s["splits"]) for s in r.get("subjects", [])])
