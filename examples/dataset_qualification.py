"""TOML-driven matched baseline qualification across long-context datasets.

This is a dataset screen, not a Self-Harness optimization run. It executes the
fixed H0, H0*, and pinned lambda-RLM methods on matched instances within each
dataset, then reports quality, score coverage, failures, cost, calls, and
recursion. Dataset-specific metrics remain separate: OBLIQ uses NDCG@10 and
OOLONG-Pairs uses pair-set F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from rlm.core.types import RLMChatCompletion
from shrlm.baselines.lambda_rlm import LambdaBaselineConfig
from shrlm.baselines.lambda_runner import LambdaRoundConfig, run_governed_lambda_round
from shrlm.environments.obliq_bench_math import (
    ObliqBenchMathVerifier,
    load_obliq_bench_math,
    recorded_ndcg,
)
from shrlm.environments.oolong_pairs import (
    OolongPairsVerifier,
    load_oolong_pairs,
    recorded_pair_metrics,
)
from shrlm.experiment.config import load_config, round_config_kwargs
from shrlm.optimization.costs import CandidateSpendBreaker, ValidationCaps
from shrlm.optimization.driver import RoundConfig, run_round
from shrlm.optimization.types import Verdict
from shrlm.optimization.walker import walk
from shrlm.rlm_harness import HARNESSES

DEFAULT_STUDY_CONFIG = Path("configs/dataset_qualification_DeepSeekV4Flash.toml")
METHODS = ("H0", "H0*", "lambda_rlm")
METHOD_DIRS = {"H0": "h0", "H0*": "h0_star", "lambda_rlm": "lambda_rlm"}
TECHNICAL_FAILURES = frozenset(
    {"wrong_format", "resource_terminated", "runtime_error", "content_filtered"}
)


@dataclass(frozen=True)
class StudySettings:
    experiment_config: Path
    methods: tuple[str, ...]
    attempts: int
    seed: int
    max_budget: float
    max_timeout: float


@dataclass(frozen=True)
class ObliqSettings:
    enabled: bool
    query_ids: tuple[str, ...]
    candidate_pool_size: int | None


@dataclass(frozen=True)
class OolongPairsSettings:
    enabled: bool
    n: int
    task_ids: tuple[int, ...]
    context_length: str


@dataclass(frozen=True)
class QualificationThresholds:
    min_metric_coverage: float
    max_technical_failure_rate: float
    min_method_score_spread: float
    min_best_method_score: float
    max_best_method_score: float


@dataclass(frozen=True)
class QualificationConfig:
    study: StudySettings
    obliq_math: ObliqSettings
    oolong_pairs: OolongPairsSettings
    thresholds: QualificationThresholds


def require_keys(table: dict[str, Any], expected: set[str], name: str) -> None:
    """Reject missing or unknown keys in a standalone study table."""
    missing = expected - set(table)
    unknown = set(table) - expected
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


def load_study_config(path: Path) -> QualificationConfig:
    """Load the strict qualification schema without extending ExperimentConfig."""
    raw = tomllib.loads(path.read_text())
    require_keys(raw, {"study", "obliq_math", "oolong_pairs", "thresholds"}, "config")

    study = raw["study"]
    require_keys(
        study,
        {"experiment_config", "methods", "attempts", "seed", "max_budget", "max_timeout"},
        "study",
    )
    methods = tuple(str(method) for method in study["methods"])
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("study.methods must contain unique method names")
    unknown_methods = set(methods) - set(METHODS)
    if unknown_methods:
        raise ValueError(f"study.methods contains unknown methods: {sorted(unknown_methods)}")

    attempts = int(study["attempts"])
    max_budget = float(study["max_budget"])
    max_timeout = float(study["max_timeout"])
    if attempts < 1 or max_budget <= 0 or max_timeout <= 0:
        raise ValueError("study attempts, max_budget, and max_timeout must be positive")

    obliq = raw["obliq_math"]
    require_keys(obliq, {"enabled", "query_ids", "candidate_pool_size"}, "obliq_math")
    query_ids = tuple(str(value) for value in obliq["query_ids"])
    pool = int(obliq["candidate_pool_size"])
    if bool(obliq["enabled"]) and not query_ids:
        raise ValueError("obliq_math.query_ids must not be empty when enabled")
    if pool < 0:
        raise ValueError("obliq_math.candidate_pool_size must be 0 (full) or positive")

    oolong = raw["oolong_pairs"]
    require_keys(oolong, {"enabled", "n", "task_ids", "context_length"}, "oolong_pairs")
    task_ids = tuple(int(value) for value in oolong["task_ids"])
    context_length = str(oolong["context_length"])
    if int(oolong["n"]) < 1 or not task_ids:
        raise ValueError("oolong_pairs.n and task_ids must be non-empty and positive")
    if context_length not in {"short", "long"}:
        raise ValueError("oolong_pairs.context_length must be 'short' or 'long'")

    thresholds = raw["thresholds"]
    threshold_keys = {
        "min_metric_coverage",
        "max_technical_failure_rate",
        "min_method_score_spread",
        "min_best_method_score",
        "max_best_method_score",
    }
    require_keys(thresholds, threshold_keys, "thresholds")
    threshold_values = {key: float(thresholds[key]) for key in threshold_keys}
    if any(not 0 <= value <= 1 for value in threshold_values.values()):
        raise ValueError("all qualification thresholds must be in [0, 1]")
    if threshold_values["min_best_method_score"] > threshold_values["max_best_method_score"]:
        raise ValueError("minimum best-method score exceeds maximum")

    return QualificationConfig(
        study=StudySettings(
            experiment_config=Path(str(study["experiment_config"])),
            methods=methods,
            attempts=attempts,
            seed=int(study["seed"]),
            max_budget=max_budget,
            max_timeout=max_timeout,
        ),
        obliq_math=ObliqSettings(
            enabled=bool(obliq["enabled"]),
            query_ids=query_ids,
            candidate_pool_size=None if pool == 0 else pool,
        ),
        oolong_pairs=OolongPairsSettings(
            enabled=bool(oolong["enabled"]),
            n=int(oolong["n"]),
            task_ids=task_ids,
            context_length=context_length,
        ),
        thresholds=QualificationThresholds(**threshold_values),
    )


def run_method(
    method: str,
    instances: list[dict[str, Any]],
    verifier: Callable[[dict[str, Any], str], Verdict],
    out_dir: Path,
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one fixed method with identical attempts and limits."""
    if method != "lambda_rlm":
        return run_round(
            RoundConfig(
                round_index=0,
                harness=HARNESSES[method],
                instances=instances,
                verifier=verifier,
                out_dir=out_dir,
                **kwargs,
            )
        )

    caps = ValidationCaps(
        max_depth=int(kwargs["max_depth"]),
        max_iterations=int(kwargs["max_iterations"]),
        max_budget=float(kwargs["max_budget"]),
        max_timeout=float(kwargs["max_timeout"]),
        candidate_budget=len(instances) * int(kwargs["attempts"]) * float(kwargs["max_budget"]),
    )
    config = LambdaRoundConfig(
        round_index=0,
        method=LambdaBaselineConfig(),
        instances=instances,
        verifier=verifier,
        out_dir=out_dir,
        backend=kwargs["backend"],
        backend_kwargs=kwargs["backend_kwargs"],
        attempts=int(kwargs["attempts"]),
        max_budget=float(kwargs["max_budget"]),
        max_timeout=float(kwargs["max_timeout"]),
    )
    return run_governed_lambda_round(config, CandidateSpendBreaker(caps)).entries


def trace_diagnostics(round_path: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate model calls and observable recursion from persisted traces."""
    total_calls = 0
    rlm_children = 0
    llm_leaves = 0
    lambda_phases: Counter[str] = Counter()
    filter_decisions: Counter[str] = Counter()
    for entry in entries:
        completion = RLMChatCompletion.from_dict(
            json.loads((round_path / str(entry["trace_path"])).read_text())
        )
        total_calls += completion.usage_summary.total_calls
        metadata = completion.metadata or {}
        audit = metadata.get("lambda_subcall_audit")
        if audit is not None:
            lambda_phases.update(audit.get("phase_counts") or {})
            filter_decisions.update(audit.get("filter_decisions") or {})
            continue
        try:
            _, stats = walk(completion)
        except ValueError:
            continue
        rlm_children += stats.n_rlm_children
        llm_leaves += stats.n_llm_leaves
    return {
        "total_model_calls": total_calls,
        "rlm_children": rlm_children,
        "llm_leaves": llm_leaves,
        "lambda_phase_counts": dict(sorted(lambda_phases.items())),
        "lambda_filter_decisions": dict(sorted(filter_decisions.items())),
    }


def summarize_method(
    dataset: str,
    entries: list[dict[str, Any]],
    round_path: Path,
) -> dict[str, Any]:
    """Summarize one method without treating missing metrics as successful zeros."""
    values: list[float] = []
    causes: Counter[str] = Counter()
    for entry in entries:
        verdict = Verdict.from_dict(entry["verdict"])
        if entry["cause"] is not None:
            causes[str(entry["cause"])] += 1
        if dataset == "obliq_math":
            value = recorded_ndcg(verdict)
        else:
            metrics = recorded_pair_metrics(verdict)
            value = None if metrics is None else float(metrics["f1"])
        if value is not None:
            values.append(value)
    n = len(entries)
    technical_failures = sum(causes[cause] for cause in TECHNICAL_FAILURES)
    return {
        "n_runs": n,
        "passes": sum(bool(entry["passed"]) for entry in entries),
        "metric_coverage": len(values),
        "metric_coverage_rate": len(values) / n if n else 0.0,
        "mean_metric_scored": mean(values) if values else None,
        "mean_metric_all_runs": sum(values) / n if n else None,
        "technical_failures": technical_failures,
        "technical_failure_rate": technical_failures / n if n else 0.0,
        "failure_causes": dict(sorted(causes.items())),
        "total_cost": sum(float(entry["cost"] or 0.0) for entry in entries),
        "wall_seconds": sum(float(entry["execution_time"]) for entry in entries),
        "input_tokens": sum(int(entry["input_tokens"]) for entry in entries),
        "output_tokens": sum(int(entry["output_tokens"]) for entry in entries),
        **trace_diagnostics(round_path, entries),
    }


def qualification_verdict(
    methods: dict[str, dict[str, Any]], thresholds: QualificationThresholds
) -> dict[str, Any]:
    """Apply preregistered dataset-utility gates to aggregate method results."""
    coverages = [float(summary["metric_coverage_rate"]) for summary in methods.values()]
    failure_rates = [float(summary["technical_failure_rate"]) for summary in methods.values()]
    scores = [float(summary["mean_metric_all_runs"] or 0.0) for summary in methods.values()]
    best = max(scores)
    spread = max(scores) - min(scores)
    checks = {
        "metric_coverage": min(coverages) >= thresholds.min_metric_coverage,
        "technical_reliability": max(failure_rates) <= thresholds.max_technical_failure_rate,
        "method_separation": spread >= thresholds.min_method_score_spread,
        "not_floor": best >= thresholds.min_best_method_score,
        "not_ceiling": best <= thresholds.max_best_method_score,
    }
    return {
        "qualified": all(checks.values()),
        "checks": checks,
        "observed": {
            "minimum_coverage": min(coverages),
            "maximum_technical_failure_rate": max(failure_rates),
            "best_method_score": best,
            "method_score_spread": spread,
        },
    }


def prepare_study_directory(out_dir: Path, config: QualificationConfig) -> None:
    """Persist study identity and refuse to mix configurations on resume."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    experiment_path = config.study.experiment_config
    payload["study"]["experiment_config_sha256"] = hashlib.sha256(
        experiment_path.read_bytes()
    ).hexdigest()
    expected = json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n"
    path = out_dir / "study.json"
    if path.exists() and path.read_text() != expected:
        raise ValueError(f"{path} belongs to a different qualification configuration")
    path.write_text(expected)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=Path("./dataset_qualification_dsv4f_v1"))
    parser.add_argument("--live", action="store_true", help="load datasets and spend money")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_study_config(args.config)
    study = config.study
    enabled_counts = {
        "obliq_math": len(config.obliq_math.query_ids) if config.obliq_math.enabled else 0,
        "oolong_pairs": config.oolong_pairs.n if config.oolong_pairs.enabled else 0,
    }
    total_instances = sum(enabled_counts.values())
    worst_case = total_instances * len(study.methods) * study.attempts * study.max_budget
    print("=== Dataset qualification study ===")
    print(f"methods={study.methods} attempts={study.attempts} seed={study.seed}")
    print(f"instances={enabled_counts}")
    print(f"per-run caps: ${study.max_budget:.2f}, {study.max_timeout:.0f}s")
    print(f"worst-case spend: ${worst_case:.2f}")
    print(f"out_dir={args.out_dir.resolve()}")
    if not args.live:
        print("--live not passed: preflight complete; nothing loaded and nothing spent.")
        return 0

    load_dotenv()
    prepare_study_directory(args.out_dir, config)
    experiment = load_config("full", path=study.experiment_config)
    kwargs = round_config_kwargs(experiment)
    kwargs.update(
        attempts=study.attempts,
        max_budget=study.max_budget,
        max_timeout=study.max_timeout,
    )

    datasets: list[
        tuple[str, str, list[dict[str, Any]], Callable[[dict[str, Any], str], Verdict]]
    ] = []
    if config.obliq_math.enabled:
        settings = config.obliq_math
        instances = load_obliq_bench_math(
            n=len(settings.query_ids),
            seed=study.seed,
            query_ids=settings.query_ids,
            candidate_pool_size=settings.candidate_pool_size,
        )
        datasets.append(("obliq_math", "ndcg_at_10", instances, ObliqBenchMathVerifier()))
    if config.oolong_pairs.enabled:
        settings = config.oolong_pairs
        env = experiment.environments.oolong_pairs
        context_length = (
            env.context_length_short
            if settings.context_length == "short"
            else env.context_length_long
        )
        instances = load_oolong_pairs(
            task_ids=settings.task_ids,
            context_lengths=(context_length,),
            n=settings.n,
            seed=study.seed,
            max_scan=env.max_scan,
            split="validation",
            revision=env.dataset_revision,
        )
        datasets.append(("oolong_pairs", "f1", instances, OolongPairsVerifier()))

    report: dict[str, Any] = {
        "format": "shrlm-dataset-qualification/v1",
        "study": {
            "methods": list(study.methods),
            "attempts": study.attempts,
            "seed": study.seed,
            "max_budget": study.max_budget,
            "max_timeout": study.max_timeout,
            "worst_case_spend": worst_case,
        },
        "thresholds": asdict(config.thresholds),
        "datasets": {},
    }
    for dataset, metric, instances, verifier in datasets:
        dataset_dir = args.out_dir / dataset
        method_summaries: dict[str, dict[str, Any]] = {}
        print(f"\n=== {dataset}: {len(instances)} instances, metric={metric} ===")
        for method in study.methods:
            method_dir = dataset_dir / METHOD_DIRS[method]
            print(f"Running {method}...")
            entries = run_method(method, instances, verifier, method_dir, kwargs)
            summary = summarize_method(dataset, entries, method_dir / "round_00")
            method_summaries[method] = summary
            print(
                f"{method}: mean_all={summary['mean_metric_all_runs']:.3f} "
                f"coverage={summary['metric_coverage']}/{summary['n_runs']} "
                f"cost=${summary['total_cost']:.4f}"
            )
        report["datasets"][dataset] = {
            "metric": metric,
            "n_instances": len(instances),
            "instance_ids": [str(instance["id"]) for instance in instances],
            "methods": method_summaries,
            "qualification": qualification_verdict(method_summaries, config.thresholds),
        }
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "summary.json").write_text(
            json.dumps(report["datasets"][dataset], indent=2, sort_keys=True) + "\n"
        )

    report_path = args.out_dir / "dataset_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nReport persisted at: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
