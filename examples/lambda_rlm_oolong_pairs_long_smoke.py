"""Minimal live smoke test for fixed baselines on OOLONG-Pairs.

Narrower than ``examples/experiment_smoke.py``: it exercises exactly one
condition x environment slice -- ``PaperLambdaRLM`` (``shrlm/baselines/
paper_lambda_rlm.py``) against real 262,144-token OOLONG-Pairs windows from
the pinned upstream dataset -- instead of the full mining/validation/
evaluation pipeline across every baseline and both environments. Use this to
check that the SPLIT -> MAP(classify) -> PARSE -> FILTER -> CROSS path
actually runs end to end against a live model before trusting it inside a
full experiment run.

It reuses the real production wiring -- ``round_config_kwargs`` for the
backend/model/sampling args from ``configs/experiment.toml``, and
``run_lambda_round``/``LambdaRoundConfig`` (the exact code path
``shrlm/experiment/evaluation.py`` uses for the λ-RLM condition) for
execution, persistence, and per-run budget/timeout enforcement -- so a green
run here is a faithful signal, not a simulation.

Cost: bounded at ``n x conditions x caps.max_budget`` (shipped config:
$0.50/run), and the shipped Qwen3-30B-A3B-Instruct-2507 pricing puts a real
262k-token pairwise run far under that (classification batches cost
proportionally to input tokens; output per batch is a few short lines, not a
full completion).

Usage:
    # pre-flight only: prints the config identity and instance count, spends
    # nothing.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py

    # the real run.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live

    # more coverage: two long instances at two different task ids.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live \\
        --n 2 --task-ids 1,11

    # matched author-style H0* versus λ-RLM comparison, with $4 in configured caps.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live --compare-h0-star \\
        --n 4 --task-ids 1,6,11,16 --out-dir ./lambda_vs_h0_star_long_smoke

    # Add --compare-b1 to include the sparse H0 starting harness as a third condition.

    # Isolate H0* on one short instance before attempting the long comparison.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live \\
        --conditions h0_star --context-length short --n 1 --task-ids 1 \\
        --out-dir ./h0_star_oolong_short_sanity
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv

from shrlm.baselines.lambda_rlm import LambdaBaselineConfig
from shrlm.baselines.lambda_runner import (
    LambdaRoundConfig,
    require_lambda_backend_credential,
    run_lambda_round,
)
from shrlm.environments.oolong_pairs import (
    OolongPairsVerifier,
    load_oolong_pairs,
    recorded_pair_metrics,
)
from shrlm.experiment.config import load_config, round_config_kwargs
from shrlm.optimization.driver import RoundConfig, run_round
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import HARNESSES

DEFAULT_TASK_IDS = (1,)

B1_CONDITION = "b1"
H0_STAR_CONDITION = "h0_star"
LAMBDA_CONDITION = "lambda_rlm"
COMPARISON_FILENAME = "comparison.json"
REFERENCE_HARNESSES = {
    B1_CONDITION: "H0",
    H0_STAR_CONDITION: "H0*",
}
KNOWN_CONDITIONS = (B1_CONDITION, H0_STAR_CONDITION, LAMBDA_CONDITION)
CONTEXT_LENGTH_CHOICES = ("short", "long")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="actually spend money")
    parser.add_argument(
        "--compare-b1",
        action="store_true",
        help="run the sparse B1/H0 harness on the exact same instances before λ-RLM",
    )
    parser.add_argument(
        "--compare-h0-star",
        action="store_true",
        help="run the upstream author-style H0* harness on the same instances before λ-RLM",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="comma-separated conditions: b1,h0_star,lambda_rlm",
    )
    parser.add_argument(
        "--context-length",
        choices=CONTEXT_LENGTH_CHOICES,
        default="long",
        help="use the short or long OOLONG-Pairs context (default: long)",
    )
    parser.add_argument("--n", type=int, default=1, help="number of instances (default 1)")
    parser.add_argument(
        "--task-ids",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TASK_IDS),
        help="comma-separated OOLONG-Pairs task ids to sample from (default: 1)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./lambda_rlm_oolong_pairs_long_smoke"),
        help="where to persist the round (resumable, like a real evaluation round)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def selected_conditions(compare_b1: bool, compare_h0_star: bool) -> tuple[str, ...]:
    """Return reference conditions first and λ-RLM last."""
    references = []
    if compare_b1:
        references.append(B1_CONDITION)
    if compare_h0_star:
        references.append(H0_STAR_CONDITION)
    return (*references, LAMBDA_CONDITION)


def resolve_conditions(
    value: str | None,
    compare_b1: bool,
    compare_h0_star: bool,
) -> tuple[str, ...]:
    if value is None:
        return selected_conditions(compare_b1, compare_h0_star)

    if compare_b1 or compare_h0_star:
        raise ValueError("--conditions cannot be combined with --compare-b1 or --compare-h0-star")

    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested) - set(KNOWN_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")

    if not requested:
        raise ValueError("--conditions must select at least one condition")
    if len(set(requested)) != len(requested):
        raise ValueError("--conditions must not contain duplicates")

    return tuple(condition for condition in KNOWN_CONDITIONS if condition in requested)


def worst_case_spend(n: int, max_budget: float | None, conditions: tuple[str, ...]) -> float:
    """Return the sum of configured per-run caps for the selected conditions."""
    return n * len(conditions) * (max_budget or 0.0)


def summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate persisted outcomes without rescoring model responses."""
    recorded = []
    for entry in entries:
        metrics = recorded_pair_metrics(Verdict.from_dict(entry["verdict"]))
        if metrics is not None:
            recorded.append(metrics)

    f1_values = [float(metrics["f1"]) for metrics in recorded]
    return {
        "n_runs": len(entries),
        "exact_passes": sum(1 for entry in entries if entry["passed"]),
        "metric_coverage": len(recorded),
        "mean_precision": (
            mean(float(metrics["precision"]) for metrics in recorded) if recorded else None
        ),
        "mean_recall": (
            mean(float(metrics["recall"]) for metrics in recorded) if recorded else None
        ),
        "mean_f1": mean(f1_values) if f1_values else None,
        "median_f1": median(f1_values) if f1_values else None,
        "total_cost": sum(float(entry["cost"] or 0.0) for entry in entries),
        "input_tokens": sum(int(entry["input_tokens"]) for entry in entries),
        "output_tokens": sum(int(entry["output_tokens"]) for entry in entries),
        "wall_seconds": sum(float(entry["execution_time"]) for entry in entries),
    }


def comparison_payload(
    entries_by_condition: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build aggregate and paired diagnostics from matched persisted manifests."""
    ids_by_condition = {
        condition: [str(entry["instance_id"]) for entry in entries]
        for condition, entries in entries_by_condition.items()
    }
    expected_ids = next(iter(ids_by_condition.values()))
    if any(ids != expected_ids for ids in ids_by_condition.values()):
        raise ValueError(
            "reference and λ-RLM manifests are not aligned by instance; "
            "refusing an unmatched comparison"
        )

    per_instance = []
    for index, instance_id in enumerate(expected_ids):
        row: dict[str, Any] = {"instance_id": instance_id}
        for condition, entries in entries_by_condition.items():
            entry = entries[index]
            metrics = recorded_pair_metrics(Verdict.from_dict(entry["verdict"]))
            row[condition] = {
                "passed": bool(entry["passed"]),
                "cause": entry["cause"],
                "f1": None if metrics is None else metrics["f1"],
                "cost": entry["cost"],
            }
        per_instance.append(row)

    return {
        "format": "shrlm-lambda-reference-smoke-comparison/v1",
        "conditions": {
            condition: summarize_entries(entries)
            for condition, entries in entries_by_condition.items()
        },
        "per_instance": per_instance,
    }


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_comparison(payload: dict[str, Any]) -> None:
    """Print the compact paired report; the JSON artifact retains all fields."""
    print("\n=== Matched comparison ===")
    reference_conditions = [
        condition for condition in payload["conditions"] if condition != LAMBDA_CONDITION
    ]
    paired_references = reference_conditions if LAMBDA_CONDITION in payload["conditions"] else ()
    for reference in paired_references:
        print(f"\n{reference} versus {LAMBDA_CONDITION}")
        print(f"{'instance':42} {f'{reference} F1':>10} {'λ-RLM F1':>10} {'delta':>8}")
        for row in payload["per_instance"]:
            reference_f1 = row[reference]["f1"]
            lambda_f1 = row[LAMBDA_CONDITION]["f1"]
            delta = None if reference_f1 is None or lambda_f1 is None else lambda_f1 - reference_f1
            print(
                f"{row['instance_id'][:42]:42} {format_metric(reference_f1):>10} "
                f"{format_metric(lambda_f1):>10} {format_metric(delta):>8}"
            )

    for condition in payload["conditions"]:
        summary = payload["conditions"][condition]
        print(
            f"{condition}: exact={summary['exact_passes']}/{summary['n_runs']} "
            f"mean_f1={format_metric(summary['mean_f1'])} "
            f"median_f1={format_metric(summary['median_f1'])} "
            f"metrics={summary['metric_coverage']}/{summary['n_runs']} "
            f"cost=${summary['total_cost']:.4f} wall={summary['wall_seconds']:.1f}s"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_ids = tuple(int(t) for t in args.task_ids.split(","))
    if args.n < 1:
        raise ValueError(f"--n must be >= 1, got {args.n}")

    load_dotenv()
    config = load_config("full")
    oolong_cfg = config.environments.oolong_pairs
    context_length = (
        oolong_cfg.context_length_short
        if args.context_length == "short"
        else oolong_cfg.context_length_long
    )
    kwargs = round_config_kwargs(config)

    round_config = LambdaRoundConfig(
        round_index=0,
        instances=[],  # filled in below, after the pre-flight prints
        verifier=OolongPairsVerifier(),
        out_dir=args.out_dir,
        method=LambdaBaselineConfig(),
        backend=kwargs["backend"],
        backend_kwargs=kwargs["backend_kwargs"],
        attempts=1,
        max_budget=kwargs["max_budget"],
        max_timeout=kwargs["max_timeout"],
    )

    conditions = resolve_conditions(
        args.conditions,
        args.compare_b1,
        args.compare_h0_star,
    )
    print("=== OOLONG-Pairs baseline smoke ===")
    print(f"backend={round_config.backend} model={round_config.backend_kwargs.get('model_name')}")
    print(f"conditions={conditions}")
    print(f"task_ids={task_ids} n={args.n} context_length={args.context_length} ({context_length})")
    print(
        f"per-run cap: max_budget=${round_config.max_budget} max_timeout={round_config.max_timeout}s"
    )
    print(f"worst-case spend: ${worst_case_spend(args.n, round_config.max_budget, conditions):.2f}")
    print(f"out_dir={args.out_dir.resolve()}")

    if not args.live:
        print("\n--live not passed: nothing loaded, nothing spent. Re-run with --live to execute.")
        return 0

    require_lambda_backend_credential(round_config)

    print(f"\nStreaming real OOLONG-Pairs {args.context_length} window(s) from the dataset...")
    instances = load_oolong_pairs(
        task_ids=task_ids,
        context_lengths=(context_length,),
        n=args.n,
        seed=args.seed,
        max_scan=oolong_cfg.max_scan,
        split="validation",
        revision=oolong_cfg.dataset_revision,
    )
    round_config = replace(round_config, instances=instances)
    print(f"loaded {len(instances)} instance(s): " + ", ".join(i["id"] for i in instances))

    entries_by_condition: dict[str, list[dict[str, Any]]] = {}
    multiple_conditions = len(conditions) > 1
    reference_conditions = tuple(
        condition for condition in conditions if condition in REFERENCE_HARNESSES
    )
    for condition in reference_conditions:
        harness_name = REFERENCE_HARNESSES[condition]
        print(f"\nRunning {condition}/{harness_name}...")
        condition_out_dir = args.out_dir / condition if multiple_conditions else args.out_dir
        reference_config = RoundConfig(
            round_index=0,
            harness=HARNESSES[harness_name],
            instances=instances,
            verifier=OolongPairsVerifier(),
            out_dir=condition_out_dir,
            **kwargs,
        )
        entries_by_condition[condition] = run_round(reference_config)

    if LAMBDA_CONDITION in conditions:
        if multiple_conditions:
            round_config = replace(round_config, out_dir=args.out_dir / LAMBDA_CONDITION)
        print("\nRunning λ-RLM...")
        entries_by_condition[LAMBDA_CONDITION] = run_lambda_round(round_config)

    print("\n=== Results ===")
    total_cost = 0.0
    for condition, entries in entries_by_condition.items():
        print(f"\n{condition}:")
        condition_cost = 0.0
        for entry in entries:
            cost = float(entry["cost"] or 0.0)
            condition_cost += cost
            print(
                f"{entry['instance_id']}: passed={entry['passed']} cause={entry['cause']} "
                f"cost=${cost:.4f} detail={entry['verdict'].get('detail')}"
            )
        total_cost += condition_cost
        condition_out_dir = args.out_dir / condition if multiple_conditions else args.out_dir
        print(f"cost=${condition_cost:.4f}")
        print(f"traces={(condition_out_dir / 'round_00').resolve()}")
    print(f"\ntotal cost: ${total_cost:.4f}")
    print(
        json.dumps(
            {"n": len(instances), "conditions": conditions, "total_cost": total_cost}, indent=2
        )
    )

    if multiple_conditions:
        payload = comparison_payload(entries_by_condition)
        comparison_path = args.out_dir / COMPARISON_FILENAME
        comparison_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print_comparison(payload)
        print(f"comparison persisted at: {comparison_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
