"""Minimal live smoke test for a baseline method on OBLIQ-Bench Math.

Narrower than a full experiment run: it exercises one or more baseline
methods (default ``H0*``, the RLM authors' own hand-written system prompt
plus ``ORCHESTRATOR_ADDENDUM``, byte-identical and unmodified since upstream
PR #165 -- see ``shrlm/rlm_harness.py``) against a real sample of OBLIQ-Bench
Math (https://huggingface.co/datasets/dianetc/OBLIQ-Bench, ``analogues/math``
subset) queries. Harness methods use the production
``RoundConfig``/``run_round`` wiring; λ-RLM uses
``LambdaRoundConfig``/``run_governed_lambda_round`` with the same instances
and verifier. No self-harness optimization loop
(mining/proposal/promotion) is involved.

``--method H0*R`` is also available: a locally-authored variant of ``H0*``
that makes ``rlm_query`` legible (a live OOLONG mining run on plain ``H0*``
issued zero ``rlm_query`` calls). It is NOT a peer reference baseline in this
repo -- it's absent from ``shrlm/baselines/README.md`` and from the λ-RLM
smoke script's own ``REFERENCE_HARNESSES`` -- so it defaults off here; use it
explicitly, not as a silent substitute for the real reference.

By default each instance's prompt carries the ENTIRE ~277k-token math corpus
(minus that query's excluded ids); use ``--candidate-pool-size`` for a much
cheaper sanity pass first (gold ids plus that many random negatives).

Usage:
    # pre-flight only: prints the config identity and instance count, spends
    # nothing.
    uv run python examples/obliq_bench_math_smoke.py --n 3

    # cheap sanity pass: one query, a 200-doc pool instead of the full corpus.
    uv run python examples/obliq_bench_math_smoke.py --live --n 1 \\
        --candidate-pool-size 200 --out-dir ./obliq_math_pool_smoke

    # the real small sample, full corpus.
    uv run python examples/obliq_bench_math_smoke.py --live --n 3 \\
        --out-dir ./obliq_math_smoke

    # generic λ-RLM combinator pipeline on one cheap matched instance.
    uv run python examples/obliq_bench_math_smoke.py --live \\
        --method lambda_rlm --query-ids q01522 --candidate-pool-size 200 \\
        --config configs/experiment.toml \\
        --max-budget 0.10 --max-timeout 900 \\
        --out-dir ./experiment_obliq_math_lambda_sanity

    # matched comparison: load one instance set, then run every selected method.
    uv run python examples/obliq_bench_math_smoke.py --live \\
        --methods 'H0,H0*,lambda_rlm' --n 10 --candidate-pool-size 200 \\
        --max-budget 0.10 --max-timeout 900 \\
        --out-dir ./experiment_obliq_math_matched
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv

from shrlm.baselines.lambda_rlm import LambdaBaselineConfig
from shrlm.baselines.lambda_runner import LambdaRoundConfig, run_governed_lambda_round
from shrlm.environments.obliq_bench_math import (
    ObliqBenchMathVerifier,
    load_obliq_bench_math,
    recorded_ndcg,
)
from shrlm.experiment.config import load_config, round_config_kwargs
from shrlm.optimization.costs import CandidateSpendBreaker, ValidationCaps
from shrlm.optimization.driver import RoundConfig, run_round
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import HARNESSES

DEFAULT_CONFIG = Path("configs/experiment_obliq_bench_math_DeepSeekV4Flash.toml")
HARNESS_CHOICES = ("H0", "H0*", "H0*R")
LAMBDA_METHOD = "lambda_rlm"
METHOD_CHOICES = (*HARNESS_CHOICES, LAMBDA_METHOD)
METHOD_OUT_DIRS = {
    "H0": "h0",
    "H0*": "h0_star",
    "H0*R": "h0_star_r",
    LAMBDA_METHOD: LAMBDA_METHOD,
}
COMPARISON_FILENAME = "comparison.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="actually spend money")
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default=None,
        help="baseline method to execute (default: H0*)",
    )
    method_group.add_argument(
        "--harness",
        choices=HARNESS_CHOICES,
        default=None,
        help="backward-compatible alias for selecting an RLM harness",
    )
    method_group.add_argument(
        "--methods",
        type=str,
        default=None,
        help="comma-separated matched methods: H0,H0*,H0*R,lambda_rlm",
    )
    parser.add_argument("--n", type=int, default=3, help="number of queries to sample")
    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="comma-separated OBLIQ-Bench Math query ids, e.g. q00816,q02193 (overrides --n)",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="cap each instance's corpus pool to this many docs (gold + random negatives); "
        "default is the full ~3,508-doc corpus",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./obliq_bench_math_smoke"),
        help="where to persist the round (resumable, like a real evaluation round)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-budget", type=float, default=None, help="override caps.max_budget")
    parser.add_argument("--max-timeout", type=float, default=None, help="override caps.max_timeout")
    return parser.parse_args(argv)


def selected_method(args: argparse.Namespace) -> str:
    """Resolve the method selector and the backward-compatible harness alias."""
    return str(args.method or args.harness or "H0*")


def selected_methods(args: argparse.Namespace) -> tuple[str, ...]:
    """Resolve one legacy selector or a validated matched method list."""
    if args.methods is None:
        return (selected_method(args),)

    requested = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    unknown = set(requested) - set(METHOD_CHOICES)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not requested:
        raise ValueError("--methods must select at least one method")
    if len(set(requested)) != len(requested):
        raise ValueError("--methods must not contain duplicates")
    return tuple(method for method in METHOD_CHOICES if method in requested)


def summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate persisted outcomes without rescoring model responses."""
    ndcg_values = [
        ndcg
        for entry in entries
        if (ndcg := recorded_ndcg(Verdict.from_dict(entry["verdict"]))) is not None
    ]
    return {
        "n_runs": len(entries),
        "passes": sum(1 for entry in entries if entry["passed"]),
        "ndcg_coverage": len(ndcg_values),
        "mean_ndcg_at_10": mean(ndcg_values) if ndcg_values else None,
        "mean_ndcg_at_10_all_runs": sum(ndcg_values) / len(entries) if entries else None,
        "median_ndcg_at_10": median(ndcg_values) if ndcg_values else None,
        "total_cost": sum(float(entry["cost"] or 0.0) for entry in entries),
        "input_tokens": sum(int(entry["input_tokens"]) for entry in entries),
        "output_tokens": sum(int(entry["output_tokens"]) for entry in entries),
        "wall_seconds": sum(float(entry["execution_time"]) for entry in entries),
    }


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def comparison_payload(
    entries_by_method: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build aggregate and paired diagnostics from matched manifests."""
    ids_by_method = {
        method: [str(entry["instance_id"]) for entry in entries]
        for method, entries in entries_by_method.items()
    }
    expected_ids = next(iter(ids_by_method.values()))
    if any(ids != expected_ids for ids in ids_by_method.values()):
        raise ValueError("method manifests are not aligned; refusing an unmatched comparison")

    per_instance = []
    for index, instance_id in enumerate(expected_ids):
        row: dict[str, Any] = {"instance_id": instance_id}
        for method, entries in entries_by_method.items():
            entry = entries[index]
            ndcg = recorded_ndcg(Verdict.from_dict(entry["verdict"]))
            row[method] = {
                "passed": bool(entry["passed"]),
                "cause": entry["cause"],
                "ndcg_at_10": ndcg,
                "cost": entry["cost"],
                "wall_seconds": entry["execution_time"],
            }
        per_instance.append(row)

    return {
        "format": "shrlm-obliq-math-matched-comparison/v1",
        "methods": {
            method: summarize_entries(entries) for method, entries in entries_by_method.items()
        },
        "per_instance": per_instance,
    }


def print_comparison(payload: dict[str, Any]) -> None:
    """Print a compact paired report; the JSON artifact retains all fields."""
    methods = tuple(payload["methods"])
    print("\n=== Matched comparison ===")
    print(f"{'instance':12} " + " ".join(f"{method + ' NDCG':>14}" for method in methods))
    for row in payload["per_instance"]:
        metrics = " ".join(f"{format_metric(row[method]['ndcg_at_10']):>14}" for method in methods)
        print(f"{row['instance_id'][:12]:12} {metrics}")

    for method, summary in payload["methods"].items():
        print(
            f"{method}: passes={summary['passes']}/{summary['n_runs']} "
            f"mean_ndcg_all={format_metric(summary['mean_ndcg_at_10_all_runs'])} "
            f"mean_ndcg_scored={format_metric(summary['mean_ndcg_at_10'])} "
            f"coverage={summary['ndcg_coverage']}/{summary['n_runs']} "
            f"cost=${summary['total_cost']:.4f} wall={summary['wall_seconds']:.1f}s"
        )


def run_method(
    method: str,
    instances: list[dict[str, Any]],
    out_dir: Path,
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one method through its production adapter on matched instances."""
    verifier = ObliqBenchMathVerifier()
    if method != LAMBDA_METHOD:
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

    max_budget = kwargs["max_budget"]
    max_timeout = kwargs["max_timeout"]
    if max_budget is None or max_timeout is None:
        raise ValueError("lambda_rlm requires finite max_budget and max_timeout caps")
    caps = ValidationCaps(
        max_depth=int(kwargs["max_depth"]),
        max_iterations=int(kwargs["max_iterations"]),
        max_budget=float(max_budget),
        max_timeout=float(max_timeout),
        candidate_budget=len(instances) * float(max_budget),
    )
    lambda_config = LambdaRoundConfig(
        round_index=0,
        method=LambdaBaselineConfig(),
        instances=instances,
        verifier=verifier,
        out_dir=out_dir,
        backend=kwargs["backend"],
        backend_kwargs=kwargs["backend_kwargs"],
        attempts=kwargs["attempts"],
        max_budget=max_budget,
        max_timeout=max_timeout,
    )
    return run_governed_lambda_round(
        lambda_config,
        CandidateSpendBreaker(caps),
    ).entries


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    methods = selected_methods(args)
    if args.n < 1:
        raise ValueError(f"--n must be >= 1, got {args.n}")
    query_ids = (
        None
        if args.query_ids is None
        else tuple(part.strip() for part in args.query_ids.split(",") if part.strip())
    )

    load_dotenv()
    config = load_config("full", path=args.config)
    kwargs = round_config_kwargs(config)
    if args.max_budget is not None:
        kwargs["max_budget"] = args.max_budget
    if args.max_timeout is not None:
        kwargs["max_timeout"] = args.max_timeout
    n_instances = args.n if query_ids is None else len(query_ids)

    print("=== OBLIQ-Bench Math baseline smoke ===")
    print(
        f"methods={methods} backend={kwargs['backend']} "
        f"model={kwargs['backend_kwargs'].get('model_name')}"
    )
    print(
        f"n={n_instances} query_ids={query_ids} "
        f"candidate_pool_size={args.candidate_pool_size or 'full corpus'} "
        f"seed={args.seed}"
    )
    print(f"per-run cap: max_budget=${kwargs['max_budget']} max_timeout={kwargs['max_timeout']}s")
    worst_case_spend = n_instances * len(methods) * (kwargs["max_budget"] or 0.0)
    print(f"worst-case spend: ${worst_case_spend:.2f}")
    print(f"out_dir={args.out_dir.resolve()}")

    if not args.live:
        print("\n--live not passed: nothing loaded, nothing spent. Re-run with --live to execute.")
        return 0

    print("\nDownloading OBLIQ-Bench Math instance(s) from Hugging Face...")
    instances = load_obliq_bench_math(
        n=args.n,
        seed=args.seed,
        query_ids=query_ids,
        candidate_pool_size=args.candidate_pool_size,
    )
    print(
        f"loaded {len(instances)} instance(s): "
        + ", ".join(
            f"{i['id']} (pool={i['pool_size']}, gold={len(i['gold_relevant_ids'])})"
            for i in instances
        )
    )

    multiple_methods = len(methods) > 1
    entries_by_method: dict[str, list[dict[str, Any]]] = {}
    print("\n=== Results ===")
    for method in methods:
        method_out_dir = (
            args.out_dir / METHOD_OUT_DIRS[method] if multiple_methods else args.out_dir
        )
        print(f"\nRunning {method}...")
        entries = run_method(method, instances, method_out_dir, kwargs)
        entries_by_method[method] = entries
        for entry in entries:
            print(
                f"{entry['instance_id']}: passed={entry['passed']} cause={entry['cause']} "
                f"cost=${float(entry['cost'] or 0.0):.4f} "
                f"detail={entry['verdict'].get('detail')}"
            )
        summary = summarize_entries(entries)
        print(
            f"passes={summary['passes']}/{summary['n_runs']} "
            f"mean_ndcg@10={format_metric(summary['mean_ndcg_at_10'])} "
            f"mean_all_runs={format_metric(summary['mean_ndcg_at_10_all_runs'])} "
            f"median_ndcg@10={format_metric(summary['median_ndcg_at_10'])} "
            f"ndcg_coverage={summary['ndcg_coverage']}/{summary['n_runs']} "
            f"cost=${summary['total_cost']:.4f} wall={summary['wall_seconds']:.1f}s"
        )
        print(f"traces={(method_out_dir / 'round_00').resolve()}")
        summary_path = method_out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"summary persisted at: {summary_path.resolve()}")

    if multiple_methods:
        payload = comparison_payload(entries_by_method)
        comparison_path = args.out_dir / COMPARISON_FILENAME
        comparison_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print_comparison(payload)
        print(f"comparison persisted at: {comparison_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
