"""Minimal live smoke test for a baseline harness on OBLIQ-Bench Math.

Narrower than a full experiment run: it exercises exactly one baseline
harness (default ``H0*``, the RLM authors' own hand-written system prompt
plus ``ORCHESTRATOR_ADDENDUM``, byte-identical and unmodified since upstream
PR #165 -- see ``shrlm/rlm_harness.py``) against a real sample of OBLIQ-Bench
Math (https://huggingface.co/datasets/dianetc/OBLIQ-Bench, ``analogues/math``
subset) queries, via the exact production wiring
(``round_config_kwargs``/``RoundConfig``/``run_round``) an experiment round
would use -- so a green run here is a faithful signal, not a simulation. No
self-harness optimization loop (mining/proposal/promotion) is involved.

``--harness H0*R`` is also available: a locally-authored variant of ``H0*``
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
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv

from shrlm.environments.obliq_bench_math import (
    ObliqBenchMathVerifier,
    load_obliq_bench_math,
    recorded_ndcg,
)
from shrlm.experiment.config import load_config, round_config_kwargs
from shrlm.optimization.driver import RoundConfig, run_round
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import HARNESSES

DEFAULT_CONFIG = Path("configs/experiment_obliq_bench_math_DeepSeekV4Flash.toml")
HARNESS_CHOICES = ("H0", "H0*", "H0*R")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="actually spend money")
    parser.add_argument("--harness", choices=HARNESS_CHOICES, default="H0*")
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
        "median_ndcg_at_10": median(ndcg_values) if ndcg_values else None,
        "total_cost": sum(float(entry["cost"] or 0.0) for entry in entries),
        "input_tokens": sum(int(entry["input_tokens"]) for entry in entries),
        "output_tokens": sum(int(entry["output_tokens"]) for entry in entries),
        "wall_seconds": sum(float(entry["execution_time"]) for entry in entries),
    }


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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

    print("=== OBLIQ-Bench Math baseline smoke ===")
    print(
        f"harness={args.harness} backend={kwargs['backend']} model={kwargs['backend_kwargs'].get('model_name')}"
    )
    print(
        f"n={args.n} query_ids={query_ids} candidate_pool_size={args.candidate_pool_size or 'full corpus'} "
        f"seed={args.seed}"
    )
    print(f"per-run cap: max_budget=${kwargs['max_budget']} max_timeout={kwargs['max_timeout']}s")
    n_instances = args.n if query_ids is None else len(query_ids)
    print(f"worst-case spend: ${n_instances * (kwargs['max_budget'] or 0.0):.2f}")
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

    print(f"\nRunning {args.harness}...")
    round_config = RoundConfig(
        round_index=0,
        harness=HARNESSES[args.harness],
        instances=instances,
        verifier=ObliqBenchMathVerifier(),
        out_dir=args.out_dir,
        **kwargs,
    )
    entries = run_round(round_config)

    print("\n=== Results ===")
    for entry in entries:
        print(
            f"{entry['instance_id']}: passed={entry['passed']} cause={entry['cause']} "
            f"cost=${float(entry['cost'] or 0.0):.4f} detail={entry['verdict'].get('detail')}"
        )
    summary = summarize_entries(entries)
    print(
        f"\npasses={summary['passes']}/{summary['n_runs']} "
        f"mean_ndcg@10={format_metric(summary['mean_ndcg_at_10'])} "
        f"median_ndcg@10={format_metric(summary['median_ndcg_at_10'])} "
        f"ndcg_coverage={summary['ndcg_coverage']}/{summary['n_runs']} "
        f"cost=${summary['total_cost']:.4f} wall={summary['wall_seconds']:.1f}s"
    )
    print(f"traces={(args.out_dir / 'round_00').resolve()}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary persisted at: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
