"""Minimal live smoke test: λ-RLM's OOLONG-Pairs paper reconstruction, long split only.

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

Cost: bounded at ``n x caps.max_budget`` (shipped config: $0.50/run), and the
shipped Qwen3-30B-A3B-Instruct-2507 pricing puts a real 262k-token pairwise
run far under that (classification batches cost proportionally to input
tokens; output per batch is a few short lines, not a full completion).

Usage:
    # pre-flight only: prints the config identity and instance count, spends
    # nothing.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py

    # the real run.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live

    # more coverage: two long instances at two different task ids.
    uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live \\
        --n 2 --task-ids 1,11
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from shrlm.baselines.lambda_rlm import LambdaBaselineConfig
from shrlm.baselines.lambda_runner import (
    LambdaRoundConfig,
    require_lambda_backend_credential,
    run_lambda_round,
)
from shrlm.environments.oolong_pairs import OolongPairsVerifier, load_oolong_pairs
from shrlm.experiment.config import load_config, round_config_kwargs

DEFAULT_TASK_IDS = (1,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="actually spend money")
    parser.add_argument("--n", type=int, default=1, help="number of long instances (default 1)")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_ids = tuple(int(t) for t in args.task_ids.split(","))
    if args.n < 1:
        raise ValueError(f"--n must be >= 1, got {args.n}")

    load_dotenv()
    config = load_config("full")
    oolong_cfg = config.environments.oolong_pairs
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

    print("=== λ-RLM OOLONG-Pairs long smoke ===")
    print(f"backend={round_config.backend} model={round_config.backend_kwargs.get('model_name')}")
    print(f"task_ids={task_ids} n={args.n} context_length_long={oolong_cfg.context_length_long}")
    print(
        f"per-run cap: max_budget=${round_config.max_budget} max_timeout={round_config.max_timeout}s"
    )
    print(f"worst-case spend: ${args.n * (round_config.max_budget or 0):.2f}")
    print(f"out_dir={args.out_dir.resolve()}")

    if not args.live:
        print("\n--live not passed: nothing loaded, nothing spent. Re-run with --live to execute.")
        return 0

    require_lambda_backend_credential(round_config)

    print("\nStreaming real OOLONG-Pairs long window(s) from the upstream dataset...")
    instances = load_oolong_pairs(
        task_ids=task_ids,
        context_lengths=(oolong_cfg.context_length_long,),
        n=args.n,
        seed=args.seed,
        max_scan=oolong_cfg.max_scan,
        split="validation",
        revision=oolong_cfg.dataset_revision,
    )
    round_config = replace(round_config, instances=instances)
    print(f"loaded {len(instances)} instance(s): " + ", ".join(i["id"] for i in instances))

    print("\nRunning...")
    entries = run_lambda_round(round_config)

    print("\n=== Results ===")
    total_cost = 0.0
    for entry in entries:
        cost = entry["cost"]
        total_cost += cost
        print(
            f"{entry['instance_id']}: passed={entry['passed']} cause={entry['cause']} "
            f"cost=${cost:.4f} detail={entry['verdict'].get('detail')}"
        )
    print(f"\ntotal cost: ${total_cost:.4f}")
    print(f"traces persisted under: {(args.out_dir / 'round_00').resolve()}")
    print(json.dumps({"n": len(entries), "total_cost": total_cost}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
