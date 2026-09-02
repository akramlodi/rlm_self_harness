"""
MRCRv2 smoke test: generate instances, run H0 (unoptimized) over the short
split, and report cost/tokens/pass-rate/grounding stats before trusting a
full pilot.

Unlike ``examples/graphwalks_example.py`` (a standalone demo using the bare
``rlm.RLM`` API) this goes through the SAME execution path the optimization
loop itself uses -- ``shrlm.optimization.driver.run_round`` with an
``ExperimentConfig``-driven harness/backend -- because the deliverable is
"run the current harness (H0, unoptimized)", which means the harness registry
entry, not an ad hoc RLM() call. It stops short of the budget-governed
structural-assertion tier ``experiment_smoke.py`` occupies: this script's job
is descriptive stats over one real short-split run, not proving the whole
pipeline's cost arithmetic.

Usage (spends nothing by default -- prints the plan and exits):
    uv run python -m examples.mrcrv2_smoke --config configs/experiment_mrcrv2.toml

Usage (spends real money against the configured backend):
    uv run python -m examples.mrcrv2_smoke --live \\
        --config configs/experiment_mrcrv2.toml --n-short 50 --n-long 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from shrlm.environments.mrcrv2 import Mrcrv2SubVerifier, Mrcrv2Verifier, load_mrcrv2_from_config
from shrlm.experiment.config import backend_kwargs_for, load_config
from shrlm.optimization.driver import RoundConfig, _BACKEND_ENV_KEYS, load_round, run_round
from shrlm.optimization.grounding import FailingLevel, apply_sub_verifier, derive_failing_level
from shrlm.optimization.walker import walk
from shrlm.rlm_harness import HARNESSES

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_mrcrv2.toml"


def missing_credentials(backend: str) -> list[str]:
    return [key for key in _BACKEND_ENV_KEYS.get(backend, ()) if not os.getenv(key)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "mrcrv2_smoke_out")
    parser.add_argument("--n-short", type=int, default=50)
    parser.add_argument("--n-long", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--harness", default="H0", help="Registry harness name to run (default H0, unoptimized).")
    parser.add_argument("--live", action="store_true", help="Actually execute the short split (spends real money).")
    args = parser.parse_args()

    config = load_config(profile=args.profile, path=args.config)
    if args.harness not in HARNESSES:
        print(f"Error: {args.harness!r} is not a registry harness; known: {sorted(HARNESSES)}")
        sys.exit(1)
    harness = HARNESSES[args.harness]

    print(f"Generating {args.n_short} short (2-needle, ~{config.environments.mrcrv2.short_target_tokens:,} "
          f"token) and {args.n_long} long (8-needle, ~{config.environments.mrcrv2.long_target_tokens:,} "
          f"token) MRCRv2 instances (seed={args.seed})...")
    short_instances = load_mrcrv2_from_config(config, n=args.n_short, seed=args.seed, length="short")
    long_instances = load_mrcrv2_from_config(config, n=args.n_long, seed=args.seed, length="long")
    print(f"  short: {len(short_instances)} instances, mean {mean(i['prompt_chars'] for i in short_instances):,.0f} chars")
    print(f"  long:  {len(long_instances)} instances, mean {mean(i['prompt_chars'] for i in long_instances):,.0f} chars")
    print("  (long split is generated for sizing only -- this smoke test runs H0 over the short split.)\n")

    backend = config.backends.runner.backend
    missing = missing_credentials(backend)
    if not args.live:
        print(f"Dry run (no --live): would execute {len(short_instances)} short-split runs under "
              f"harness={args.harness!r} on backend={backend!r}. Re-run with --live to spend.")
        return
    if missing:
        print(f"Error: --live requires backend {backend!r} credentials; missing env var(s): {missing}")
        sys.exit(1)

    round_config = RoundConfig(
        round_index=0,
        harness=harness,
        instances=short_instances,
        verifier=Mrcrv2Verifier(),
        out_dir=args.out_dir,
        backend=backend,
        backend_kwargs=backend_kwargs_for(config, "runner"),
        attempts=1,
        max_iterations=config.caps.max_iterations,
        max_depth=config.caps.max_depth,
        max_budget=config.caps.max_budget,
        max_timeout=config.caps.max_timeout,
    )
    print(f"Running {len(short_instances)} runs under harness={args.harness!r} "
          f"backend={backend!r} model={config.backends.runner.model!r}...")
    entries = run_round(round_config)
    report(args.out_dir, entries)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def report(out_dir: Path, entries: list[dict]) -> None:
    n = len(entries)
    total_cost = sum(float(e["cost"]) for e in entries)
    total_input = sum(int(e["input_tokens"]) for e in entries)
    total_output = sum(int(e["output_tokens"]) for e in entries)
    n_passed = sum(1 for e in entries if e["passed"])

    runs, verdicts, _envelope, _entries = load_round(out_dir, 0)
    sub_verifier = Mrcrv2SubVerifier()
    n_checkable = 0
    n_no_recursion = 0
    for instance, completion in runs:
        root, _stats = walk(completion)
        if derive_failing_level(root, {}) is FailingLevel.NO_RECURSION:
            n_no_recursion += 1
        grounding = apply_sub_verifier(instance, root, sub_verifier)
        if any(verdict is not None for verdict in grounding.verdicts.values()):
            n_checkable += 1

    print("\n" + "=" * 60)
    print("MRCRv2 SMOKE SUMMARY (short split, H0 unoptimized)")
    print("=" * 60)
    print(f"  runs:                    {n}")
    print(f"  total cost:              ${total_cost:.4f}")
    print(f"  mean cost/run:           ${total_cost / n:.4f}" if n else "  mean cost/run:           n/a")
    print(f"  mean tokens/run (in/out):{total_input / n:,.0f} / {total_output / n:,.0f}" if n else "")
    print(f"  verifier pass rate:      {n_passed}/{n} ({n_passed / n:.1%})" if n else "")
    print(f"  sub-verifier coverage:   {n_checkable}/{n} runs with >=1 checkable sub-verdict "
          f"({n_checkable / n:.1%})" if n else "")
    print(f"  single-pass (no sub-calls): {n_no_recursion}/{n}")
    print(f"  decomposed (>=1 sub-call):  {n - n_no_recursion}/{n}")


if __name__ == "__main__":
    main()
