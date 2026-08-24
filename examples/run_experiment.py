"""Run the REAL Self-Harness experiment: the full profile, end to end (R7).

A thin CLI over ``shrlm.experiment.orchestrator.run_experiment``: it loads one
config profile from ``configs/experiment.toml``, refuses to start unless every
credential the configured backends read is present (nothing is created and
nothing is spent on the decline path), prints the experiment identity, and
hands off to the orchestrator with the real defaults (GraphWalksVerifier,
configured attributor/proposer clients, real dataset loaders).

There is deliberately NO script-level spend ceiling here -- this is the real
experiment, and the configured caps govern: every run is cut off at
``caps.max_budget`` / ``caps.max_timeout``, and every candidate's cumulative
validation spend at ``caps.candidate_budget`` (the circuit breakers). Rough
exposure math: worst-case spend is bounded by the config's total run count
times ``caps.max_budget`` -- at the shipped full profile that is
(m*n_in + v*(k+1)*(n_in+n_ho) + p_merge*v*(n_in+n_ho)) = 1,456 runs/round x
t = 3 rounds = 4,368 runs x $0.50 = $2,184 as an absolute ceiling, with the
per-candidate ``candidate_budget`` breakers binding far earlier in practice
(prior smoke measurements put the mean run at $0.0024-$0.0134).

The experiment is resumable (persist-first): re-invoking with the same
``--out-dir`` verifies the config identity, replays completed rounds from
their markers, and resumes an interrupted round at its exact stage boundary.
A changed config refuses to resume (R3) -- use a fresh --out-dir instead.
An interrupt (Ctrl-C) therefore loses nothing: re-run the same command.

Usage:
    # pre-flight only: env gate, config identity, per-role summary; spends $0.
    uv run python examples/run_experiment.py --out-dir ./experiment_full --dry-run

    # the real run (downloads the pinned datasets, spends real money).
    uv run python examples/run_experiment.py --out-dir ./experiment_full
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from shrlm.experiment.config import CLIENT_ROLES, ExperimentConfig, identity_hash, load_config
from shrlm.experiment.orchestrator import run_experiment
from shrlm.optimization.driver import _BACKEND_ENV_KEYS

load_dotenv()


def missing_env_keys(config: ExperimentConfig) -> list[str]:
    """Every unset credential variable the configured backends read."""
    required = sorted(
        {
            env_key
            for role in CLIENT_ROLES
            for env_key in _BACKEND_ENV_KEYS.get(getattr(config.backends, role).backend, ())
        }
    )
    return [env_key for env_key in required if not os.getenv(env_key)]


def print_preflight(config: ExperimentConfig, out_dir: Path) -> None:
    """The pre-flight summary: identity, per-role backends, loop counts."""
    print(f"Profile:  {config.profile}")
    print(f"Identity: {identity_hash(config)}")
    for role in CLIENT_ROLES:
        endpoint = getattr(config.backends, role)
        print(f"  {role:<10} {endpoint.backend} / {endpoint.model}")
    loop = config.loop
    print(f"Loop:     m={loop.m}, v={loop.v}, k={loop.k}, t={loop.t}, patience={loop.patience}")
    print(
        f"Resume:   re-invoking with --out-dir {out_dir} resumes this experiment; "
        "a changed config refuses (R3) -- use a fresh --out-dir."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python examples/run_experiment.py",
        description="Run (or resume) the real Self-Harness experiment; caps govern spend.",
    )
    parser.add_argument("--out-dir", required=True, help="the experiment directory")
    parser.add_argument(
        "--profile",
        default="full",
        choices=["full", "smoke"],
        help="the config profile to run (default: full)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pre-flight only: env gate, config load, identity print; spends nothing",
    )
    args = parser.parse_args(argv)

    config = load_config(args.profile)

    missing = missing_env_keys(config)
    if missing:
        print(
            f"Declining to run: {', '.join(missing)} "
            f"{'are' if len(missing) > 1 else 'is'} not set (the configured backends "
            "read credentials from the environment only). Set them (e.g. in a .env file) "
            "and re-run. Nothing was spent."
        )
        return 1

    # The pricing attestation guards the cost-SYNTHESIZING backend only: an
    # azure_foundry role bills against the configured [pricing.list_price], so
    # the attested portal rate must match it; openrouter roles report provider
    # costs directly and demand no attestation (the R4/KTD9 fallback).
    if any(getattr(config.backends, role).backend == "azure_foundry" for role in CLIENT_ROLES):
        from shrlm.experiment.live_gates import pricing_attestation_mismatch

        pricing_reason = pricing_attestation_mismatch(
            os.getenv("SHRLM_VERIFIED_PRICING"),
            config.pricing.list_price.input_per_million,
            config.pricing.list_price.output_per_million,
        )
        if pricing_reason is not None:
            print(f"Declining to run: {pricing_reason}. Nothing was spent.")
            return 1

    out_dir = Path(args.out_dir)
    print_preflight(config, out_dir)

    if args.dry_run:
        print("Dry run: pre-flight passed; nothing was created and nothing was spent.")
        return 0

    print(f"Running up to {config.loop.t} optimization round(s) into {out_dir}...")
    result = run_experiment(config, out_dir)

    for outcome in result.rounds:
        promoted = (
            f"promoted {outcome.promoted_harness_hash}" if outcome.promoted else "no promotion"
        )
        print(f"  round {outcome.round_index}: {promoted}")
    print(f"Stopped on {result.stopped}; final harness {result.final_harness_hash}")
    print(f"Frozen harness: {result.frozen_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted -- re-run the same command to resume.")
        raise SystemExit(130) from None
