"""Evaluate saved harnesses on frozen test splits, with resumable process workers.

Example (uses the experiment's saved repetition count and spend limits)::

    uv run python examples/run_evaluation.py --config configs/experiment.toml \
        --out-dir experiment_full --conditions initial sh_rlm \
        --test-sets oolong_pairs_long --workers 3

``initial`` is the exact harness saved by the first mining round; ``sh_rlm``
is the frozen final harness. Conditions execute in order with at most
``--workers`` simultaneous attempts. Re-running the same command resumes
without repeating persisted attempts. ``--dry-run`` verifies the inputs and
prints identities, run counts and caps without making model calls.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from shrlm.experiment.config import CONFIG_PATH, identity_hash, load_config, validation_caps
from shrlm.experiment.evaluation import (
    CONDITION_INITIAL,
    CONDITION_LAMBDA_RLM,
    CONDITION_SH_RLM,
    CONDITIONS,
    resolve_conditions,
    run_evaluation,
    test_sets,
)
from shrlm.experiment.live_gates import pricing_attestation_mismatch
from shrlm.experiment.orchestrator import CONFIG_FILENAME, check_identity
from shrlm.experiment.splits import DEFAULT_LOADERS, MANIFEST_FILE, SPLITS_DIR, materialize_splits
from shrlm.optimization.driver import _BACKEND_ENV_KEYS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=[CONDITION_INITIAL, CONDITION_SH_RLM],
    )
    parser.add_argument("--test-sets", nargs="+", help="default: all frozen test sets")
    parser.add_argument("--workers", type=int, default=1, help="concurrent harness runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers > 1 and CONDITION_LAMBDA_RLM in args.conditions:
        parser.error("lambda_rlm requires --workers 1")
    if not (args.out_dir / CONFIG_FILENAME).is_file():
        parser.error("--out-dir must be an existing experiment with a saved config identity")
    manifest_path = args.out_dir / SPLITS_DIR / MANIFEST_FILE
    if not manifest_path.is_file():
        parser.error("--out-dir must contain frozen splits")

    load_dotenv()
    config = load_config(args.profile, args.config)
    check_identity(config, args.out_dir)
    backend = config.backends.runner.backend
    missing = [key for key in _BACKEND_ENV_KEYS.get(backend, ()) if not os.getenv(key)]
    if missing:
        parser.error(f"missing runner credentials: {', '.join(missing)}")
    if backend == "azure_foundry":
        pricing = config.pricing.list_price
        mismatch = pricing_attestation_mismatch(
            os.getenv("SHRLM_VERIFIED_PRICING"),
            pricing.input_per_million,
            pricing.output_per_million,
        )
        if mismatch:
            parser.error(mismatch)

    # Verify only environments already frozen by this experiment; evaluation
    # must not download or draw missing datasets, including during preflight.
    manifest = json.loads(manifest_path.read_text())
    frozen_loaders = {name: DEFAULT_LOADERS[name] for name in manifest["environments"]}
    splits_dir = materialize_splits(config, args.out_dir, loaders=frozen_loaders)
    sets = test_sets(splits_dir, set_ids=args.test_sets)
    sources = resolve_conditions(args.conditions)
    caps = validation_caps(config)
    print(f"Identity: {identity_hash(config)}", flush=True)
    # Verify both envelopes before either condition spends anything. Temporary
    # modules avoid writing into the evaluation output during a dry run.
    with TemporaryDirectory(prefix="shrlm-evaluation-preflight-") as temporary:
        for condition, source in sources:
            method = source.load(args.out_dir, Path(temporary) / condition)
            method.validate_caps(condition, caps)
            print(f"{condition}: {method.method_hash}", flush=True)
    total = 0
    for test_set in sets:
        with (splits_dir / test_set.file_name).open() as stream:
            count = sum(bool(line.strip()) for line in stream)
        runs = count * config.operational.eval_repetitions * len(sources)
        total += runs
        print(f"{test_set.set_id}: {count} tasks; {runs} runs across conditions", flush=True)
    print(
        f"Total: {total} runs; {config.operational.eval_repetitions} attempts/task; "
        f"{args.workers} workers; ${caps.max_budget}/run; "
        f"${caps.candidate_budget}/condition; {caps.max_timeout}s/run",
        flush=True,
    )
    if args.dry_run:
        return 0
    result = run_evaluation(
        config,
        args.conditions,
        args.out_dir,
        test_set_ids=args.test_sets,
        run_workers=args.workers,
        loaders=frozen_loaders,
    )
    print(f"Evaluation summary: {result.summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
