"""Launcher for the 2-round OBLIQ-Bench Math calibration probe.

Reuses the REAL stage1_pool200/stage2_pool500 TOML configs as-is, overriding
only loop.t (7/8 -> 2) via dataclasses.replace() -- no separate "calib"
config files, since t is just a dataclass field and every other setting
should stay byte-identical to what the real run will actually use for its
evidence to generalize. Fresh --out-dirs keep this probe's artifacts from
ever colliding with the real stage1/stage2 run; stage 2's config hardcodes
loop.initial_harness = "STAGE1_WINNER" as a literal string (validated
against the shrlm.rlm_harness.HARNESSES registry at load time), so this
script registers the calibration run's own stage-1 winner under that same
key -- safe because each --stage invocation is a separate process with its
own in-memory registry, never run concurrently with the real optimization.

Purpose: gather real promotion-rule evidence (baseline's own repeated-
attempt variance, and real baseline-vs-candidate deltas) to feed
calibrate_obliq_tau.py, and to see whether refinement ever actually triggers
before deciding whether to keep it in the real configs at all.

Usage:
    uv run python examples/obliq_two_stage/run_calibration.py --stage 1
    uv run python examples/obliq_two_stage/run_calibration.py --stage 2
"""

import argparse
import dataclasses
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent

STAGE1_CONFIG = REPO_ROOT / "configs/experiment_obliq_bench_math_openrouter_stage1_pool200.toml"
STAGE2_CONFIG = REPO_ROOT / "configs/experiment_obliq_bench_math_openrouter_stage2_pool500.toml"
STAGE1_OUT_DIR = REPO_ROOT / "experiment_obliq_bench/calib_stage1"
STAGE2_OUT_DIR = REPO_ROOT / "experiment_obliq_bench/calib_stage2"

PARTITION_PATH = HERE / "partition.json"
SEED = 0  # manifest consistency tag only -- see the real configs' own header comments
CALIBRATION_ROUNDS = 2


def load_partition() -> dict[str, list[str]]:
    if not PARTITION_PATH.exists():
        raise FileNotFoundError(
            f"{PARTITION_PATH} missing -- run plan_partition.py first to compute the "
            "disjoint 151-query partition."
        )
    return json.loads(PARTITION_PATH.read_text())


def _require_openrouter_key() -> None:
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set; nothing was spent")


def _with_calibration_rounds(config):
    return dataclasses.replace(config, loop=dataclasses.replace(config.loop, t=CALIBRATION_ROUNDS))


def run_stage1() -> None:
    from dotenv import load_dotenv

    from freeze_splits import freeze_stage_splits
    from shrlm.experiment.config import load_config
    from shrlm.experiment.orchestrator import run_experiment

    load_dotenv()
    _require_openrouter_key()
    partition = load_partition()

    STAGE1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    freeze_stage_splits(
        out_dir=STAGE1_OUT_DIR,
        seed=SEED,
        candidate_pool_size=200,
        mining_ids=partition["stage1_mining"],
        validation_ids=partition["stage1_validation"],
        promotion_ids=partition["stage1_promotion"],
        test_ids=partition["full_eval_test"],
    )

    config = _with_calibration_rounds(load_config("full", STAGE1_CONFIG))

    print(
        f"Running calibration stage 1 (candidate_pool_size=200, t={CALIBRATION_ROUNDS}) "
        f"into {STAGE1_OUT_DIR}..."
    )
    result = run_experiment(config, STAGE1_OUT_DIR)
    for outcome in result.rounds:
        promoted = (
            f"promoted {outcome.promoted_harness_hash}" if outcome.promoted else "no promotion"
        )
        print(f"  round {outcome.round_index}: {promoted}")
    print(f"Stopped on {result.stopped}; final harness {result.final_harness_hash}")
    print(f"Frozen harness: {result.frozen_path}")


def run_stage2() -> None:
    from dotenv import load_dotenv

    from freeze_splits import freeze_stage_splits
    from shrlm.experiment.config import load_config
    from shrlm.experiment.orchestrator import run_experiment
    from shrlm.optimization.candidates import materialize_harness
    from shrlm.rlm_harness import HARNESSES

    load_dotenv()
    _require_openrouter_key()
    partition = load_partition()

    frozen_path = STAGE1_OUT_DIR / "sh_rlm" / "harness.json"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"{frozen_path} missing -- calibration stage 1 has not finished "
            "(run --stage 1 first)."
        )
    envelope = json.loads(frozen_path.read_text())
    module_path = STAGE2_OUT_DIR / "calib_stage1_winner_module.py"
    STAGE2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_winner = materialize_harness(envelope["harness"], module_path)
    # Stage 2's TOML hardcodes initial_harness = "STAGE1_WINNER" literally;
    # register under that exact key (not a calib-specific one -- see module
    # docstring for why that's safe here).
    HARNESSES["STAGE1_WINNER"] = stage1_winner
    print(f"registered calibration stage 1's winner ({envelope['hash']}) as HARNESSES['STAGE1_WINNER']")

    freeze_stage_splits(
        out_dir=STAGE2_OUT_DIR,
        seed=SEED,
        candidate_pool_size=500,
        mining_ids=partition["stage2_mining"],
        validation_ids=partition["stage2_validation"],
        promotion_ids=partition["stage2_promotion"],
        test_ids=partition["full_eval_test"],
    )

    config = _with_calibration_rounds(load_config("full", STAGE2_CONFIG))

    print(
        f"Running calibration stage 2 (candidate_pool_size=500, t={CALIBRATION_ROUNDS}) "
        f"into {STAGE2_OUT_DIR}..."
    )
    result = run_experiment(config, STAGE2_OUT_DIR)
    for outcome in result.rounds:
        promoted = (
            f"promoted {outcome.promoted_harness_hash}" if outcome.promoted else "no promotion"
        )
        print(f"  round {outcome.round_index}: {promoted}")
    print(f"Stopped on {result.stopped}; final harness {result.final_harness_hash}")
    print(f"Frozen harness: {result.frozen_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    if args.stage == 1:
        run_stage1()
    else:
        run_stage2()
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
