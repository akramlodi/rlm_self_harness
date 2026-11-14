"""Launcher for the 2(-3)-stage OBLIQ-Bench Math optimization plan.

--stage 1: freeze stage 1's splits from the disjoint partition, then run
           configs/experiment_obliq_bench_math_openrouter_stage1_pool200.toml
           for real (examples/run_experiment.py's own main() body, inlined,
           since this needs an extra pre-step run_experiment.py doesn't do).

--stage 2: materialize stage 1's frozen final harness
           (<stage1_out_dir>/sh_rlm/harness.json), register it into
           shrlm.rlm_harness.HARNESSES under "STAGE1_WINNER" (the only way to
           point loop.initial_harness at a dynamically produced harness --
           it's a plain registry-key string, config files cannot name one
           directly), freeze stage 2's splits, then run stage 2's config.

Usage:
    uv run python examples/obliq_two_stage/run_two_stage.py --stage 1
    uv run python examples/obliq_two_stage/run_two_stage.py --stage 2
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent

STAGE1_CONFIG = REPO_ROOT / "configs/experiment_obliq_bench_math_openrouter_stage1_pool200.toml"
STAGE2_CONFIG = REPO_ROOT / "configs/experiment_obliq_bench_math_openrouter_stage2_pool500.toml"
STAGE1_OUT_DIR = REPO_ROOT / "experiment_obliq_bench/stage1_pool200"
STAGE2_OUT_DIR = REPO_ROOT / "experiment_obliq_bench/stage2_pool500"

PARTITION_PATH = HERE / "partition.json"
SEED = 0  # manifest consistency tag only -- see configs' own header comments


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

    config = load_config("full", STAGE1_CONFIG)

    print(f"Running stage 1 (candidate_pool_size=200) into {STAGE1_OUT_DIR}...")
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
            f"{frozen_path} missing -- stage 1 has not finished (run --stage 1 first)."
        )
    envelope = json.loads(frozen_path.read_text())
    module_path = STAGE2_OUT_DIR / "stage1_winner_module.py"
    STAGE2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_winner = materialize_harness(envelope["harness"], module_path)
    HARNESSES["STAGE1_WINNER"] = stage1_winner
    print(f"registered stage 1's winner ({envelope['hash']}) as HARNESSES['STAGE1_WINNER']")

    freeze_stage_splits(
        out_dir=STAGE2_OUT_DIR,
        seed=SEED,
        candidate_pool_size=500,
        mining_ids=partition["stage2_mining"],
        validation_ids=partition["stage2_validation"],
        promotion_ids=partition["stage2_promotion"],
        test_ids=partition["full_eval_test"],
    )

    config = load_config("full", STAGE2_CONFIG)  # validates initial_harness now that it's registered

    print(f"Running stage 2 (candidate_pool_size=500) into {STAGE2_OUT_DIR}...")
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
