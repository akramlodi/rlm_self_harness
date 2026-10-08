"""Run the frozen-H0 OOLONG-Pairs pilot used to choose promotion tau.

The default design is 4 frozen instances at each of 8k, 16k, and 32k, with
3 attempts per instance (36 runs). It writes one ordinary ``round_00`` per
length and then prints the paired F1 noise report. Run the dry preflight first;
``--live`` is the only mode that loads data or spends money.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from shrlm.environments.oolong_pairs import OolongPairsVerifier, load_oolong_pairs
from shrlm.experiment.calibrate_promotion_tau import calibrate
from shrlm.experiment.config import CONFIG_PATH, load_config, round_config_kwargs
from shrlm.optimization.driver import RoundConfig, run_round
from shrlm.rlm_harness import H0

DEFAULT_LENGTHS = (8192, 16384, 32768)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("oolong_pairs_tau_calibration"))
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--n-per-length", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    config = load_config(path=args.config)
    lengths = tuple(int(value) for value in args.lengths.split(","))
    if args.n_per_length < 1 or args.attempts < 2:
        raise ValueError("n-per-length must be >= 1 and attempts must be >= 2")

    run_cap = config.caps.max_budget
    n_runs = len(lengths) * args.n_per_length * args.attempts
    print(f"lengths={lengths} n_per_length={args.n_per_length} attempts={args.attempts}")
    print(f"planned_runs={n_runs} worst_case_spend=${n_runs * run_cap:.2f}")
    print(f"out_dir={args.out_dir.resolve()}")
    if not args.live:
        print("--live not passed: nothing loaded and nothing spent")
        return 0

    kwargs = round_config_kwargs(config)
    kwargs["attempts"] = args.attempts
    split_dirs: list[Path] = []
    env = config.environments.oolong_pairs
    for length in lengths:
        split_dir = args.out_dir / f"context_{length}"
        instances = load_oolong_pairs(
            task_ids=env.task_ids,
            context_lengths=(length,),
            n=args.n_per_length,
            seed=args.seed,
            max_scan=env.max_scan,
            split="validation",
            revision=env.dataset_revision,
        )
        run_round(
            RoundConfig(
                round_index=0,
                harness=H0,
                instances=instances,
                verifier=OolongPairsVerifier(),
                out_dir=split_dir,
                **kwargs,
            )
        )
        split_dirs.append(split_dir)

    report = calibrate(split_dirs, args.quantile)
    report_path = args.out_dir / "tau_calibration.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
