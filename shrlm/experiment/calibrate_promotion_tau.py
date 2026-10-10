"""Estimate OOLONG-Pairs promotion noise from repeated frozen-H0 runs.

Each input is a split directory containing ``round_00`` (for example
``.../baseline/heldout``). Runs are paired by their persisted attempt number.
The report recommends a conservative absolute null-delta quantile for the
overall macro-F1 improvement threshold and for the per-length regression
allowance. Freeze the chosen values in the experiment TOML before optimization.
"""

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from shrlm.environments.diagnostics import SET_QUALITY
from shrlm.optimization.driver import load_round
from shrlm.optimization.validation import EVAL_ROUND_INDEX


def quality_value(entry: dict[str, Any]) -> float:
    """Return persisted F1, using only verifier-registered terminal values."""
    measurement = entry["verdict"].get("quality")
    if measurement is not None:
        if measurement.get("definition_id") != SET_QUALITY.identifier:
            raise ValueError(
                f"expected {SET_QUALITY.identifier!r}, got {measurement.get('definition_id')!r}"
            )
        return float(measurement["value"])
    cause = entry.get("cause")
    if cause not in SET_QUALITY.terminal_values:
        raise ValueError(
            f"run {entry.get('run_id')!r} has no F1 and cause {cause!r} has no terminal value"
        )
    return float(SET_QUALITY.terminal_values[cause])


def upper_quantile(values: list[float], quantile: float) -> float:
    """Conservative nearest-rank quantile, requiring finite non-empty input."""
    if not values:
        raise ValueError("cannot estimate a quantile from no deltas")
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def calibrate(split_dirs: list[Path], quantile: float = 0.95) -> dict[str, Any]:
    """Compute paired attempt-level null deltas across context lengths."""
    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for split_dir in split_dirs:
        runs, _verdicts, _envelope, entries = load_round(split_dir, EVAL_ROUND_INDEX)
        for entry, (instance, _completion) in zip(entries, runs, strict=True):
            if "context_len" not in instance:
                raise ValueError(f"instance {instance.get('id')!r} has no context_len")
            values[int(entry["attempt"])][str(instance["context_len"])].append(quality_value(entry))

    attempts = sorted(values)
    if len(attempts) < 2:
        raise ValueError("tau calibration needs at least two attempts; use three or more")
    lengths = sorted(values[attempts[0]], key=int)
    for attempt in attempts:
        if sorted(values[attempt], key=int) != lengths:
            raise ValueError("every attempt must cover the same context lengths")

    means = {
        attempt: {
            length: sum(values[attempt][length]) / len(values[attempt][length])
            for length in lengths
        }
        for attempt in attempts
    }
    macro = {attempt: sum(means[attempt].values()) / len(lengths) for attempt in attempts}
    pairs = list(combinations(attempts, 2))
    macro_deltas = [abs(macro[right] - macro[left]) for left, right in pairs]
    by_length_deltas = {
        length: [abs(means[right][length] - means[left][length]) for left, right in pairs]
        for length in lengths
    }
    return {
        "schema": "shrlm-promotion-tau-calibration/v1",
        "quality_definition": SET_QUALITY.to_dict(),
        "quantile": quantile,
        "attempts": attempts,
        "attempt_macro_f1": {str(key): value for key, value in macro.items()},
        "attempt_f1_by_context_length": {str(attempt): means[attempt] for attempt in attempts},
        "absolute_pairwise_macro_deltas": macro_deltas,
        "absolute_pairwise_deltas_by_context_length": by_length_deltas,
        "recommended": {
            "tau_improvement": upper_quantile(macro_deltas, quantile),
            # Per context length: noise is not uniform (observed macro deltas on
            # a frozen harness can differ by over 30x between lengths), so a
            # single collapsed-to-worst-case margin is either far too loose at a
            # quiet length or still too tight at a noisy one. PromotionConfig's
            # per-length regression check consumes this mapping directly.
            "tau_regression": {
                length: upper_quantile(deltas, quantile)
                for length, deltas in by_length_deltas.items()
            },
        },
        "warning": (
            "Three attempts yield only three non-independent pairwise deltas; treat this "
            "as a pilot noise floor, freeze the result, and do not tune it on candidates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        action="append",
        required=True,
        type=Path,
        help="split directory containing round_00; repeat for 8k, 16k, and 32k",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.split_dir, args.quantile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
