"""Estimate OBLIQ-Bench Math promotion noise from the calibration probe's
own repeated-attempt data, and report real baseline-vs-candidate deltas.

Adapted from shrlm.experiment.calibrate_promotion_tau, which is OOLONG-Pairs-
specific (hardcoded to F1 and a context-length axis OBLIQ-Bench Math has no
equivalent of -- it is a single, unstratified 15-instance pool). This script
reuses the same two cross-checking statistical methods (a pilot nearest-rank
quantile over pairwise attempt-mean deltas, and a closed-form variance-
decomposition estimate) against NDCG@10 instead of F1, with no per-length
split: tau_regression and tau_improvement collapse to the SAME quantity here
(both are just "the noise floor of one pooled-mean comparison"), unlike
OOLONG-Pairs where tau_improvement is a single macro figure but tau_regression
is a dict of per-length figures.

Usage (baseline-only noise-floor calibration):
    uv run python examples/obliq_two_stage/calibrate_obliq_tau.py \\
        --baseline-split-dir experiment_real_runs/experiment_obliq_bench_math_calib_stage1/opt/round_01/validation/round_01/baseline/heldout

Usage (also reporting real baseline-vs-candidate deltas from a round that
had exactly one measured candidate -- the shape every obliq round takes):
    uv run python examples/obliq_two_stage/calibrate_obliq_tau.py \\
        --baseline-split-dir .../baseline/heldout \\
        --candidate-split-dir .../r01-c01-sN/heldout
"""

import argparse
import json
import math
from statistics import NormalDist
from typing import Any

from shrlm.environments.diagnostics import NDCG_QUALITY
from shrlm.optimization.driver import load_round
from shrlm.optimization.validation import EVAL_ROUND_INDEX


def quality_value(entry: dict[str, Any]) -> float:
    """Persisted NDCG@10, using only verifier-registered terminal values."""
    measurement = entry["verdict"].get("quality")
    if measurement is not None:
        if measurement.get("definition_id") != NDCG_QUALITY.identifier:
            raise ValueError(
                f"expected {NDCG_QUALITY.identifier!r}, got {measurement.get('definition_id')!r}"
            )
        return float(measurement["value"])
    cause = entry.get("cause")
    if cause not in NDCG_QUALITY.terminal_values:
        raise ValueError(
            f"run {entry.get('run_id')!r} has no NDCG and cause {cause!r} has no terminal value"
        )
    return float(NDCG_QUALITY.terminal_values[cause])


def upper_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot estimate a quantile from no deltas")
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _by_instance_attempt(split_dirs: list[str]) -> dict[str, dict[int, float]]:
    """NDCG for every (instance, attempt) across one or more split dirs."""
    values: dict[str, dict[int, float]] = {}
    for split_dir in split_dirs:
        runs, _verdicts, _envelope, entries = load_round(split_dir, EVAL_ROUND_INDEX)
        for entry, (instance, _completion) in zip(entries, runs, strict=True):
            instance_id = str(entry["instance_id"])
            values.setdefault(instance_id, {})[int(entry["attempt"])] = quality_value(entry)
    return values


def _matrix(split_dirs: list[str]) -> tuple[list[str], list[list[float]]]:
    """n_instances x n_attempts NDCG matrix, shape-checked."""
    by_instance = _by_instance_attempt(split_dirs)
    instance_ids = sorted(by_instance)
    n_attempts = None
    rows = []
    for instance_id in instance_ids:
        attempts = sorted(by_instance[instance_id])
        if n_attempts is None:
            n_attempts = len(attempts)
        elif len(attempts) != n_attempts:
            raise ValueError("every instance must share the same attempt count")
        rows.append([by_instance[instance_id][a] for a in attempts])
    return instance_ids, rows


def pilot_calibrate(split_dirs: list[str], quantile: float = 0.95) -> dict[str, Any]:
    """Pairwise attempt-mean null deltas -- a 3-attempt pilot estimate."""
    _, rows = _matrix(split_dirs)
    n_instances, n_attempts = len(rows), len(rows[0])
    attempt_means = [sum(row[a] for row in rows) / n_instances for a in range(n_attempts)]
    from itertools import combinations

    pairs = list(combinations(range(n_attempts), 2))
    deltas = [abs(attempt_means[j] - attempt_means[i]) for i, j in pairs]
    return {
        "schema": "shrlm-obliq-tau-calibration/v1-pilot",
        "n_instances": n_instances,
        "n_attempts": n_attempts,
        "attempt_means": attempt_means,
        "pairwise_deltas": deltas,
        "recommended_tau": upper_quantile(deltas, quantile),
        "warning": (
            "A handful of attempts yield only a few non-independent pairwise deltas; "
            "treat this as a cross-check against parametric_calibrate, not the primary estimate."
        ),
    }


def parametric_calibrate(split_dirs: list[str], quantile: float = 0.95) -> dict[str, Any]:
    """Closed-form variance-decomposition estimate (primary method)."""
    _, rows = _matrix(split_dirs)
    n_instances, n_attempts = len(rows), len(rows[0])

    def sample_variance(values: list[float]) -> float:
        n = len(values)
        mean = sum(values) / n
        return sum((v - mean) ** 2 for v in values) / (n - 1)

    instance_means = [sum(row) / n_attempts for row in rows]
    within_var = sum(sample_variance(row) for row in rows) / n_instances
    between_var = max(0.0, sample_variance(instance_means) - within_var / n_attempts)
    pooled_mean_var = between_var / n_instances + within_var / (n_instances * n_attempts)
    z = NormalDist().inv_cdf((1 + quantile) / 2)
    # x sqrt(2): the promotion check differences two INDEPENDENT pooled
    # means (baseline vs candidate), each with variance pooled_mean_var.
    tau = z * math.sqrt(2 * pooled_mean_var)
    return {
        "schema": "shrlm-obliq-tau-calibration/v1-parametric",
        "n_instances": n_instances,
        "n_attempts": n_attempts,
        "within_instance_variance": within_var,
        "between_instance_variance": between_var,
        "pooled_mean_variance": pooled_mean_var,
        "recommended_tau": tau,
        "note": (
            "tau_regression and tau_improvement are the SAME number here -- OBLIQ-Bench "
            "Math has no per-length axis to distinguish them, unlike OOLONG-Pairs."
        ),
    }


def real_delta(baseline_split_dirs: list[str], candidate_split_dirs: list[str]) -> dict[str, Any]:
    """The actual observed baseline-vs-candidate NDCG delta, for comparison
    against the calibrated noise floor above."""
    _, base_rows = _matrix(baseline_split_dirs)
    _, cand_rows = _matrix(candidate_split_dirs)
    base_mean = sum(v for row in base_rows for v in row) / (len(base_rows) * len(base_rows[0]))
    cand_mean = sum(v for row in cand_rows for v in row) / (len(cand_rows) * len(cand_rows[0]))
    return {
        "baseline_mean_ndcg": base_mean,
        "candidate_mean_ndcg": cand_mean,
        "delta": cand_mean - base_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-split-dir", action="append", required=True, type=str)
    parser.add_argument("--candidate-split-dir", action="append", default=None, type=str)
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "pilot": pilot_calibrate(args.baseline_split_dir, args.quantile),
        "parametric": parametric_calibrate(args.baseline_split_dir, args.quantile),
    }
    if args.candidate_split_dir:
        report["real_observed_delta"] = real_delta(
            args.baseline_split_dir, args.candidate_split_dir
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
