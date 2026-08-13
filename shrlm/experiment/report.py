"""The cost/time report: measured usage -> full-experiment API-vs-GPU decision (R6).

``build_report`` reads one experiment directory -- the optimization manifests
under ``opt/``, the append-only ``stage_usage.jsonl``, and
``eval/eval_summary.json`` -- and turns them into the artifact the hosted-API
vs rented-GPU decision rests on. Nothing here executes a run or mutates
experiment state; the report is a pure function of persisted bytes plus the
configuration, emitted as markdown on stdout and as ``report.json`` in the
experiment directory.

This module holds measurement, run counts, and the projections built from
them. Scenario pricing (API tiers, GPU-hour projections) and the
recommendation policy live in ``shrlm.experiment.scenarios``; markdown
rendering lives in ``shrlm.experiment.render``. Both are re-exported here so
``build_report``, ``write_report``, ``render_markdown``, and the CLI stay
importable from this module exactly as before the split.

Measured section
    A per-stage table (runs, tokens in/out, wall time, USD, cache hits,
    lower-bound and resumed annotations) aggregated from ``stage_usage.jsonl``
    through ``read_stage_usage``'s exactly-once rule, plus the extrapolation
    basis: per-run means split by environment AND length. Optimization runs
    are graphwalks source-short by construction (the orchestrator mines and
    validates that split alone); evaluation runs carry their environment and
    length in ``eval_summary.json``.

Extrapolation (KTD6)
    Run counts are derived from config, never hardcoded::

        runs/round = m*n_in + v(K+1)(n_in+n_ho) + p_merge*v(n_in+n_ho)
        eval runs at a length = eval_conditions
            * sum(env test-role size at that length, from splits.split_plan)
            * eval_repetitions

    ``report.eval_conditions`` is the FULL-experiment grid (B1, H1, SH-RLM)
    and is deliberately independent of how many conditions the scaffold
    actually evaluated. The experiment evaluates every configured
    environment -- source GraphWalks and target OOLONG-Pairs -- so each
    eval leg sums the test-role size across every environment at that
    length rather than the source split alone. Those counts multiply
    measured per-run means *at each length* -- long-run cost is superlinear
    in context length, so a scaled smoke total would understate it. The
    optimization leg and the short evaluation leg use the short buckets; the
    long evaluation leg uses the long buckets. A missing long bucket is never
    a zero: the projection is labeled short-only and the unprojected long
    runs are named in a warning.

Pessimistic scenario
    Promotion admits a candidate costing up to ``promotion.cost_band[1]`` x
    its incumbent, and that ceiling compounds round over round. The
    pessimistic projection therefore replaces the optimization leg's ``x T``
    with ``sum(ceiling**i for i in range(T))`` -- the drift-adjusted upper
    bound, reported alongside the point estimate, never instead of it. The
    evaluation legs are unscaled: they run one frozen harness, not a chain of
    promotions.

Recommendation policy
    See ``shrlm.experiment.scenarios`` for the full policy (validity gates,
    quantization deprioritization, and how a scenario becomes the
    recommendation or a labeled provisional comparison).

Thin and lower-bound inputs are flagged, not hidden: fewer than
``MIN_LONG_RUN_SAMPLES`` measured long runs in a bucket, or any terminated run
contributing to a mean, produces a warning that travels into ``report.json``.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shrlm.experiment.config import CONFIG_PATH, ExperimentConfig, load_config
from shrlm.experiment.evaluation import EVAL_DIR, EVAL_SUMMARY_FILENAME, STAGE_EVAL
from shrlm.experiment.orchestrator import (
    MINING_DIR,
    OPT_DIR,
    SPLIT_ENVIRONMENT,
    SPLIT_LENGTH,
    STAGE_ATTRIBUTION,
    STAGE_MINING,
    STAGE_PROPOSAL,
    STAGE_VALIDATION,
    VALIDATION_DIR,
)
from shrlm.experiment.render import render_markdown
from shrlm.experiment.scenarios import (
    STATUS_PROVISIONAL,
    STATUS_RECOMMENDED,
    Recommendation,
    ReportInputError,
    Scenario,
    build_scenarios,
    gpu_hours,
    recommend,
    validity_gates,
)
from shrlm.experiment.splits import LENGTHS, split_plan
from shrlm.experiment.usage import (
    STAGE_USAGE_FILE,
    UsageTotals,
    aggregate_manifest_usage,
    read_jsonl,
    read_stage_usage,
    read_usage_records,
)
from shrlm.optimization.driver import MANIFEST_FILE

REPORT_FILENAME = "report.json"
REPORT_FORMAT = "shrlm-cost-report/v1"

# Fewer measured long runs than this in one bucket makes its per-run mean a
# thin input: long-run cost has the widest spread, so a one- or two-run mean
# is reported with a warning rather than presented as a measurement.
MIN_LONG_RUN_SAMPLES = 3

SHORT, LONG = LENGTHS

SOURCE_OPTIMIZATION = "optimization"
SOURCE_EVAL = "eval"

LEG_OPTIMIZATION = "optimization"
LEG_EVAL_SHORT = "eval_short"
LEG_EVAL_LONG = "eval_long"

LABEL_POINT = "point"
LABEL_PESSIMISTIC = "pessimistic"

# Presentation order for the measured stage table; a stage outside this tuple
# sorts after it by name (the report renders whatever the sidecar holds).
STAGE_ORDER: tuple[str, ...] = (
    STAGE_MINING,
    STAGE_ATTRIBUTION,
    STAGE_PROPOSAL,
    STAGE_VALIDATION,
    STAGE_EVAL,
)


# ---------------------------------------------------------------------------
# Measured: stages, per-run buckets, disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageMeasurement:
    """One pipeline stage's measured totals across the whole experiment."""

    stage: str
    runs: int
    input_tokens: int
    output_tokens: int
    cost: float
    wall_seconds: float
    cache_hits: int
    resumed_attempts: int
    lower_bound: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "runs": self.runs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "wall_seconds": self.wall_seconds,
            "cache_hits": self.cache_hits,
            "resumed_attempts": self.resumed_attempts,
            "lower_bound": self.lower_bound,
        }


@dataclass(frozen=True)
class RunBucket:
    """Measured runs at one (environment, length), the extrapolation basis.

    Constructed only for buckets holding at least one run, so every mean is
    well defined.
    """

    environment: str
    length: str
    n_runs: int
    input_tokens: int
    output_tokens: int
    cost: float
    wall_seconds: float
    lower_bound: bool
    sources: dict[str, int]

    @property
    def mean_input_tokens(self) -> float:
        return self.input_tokens / self.n_runs

    @property
    def mean_output_tokens(self) -> float:
        return self.output_tokens / self.n_runs

    @property
    def mean_cost(self) -> float:
        return self.cost / self.n_runs

    @property
    def mean_wall_seconds(self) -> float:
        return self.wall_seconds / self.n_runs

    @property
    def thin(self) -> bool:
        return self.length == LONG and self.n_runs < MIN_LONG_RUN_SAMPLES

    def to_payload(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "length": self.length,
            "n_runs": self.n_runs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "wall_seconds": self.wall_seconds,
            "mean_input_tokens": self.mean_input_tokens,
            "mean_output_tokens": self.mean_output_tokens,
            "mean_cost": self.mean_cost,
            "mean_wall_seconds": self.mean_wall_seconds,
            "lower_bound": self.lower_bound,
            "thin": self.thin,
            "sources": dict(sorted(self.sources.items())),
        }


def optimization_manifests(out_dir: Path) -> list[Path]:
    """Every optimization-stage run manifest: mining rounds and validation trees."""
    opt_dir = out_dir / OPT_DIR
    if not opt_dir.exists():
        return []
    paths: list[Path] = []
    for round_path in sorted(opt_dir.iterdir()):
        paths.extend(sorted((round_path / MINING_DIR).rglob(MANIFEST_FILE)))
        paths.extend(sorted((round_path / VALIDATION_DIR).rglob(MANIFEST_FILE)))
    return paths


def read_eval_summary(out_dir: Path) -> dict[str, Any] | None:
    """The persisted ``eval/eval_summary.json``; ``None`` when eval never ran.

    An ``eval/`` directory without its summary is a contradiction (the runner
    writes the summary at the end of every invocation) and is refused rather
    than reported as an unevaluated experiment.
    """
    eval_dir = out_dir / EVAL_DIR
    if not eval_dir.exists():
        return None
    path = eval_dir / EVAL_SUMMARY_FILENAME
    if not path.exists():
        raise ReportInputError(
            f"{eval_dir} exists but {path} does not; an interrupted evaluation cannot be "
            "reported on -- re-invoke run_evaluation to complete its summary."
        )
    return json.loads(path.read_text())


@dataclass
class _BucketAccumulator:
    """Mutable accumulator behind one immutable ``RunBucket``."""

    n_runs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    wall_seconds: float = 0.0
    lower_bound: bool = False
    sources: Counter[str] = field(default_factory=Counter)

    def add(self, source: str, n_runs: int, usage: UsageTotals) -> None:
        self.n_runs += n_runs
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost += usage.cost
        self.wall_seconds += usage.wall_seconds
        self.lower_bound = self.lower_bound or usage.lower_bound
        self.sources[source] += n_runs


def run_buckets(out_dir: Path) -> list[RunBucket]:
    """Measured per-(environment, length) run aggregates, the extrapolation basis.

    Optimization manifests land in the ``(SPLIT_ENVIRONMENT, SPLIT_LENGTH)``
    bucket -- the orchestrator mines and validates that split and no other --
    and evaluation aggregates land in the bucket their ``eval_summary.json``
    entry names.
    """
    accumulators: dict[tuple[str, str], _BucketAccumulator] = {}

    def accumulator(environment: str, length: str) -> _BucketAccumulator:
        return accumulators.setdefault((environment, length), _BucketAccumulator())

    for path in optimization_manifests(out_dir):
        entries = read_jsonl(path)
        if not entries:
            continue
        accumulator(SPLIT_ENVIRONMENT, SPLIT_LENGTH).add(
            SOURCE_OPTIMIZATION, len(entries), aggregate_manifest_usage(entries)
        )

    summary = read_eval_summary(out_dir)
    if summary is not None:
        for condition in summary["conditions"].values():
            for aggregate in condition["test_sets"].values():
                accumulator(str(aggregate["environment"]), str(aggregate["length"])).add(
                    SOURCE_EVAL,
                    int(aggregate["n_runs"]),
                    UsageTotals(
                        input_tokens=int(aggregate["input_tokens"]),
                        output_tokens=int(aggregate["output_tokens"]),
                        cost=float(aggregate["total_cost"]),
                        wall_seconds=float(aggregate["wall_seconds"]),
                        lower_bound=bool(aggregate["usage_lower_bound"]),
                    ),
                )

    return [
        RunBucket(
            environment=environment,
            length=length,
            n_runs=accumulated.n_runs,
            input_tokens=accumulated.input_tokens,
            output_tokens=accumulated.output_tokens,
            cost=accumulated.cost,
            wall_seconds=accumulated.wall_seconds,
            lower_bound=accumulated.lower_bound,
            sources=dict(accumulated.sources),
        )
        for (environment, length), accumulated in sorted(accumulators.items())
        if accumulated.n_runs > 0
    ]


def stage_run_counts(out_dir: Path) -> dict[str, int]:
    """Executed runs per stage: manifest lines for the run-executing stages."""
    counts: Counter[str] = Counter()
    for path in optimization_manifests(out_dir):
        parts = path.relative_to(out_dir).parts
        stage = STAGE_MINING if MINING_DIR in parts else STAGE_VALIDATION
        counts[stage] += len(read_jsonl(path))
    summary = read_eval_summary(out_dir)
    if summary is not None:
        counts[STAGE_EVAL] += sum(
            int(aggregate["n_runs"])
            for condition in summary["conditions"].values()
            for aggregate in condition["test_sets"].values()
        )
    return dict(counts)


def stage_measurements(out_dir: Path) -> list[StageMeasurement]:
    """The measured per-stage table, exactly-once over ``stage_usage.jsonl``."""
    usage_path = out_dir / STAGE_USAGE_FILE
    totals = read_stage_usage(usage_path)
    # Tolerant reader: a torn final line must not kill a report over work
    # already paid for. read_stage_usage already marks such totals as lower
    # bounds; the strict reader stays for splits and run manifests.
    records = read_usage_records(usage_path).records
    stage_of = {str(record["stage_work_id"]): str(record["stage"]) for record in records}
    resumed = Counter(str(record["stage_work_id"]) for record in records if bool(record["resumed"]))
    runs = stage_run_counts(out_dir)

    per_stage: dict[str, UsageTotals] = {}
    resumed_attempts: Counter[str] = Counter()
    for work_id, total in totals.items():
        stage = stage_of[work_id]
        per_stage[stage] = per_stage.get(stage, UsageTotals()) + total
        resumed_attempts[stage] += resumed[work_id]

    def order(stage: str) -> tuple[int, str]:
        return (STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER), stage)

    return [
        StageMeasurement(
            stage=stage,
            runs=runs.get(stage, 0),
            input_tokens=total.input_tokens,
            output_tokens=total.output_tokens,
            cost=total.cost,
            wall_seconds=total.wall_seconds,
            cache_hits=total.cache_hits,
            resumed_attempts=resumed_attempts[stage],
            lower_bound=total.lower_bound,
        )
        for stage, total in sorted(per_stage.items(), key=lambda item: order(item[0]))
    ]


def disk_footprint(out_dir: Path, measured_runs: int, projected_runs: float) -> dict[str, float]:
    """Measured bytes on disk, and the full experiment's projection from them.

    Traces embed the full prompt, so the compute instance's disk is a real
    constraint (plan Risks); the projection is the measured bytes-per-run
    carried to the config-derived total run count. The report artifact itself
    is excluded -- it is this function's own output, and counting it would
    make the footprint (and therefore the report) depend on whether a previous
    report had already been written.
    """
    report_path = out_dir / REPORT_FILENAME
    measured_bytes = sum(
        path.stat().st_size for path in out_dir.rglob("*") if path.is_file() and path != report_path
    )
    bytes_per_run = measured_bytes / measured_runs
    return {
        "measured_bytes": float(measured_bytes),
        "measured_runs": measured_runs,
        "bytes_per_run": bytes_per_run,
        "projected_bytes": bytes_per_run * projected_runs,
    }


# ---------------------------------------------------------------------------
# Config-derived run counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunCounts:
    """The full experiment's run counts, derived from config alone (R6)."""

    runs_per_round: float
    rounds: int
    optimization_runs: float
    eval_conditions: int
    eval_short_runs: float
    eval_long_runs: float

    @property
    def eval_runs(self) -> float:
        return self.eval_short_runs + self.eval_long_runs

    @property
    def total_runs(self) -> float:
        return self.optimization_runs + self.eval_runs

    def to_payload(self) -> dict[str, Any]:
        return {
            "runs_per_round": self.runs_per_round,
            "rounds": self.rounds,
            "optimization_runs": self.optimization_runs,
            "eval_conditions": self.eval_conditions,
            "eval_short_runs": self.eval_short_runs,
            "eval_long_runs": self.eval_long_runs,
            "eval_runs": self.eval_runs,
            "total_runs": self.total_runs,
        }


def eval_test_sizes(config: ExperimentConfig) -> dict[str, int]:
    """Test-role instance count at each length, summed across every environment.

    The experiment evaluates every configured environment -- source
    GraphWalks and target OOLONG-Pairs -- so the eval run count is not the
    source split's ``test_short``/``test_long`` alone; it is each
    environment's ``test`` role from ``splits.split_plan``, summed per
    length.
    """
    plan = split_plan(config)
    return {length: sum(roles[length]["test"] for roles in plan.values()) for length in LENGTHS}


def run_counts(config: ExperimentConfig) -> RunCounts:
    """``m*n_in + v(K+1)(n_in+n_ho) + p_merge*v(n_in+n_ho)`` per round, plus the grid.

    The merge leg is the last term: a promoting round re-evaluates the merged
    harness over both splits, up to ``v(n_in+n_ho)`` further runs, assumed to
    happen in a ``report.p_merge`` fraction of rounds. The evaluation grid uses
    ``report.eval_conditions`` -- the FULL experiment's condition count (B1,
    H1, SH-RLM) -- not however many conditions this scaffold measured -- times
    the test-role instance count summed across every configured environment
    (see ``eval_test_sizes``), not the source split alone.
    """
    loop, splits = config.loop, config.splits
    subjects = splits.n_in + splits.n_ho
    per_round = (
        loop.m * splits.n_in
        + loop.v * (loop.k + 1) * subjects
        + config.report.p_merge * loop.v * subjects
    )
    conditions = config.report.eval_conditions
    repetitions = config.operational.eval_repetitions
    eval_sizes = eval_test_sizes(config)
    return RunCounts(
        runs_per_round=float(per_round),
        rounds=loop.t,
        optimization_runs=float(per_round) * loop.t,
        eval_conditions=conditions,
        eval_short_runs=float(conditions * eval_sizes[SHORT] * repetitions),
        eval_long_runs=float(conditions * eval_sizes[LONG] * repetitions),
    )


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    """One projected leg of the experiment: run-equivalents x measured means."""

    name: str
    context: str
    runs: float
    input_tokens: float
    output_tokens: float
    wall_seconds: float
    basis: str

    @property
    def total_tokens(self) -> float:
        return self.input_tokens + self.output_tokens

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context": self.context,
            "runs": self.runs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "wall_seconds": self.wall_seconds,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class Projection:
    """One whole-experiment projection: the point estimate or the pessimistic bound."""

    label: str
    legs: tuple[Leg, ...]
    long_measured: bool

    @property
    def input_tokens(self) -> float:
        return sum(leg.input_tokens for leg in self.legs)

    @property
    def output_tokens(self) -> float:
        return sum(leg.output_tokens for leg in self.legs)

    @property
    def total_tokens(self) -> float:
        return self.input_tokens + self.output_tokens

    @property
    def wall_seconds(self) -> float:
        return sum(leg.wall_seconds for leg in self.legs)

    def tokens_by_context(self) -> dict[str, float]:
        by_context: dict[str, float] = {length: 0.0 for length in LENGTHS}
        for leg in self.legs:
            by_context[leg.context] += leg.total_tokens
        return by_context

    def to_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "legs": [leg.to_payload() for leg in self.legs],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "wall_seconds": self.wall_seconds,
            "tokens_by_context": self.tokens_by_context(),
            "long_measured": self.long_measured,
        }


def pooled_bucket(buckets: Sequence[RunBucket], length: str) -> RunBucket | None:
    """Every measured bucket at one length, pooled run-weighted; ``None`` if empty."""
    matching = [bucket for bucket in buckets if bucket.length == length]
    if not matching:
        return None
    sources: Counter[str] = Counter()
    for bucket in matching:
        sources.update(bucket.sources)
    return RunBucket(
        environment="+".join(bucket.environment for bucket in matching),
        length=length,
        n_runs=sum(bucket.n_runs for bucket in matching),
        input_tokens=sum(bucket.input_tokens for bucket in matching),
        output_tokens=sum(bucket.output_tokens for bucket in matching),
        cost=sum(bucket.cost for bucket in matching),
        wall_seconds=sum(bucket.wall_seconds for bucket in matching),
        lower_bound=any(bucket.lower_bound for bucket in matching),
        sources=dict(sources),
    )


def leg_from(name: str, context: str, runs: float, basis: RunBucket) -> Leg:
    return Leg(
        name=name,
        context=context,
        runs=runs,
        input_tokens=runs * basis.mean_input_tokens,
        output_tokens=runs * basis.mean_output_tokens,
        wall_seconds=runs * basis.mean_wall_seconds,
        basis=f"{basis.environment}/{basis.length} ({basis.n_runs} run(s))",
    )


def project(
    counts: RunCounts,
    optimization_basis: RunBucket,
    short_basis: RunBucket,
    long_basis: RunBucket | None,
    *,
    label: str,
    optimization_multiplier: float,
) -> Projection:
    """One projection: config-derived run counts x measured per-length means.

    ``optimization_multiplier`` is the number of rounds for the point estimate
    and the compounding cost-band sum for the pessimistic bound. A missing
    ``long_basis`` drops the long leg entirely -- the projection is then
    short-only and says so, rather than projecting long runs at zero cost.
    """
    legs = [
        leg_from(
            LEG_OPTIMIZATION,
            SHORT,
            counts.runs_per_round * optimization_multiplier,
            optimization_basis,
        ),
        leg_from(LEG_EVAL_SHORT, SHORT, counts.eval_short_runs, short_basis),
    ]
    if long_basis is not None:
        legs.append(leg_from(LEG_EVAL_LONG, LONG, counts.eval_long_runs, long_basis))
    return Projection(label=label, legs=tuple(legs), long_measured=long_basis is not None)


def pessimistic_multiplier(config: ExperimentConfig) -> float:
    """``sum(ceiling**i for i in range(T))``: the cost band compounding over rounds."""
    ceiling = config.promotion.cost_band[1]
    return float(sum(ceiling**index for index in range(config.loop.t)))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostReport:
    """Everything the decision needs, as one persistable, renderable object."""

    out_dir: Path
    profile: str
    stages: list[StageMeasurement]
    buckets: list[RunBucket]
    run_counts: RunCounts
    point: Projection
    pessimistic: Projection
    scenarios: list[Scenario]
    recommendation: Recommendation
    warnings: list[str]
    disk: dict[str, float]

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": REPORT_FORMAT,
            "profile": self.profile,
            "measured": {
                "stages": [stage.to_payload() for stage in self.stages],
                "buckets": [bucket.to_payload() for bucket in self.buckets],
                "disk": self.disk,
            },
            "run_counts": self.run_counts.to_payload(),
            "extrapolation": {
                "point": self.point.to_payload(),
                "pessimistic": self.pessimistic.to_payload(),
                "pessimistic_optimization_multiplier": next(
                    leg.runs for leg in self.pessimistic.legs if leg.name == LEG_OPTIMIZATION
                )
                / self.run_counts.runs_per_round,
            },
            "scenarios": [scenario.to_payload() for scenario in self.scenarios],
            "recommendation": self.recommendation.to_payload(),
            "warnings": self.warnings,
        }


def collect_warnings(
    config: ExperimentConfig,
    out_dir: Path,
    buckets: Sequence[RunBucket],
    counts: RunCounts,
    point: Projection,
    optimization_basis: RunBucket,
) -> list[str]:
    """Every caveat the extrapolation inputs carry, named rather than hidden."""
    warnings: list[str] = []
    if not point.long_measured:
        warnings.append(
            f"long unmeasured: no long-context runs are recorded, so the projection is "
            f"short-only and {counts.eval_long_runs:.0f} long evaluation run(s) are "
            "unprojected (their cost is not zero -- it is unknown)"
        )
    for bucket in buckets:
        if bucket.lower_bound:
            warnings.append(
                f"lower-bound usage in {bucket.environment}/{bucket.length}: terminated "
                "run(s) contributed, so its per-run mean understates the true cost"
            )
        if bucket.thin:
            warnings.append(
                f"thin sample in {bucket.environment}/{bucket.length}: {bucket.n_runs} "
                f"measured run(s) < {MIN_LONG_RUN_SAMPLES}, so its per-run mean carries "
                "wide uncertainty"
            )
    if (optimization_basis.environment, optimization_basis.length) != (
        SPLIT_ENVIRONMENT,
        SPLIT_LENGTH,
    ):
        warnings.append(
            f"no measured {SPLIT_ENVIRONMENT}/{SPLIT_LENGTH} optimization runs: the "
            f"optimization leg falls back to the pooled {SHORT} basis "
            f"({optimization_basis.environment})"
        )
    summary = read_eval_summary(out_dir)
    measured_conditions = len(summary["conditions"]) if summary is not None else 0
    if measured_conditions != counts.eval_conditions:
        warnings.append(
            f"measured evaluation covers {measured_conditions} condition(s) while the "
            f"full-experiment grid assumes {counts.eval_conditions} "
            "(report.eval_conditions); the projection uses the configured grid"
        )
    return warnings


def build_report(
    config: ExperimentConfig,
    out_dir: Path | str,
    *,
    accept_quantization: bool = False,
) -> CostReport:
    """Read one experiment directory into the full cost/time report (R6).

    Args:
        config: The loaded experiment configuration; every run count, price,
            and GPU profile comes from it.
        out_dir: The experiment directory the orchestrator and evaluation
            runner wrote.
        accept_quantization: Allow scenarios declaring ``changes_numerics`` to
            be recommended. They are always reported; this flag only makes
            them candidates.

    Returns:
        The ``CostReport``: measured stage table and per-run buckets, the
        config-derived run counts, the point and pessimistic projections, every
        priced scenario, and the recommendation (or provisional comparison).

    Raises:
        ReportInputError: The directory records no runs at all, holds an
            evaluation directory without its summary, or a GPU profile is
            missing a context's throughput.
    """
    out = Path(out_dir)
    stages = stage_measurements(out)
    buckets = run_buckets(out)
    if not buckets:
        raise ReportInputError(
            f"{out} records no runs; a cost report extrapolates from measured per-run "
            "means, so there is nothing to report on."
        )

    short_basis = pooled_bucket(buckets, SHORT)
    if short_basis is None:
        raise ReportInputError(
            f"{out} records no {SHORT}-context runs; the optimization loop and the short "
            "evaluation grid both extrapolate from short per-run means."
        )
    long_basis = pooled_bucket(buckets, LONG)
    optimization_basis = next(
        (
            bucket
            for bucket in buckets
            if (bucket.environment, bucket.length) == (SPLIT_ENVIRONMENT, SPLIT_LENGTH)
        ),
        short_basis,
    )

    counts = run_counts(config)
    point = project(
        counts,
        optimization_basis,
        short_basis,
        long_basis,
        label=LABEL_POINT,
        optimization_multiplier=float(counts.rounds),
    )
    pessimistic = project(
        counts,
        optimization_basis,
        short_basis,
        long_basis,
        label=LABEL_PESSIMISTIC,
        optimization_multiplier=pessimistic_multiplier(config),
    )
    scenarios = build_scenarios(config, point, pessimistic)
    gates = validity_gates(config, buckets)
    return CostReport(
        out_dir=out,
        profile=config.profile,
        stages=stages,
        buckets=buckets,
        run_counts=counts,
        point=point,
        pessimistic=pessimistic,
        scenarios=scenarios,
        recommendation=recommend(scenarios, gates, accept_quantization=accept_quantization),
        warnings=collect_warnings(config, out, buckets, counts, point, optimization_basis),
        disk=disk_footprint(out, sum(bucket.n_runs for bucket in buckets), counts.total_runs),
    )


def write_report(report: CostReport) -> Path:
    """Persist ``report.json`` in the experiment directory; returns its path.

    The payload is a pure function of the persisted measurements, so rewriting
    an unchanged experiment's report produces byte-identical bytes.
    """
    path = report.out_dir / REPORT_FILENAME
    path.write_text(
        json.dumps(report.to_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


# ---------------------------------------------------------------------------
# The report command
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """The report command: render markdown on stdout and write ``report.json``."""
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.report",
        description="Cost/time report and full-experiment API-vs-GPU extrapolation (R6).",
    )
    parser.add_argument("out_dir", help="the experiment directory to report on")
    parser.add_argument("--profile", default="full", help="config profile (full | smoke)")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="path to the experiment TOML")
    parser.add_argument(
        "--accept-quantization",
        action="store_true",
        help="allow scenarios declaring changes_numerics (the model numerics change)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.profile, args.config)
    report = build_report(config, args.out_dir, accept_quantization=args.accept_quantization)
    path = write_report(report)
    sys.stdout.write(render_markdown(report))
    sys.stdout.write(f"\nWrote {path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "MIN_LONG_RUN_SAMPLES",
    "REPORT_FILENAME",
    "REPORT_FORMAT",
    "STATUS_PROVISIONAL",
    "STATUS_RECOMMENDED",
    "CostReport",
    "Leg",
    "Projection",
    "Recommendation",
    "ReportInputError",
    "RunBucket",
    "RunCounts",
    "Scenario",
    "StageMeasurement",
    "build_report",
    "build_scenarios",
    "eval_test_sizes",
    "gpu_hours",
    "main",
    "project",
    "recommend",
    "render_markdown",
    "run_buckets",
    "run_counts",
    "stage_measurements",
    "validity_gates",
    "write_report",
]
