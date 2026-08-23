"""Tests for U7's cost/time report and full-experiment extrapolation (R6).

Every number below is hand-computable from the checked-in fixture experiment
(``tests/experiment/fixtures/report_experiment``) and the shipped
``configs/experiment.toml``. The fixture measures:

    optimization (graphwalks/short)  4 runs  x 1000 in / 100 out / 10 s
    eval graphwalks_short   2 conditions x 2 runs x 1000 in / 100 out
    eval oolong_pairs_short 2 conditions x 2 runs x 1000 in / 100 out
    eval graphwalks_long    2 conditions x 3 runs x 10000 in / 500 out
    eval oolong_pairs_long  2 conditions x 3 runs x 10000 in / 500 out

so every short bucket has a per-run mean of 1000 in / 100 out and every long
bucket 10000 in / 500 out, whichever way the buckets are pooled.

Full-profile config arithmetic (m=2, n_in=24, v=4, K=4, n_ho=40, p_merge=0.5,
T=15, eval_conditions=3, eval_repetitions=3). The evaluation grid covers both
configured environments: source GraphWalks (test_short=40, test_long=150) and
target OOLONG-Pairs (n_short=40, n_long=40), so each length's eval run count
sums both environments' test-role size, not GraphWalks alone (see
``shrlm.experiment.report.eval_test_sizes``):

    runs/round = 2*24 + 4*5*64 + 0.5*4*64 = 48 + 1280 + 128 = 1456
    optimization runs = 1456 * 15 = 21840
    eval short size = 40 (graphwalks) + 40 (oolong_pairs) = 80
    eval long size  = 150 (graphwalks) + 40 (oolong_pairs) = 190
    eval runs = 3 * (80 + 190) * 3 = 2430  (720 short + 1,710 long)

The long leg is smaller than the 2,700 runs quoted in ``paper/proposal.tex``
(Section: Feasibility and Cost): that figure assumed 150 OOLONG-Pairs long
instances, but the pinned ``trec_coarse`` validation split ships only 2 distinct
context windows at 262,144 tokens, so 2 x 20 task_ids = 40 is the entire
available pool (see ``DATASET_INVENTORY.md``) and ``n_long`` is configured at
that ceiling.
"""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.analysis_io import ANALYSIS_DIR
from shrlm.experiment.config import ExperimentConfig, GpuScenario, load_config
from shrlm.experiment.orchestrator import (
    MINING_DIR,
    ROUND_MARKER_FILENAME,
    RoundOutcome,
    experiment_round_dir,
)
from shrlm.experiment.report import (
    MIN_LONG_RUN_SAMPLES,
    REPORT_FILENAME,
    STATUS_PROVISIONAL,
    STATUS_RECOMMENDED,
    build_report,
    optimization_manifests,
    render_markdown,
    run_counts,
    stage_run_counts,
    write_report,
)
from shrlm.experiment.rounds import discover_rounds
from shrlm.optimization.bundle import round_dir
from shrlm.optimization.driver import MANIFEST_FILE

FIXTURE = Path(__file__).parent / "fixtures" / "report_experiment"

# Hand-computed from the fixture + full-profile config (see module docstring).
RUNS_PER_ROUND = 1456.0
OPTIMIZATION_RUNS = 21840.0
EVAL_SHORT_RUNS = 720.0
EVAL_LONG_RUNS = 1710.0

SHORT_MEAN_INPUT = 1000.0
SHORT_MEAN_OUTPUT = 100.0
LONG_MEAN_INPUT = 10000.0
LONG_MEAN_OUTPUT = 500.0

POINT_INPUT_TOKENS = (
    OPTIMIZATION_RUNS * SHORT_MEAN_INPUT
    + EVAL_SHORT_RUNS * SHORT_MEAN_INPUT
    + EVAL_LONG_RUNS * LONG_MEAN_INPUT
)  # 49_560_000
POINT_OUTPUT_TOKENS = (
    OPTIMIZATION_RUNS * SHORT_MEAN_OUTPUT
    + EVAL_SHORT_RUNS * SHORT_MEAN_OUTPUT
    + EVAL_LONG_RUNS * LONG_MEAN_OUTPUT
)  # 3_606_000


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config("full")


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """A writable copy of the checked-in fixture experiment."""
    out_dir = tmp_path / "experiment"
    shutil.copytree(FIXTURE, out_dir)
    return out_dir


def read_summary(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "eval" / "eval_summary.json").read_text())


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "eval" / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def edit_sets(out_dir: Path, set_id: str, **changes: Any) -> None:
    """Apply ``changes`` to one test set's aggregate in every condition."""
    summary = read_summary(out_dir)
    for condition in summary["conditions"].values():
        condition["test_sets"][set_id].update(changes)
    write_summary(out_dir, summary)


def drop_sets(out_dir: Path, length: str) -> None:
    """Remove every test set of one length from the evaluation summary."""
    summary = read_summary(out_dir)
    for condition in summary["conditions"].values():
        condition["test_sets"] = {
            set_id: aggregate
            for set_id, aggregate in condition["test_sets"].items()
            if aggregate["length"] != length
        }
    write_summary(out_dir, summary)


def write_no_op_round(out_dir: Path, round_index: int, *, n_runs: int = 1) -> Path:
    """Add a completed round that promoted nothing: marker and mined runs, no ledger.

    Exactly the shape the review flagged as invisible: ``_execute_round``
    persists ``round.json`` for every round it finishes, including one whose
    proposal stage produced no candidate -- so validation never ran and no
    ``promotions.jsonl`` exists. Its manifest lines are copies of the fixture's
    own, so the round's runs land in the same short bucket with the same means.
    """
    template = json.loads(
        (round_dir(experiment_round_dir(out_dir, 1) / MINING_DIR, 1) / MANIFEST_FILE)
        .read_text()
        .splitlines()[0]
    )
    round_path = experiment_round_dir(out_dir, round_index)
    mining_round_path = round_dir(round_path / MINING_DIR, round_index)
    mining_round_path.mkdir(parents=True)
    (mining_round_path / MANIFEST_FILE).write_text(
        "".join(
            json.dumps({**template, "instance_id": f"noop-{index}", "run_id": f"noop-{index}__a01"})
            + "\n"
            for index in range(n_runs)
        )
    )
    outcome = RoundOutcome(
        round_index=round_index,
        promoted=False,
        promoted_harness_hash=None,
        has_ledger=False,
    )
    (round_path / ROUND_MARKER_FILENAME).write_text(json.dumps(outcome.to_payload()) + "\n")
    return round_path


def scenario_by_name(report: Any, name: str) -> Any:
    matches = [scenario for scenario in report.scenarios if scenario.name == name]
    assert len(matches) == 1, f"{name} not found among {[s.name for s in report.scenarios]}"
    return matches[0]


def gate_names(report: Any) -> list[str]:
    return [gate["gate"] for gate in report.recommendation.failing_gates]


# ---------------------------------------------------------------------------
# Round discovery: the report reads the shared inventory (R1, KTD1)
# ---------------------------------------------------------------------------


class TestRoundDiscovery:
    """What the report counts is what ``rounds.discover_rounds`` reports.

    The equality assertions below are the migration's contract: they were
    written against the report's own ``opt/`` walk and must read identically
    once the walk is gone, because the swap changes how rounds are found and
    nothing about what the report says.
    """

    def test_optimization_manifests_are_the_loop_written_round_manifests(self, experiment):
        assert [
            path.relative_to(experiment).as_posix() for path in optimization_manifests(experiment)
        ] == [
            "opt/round_01/mining/round_01/runs.jsonl",
            "opt/round_01/validation/round_01/incumbent/held_in/round_00/runs.jsonl",
        ]

    def test_manifests_come_from_the_rounds_discovery_reports(self, experiment):
        rounds = discover_rounds(experiment).rounds
        manifests = optimization_manifests(experiment)

        assert [record.round_index for record in rounds] == [1]
        assert set(manifests) == {
            path
            for record in rounds
            for root in (record.mining_round_path, record.validation_round_path)
            for path in root.rglob(MANIFEST_FILE)
        }

    def test_stage_run_counts_separate_mining_from_validation(self, experiment):
        assert stage_run_counts(experiment) == {"mining": 2, "validation": 2, "eval": 20}

    def test_completed_no_op_round_is_counted(self, config, experiment):
        """A round that promoted nothing is a real outcome, not missing data.

        Discovery reports it (marker present, ``has_ledger`` false), so its
        mined runs reach the report's stage table rather than depending on
        whether a private walk happened to descend into the round.
        """
        write_no_op_round(experiment, 2, n_runs=2)

        inventory = discover_rounds(experiment)
        no_op = inventory.rounds[-1]
        assert [record.round_index for record in inventory.rounds] == [1, 2]
        assert (no_op.round_complete, no_op.has_ledger) == (True, False)

        assert no_op.mining_round_path / MANIFEST_FILE in optimization_manifests(experiment)
        assert stage_run_counts(experiment)["mining"] == 4
        report = build_report(config, experiment)
        assert {stage.stage: stage.runs for stage in report.stages}["mining"] == 4

    def test_an_entry_that_is_not_a_round_contributes_no_runs(self, config, experiment):
        """Only what the loop's own naming would have produced is a round.

        The report's former walk listed every entry under ``opt/`` and
        descended into each, so anything parked there carrying a driver tree
        -- a partial copy, a scratch directory -- was counted as measured
        optimization runs. Discovery names rounds the way the loop does, so it
        cannot.
        """
        round_path = write_no_op_round(experiment, 2, n_runs=2)
        stray = experiment / "opt" / "round_02.bak"
        shutil.copytree(round_path, stray)
        assert (round_dir(stray / MINING_DIR, 2) / MANIFEST_FILE).exists()

        assert not any(path.is_relative_to(stray) for path in optimization_manifests(experiment))
        assert stage_run_counts(experiment)["mining"] == 4  # 2 fixture + 2 no-op, not 6
        assert {stage.stage: stage.runs for stage in build_report(config, experiment).stages}[
            "mining"
        ] == 4


# ---------------------------------------------------------------------------
# Measured section
# ---------------------------------------------------------------------------


class TestMeasured:
    def test_stage_table_covers_every_metered_stage(self, config, experiment):
        report = build_report(config, experiment)
        stages = {stage.stage: stage for stage in report.stages}

        assert sorted(stages) == ["attribution", "eval", "mining", "proposal", "validation"]
        assert stages["mining"].runs == 2
        assert stages["validation"].runs == 2
        assert stages["eval"].runs == 20  # 2 conditions x (2 + 2 + 3 + 3)
        assert stages["attribution"].runs == 0
        assert stages["mining"].input_tokens == 2000
        assert stages["eval"].input_tokens == 128000
        assert stages["proposal"].cache_hits == 1

    def test_per_run_means_split_by_environment_and_length(self, config, experiment):
        report = build_report(config, experiment)
        buckets = {(b.environment, b.length): b for b in report.buckets}

        assert sorted(buckets) == [
            ("graphwalks", "long"),
            ("graphwalks", "short"),
            ("oolong_pairs", "long"),
            ("oolong_pairs", "short"),
        ]
        graphwalks_short = buckets[("graphwalks", "short")]
        assert graphwalks_short.n_runs == 8  # 4 optimization + 4 evaluation
        assert graphwalks_short.mean_input_tokens == SHORT_MEAN_INPUT
        assert graphwalks_short.mean_output_tokens == SHORT_MEAN_OUTPUT
        assert graphwalks_short.sources == {"optimization": 4, "eval": 4}

        graphwalks_long = buckets[("graphwalks", "long")]
        assert graphwalks_long.n_runs == 6
        assert graphwalks_long.mean_input_tokens == LONG_MEAN_INPUT
        assert graphwalks_long.mean_output_tokens == LONG_MEAN_OUTPUT

    def test_lower_bound_flag_propagates_from_a_manifest_line(self, config, experiment):
        manifest = next(experiment.glob("opt/round_*/mining/round_*/runs.jsonl"))
        lines = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        lines[0]["usage_lower_bound"] = True
        manifest.write_text("".join(json.dumps(line) + "\n" for line in lines))

        report = build_report(config, experiment)
        buckets = {(b.environment, b.length): b for b in report.buckets}

        assert buckets[("graphwalks", "short")].lower_bound is True
        assert any("lower-bound" in warning for warning in report.warnings)

    def test_disk_footprint_projects_from_measured_bytes_per_run(self, config, experiment):
        report = build_report(config, experiment)
        disk = report.disk

        assert disk["measured_bytes"] > 0
        assert disk["measured_runs"] == 24
        assert disk["bytes_per_run"] == pytest.approx(disk["measured_bytes"] / 24)
        assert disk["projected_bytes"] == pytest.approx(
            disk["bytes_per_run"] * report.run_counts.total_runs
        )

    def test_analysis_snapshots_do_not_inflate_the_footprint(self, config, experiment):
        """The post-round hook writes one snapshot per executed round, so counting
        them would make bytes-per-run climb with how often the analyses ran rather
        than with the experiment's own data -- and this figure feeds the projection
        the disk-constraint argument rests on."""
        before = build_report(config, experiment).disk

        snapshot_dir = experiment / ANALYSIS_DIR / "20260819T000000Z"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "surface_activity.csv").write_text("x" * 100_000)

        after = build_report(config, experiment).disk

        assert after["measured_bytes"] == before["measured_bytes"]
        assert after["projected_bytes"] == pytest.approx(before["projected_bytes"])


# ---------------------------------------------------------------------------
# Config-derived run counts and the extrapolation
# ---------------------------------------------------------------------------


class TestRunCounts:
    def test_runs_per_round_follows_the_configured_formula(self, config, experiment):
        counts = build_report(config, experiment).run_counts

        loop, splits, report_cfg = config.loop, config.splits, config.report
        expected = (
            loop.m * splits.n_in
            + loop.v * (loop.k + 1) * (splits.n_in + splits.n_ho)
            + report_cfg.p_merge * loop.v * (splits.n_in + splits.n_ho)
        )
        assert counts.runs_per_round == expected == RUNS_PER_ROUND
        assert counts.rounds == loop.t
        assert counts.optimization_runs == OPTIMIZATION_RUNS

    def test_shipped_config_eval_projection_matches_the_paper(self, config):
        """Pins the full-experiment evaluation projection so an omitted
        environment cannot silently regress it. The eval grid covers BOTH
        configured environments -- source GraphWalks and target OOLONG-Pairs
        -- not the source split alone: 3 conditions x 2 environments x
        (40 short + 150 long graphwalks, 40 short + 40 long oolong_pairs)
        x 3 repetitions = 720 short and 1,710 long
        runs, matching paper/proposal.tex's Feasibility and Cost section.
        """
        counts = run_counts(config)

        assert counts.eval_short_runs == 720.0
        assert counts.eval_long_runs == 1710.0

    def test_eval_grid_uses_configured_conditions_not_measured_conditions(self, config, experiment):
        report = build_report(config, experiment)

        measured_conditions = len(read_summary(experiment)["conditions"])
        assert measured_conditions == 2  # the scaffold evaluates b1 + sh_rlm
        assert config.report.eval_conditions == 3  # B1, H1, SH-RLM
        assert report.run_counts.eval_conditions == 3
        assert report.run_counts.eval_short_runs == EVAL_SHORT_RUNS
        assert report.run_counts.eval_long_runs == EVAL_LONG_RUNS
        assert report.run_counts.eval_runs == EVAL_SHORT_RUNS + EVAL_LONG_RUNS
        assert any("condition" in warning for warning in report.warnings)

    def test_point_estimate_is_run_counts_times_measured_per_run_means(self, config, experiment):
        point = build_report(config, experiment).point

        legs = {leg.name: leg for leg in point.legs}
        assert legs["optimization"].input_tokens == OPTIMIZATION_RUNS * SHORT_MEAN_INPUT
        assert legs["eval_short"].input_tokens == EVAL_SHORT_RUNS * SHORT_MEAN_INPUT
        assert legs["eval_long"].input_tokens == EVAL_LONG_RUNS * LONG_MEAN_INPUT
        assert point.input_tokens == POINT_INPUT_TOKENS
        assert point.output_tokens == POINT_OUTPUT_TOKENS
        assert point.long_measured is True


class TestApiScenarios:
    def test_promo_and_list_tiers_are_both_reported(self, config, experiment):
        report = build_report(config, experiment)

        promo = scenario_by_name(report, "api_promo")
        listed = scenario_by_name(report, "api_list")
        expected_promo = (
            POINT_INPUT_TOKENS / 1e6 * config.pricing.promo.input_per_million
            + POINT_OUTPUT_TOKENS / 1e6 * config.pricing.promo.output_per_million
        )
        expected_list = (
            POINT_INPUT_TOKENS / 1e6 * config.pricing.list_price.input_per_million
            + POINT_OUTPUT_TOKENS / 1e6 * config.pricing.list_price.output_per_million
        )
        assert promo.usd_point == pytest.approx(expected_promo)
        assert listed.usd_point == pytest.approx(expected_list)
        # Azure publishes a single Kimi-K2.5 tier (KTD4): the promo slot is
        # repurposed to equal the list price, so the two scenarios coincide.
        assert promo.usd_point == listed.usd_point

    def test_round_extrapolation_identity(self, config, experiment):
        """(m*n_in + v(K+1)(n_in+n_ho) + p_merge*v(n_in+n_ho)) * mean short tokens * price."""
        report = build_report(config, experiment)

        optimization = next(leg for leg in report.point.legs if leg.name == "optimization")
        per_round_input = RUNS_PER_ROUND * SHORT_MEAN_INPUT
        assert optimization.input_tokens == per_round_input * config.loop.t
        optimization_usd = (
            optimization.input_tokens / 1e6 * config.pricing.promo.input_per_million
            + optimization.output_tokens / 1e6 * config.pricing.promo.output_per_million
        )
        assert optimization_usd == pytest.approx(
            RUNS_PER_ROUND
            * config.loop.t
            * (
                SHORT_MEAN_INPUT / 1e6 * config.pricing.promo.input_per_million
                + SHORT_MEAN_OUTPUT / 1e6 * config.pricing.promo.output_per_million
            )
        )


class TestGpuScenarios:
    def test_tokens_over_throughput_times_rate(self, config, experiment):
        report = build_report(config, experiment)
        scenario = scenario_by_name(report, "h100_sxm_rented")
        profile = next(s for s in config.gpu_scenarios if s.name == "h100_sxm_rented")

        short_tokens = OPTIMIZATION_RUNS * (
            SHORT_MEAN_INPUT + SHORT_MEAN_OUTPUT
        ) + EVAL_SHORT_RUNS * (SHORT_MEAN_INPUT + SHORT_MEAN_OUTPUT)
        long_tokens = EVAL_LONG_RUNS * (LONG_MEAN_INPUT + LONG_MEAN_OUTPUT)
        hours = (
            short_tokens / profile.throughput_tokens_per_second["short"]
            + long_tokens / profile.throughput_tokens_per_second["long"]
        ) / 3600.0

        assert hours == pytest.approx(9.997222222222222)
        assert scenario.detail["gpu_hours_point"] == pytest.approx(hours)
        assert scenario.usd_point == pytest.approx(hours * profile.hourly_rate_usd)
        assert scenario.detail["usd_point_low"] == pytest.approx(
            hours * profile.sensitivity_range[0]
        )
        assert scenario.detail["usd_point_high"] == pytest.approx(
            hours * profile.sensitivity_range[1]
        )
        assert scenario.provenance == profile.provenance

    def test_unvalidated_profiles_are_scenario_only(self, config, experiment):
        report = build_report(config, experiment)

        int4 = scenario_by_name(report, "1x_a100_int4")
        tp2 = scenario_by_name(report, "2x_a100_tp2_rented")
        assert int4.eligible is False
        assert tp2.eligible is False
        assert "unvalidated" in int4.ineligible_reason
        assert int4.changes_numerics is True
        assert scenario_by_name(report, "h100_sxm_rented").eligible is True

    def test_eligibility_reads_the_declared_booleans_not_the_prose(self, config, experiment):
        """The gates are typed fields, so editing a justification cannot move them."""
        profile = next(s for s in config.gpu_scenarios if s.name == "h100_sxm_rented")
        prose_only = replace(
            profile,
            provenance="Unvalidated estimate (prose only)",
            precision_note="INT4 quantized weights change model numerics (prose only)",
        )
        report = build_report(replace(config, gpu_scenarios=(prose_only,)), experiment)

        scenario = scenario_by_name(report, "h100_sxm_rented")
        assert scenario.eligible is True
        assert scenario.changes_numerics is False

    def test_quantized_scenario_is_deprioritized_unless_accepted(self, config, experiment):
        cheap_quantized = GpuScenario(
            name="validated_int4",
            hourly_rate_usd=0.01,
            provenance="Measured on rented hardware, 2026-08",
            provenance_validated=True,
            sensitivity_range=(0.01, 0.02),
            precision_note="INT4 weights change model numerics",
            changes_numerics=True,
            throughput_tokens_per_second={"short": 5000.0, "long": 2000.0},
        )
        quantized_config = replace(config, gpu_scenarios=(cheap_quantized,))

        default = build_report(quantized_config, experiment)
        assert scenario_by_name(default, "validated_int4").changes_numerics is True
        assert default.recommendation.scenario != "validated_int4"

        accepted = build_report(quantized_config, experiment, accept_quantization=True)
        assert accepted.recommendation.scenario == "validated_int4"


class TestPessimisticScenario:
    def test_cost_band_ceiling_compounds_across_optimization_rounds(self, config, experiment):
        report = build_report(config, experiment)

        ceiling = config.promotion.cost_band[1]
        factor = sum(ceiling**index for index in range(config.loop.t))
        # Geometric in the ceiling, so this value is highly sensitive to it:
        # the shipped 1.25 gives ~109.7 over 15 rounds, where 2.0 gave 32,767.
        assert factor == pytest.approx(sum(1.25**i for i in range(15)))

        point_optimization = next(leg for leg in report.point.legs if leg.name == "optimization")
        pessimistic_optimization = next(
            leg for leg in report.pessimistic.legs if leg.name == "optimization"
        )
        assert pessimistic_optimization.input_tokens == pytest.approx(
            RUNS_PER_ROUND * factor * SHORT_MEAN_INPUT
        )
        assert pessimistic_optimization.input_tokens > point_optimization.input_tokens

        # The evaluation legs are the point estimate's: the ceiling bounds
        # optimization-round drift, not the frozen harness's evaluation cost.
        point_eval = next(leg for leg in report.point.legs if leg.name == "eval_long")
        pessimistic_eval = next(leg for leg in report.pessimistic.legs if leg.name == "eval_long")
        assert pessimistic_eval.input_tokens == point_eval.input_tokens

    def test_reported_distinctly_from_the_point_estimate(self, config, experiment):
        report = build_report(config, experiment)
        promo = scenario_by_name(report, "api_promo")

        assert promo.usd_pessimistic > promo.usd_point
        payload = report.to_payload()
        assert payload["extrapolation"]["point"]["label"] == "point"
        assert payload["extrapolation"]["pessimistic"]["label"] == "pessimistic"
        assert payload["scenarios"][0]["usd_pessimistic"] > 0.0


# ---------------------------------------------------------------------------
# Recommendation policy
# ---------------------------------------------------------------------------


class TestRecommendationPolicy:
    def test_clean_fixture_recommends_the_cheapest_valid_scenario(self, config, experiment):
        report = build_report(config, experiment)

        assert report.recommendation.status == STATUS_RECOMMENDED
        assert report.recommendation.failing_gates == []
        eligible = [s for s in report.scenarios if s.eligible and not s.changes_numerics]
        assert report.recommendation.scenario == min(eligible, key=lambda s: s.usd_point).name
        # At Azure Kimi-K2.5 rates ($0.60/$3.00 per 1M, single-tier) the rented
        # H100 undercuts both API tiers on this fixture's token volumes.
        assert report.recommendation.scenario == "h100_sxm_rented"

    def test_lower_bound_long_mean_yields_a_provisional_label(self, config, experiment):
        edit_sets(experiment, "graphwalks_long", usage_lower_bound=True)

        report = build_report(config, experiment)

        assert report.recommendation.status == STATUS_PROVISIONAL
        assert report.recommendation.scenario is None
        assert gate_names(report) == ["long_run_lower_bound"]
        assert "graphwalks/long" in report.recommendation.failing_gates[0]["detail"]

    def test_missing_long_coverage_is_named_and_the_estimate_is_short_only(
        self, config, experiment
    ):
        drop_sets(experiment, "long")

        report = build_report(config, experiment)

        assert report.point.long_measured is False
        assert report.point.input_tokens == (
            OPTIMIZATION_RUNS * SHORT_MEAN_INPUT + EVAL_SHORT_RUNS * SHORT_MEAN_INPUT
        )
        assert any("long unmeasured" in warning for warning in report.warnings)
        assert report.recommendation.status == STATUS_PROVISIONAL
        assert gate_names(report) == ["long_coverage", "long_coverage"]
        details = " ".join(gate["detail"] for gate in report.recommendation.failing_gates)
        assert "graphwalks" in details
        assert "oolong_pairs" in details

    def test_one_environment_missing_long_coverage_still_gates(self, config, experiment):
        summary = read_summary(experiment)
        for condition in summary["conditions"].values():
            condition["test_sets"].pop("oolong_pairs_long")
        write_summary(experiment, summary)

        report = build_report(config, experiment)

        assert gate_names(report) == ["long_coverage"]
        assert "oolong_pairs" in report.recommendation.failing_gates[0]["detail"]
        assert report.point.long_measured is True  # graphwalks still measures long

    def test_thin_long_sample_is_flagged_but_does_not_gate(self, config, experiment):
        edit_sets(experiment, "graphwalks_long", n_runs=1)

        report = build_report(config, experiment)

        assert MIN_LONG_RUN_SAMPLES == 3
        assert any("thin" in warning for warning in report.warnings)
        assert report.recommendation.status == STATUS_RECOMMENDED


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_report_json_round_trips_and_markdown_renders(self, config, experiment):
        report = build_report(config, experiment)
        path = write_report(report)

        payload = json.loads(path.read_text())
        assert path.name == REPORT_FILENAME
        assert payload["measured"]["stages"][0]["stage"] in {
            "mining",
            "attribution",
            "proposal",
            "validation",
            "eval",
        }
        assert payload["recommendation"]["status"] == STATUS_RECOMMENDED
        assert payload["run_counts"]["runs_per_round"] == RUNS_PER_ROUND

        markdown = render_markdown(report)
        assert "## Measured" in markdown
        assert "## Extrapolation" in markdown
        assert "## Recommendation" in markdown
        assert "api_promo" in markdown

    def test_writing_twice_is_idempotent(self, config, experiment):
        report = build_report(config, experiment)
        first = write_report(report).read_text()
        second = write_report(build_report(config, experiment)).read_text()

        assert first == second
