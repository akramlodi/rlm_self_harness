"""Tests for U4 stage-level usage metering (R4, R5).

The two KTD4 capture paths are exercised separately: run-executing stages
aggregate per-run manifest keys (``aggregate_manifest_usage`` + ``add``),
while caller-held proposer/attributor clients are snapshot-diffed
(``watch``). The exactly-once rule is pinned directly: per ``stage_work_id``
the latest record per ``attempt_index`` wins, distinct attempts sum.
"""

import json
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
from shrlm.experiment.usage import (
    StageMeter,
    UsageTotals,
    aggregate_manifest_usage,
    read_stage_usage,
    read_usage_records,
    snapshot_usage,
)
from shrlm.rlm_harness import H0
from shrlm.runner import build_harnessed_rlm
from tests.mock_lm import MockLM

FINAL = "```repl\nanswer['content'] = 'RIGHT'\nanswer['ready'] = True\n```"


def manifest_entry(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cost: float | None = 0.001,
    execution_time: float = 0.5,
    usage_lower_bound: bool = False,
) -> dict[str, Any]:
    """A U4-format manifest line reduced to its usage keys."""
    return {
        "run_id": "inst__a01",
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "execution_time": execution_time,
        "usage_lower_bound": usage_lower_bound,
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# run_metrics carries the new usage keys (R4)
# ---------------------------------------------------------------------------


class TestRunMetricsUsageKeys:
    def test_mock_completion_reports_nonzero_tokens_and_wall_time(self, monkeypatch):
        monkeypatch.setattr(
            rlm_module,
            "get_client",
            lambda backend, backend_kwargs: MockLM(responses=[FINAL]),
        )
        harnessed = build_harnessed_rlm(
            H0, backend="openai", backend_kwargs={"model_name": "mock-model"}
        )
        metrics = harnessed.completion("context").metrics

        assert metrics["input_tokens"] > 0
        assert metrics["output_tokens"] > 0
        assert metrics["execution_time"] > 0.0
        assert "cost" in metrics  # legacy key survives alongside the new ones


# ---------------------------------------------------------------------------
# aggregate_manifest_usage: the run-stage capture path
# ---------------------------------------------------------------------------


class TestAggregateManifestUsage:
    def test_sums_the_u4_keys(self):
        totals = aggregate_manifest_usage(
            [
                manifest_entry(input_tokens=10, output_tokens=5, cost=0.001, execution_time=0.5),
                manifest_entry(input_tokens=20, output_tokens=15, cost=0.002, execution_time=1.5),
            ]
        )
        assert totals == UsageTotals(
            input_tokens=30, output_tokens=20, cost=pytest.approx(0.003), wall_seconds=2.0
        )
        assert totals.lower_bound is False

    def test_pre_u4_line_contributes_zero_and_forces_lower_bound(self):
        totals = aggregate_manifest_usage(
            [manifest_entry(input_tokens=10), {"run_id": "old__a01", "cost": 0.004}]
        )
        assert totals.input_tokens == 10
        assert totals.cost == pytest.approx(0.005)
        assert totals.lower_bound is True

    def test_cost_none_contributes_zero_without_flag(self):
        totals = aggregate_manifest_usage([manifest_entry(cost=None)])
        assert totals.cost == 0.0
        assert totals.lower_bound is False

    def test_terminated_entry_propagates_the_lower_bound_flag(self):
        totals = aggregate_manifest_usage(
            [manifest_entry(), manifest_entry(usage_lower_bound=True)]
        )
        assert totals.lower_bound is True


# ---------------------------------------------------------------------------
# UsageTotals arithmetic
# ---------------------------------------------------------------------------


class TestUsageTotalsArithmetic:
    def test_subtraction_yields_the_increment(self):
        before = UsageTotals(input_tokens=10, output_tokens=10, cost=0.001)
        after = UsageTotals(input_tokens=30, output_tokens=25, cost=0.004)
        assert after - before == UsageTotals(
            input_tokens=20, output_tokens=15, cost=pytest.approx(0.003)
        )

    def test_negative_diff_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            UsageTotals(input_tokens=5) - UsageTotals(input_tokens=10)


# ---------------------------------------------------------------------------
# StageMeter: snapshot-diff path for caller-held clients
# ---------------------------------------------------------------------------


class TestSnapshotDiff:
    def test_sequential_stages_sharing_one_client_attribute_usage_disjointly(self, tmp_path):
        """MockLM.get_usage_summary is cumulative (10 tokens per call each
        way); two meters around disjoint call windows must each record only
        their own window."""
        lm = MockLM()
        path = tmp_path / "stage_usage.jsonl"

        with StageMeter(
            stage="proposal", stage_work_id="round01/proposal", round_index=1, out_path=path
        ) as meter:
            meter.watch(lm)
            lm.completion("one")
            lm.completion("two")
        with StageMeter(
            stage="attribution", stage_work_id="round01/attribution", round_index=1, out_path=path
        ) as meter:
            meter.watch(lm)
            lm.completion("three")

        first, second = read_records(path)
        assert first["input_tokens"] == 20
        assert first["output_tokens"] == 20
        assert second["input_tokens"] == 10
        assert second["output_tokens"] == 10
        assert first["wall_seconds"] > 0.0

    def test_watch_outside_the_context_body_is_refused(self, tmp_path):
        meter = StageMeter(
            stage="proposal",
            stage_work_id="round01/proposal",
            round_index=1,
            out_path=tmp_path / "stage_usage.jsonl",
        )
        with pytest.raises(RuntimeError, match="outside the context body"):
            meter.watch(MockLM())


# ---------------------------------------------------------------------------
# StageMeter: manifest-aggregate path for run stages
# ---------------------------------------------------------------------------


class TestRunStageRecords:
    def test_record_equals_the_sum_of_its_manifest_lines(self, tmp_path):
        entries = [
            manifest_entry(input_tokens=100, output_tokens=40, cost=0.01, execution_time=2.0),
            manifest_entry(input_tokens=50, output_tokens=20, cost=0.005, execution_time=1.0),
        ]
        path = tmp_path / "stage_usage.jsonl"
        with StageMeter(
            stage="mining", stage_work_id="round01/mining", round_index=1, out_path=path
        ) as meter:
            meter.add(aggregate_manifest_usage(entries))

        (record,) = read_records(path)
        assert record["input_tokens"] == 150
        assert record["output_tokens"] == 60
        assert record["cost"] == pytest.approx(0.015)
        assert record["resumed"] is False
        assert record["lower_bound"] is False
        assert record["cache_hits"] == 0

    def test_cache_hits_are_recorded(self, tmp_path):
        path = tmp_path / "stage_usage.jsonl"
        with StageMeter(
            stage="proposal", stage_work_id="round01/proposal", round_index=1, out_path=path
        ) as meter:
            meter.add_cache_hits(3)
        (record,) = read_records(path)
        assert record["cache_hits"] == 3

    def test_negative_cache_hits_are_refused(self, tmp_path):
        path = tmp_path / "stage_usage.jsonl"
        with StageMeter(
            stage="proposal", stage_work_id="round01/proposal", round_index=1, out_path=path
        ) as meter:
            with pytest.raises(ValueError, match="non-negative"):
                meter.add_cache_hits(-1)

    def test_exception_in_the_body_still_appends_a_lower_bound_record(self, tmp_path):
        path = tmp_path / "stage_usage.jsonl"
        with pytest.raises(RuntimeError, match="boom"):
            with StageMeter(
                stage="validation",
                stage_work_id="round01/validation",
                round_index=1,
                out_path=path,
            ) as meter:
                meter.add(UsageTotals(input_tokens=5, output_tokens=2, cost=0.001))
                raise RuntimeError("boom")

        (record,) = read_records(path)
        assert record["input_tokens"] == 5  # observed usage preserved, never zeroed
        assert record["lower_bound"] is True


# ---------------------------------------------------------------------------
# Resume: append-only records, exactly-once aggregation
# ---------------------------------------------------------------------------


class TestResumeAndExactlyOnce:
    def test_rerun_appends_a_second_record_flagged_resumed(self, tmp_path):
        path = tmp_path / "stage_usage.jsonl"
        first_attempt = [manifest_entry(input_tokens=100, output_tokens=40, cost=0.01)]
        # The incremental contract: the resumed attempt hands the meter only
        # the manifest lines new in that attempt.
        second_attempt = [manifest_entry(input_tokens=30, output_tokens=10, cost=0.003)]

        with StageMeter(
            stage="mining", stage_work_id="round01/mining", round_index=1, out_path=path
        ) as meter:
            meter.add(aggregate_manifest_usage(first_attempt))
        with StageMeter(
            stage="mining", stage_work_id="round01/mining", round_index=1, out_path=path
        ) as meter:
            meter.add(aggregate_manifest_usage(second_attempt))

        first, second = read_records(path)
        assert first["resumed"] is False
        assert first["attempt_index"] == 0
        assert second["resumed"] is True
        assert second["attempt_index"] == 1

        totals = read_stage_usage(path)["round01/mining"]
        assert totals.input_tokens == 130  # each attempt counted exactly once
        assert totals.output_tokens == 50
        assert totals.cost == pytest.approx(0.013)

    def test_duplicate_attempt_record_is_superseded_not_summed(self, tmp_path):
        """A re-appended record for the same (stage_work_id, attempt_index)
        supersedes its predecessor: the latest write wins, so a crash between
        append and stage-completion marker can never double-count."""
        path = tmp_path / "stage_usage.jsonl"
        with StageMeter(
            stage="mining", stage_work_id="round01/mining", round_index=1, out_path=path
        ) as meter:
            meter.add(UsageTotals(input_tokens=100, output_tokens=40, cost=0.01))

        stale = read_records(path)[0]
        duplicate = dict(stale, input_tokens=120, output_tokens=45, cost=0.012)
        with open(path, "a") as handle:
            handle.write(json.dumps(duplicate, sort_keys=True) + "\n")

        totals = read_stage_usage(path)["round01/mining"]
        assert totals.input_tokens == 120
        assert totals.output_tokens == 45
        assert totals.cost == pytest.approx(0.012)

    def test_explicit_resumed_flag_overrides_the_inference(self, tmp_path):
        """A caller that knows a prior attempt died before its append can
        force resumed=True on what the sidecar sees as attempt zero."""
        path = tmp_path / "stage_usage.jsonl"
        with StageMeter(
            stage="mining",
            stage_work_id="round01/mining",
            round_index=1,
            out_path=path,
            resumed=True,
        ) as meter:
            meter.add(UsageTotals(input_tokens=1))
        (record,) = read_records(path)
        assert record["attempt_index"] == 0
        assert record["resumed"] is True

    def test_a_torn_final_line_neither_raises_nor_swallows_later_records(self, tmp_path):
        """A kill mid-append leaves an unterminated, unparsable line. Every
        later ``StageMeter.__enter__``, report build, and spend check parses
        the whole sidecar, so that line must be quarantined rather than raised
        on -- and it must not swallow the next record either."""
        path = tmp_path / "stage_usage.jsonl"
        with StageMeter(
            stage="mining", stage_work_id="round01/mining", round_index=1, out_path=path
        ) as meter:
            meter.add(UsageTotals(input_tokens=100, output_tokens=40, cost=0.01))

        with open(path, "a") as handle:
            handle.write('{"attempt_index": 1, "cache_hits": 0, "cost": 0.003, "input_to')

        with StageMeter(
            stage="mining", stage_work_id="round01/mining", round_index=1, out_path=path
        ) as meter:
            meter.add(UsageTotals(input_tokens=30, output_tokens=10, cost=0.003))

        sidecar = read_usage_records(path)
        assert sidecar.torn_lines == 1
        assert len(sidecar.records) == 2, "the fence keeps the torn line from eating the next one"

        totals = read_stage_usage(path)["round01/mining"]
        assert totals.input_tokens == 130
        assert totals.output_tokens == 50
        assert totals.lower_bound is True, "unattributable torn usage makes every total a bound"

    def test_intact_sidecar_reports_no_torn_lines_and_exact_totals(self, tmp_path):
        path = tmp_path / "stage_usage.jsonl"
        with StageMeter(
            stage="mining", stage_work_id="round01/mining", round_index=1, out_path=path
        ) as meter:
            meter.add(UsageTotals(input_tokens=7, output_tokens=3, cost=0.001))
        assert read_usage_records(path).torn_lines == 0
        assert read_stage_usage(path)["round01/mining"].lower_bound is False

    def test_snapshot_usage_reads_the_cumulative_client_summary(self):
        lm = MockLM()
        lm.completion("one")
        totals = snapshot_usage(lm)
        assert totals == UsageTotals(input_tokens=10, output_tokens=10, cost=0.0)
