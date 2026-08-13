"""Tests for the U6 evaluation runner: fixed conditions over frozen test splits.

Everything runs offline through the repo-standard seam: ``rlm.core.rlm.
get_client`` is monkeypatched with the cost-stubbed scripted client factory
from ``tests.optimization.test_driver`` (a plain ``MockLM`` persists no cost,
and the governed eval stage refuses cost-less runs), splits come from the
injected offline loaders of ``tests.experiment.test_orchestrator``, and the
frozen ``sh_rlm/harness.json`` is written directly -- the eval runner must
consume the freeze envelope exactly as the orchestrator leaves it.

The load-bearing claims: every condition consumes byte-identical instance
bytes (R8), repetitions are attempts on one round dir (KTD8), the frozen
condition round-trips through ``materialize_harness`` under a hash check, the
spend breaker leaves resumable partial state (R12), and ``eval_summary.json``
is a pure function of the persisted rounds.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.config import ExperimentConfig
from shrlm.experiment.evaluation import (
    CONDITION_B1,
    CONDITION_SH_RLM,
    EVAL_DIR,
    EVAL_SUMMARY_FILENAME,
    EvaluationPersistenceError,
    run_evaluation,
)
from shrlm.experiment.orchestrator import FROZEN_DIR, FROZEN_HARNESS_FILENAME
from shrlm.experiment.splits import SPLITS_DIR, LoaderFn
from shrlm.experiment.usage import STAGE_USAGE_FILE, read_stage_usage
from shrlm.harness_identity import harness_hash, write_harness_json
from shrlm.optimization.costs import OUTCOME_COMPLETED, OUTCOME_OVER_BUDGET
from shrlm.rlm_harness import H0
from tests.experiment.test_orchestrator import fake_loader, make_config, patch_runner
from tests.optimization.test_driver import ClientFactory, GoldVerifier, final

# Only GraphWalks is materialized, so the persisted test sets are exactly two:
# ``graphwalks_short`` and ``graphwalks_long`` (one instance each at the
# template's scale).
LOADERS: dict[str, LoaderFn] = {"graphwalks": fake_loader("graphwalks")}

SET_SHORT = "graphwalks_short"
SET_LONG = "graphwalks_long"
BOTH_SETS = (SET_SHORT, SET_LONG)
BOTH_CONDITIONS = (CONDITION_B1, CONDITION_SH_RLM)

FROZEN_TEXT = "Verify every claim against the stored evidence before answering. [frozen]"


def eval_config(tmp_path: Path, *, repetitions: int = 1, **kwargs: Any) -> ExperimentConfig:
    """The orchestrator's test config with an evaluation repetition count.

    ``eval_repetitions`` is operational (KTD8), so overriding it leaves the
    identity hash -- and therefore an existing experiment directory -- intact.
    """
    config = make_config(tmp_path, **kwargs)
    return replace(config, operational=replace(config.operational, eval_repetitions=repetitions))


def freeze(out_dir: Path, harness: Any = None) -> str:
    """Write the ``sh_rlm/harness.json`` envelope the orchestrator freezes."""
    frozen = harness if harness is not None else replace(H0, verification_instruction=FROZEN_TEXT)
    path = out_dir / FROZEN_DIR / FROZEN_HARNESS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    write_harness_json(frozen, path)
    return harness_hash(frozen)


def evaluate(
    config: ExperimentConfig,
    out_dir: Path,
    conditions: tuple[str, ...] = BOTH_CONDITIONS,
) -> Any:
    return run_evaluation(
        config,
        conditions,
        out_dir,
        verifiers={"graphwalks": GoldVerifier()},
        loaders=LOADERS,
    )


def set_round_dir(out_dir: Path, condition: str, set_id: str) -> Path:
    return out_dir / EVAL_DIR / condition / set_id / "round_00"


def manifest_lines(out_dir: Path, condition: str, set_id: str) -> list[dict[str, Any]]:
    path = set_round_dir(out_dir, condition, set_id) / "runs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary_of(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME).read_text())


def scripted(monkeypatch: pytest.MonkeyPatch, *responses: str) -> ClientFactory:
    return patch_runner(monkeypatch, [final(response) for response in responses])


# Execution order is (condition, set): b1 short, b1 long, sh_rlm short, sh_rlm long.
ONE_ATTEMPT_SCRIPT = ("RIGHT", "WRONG", "RIGHT", "RIGHT")


# ---------------------------------------------------------------------------
# The grid: conditions x test sets, over byte-identical instances (R8)
# ---------------------------------------------------------------------------


class TestConditionGrid:
    def test_two_conditions_over_two_test_sets_produce_four_round_dirs(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        factory = scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)

        result = evaluate(config, out)

        assert factory.total_calls == 4
        assert [condition.condition_id for condition in result.conditions] == list(BOTH_CONDITIONS)
        for condition in BOTH_CONDITIONS:
            for set_id in BOTH_SETS:
                path = set_round_dir(out, condition, set_id)
                assert (path / "runs.jsonl").exists(), path
                assert (path / "harness.json").exists(), path
                assert (path / "instances.jsonl").exists(), path

    def test_every_condition_consumes_byte_identical_instances(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)

        result = evaluate(config, out)

        for set_id in BOTH_SETS:
            split_bytes = (out / SPLITS_DIR / f"{set_id}_test.jsonl").read_bytes()
            persisted = {
                set_round_dir(out, condition, set_id).joinpath("instances.jsonl").read_bytes()
                for condition in BOTH_CONDITIONS
            }
            assert persisted == {split_bytes}
            shas = {
                condition.test_sets[set_id]["instances_sha256"] for condition in result.conditions
            }
            assert len(shas) == 1

    def test_unknown_condition_raises_naming_the_known_ones(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        factory = scripted(monkeypatch)
        with pytest.raises(ValueError, match="h1"):
            evaluate(config, out, conditions=("b1", "h1"))
        assert factory.total_calls == 0

    def test_duplicate_condition_raises(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        scripted(monkeypatch)
        with pytest.raises(ValueError, match="duplicate"):
            evaluate(config, out, conditions=(CONDITION_B1, CONDITION_B1))


# ---------------------------------------------------------------------------
# Repetitions: unseeded temperature samples as attempts (KTD8)
# ---------------------------------------------------------------------------


class TestRepetitions:
    def test_three_repetitions_yield_three_runs_per_instance_with_distinct_run_ids(
        self, tmp_path, monkeypatch
    ):
        config = eval_config(tmp_path, repetitions=3)
        out = tmp_path / "exp"
        freeze(out)
        factory = scripted(monkeypatch, *(["RIGHT"] * 12))

        result = evaluate(config, out)

        assert factory.total_calls == 12  # 2 conditions x 2 sets x 1 instance x 3 attempts
        for condition in BOTH_CONDITIONS:
            for set_id in BOTH_SETS:
                entries = manifest_lines(out, condition, set_id)
                assert len(entries) == 3
                run_ids = [entry["run_id"] for entry in entries]
                assert len(set(run_ids)) == 3
                assert sorted(entry["attempt"] for entry in entries) == [1, 2, 3]
        assert result.summary["eval_repetitions"] == 3
        assert result.conditions[0].test_sets[SET_SHORT]["n_runs"] == 3


# ---------------------------------------------------------------------------
# Aggregation: split_aggregate's shape plus tokens and wall time
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_pass_counts_costs_tokens_and_wall_time_per_condition_per_set(
        self, tmp_path, monkeypatch
    ):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)

        result = evaluate(config, out)
        b1, sh_rlm = result.conditions

        assert b1.test_sets[SET_SHORT]["pass_count"] == 1
        assert b1.test_sets[SET_LONG]["pass_count"] == 0
        assert sh_rlm.test_sets[SET_SHORT]["pass_count"] == 1
        assert sh_rlm.test_sets[SET_LONG]["pass_count"] == 1

        for condition in result.conditions:
            for set_id in BOTH_SETS:
                aggregate = condition.test_sets[set_id]
                assert aggregate["n_runs"] == 1
                assert aggregate["outcome"] == OUTCOME_COMPLETED
                assert aggregate["total_cost"] == pytest.approx(0.001)
                assert aggregate["mean_cost"] == pytest.approx(0.001)
                assert aggregate["input_tokens"] == 10
                assert aggregate["output_tokens"] == 10
                assert aggregate["wall_seconds"] > 0.0
                assert aggregate["total_sub_calls"] == 0
                assert aggregate["usage_lower_bound"] is False

    def test_harness_hashes_distinguish_the_conditions(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        frozen_hash = freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)

        result = evaluate(config, out)
        b1, sh_rlm = result.conditions

        assert b1.harness_hash == harness_hash(H0)
        assert sh_rlm.harness_hash == frozen_hash
        for set_id in BOTH_SETS:
            assert b1.test_sets[set_id]["harness_hash"] == harness_hash(H0)
            assert sh_rlm.test_sets[set_id]["harness_hash"] == frozen_hash

    def test_stage_usage_records_one_work_id_per_condition_per_set(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)

        evaluate(config, out)

        totals = read_stage_usage(out / STAGE_USAGE_FILE)
        expected = {
            f"{EVAL_DIR}/{condition}/{set_id}"
            for condition in BOTH_CONDITIONS
            for set_id in BOTH_SETS
        }
        assert set(totals) == expected
        for work_id, usage in totals.items():
            assert usage.input_tokens == 10, work_id
            assert usage.output_tokens == 10, work_id
            assert usage.wall_seconds > 0.0, work_id


# ---------------------------------------------------------------------------
# The frozen condition: envelope -> materialize_harness -> hash check
# ---------------------------------------------------------------------------


class TestFrozenCondition:
    def test_frozen_harness_loads_from_the_freeze_envelope(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        frozen_hash = freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)

        result = evaluate(config, out)

        recorded = json.loads(
            (set_round_dir(out, CONDITION_SH_RLM, SET_SHORT) / "harness.json").read_text()
        )
        assert recorded["hash"] == frozen_hash
        assert result.conditions[1].harness_hash == frozen_hash

    def test_tampered_envelope_raises_before_any_run(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        path = out / FROZEN_DIR / FROZEN_HARNESS_FILENAME
        envelope = json.loads(path.read_text())
        envelope["harness"]["surfaces"]["S4_verification_instruction"] = "tampered instruction"
        path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")

        factory = scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)
        with pytest.raises(EvaluationPersistenceError, match="hash"):
            evaluate(config, out, conditions=(CONDITION_SH_RLM,))
        assert factory.total_calls == 0

    def test_missing_freeze_envelope_raises(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        factory = scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)
        with pytest.raises(EvaluationPersistenceError, match=FROZEN_HARNESS_FILENAME):
            evaluate(config, out, conditions=(CONDITION_SH_RLM,))
        assert factory.total_calls == 0


# ---------------------------------------------------------------------------
# The spend breaker (R12) and resume
# ---------------------------------------------------------------------------


class TestSpendBreaker:
    def test_trip_persists_partial_state_and_resume_runs_nothing(self, tmp_path, monkeypatch):
        # Every run costs 0.001, so a 0.0005 cumulative budget trips after the
        # first run of each condition: the rest of that condition is skipped,
        # never silently dropped.
        config = eval_config(tmp_path, candidate_budget=0.0005)
        out = tmp_path / "exp"
        freeze(out)
        factory = scripted(monkeypatch, "RIGHT", "RIGHT")

        result = evaluate(config, out)

        assert factory.total_calls == 2  # one run per condition, then the breaker
        for condition in result.conditions:
            assert condition.outcome == OUTCOME_OVER_BUDGET
            assert condition.over_budget
            assert condition.test_sets[SET_SHORT]["n_runs"] == 1
            assert condition.test_sets[SET_LONG]["n_runs"] == 0
            assert condition.test_sets[SET_LONG]["skipped_run_ids"] == ["graphwalks-long-0__a01"]
            assert condition.test_sets[SET_LONG]["pass_rate"] is None

        idle = scripted(monkeypatch)
        resumed = evaluate(config, out)
        assert idle.total_calls == 0
        assert resumed.summary["conditions"] == result.summary["conditions"]

    def test_completed_evaluation_reruns_nothing(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)
        evaluate(config, out)

        idle = scripted(monkeypatch)
        evaluate(config, out)
        assert idle.total_calls == 0


# ---------------------------------------------------------------------------
# eval_summary.json: a pure function of the persisted rounds
# ---------------------------------------------------------------------------


class TestEvalSummary:
    def test_resume_rewrites_byte_identical_bytes(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)
        evaluate(config, out)
        first = (out / EVAL_DIR / EVAL_SUMMARY_FILENAME).read_bytes()

        scripted(monkeypatch)
        evaluate(config, out)
        assert (out / EVAL_DIR / EVAL_SUMMARY_FILENAME).read_bytes() == first

    def test_two_fresh_experiments_agree_on_everything_but_wall_clock(self, tmp_path, monkeypatch):
        def summarize(name: str) -> dict[str, Any]:
            config = eval_config(tmp_path / name)
            out = tmp_path / name / "exp"
            freeze(out)
            scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)
            evaluate(config, out)
            summary = summary_of(out)
            for condition in summary["conditions"].values():
                for aggregate in condition["test_sets"].values():
                    aggregate.pop("wall_seconds")
            return summary

        assert summarize("first") == summarize("second")

    def test_summary_records_the_condition_sources_and_identity(self, tmp_path, monkeypatch):
        config = eval_config(tmp_path)
        out = tmp_path / "exp"
        freeze(out)
        scripted(monkeypatch, *ONE_ATTEMPT_SCRIPT)

        result = evaluate(config, out)
        summary = summary_of(out)

        assert summary == result.summary
        assert summary["profile"] == "full"
        assert summary["conditions"][CONDITION_B1]["source"]["kind"] == "registry"
        assert summary["conditions"][CONDITION_SH_RLM]["source"]["kind"] == "frozen"
        # The evaluation order lives in the top-level list; the per-condition
        # mapping is written with sorted keys (deterministic bytes).
        assert summary["test_sets"] == list(BOTH_SETS)
        assert set(summary["conditions"][CONDITION_B1]["test_sets"]) == set(BOTH_SETS)
