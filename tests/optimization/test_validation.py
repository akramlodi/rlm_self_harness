"""Tests for the U3 candidate evaluation driver: persisted, resumable, comparable rounds.

Each (subject x split) evaluation is one persist-first ``run_round`` executed
through the cost governor's ``run_governed_round`` (R5; KTD4): baseline and
every candidate get their own nested ``round_00`` with a ``harness.json``
identity, sha-linked traces, and a manifest whose pass counts and costs are
recomputable from disk alone. The per-subject ``summary.json`` is the
validation-owned aggregate the band check and the ledger consume, so neither
ever re-reads traces.

The offline seam is the repo's usual one: patch ``rlm.core.rlm.get_client``
with a factory of scripted clients that stub per-call costs.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
from rlm.core.types import ModelUsageSummary, UsageSummary
from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import GATE_CAPS, CandidateRejection, LoadedCandidate
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    ValidationCaps,
    governed_limits,
)
from shrlm.optimization.driver import RoundConfig, run_round
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.optimization.validation import (
    BASELINE_ID,
    EVAL_ROUND_INDEX,
    SPLIT_HELDIN,
    SPLIT_HELDOUT,
    SUMMARY_FILENAME,
    EvaluationConfig,
    SubjectEvaluation,
    ValidationSplits,
    evaluate_subject,
    evaluate_validation_round,
    load_summary,
    split_aggregate,
    split_dir,
    subject_dir,
)
from shrlm.rlm_harness import H0, build_runtime_policy
from tests.mock_lm import MockLM

COST_PER_CALL = 0.001

# Per-run budget 0.0015: one scripted call per run stays under it, a two-call
# run crosses it after its second iteration (BudgetExceededError, spent 0.002).
CAPS = ValidationCaps(
    max_depth=2,
    max_iterations=3,
    max_budget=0.0015,
    max_timeout=60.0,
    candidate_budget=0.01,
)


# ---------------------------------------------------------------------------
# Offline seam: scripted clients with stubbed per-call cost
# ---------------------------------------------------------------------------


class ScriptedLM(MockLM):
    """A ``MockLM`` popping from a shared script, with a stubbed per-call cost.

    Exhausting the script raises rather than improvising: a test whose scripted
    turns run out is a broken test, not a passing run.
    """

    def __init__(self, script: list[str], cost_per_call: float):
        super().__init__(model_name="mock-model")
        self._script = script
        self._cost_per_call = cost_per_call

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self._call_count += 1
        if not self._script:
            raise IndexError("ScriptedLM: script exhausted")
        return self._script.pop(0)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._call_count,
                    total_input_tokens=self._call_count * 10,
                    total_output_tokens=self._call_count * 10,
                    total_cost=self._cost_per_call * self._call_count,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=10,
            total_output_tokens=10,
            total_cost=self._cost_per_call,
        )


class ClientFactory:
    """Stands in for ``get_client``: fresh client per call, one shared script."""

    def __init__(self, script: list[str], cost_per_call: float = COST_PER_CALL):
        self.script = list(script)
        self.cost_per_call = cost_per_call
        self.clients: list[ScriptedLM] = []

    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> ScriptedLM:
        client = ScriptedLM(self.script, self.cost_per_call)
        self.clients.append(client)
        return client

    @property
    def total_calls(self) -> int:
        return sum(client._call_count for client in self.clients)


def final(content: str) -> str:
    """A ```repl``` block that returns ``content`` from the answer variable."""
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


@dataclass
class GoldVerifier:
    """Deterministic string-match verifier."""

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        gold = str(instance["gold"])
        if produced == gold:
            return Verdict(passed=True, cause=None, gold=gold, produced=produced)
        return Verdict(passed=False, cause=VerifierCause.WRONG_VALUE, gold=gold, produced=produced)


# ---------------------------------------------------------------------------
# Fixtures: splits, configs, fabricated candidates
# ---------------------------------------------------------------------------


def make_instances(prefix: str, n: int = 2) -> list[dict[str, Any]]:
    return [
        {"id": f"{prefix}-{i}", "prompt": f"{prefix} context {i}", "gold": "RIGHT"}
        for i in range(1, n + 1)
    ]


def make_splits(n: int = 2) -> ValidationSplits:
    return ValidationSplits(heldin=make_instances("hi", n), heldout=make_instances("ho", n))


def make_config(tmp_path: Path, caps: ValidationCaps = CAPS, **overrides: Any) -> EvaluationConfig:
    values: dict[str, Any] = {
        "splits": make_splits(),
        "verifier": GoldVerifier(),
        "caps": caps,
        "out_dir": tmp_path / "validation",
        "round_index": 1,
        "repetitions": 2,
        "backend": "openai",
        "backend_kwargs": {"model_name": "validation-test"},
    }
    values.update(overrides)
    return EvaluationConfig(**values)


def candidate_harness(tag: str) -> Any:
    """A harness whose serialization (and hash) differs from H0's by one edit."""
    return replace(H0, decomposition_instruction=H0.decomposition_instruction + f"\n[{tag}]")


def fake_candidate(candidate_id: str, harness: Any) -> LoadedCandidate:
    """A ``LoadedCandidate`` as the U1 loader would hand it over, sans gates."""
    return LoadedCandidate(
        candidate_id=candidate_id,
        path=Path("/nonexistent/proposal.json"),
        surface="S2",
        proposal={},
        harness=harness,
        harness_hash=harness_hash(harness),
        module_path=Path("/nonexistent/surfaces.py"),
    )


def read_manifest(round_path: Path) -> list[dict[str, Any]]:
    lines = (round_path / "runs.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def full_round_script() -> list[str]:
    """Baseline all-pass; cand-a passes 3/4 held-in and 2/4 held-out; cand-b all-pass.

    Run order per split is instance-major, attempt-minor: hi-1 a1, hi-1 a2,
    hi-2 a1, hi-2 a2.
    """
    return (
        [final("RIGHT")] * 8
        + [final("RIGHT"), final("RIGHT"), final("WRONG"), final("RIGHT")]
        + [final("WRONG"), final("WRONG"), final("RIGHT"), final("RIGHT")]
        + [final("RIGHT")] * 8
    )


def run_full_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EvaluationConfig, Any, ClientFactory]:
    factory = ClientFactory(full_round_script())
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_config(tmp_path)
    candidates = [
        fake_candidate("cand-a", candidate_harness("cand-a")),
        fake_candidate("cand-b", candidate_harness("cand-b")),
    ]
    result = evaluate_validation_round(H0, candidates, config)
    return config, result, factory


# ---------------------------------------------------------------------------
# Layout and split validation
# ---------------------------------------------------------------------------


class TestLayoutAndSplits:
    def test_identical_instance_in_both_splits_is_rejected(self):
        shared = {"id": "hi-1", "prompt": "context", "gold": "RIGHT"}
        with pytest.raises(ValueError, match="disjoint"):
            ValidationSplits(heldin=[shared], heldout=[dict(shared)])

    def test_empty_split_is_rejected(self):
        with pytest.raises(ValueError, match="heldout"):
            ValidationSplits(heldin=make_instances("hi"), heldout=[])

    def test_split_dirs_nest_a_round_00_with_the_subject_identity(self, tmp_path, monkeypatch):
        factory = ClientFactory([final("RIGHT")] * 8)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path)

        evaluation = evaluate_subject(BASELINE_ID, H0, config)

        assert isinstance(evaluation, SubjectEvaluation)
        for split_id in (SPLIT_HELDIN, SPLIT_HELDOUT):
            nested = (
                split_dir(config.out_dir, config.round_index, BASELINE_ID, split_id) / "round_00"
            )
            assert (nested / "harness.json").is_file()
            assert (nested / "instances.jsonl").is_file()
            assert (nested / "runs.jsonl").is_file()
            assert json.loads((nested / "harness.json").read_text())["hash"] == harness_hash(H0)
        assert evaluation.summary_path == (
            subject_dir(config.out_dir, config.round_index, BASELINE_ID) / SUMMARY_FILENAME
        )
        assert evaluation.summary_path.is_file()

    def test_unsafe_subject_id_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="filesystem-safe"):
            subject_dir(tmp_path, 1, "../escape")

    def test_candidate_named_baseline_is_rejected(self, tmp_path, monkeypatch):
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        candidate = fake_candidate(BASELINE_ID, candidate_harness("x"))
        with pytest.raises(ValueError, match="baseline"):
            evaluate_validation_round(H0, [candidate], make_config(tmp_path))
        assert idle.total_calls == 0

    def test_duplicate_candidate_ids_are_rejected(self, tmp_path, monkeypatch):
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        candidates = [
            fake_candidate("cand-a", candidate_harness("a")),
            fake_candidate("cand-a", candidate_harness("b")),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            evaluate_validation_round(H0, candidates, make_config(tmp_path))
        assert idle.total_calls == 0


# ---------------------------------------------------------------------------
# The full round: baseline + 2 candidates x 2 splits x 2 reps
# ---------------------------------------------------------------------------


class TestFullEvaluation:
    def test_every_subject_split_persists_its_runs(self, tmp_path, monkeypatch):
        config, result, factory = run_full_evaluation(tmp_path, monkeypatch)

        assert factory.total_calls == 24
        for evaluation in [result.baseline, *result.candidates]:
            assert isinstance(evaluation, SubjectEvaluation)
            for split_id in (SPLIT_HELDIN, SPLIT_HELDOUT):
                nested = (
                    split_dir(config.out_dir, config.round_index, evaluation.subject_id, split_id)
                    / f"round_{EVAL_ROUND_INDEX:02d}"
                )
                entries = read_manifest(nested)
                assert len(entries) == 4
                recorded = json.loads((nested / "harness.json").read_text())["hash"]
                assert recorded == evaluation.harness_hash
                for entry in entries:
                    assert (nested / entry["trace_path"]).is_file()

    def test_summaries_carry_the_scripted_pass_counts(self, tmp_path, monkeypatch):
        _config, result, _factory = run_full_evaluation(tmp_path, monkeypatch)
        by_id = {
            evaluation.subject_id: evaluation
            for evaluation in [result.baseline, *result.candidates]
        }

        expectations = {
            BASELINE_ID: {SPLIT_HELDIN: 4, SPLIT_HELDOUT: 4},
            "cand-a": {SPLIT_HELDIN: 3, SPLIT_HELDOUT: 2},
            "cand-b": {SPLIT_HELDIN: 4, SPLIT_HELDOUT: 4},
        }
        for subject_id, expected in expectations.items():
            summary = by_id[subject_id].summary
            assert summary["outcome"] == OUTCOME_COMPLETED
            assert summary["spent"] == pytest.approx(8 * COST_PER_CALL)
            for split_id, pass_count in expected.items():
                split_summary = summary["splits"][split_id]
                assert split_summary["n_runs"] == 4
                assert split_summary["pass_count"] == pass_count
                assert split_summary["pass_rate"] == pytest.approx(pass_count / 4)
                assert split_summary["mean_cost"] == pytest.approx(COST_PER_CALL)
                assert split_summary["skipped_run_ids"] == []

    def test_aggregates_are_recomputable_from_disk_alone(self, tmp_path, monkeypatch):
        config, result, _factory = run_full_evaluation(tmp_path, monkeypatch)
        for evaluation in [result.baseline, *result.candidates]:
            for split_id in (SPLIT_HELDIN, SPLIT_HELDOUT):
                split_path = split_dir(
                    config.out_dir, config.round_index, evaluation.subject_id, split_id
                )
                recomputed = split_aggregate(split_path)
                recorded = evaluation.summary["splits"][split_id]
                for key, value in recomputed.items():
                    assert recorded[key] == value
                # Pass counts and costs are independently recomputable from
                # the manifest alone, with no summary in the loop.
                manifest = read_manifest(split_path / f"round_{EVAL_ROUND_INDEX:02d}")
                assert recomputed["pass_count"] == sum(1 for e in manifest if e["passed"])
                assert recomputed["total_cost"] == pytest.approx(sum(e["cost"] for e in manifest))

    def test_summary_file_round_trips_through_load_summary(self, tmp_path, monkeypatch):
        config, result, _factory = run_full_evaluation(tmp_path, monkeypatch)
        for evaluation in [result.baseline, *result.candidates]:
            loaded = load_summary(
                subject_dir(config.out_dir, config.round_index, evaluation.subject_id)
            )
            assert loaded == evaluation.summary
            # The summary's round links resolve relative to the subject dir.
            for split_summary in loaded["splits"].values():
                linked = evaluation.path / split_summary["round_path"]
                assert (linked / "harness.json").is_file()

    def test_rerun_is_idempotent_and_makes_no_model_calls(self, tmp_path, monkeypatch):
        _config, first, _factory = run_full_evaluation(tmp_path, monkeypatch)

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        config = make_config(tmp_path)
        candidates = [
            fake_candidate("cand-a", candidate_harness("cand-a")),
            fake_candidate("cand-b", candidate_harness("cand-b")),
        ]
        second = evaluate_validation_round(H0, candidates, config)

        assert idle.total_calls == 0
        assert second.baseline.summary == first.baseline.summary
        for before, after in zip(first.candidates, second.candidates, strict=True):
            assert after.summary == before.summary


# ---------------------------------------------------------------------------
# Resume: a crashed candidate picks up exactly where it left off
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_mid_candidate_executes_only_the_missing_runs(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        harness = candidate_harness("cand-a")
        limits = governed_limits("cand-a", harness.runtime_policy, CAPS)
        assert not isinstance(limits, CandidateRejection)

        # Simulate a crash after three of the four held-in runs persisted.
        first = ClientFactory([final("RIGHT")] * 3)
        monkeypatch.setattr(rlm_module, "get_client", first)
        run_round(
            RoundConfig(
                round_index=EVAL_ROUND_INDEX,
                harness=harness,
                instances=config.splits.heldin,
                verifier=GoldVerifier(),
                out_dir=split_dir(config.out_dir, config.round_index, "cand-a", SPLIT_HELDIN),
                backend="openai",
                backend_kwargs={"model_name": "validation-test"},
                attempts=config.repetitions,
                **limits,
            ),
            stop_after=3,
        )
        assert first.total_calls == 3

        resumed = ClientFactory([final("RIGHT")] * 5)
        monkeypatch.setattr(rlm_module, "get_client", resumed)
        evaluation = evaluate_subject("cand-a", harness, config)

        assert isinstance(evaluation, SubjectEvaluation)
        # Only the missing runs executed: 1 held-in + 4 held-out.
        assert resumed.total_calls == 5
        assert evaluation.summary["splits"][SPLIT_HELDIN]["n_runs"] == 4
        assert evaluation.summary["splits"][SPLIT_HELDOUT]["n_runs"] == 4
        assert evaluation.summary["outcome"] == OUTCOME_COMPLETED
        assert evaluation.summary["spent"] == pytest.approx(8 * COST_PER_CALL)


# ---------------------------------------------------------------------------
# Budget: terminated runs stay aggregable; the breaker spans both splits
# ---------------------------------------------------------------------------


class TestBudget:
    def test_all_runs_terminating_on_budget_still_aggregate(self, tmp_path, monkeypatch):
        # Every run burns two iterations into the per-run cap: persisted as
        # RESOURCE_TERMINATED with the exception's spent figure as its cost.
        factory = ClientFactory(["Scanning part one.", "Scanning part two."] * 2)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path, splits=make_splits(1), repetitions=1)

        evaluation = evaluate_subject("cand-burn", candidate_harness("burn"), config)

        assert isinstance(evaluation, SubjectEvaluation)
        assert evaluation.summary["outcome"] == OUTCOME_COMPLETED
        for split_id in (SPLIT_HELDIN, SPLIT_HELDOUT):
            split_summary = evaluation.summary["splits"][split_id]
            assert split_summary["n_runs"] == 1
            assert split_summary["pass_count"] == 0
            assert split_summary["n_resource_terminated"] == 1
            assert split_summary["mean_cost"] == pytest.approx(2 * COST_PER_CALL)
            split_path = split_dir(config.out_dir, config.round_index, "cand-burn", split_id)
            entries = read_manifest(split_path / f"round_{EVAL_ROUND_INDEX:02d}")
            assert entries[0]["cause"] == VerifierCause.RESOURCE_TERMINATED.value
            assert split_aggregate(split_path)["n_resource_terminated"] == 1

    def test_breaker_is_cumulative_across_both_splits(self, tmp_path, monkeypatch):
        caps = replace(CAPS, candidate_budget=0.0015)
        factory = ClientFactory([final("RIGHT")] * 4)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path, caps=caps, repetitions=1)

        evaluation = evaluate_subject("cand-a", candidate_harness("cand-a"), config)

        assert isinstance(evaluation, SubjectEvaluation)
        # The breaker tripped after the second held-in run; held-out never ran.
        assert factory.total_calls == 2
        assert evaluation.summary["outcome"] == OUTCOME_OVER_BUDGET
        assert evaluation.over_budget
        heldin = evaluation.summary["splits"][SPLIT_HELDIN]
        heldout = evaluation.summary["splits"][SPLIT_HELDOUT]
        assert heldin["n_runs"] == 2
        assert heldin["skipped_run_ids"] == []
        assert heldout["n_runs"] == 0
        assert heldout["pass_rate"] is None
        assert heldout["skipped_run_ids"] == ["ho-1__a01", "ho-2__a01"]

    def test_over_cap_policy_comes_back_as_a_structured_rejection(self, tmp_path, monkeypatch):
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        policy = build_runtime_policy() | {"enabled": True, "max_depth": 3}
        harness = replace(H0, runtime_policy=policy)
        config = make_config(tmp_path)

        result = evaluate_subject("cand-deep", harness, config)

        assert isinstance(result, CandidateRejection)
        assert result.candidate_id == "cand-deep"
        assert result.gate == GATE_CAPS
        assert idle.total_calls == 0
        assert not subject_dir(config.out_dir, config.round_index, "cand-deep").exists()


# ---------------------------------------------------------------------------
# Sub-call counts come from rehydrated traces, never from process state
# ---------------------------------------------------------------------------


class TestSubCallAggregation:
    def test_sub_calls_are_counted_from_persisted_traces(self, tmp_path, monkeypatch):
        caps = ValidationCaps(
            max_depth=2,
            max_iterations=3,
            max_budget=0.01,
            max_timeout=60.0,
            candidate_budget=0.1,
        )
        # Held-in run: one llm_query sub-call, then the final answer. Held-out
        # run: a direct final answer, no sub-calls.
        script = [
            "```repl\npart = llm_query('probe the header')\nprint(part)\n```",
            "leaf: header",
            final("RIGHT"),
            final("RIGHT"),
        ]
        factory = ClientFactory(script)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path, caps=caps, splits=make_splits(1), repetitions=1)

        evaluation = evaluate_subject(BASELINE_ID, H0, config)

        assert isinstance(evaluation, SubjectEvaluation)
        heldin = evaluation.summary["splits"][SPLIT_HELDIN]
        heldout = evaluation.summary["splits"][SPLIT_HELDOUT]
        assert heldin["total_sub_calls"] == 1
        assert heldin["mean_sub_calls"] == pytest.approx(1.0)
        assert heldout["total_sub_calls"] == 0
        # Recomputable from the persisted traces alone.
        split_path = split_dir(config.out_dir, config.round_index, BASELINE_ID, SPLIT_HELDIN)
        assert split_aggregate(split_path)["total_sub_calls"] == 1


# ---------------------------------------------------------------------------
# The summary file: written once, divergence refused
# ---------------------------------------------------------------------------


class TestSummaryPersistence:
    def test_divergent_summary_rewrite_is_refused(self, tmp_path, monkeypatch):
        factory = ClientFactory([final("RIGHT")] * 2)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path, splits=make_splits(1), repetitions=1)
        evaluation = evaluate_subject(BASELINE_ID, H0, config)
        assert isinstance(evaluation, SubjectEvaluation)

        tampered = dict(evaluation.summary)
        tampered["spent"] = 999.0
        evaluation.summary_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        with pytest.raises(ValueError, match="summary"):
            evaluate_subject(BASELINE_ID, H0, config)
        assert idle.total_calls == 0


if __name__ == "__main__":
    pytest.main([__file__])
