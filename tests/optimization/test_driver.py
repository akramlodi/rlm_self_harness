"""Tests for the U5 mining driver: run, persist, then mine.

Persist-first is the design center (KTD5). Every run -- pass, fail, or resource
termination -- must leave a trace file and a manifest line on disk the moment it
completes, so a crash cannot lose paid runs and the round's pass set is
auditable from the manifest alone. Mining then consumes only what was
persisted: ``mine_round`` reads the manifest and trace files back and feeds the
``WeaknessMiner`` the recorded verdicts, never recomputing them.

The offline seam is the one every behavioral test in this repo uses: patch
``rlm.core.rlm.get_client`` to hand the runtime a scripted ``MockLM``. The
factory returns a fresh client per ``get_client`` call (the runtime constructs
one per completion) while all clients pop from one shared script, so per-run
budget accounting stays per-run.
"""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm.utils.exceptions import TimeoutExceededError
from shrlm.harness_identity import harness_hash
from shrlm.optimization.attribution import VALIDATOR_VERSION, AttributionCache, LLMAttributor
from shrlm.optimization.audit import AuditReport, run_audited_round
from shrlm.optimization.bundle import write_bundle
from shrlm.optimization.driver import (
    ACCOUNTING_VERSION,
    ACCOUNTING_VERSION_KEY,
    RoundConfig,
    RoundPersistenceError,
    build_round_rlm,
    execute_run,
    load_round,
    mine_round,
    persist_interrupted_run,
    prepare_round,
    round_dir,
    run_id_for,
    run_round,
)
from shrlm.optimization.mining import MiningResult, WeaknessMiner
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import H0
from tests.mock_lm import MockLM

# ---------------------------------------------------------------------------
# Offline seam: scripted clients with per-call cost
# ---------------------------------------------------------------------------

COST_PER_CALL = 0.001
# Two one-call runs stay under this; a two-call run crosses it after its second
# iteration, which is what makes the runtime raise BudgetExceededError mid-run.
MAX_BUDGET = 0.0015


class ScriptedLM(MockLM):
    """A ``MockLM`` popping from a shared script, with a per-call cost.

    Exhausting the script raises rather than improvising: a test whose scripted
    turns run out is a broken test, not a passing run. ``cost_source`` mirrors
    the azure_foundry client's provenance reporting; the default ``None``
    models a client that reports no source (openrouter today), so pre-existing
    tests' persisted bytes keep their exact shape.
    """

    def __init__(
        self,
        script: list[str],
        calls: list[str],
        cost_per_call: float,
        cost_source: str | None = None,
    ):
        super().__init__(model_name="mock-model")
        self._script = script
        self._calls = calls
        self._cost_per_call = cost_per_call
        self._cost_source = cost_source

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self._call_count += 1
        self._calls.append("call")
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
                    cost_source=self._cost_source,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=10,
            total_output_tokens=10,
            total_cost=self._cost_per_call,
            cost_source=self._cost_source,
        )


class ClientFactory:
    """Stands in for ``get_client``: fresh client per call, one shared script."""

    def __init__(
        self,
        script: list[str],
        cost_per_call: float = COST_PER_CALL,
        cost_source: str | None = None,
    ):
        self.script = list(script)
        self.calls: list[str] = []
        self.cost_per_call = cost_per_call
        self.cost_source = cost_source

    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> ScriptedLM:
        return ScriptedLM(self.script, self.calls, self.cost_per_call, self.cost_source)

    @property
    def total_calls(self) -> int:
        return len(self.calls)


def final(content: str) -> str:
    """A ```repl``` block that returns ``content`` from the answer variable."""
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


# ---------------------------------------------------------------------------
# Instances, verifier, attributor
# ---------------------------------------------------------------------------


@dataclass
class GoldVerifier:
    """Deterministic string-match verifier; counts calls so tests can pin usage."""

    calls: int = 0

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        self.calls += 1
        gold = str(instance["gold"])
        if produced == gold:
            return Verdict(passed=True, cause=None, gold=gold, produced=produced)
        return Verdict(
            passed=False,
            cause=VerifierCause.WRONG_VALUE,
            gold=gold,
            produced=produced,
            detail="string mismatch",
        )


@dataclass
class BoomVerifier:
    """A verifier that must never run: mining consumes persisted verdicts only."""

    calls: int = 0

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        raise AssertionError("mine_round must use persisted verdicts, not the verifier")


def make_instances() -> list[dict[str, Any]]:
    return [
        {"id": "inst-pass", "question": "q one", "prompt": "context one", "gold": "RIGHT"},
        {"id": "inst-fail", "question": "q two", "prompt": "context two", "gold": "RIGHT"},
        {"id": "inst-term", "question": "q three", "prompt": "context three", "gold": "RIGHT"},
    ]


def full_script() -> list[str]:
    """Run 1 passes, run 2 fails, run 3 burns two iterations into the budget cap."""
    return [
        final("RIGHT"),
        final("WRONG"),
        "Scanning part one before answering.",
        "Scanning part two before answering.",
    ]


def make_round_config(tmp_path: Path, **overrides: Any) -> RoundConfig:
    values: dict[str, Any] = {
        "round_index": 1,
        "harness": H0,
        "instances": make_instances(),
        "verifier": GoldVerifier(),
        "out_dir": tmp_path,
        "backend": "openai",
        "backend_kwargs": {"model_name": "driver-test"},
        "attempts": 1,
        "max_iterations": 3,
        "max_budget": MAX_BUDGET,
    }
    values.update(overrides)
    return RoundConfig(**values)


# The attributor runs ungrounded (no sub-verifier), so the canned response must
# carry ``failing_level``. Node id "r" is the root and exists in every tree.
CANNED_ATTRIBUTION = (
    "```json\n"
    + json.dumps(
        {
            "causal_status": "causal",
            "agent_mechanism": "lossy_aggregation",
            "failing_level": "root",
            "evidence_node_ids": ["r"],
            "symptom_summary": "the merge step dropped a sub-result",
        }
    )
    + "\n```"
)


class NoneSubVerifier:
    """A sub-verifier stub for which no child is checkable (always None).

    Over the narrow MockLM trees the driver tests produce (root only, no
    sub-calls), ``derive_failing_level`` returns NO_RECURSION, so every record
    is grounded and the attributor renders the grounded prompt variant -- a
    second, distinct variant for the very same persisted round.
    """

    def __call__(self, instance: dict[str, Any], node: Any) -> bool | None:
        return None


def make_miner(
    verifier: Any,
    responses: list[str] | None = None,
    sub_verifier: Any = None,
    cache: AttributionCache | None = None,
) -> WeaknessMiner:
    lm = MockLM(responses=responses if responses is not None else [CANNED_ATTRIBUTION] * 8)
    return WeaknessMiner(
        verifier=verifier, attributor=LLMAttributor(lm, cache=cache), sub_verifier=sub_verifier
    )


def read_manifest(round_path: Path) -> list[dict[str, Any]]:
    lines = (round_path / "runs.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_full_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RoundConfig:
    factory = ClientFactory(full_script())
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_round_config(tmp_path)
    run_round(config)
    return config


@dataclass
class DualMineOutcome:
    """What ``run_dual_mined_round`` hands back from its two mining passes."""

    first_result: MiningResult
    first_report: AuditReport
    second_miner: WeaknessMiner
    second_result: MiningResult
    second_report: AuditReport


def run_dual_mined_round(
    monkeypatch: pytest.MonkeyPatch,
    script: list[str],
    first_config: RoundConfig,
    first_miner: WeaknessMiner,
    second_config: RoundConfig,
    second_miner_factory: Callable[[], WeaknessMiner],
    second_label: str,
) -> DualMineOutcome:
    """The dual-mine core shared by the ablation and labeled-destination tests.

    One scripted round is run and mined into the round root, then the
    now-complete persisted round is mined a second time into
    ``bundles/<second_label>/``. The runtime seam is silenced for the second
    pass and asserted silent: a complete round resumes as a no-op, so only
    the attributor may call out. ``second_miner_factory`` is called only
    after the first pass finishes, so a file-backed ``AttributionCache`` it
    constructs loads the entries the first pass persisted -- the shared-cache
    semantics both callers test. The materially different parts (scripts,
    instances, each pass's mining mode) stay at the call sites.
    """
    factory = ClientFactory(script)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    first_result, first_report = run_audited_round(
        first_config, first_miner, split_id="held_in_v1", created_at="2026-01-01T00:00:00"
    )

    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    second_miner = second_miner_factory()
    second_result, second_report = run_audited_round(
        second_config,
        second_miner,
        split_id="held_in_v1",
        created_at="2026-02-02T00:00:00",
        bundle_label=second_label,
    )
    assert idle.total_calls == 0

    return DualMineOutcome(
        first_result=first_result,
        first_report=first_report,
        second_miner=second_miner,
        second_result=second_result,
        second_report=second_report,
    )


# ---------------------------------------------------------------------------
# Persist-first: every run leaves a manifest line and a matching trace
# ---------------------------------------------------------------------------


class TestPersistFirst:
    def test_three_runs_yield_three_manifest_lines(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        entries = read_manifest(round_path)

        assert [entry["instance_id"] for entry in entries] == [
            "inst-pass",
            "inst-fail",
            "inst-term",
        ]
        assert [entry["passed"] for entry in entries] == [True, False, False]

    def test_termination_line_carries_resource_terminated(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        terminated = entries[2]

        assert terminated["passed"] is False
        assert terminated["cause"] == VerifierCause.RESOURCE_TERMINATED.value
        assert terminated["verdict"]["cause"] == VerifierCause.RESOURCE_TERMINATED.value
        assert "BudgetExceededError" in terminated["verdict"]["detail"]
        # The per-completion usage summary is gone when the root raises, but
        # the exception carries what the run spent; the driver persists that
        # figure so the validation stage's spend accounting never undercounts
        # a paid termination.
        assert terminated["cost"] == pytest.approx(2 * COST_PER_CALL)

    def test_pass_rate_is_recomputable_from_the_manifest_alone(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        assert sum(1 for entry in entries if entry["passed"]) / len(entries) == pytest.approx(1 / 3)

    def test_trace_files_exist_and_shas_match(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        for entry in read_manifest(round_path):
            trace = round_path / entry["trace_path"]
            assert trace.exists()
            assert hashlib.sha256(trace.read_bytes()).hexdigest() == entry["trace_sha256"]
            completion = RLMChatCompletion.from_dict(json.loads(trace.read_text()))
            assert completion.metadata is not None

    def test_terminated_trace_holds_the_partial_trajectory(self, tmp_path, monkeypatch):
        """The budget check runs before the terminating iteration is logged
        (rlm/core/rlm.py), so the partial trajectory holds only the iterations
        completed before it -- here, one of the two the run executed."""
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        entry = read_manifest(round_path)[2]
        completion = RLMChatCompletion.from_dict(
            json.loads((round_path / entry["trace_path"]).read_text())
        )
        assert completion.error is not None and "BudgetExceededError" in completion.error
        assert len(completion.metadata["iterations"]) == 1

    def test_successful_runs_record_their_cost(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        assert entries[0]["cost"] == pytest.approx(COST_PER_CALL)
        assert entries[1]["cost"] == pytest.approx(COST_PER_CALL)


# ---------------------------------------------------------------------------
# U4 manifest usage keys: tokens, wall time, lower-bound flag (R4)
# ---------------------------------------------------------------------------


class TestManifestUsageKeys:
    def test_successful_line_carries_tokens_time_and_legacy_cost(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        # Run 1 is a single scripted call; ScriptedLM reports 10 tokens per
        # call in each direction.
        assert entries[0]["input_tokens"] == 10
        assert entries[0]["output_tokens"] == 10
        assert entries[0]["execution_time"] > 0.0
        assert entries[0]["usage_lower_bound"] is False
        assert entries[0]["cost"] == pytest.approx(COST_PER_CALL)

    def test_terminated_line_preserves_observed_usage_as_lower_bound(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        terminated = entries[2]
        # The completion context publishes the run's total before the handler
        # holding it is stopped, so a terminated run carries the calls, tokens
        # and cost the runtime actually recorded -- two scripted calls at 10
        # tokens each way. The flag stays true regardless: a request killed in
        # flight or a route that reports no cost is still unrecorded, so the
        # figure is a lower bound even though it is no longer empty.
        assert terminated["usage_lower_bound"] is True
        assert terminated["cost"] == pytest.approx(2 * COST_PER_CALL)
        assert terminated["execution_time"] > 0.0
        assert terminated["input_tokens"] == 20
        assert terminated["output_tokens"] == 20

    def test_cost_source_is_persisted_when_the_client_reports_one(self, tmp_path, monkeypatch):
        """R7 provenance: a client reporting cost_source (the azure_foundry
        synthesis path) has it recorded on every completed run's manifest
        line -- the terminated run included, since its usage is now the
        runtime's own published total rather than a figure rebuilt from an
        exception that carried no provenance."""
        factory = ClientFactory(full_script(), cost_source="synthesized")
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path)
        run_round(config)

        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        assert entries[0]["cost_source"] == "synthesized"
        assert entries[1]["cost_source"] == "synthesized"
        assert entries[2]["cost_source"] == "synthesized"  # terminated: published with provenance

    def test_cost_source_is_omitted_when_the_client_reports_none(self, tmp_path, monkeypatch):
        """A client with no provenance (openrouter today) changes no persisted
        byte: the additive key is simply absent, keeping prior line shapes."""
        config = run_full_round(tmp_path, monkeypatch)
        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        assert all("cost_source" not in entry for entry in entries)


class TestSharedRunExecutionBody:
    """KTD14: one implementation of "execute one run and describe its outcome".

    A run child cannot call ``run_round`` -- that writes the shared files the
    parent owns -- so without extraction it would re-implement the timing
    window, the limit handling, and the lower-bound flag, and the two copies
    would drift apart silently. These pin that the extracted helpers produce
    exactly what the loop produces.
    """

    def test_the_preamble_returns_the_pending_list_in_dispatch_order(self, tmp_path, monkeypatch):
        """R7: instance-major, attempt-minor -- the order an early stop truncates."""
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path, attempts=2)

        path, existing, pending = prepare_round(config)

        assert existing == []
        assert path == round_dir(config.out_dir, config.round_index)
        assert [(str(instance["id"]), attempt) for instance, attempt in pending] == [
            (str(instance["id"]), attempt) for instance in config.instances for attempt in (1, 2)
        ]

    def test_the_preamble_omits_runs_already_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path)
        run_round(config, stop_after=1)

        _path, existing, pending = prepare_round(config)

        assert len(existing) == 1
        assert str(config.instances[0]["id"]) not in {str(i["id"]) for i, _a in pending}

    def test_the_helper_produces_what_the_loop_persists(self, tmp_path, monkeypatch):
        """Same scripted turn through the helper and through run_round."""
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path)
        run_round(config, stop_after=1)
        looped = read_manifest(round_dir(config.out_dir, config.round_index))[0]

        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        direct_config = make_round_config(tmp_path / "direct")
        harnessed = build_round_rlm(direct_config)
        outcome = execute_run(
            harnessed,
            direct_config.instances[0],
            model_name="driver-test",
            verifier=direct_config.verifier,
        )

        assert outcome.usage_lower_bound is looped["usage_lower_bound"]
        assert outcome.verdict.passed is looped["passed"]
        assert outcome.completion.usage_summary.total_cost == pytest.approx(looped["cost"])

    def test_a_verifier_that_raises_a_limit_is_caught_as_a_terminated_run(
        self, tmp_path, monkeypatch
    ):
        """A deadline landing during verification must not escape the round.

        Verification runs inside the per-run limit handler precisely so this
        case persists a terminated line rather than tearing down the round with
        the in-flight run unaccounted for.
        """
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path)
        harnessed = build_round_rlm(config)

        def verifier_that_times_out(instance, response):
            raise TimeoutExceededError(elapsed=9.0, timeout=1.0)

        outcome = execute_run(
            harnessed,
            config.instances[0],
            model_name="driver-test",
            verifier=verifier_that_times_out,
        )

        assert outcome.usage_lower_bound is True
        assert outcome.verdict.cause is VerifierCause.RESOURCE_TERMINATED
        assert "TimeoutExceededError" in outcome.verdict.detail

    def test_omitting_the_verifier_leaves_the_verdict_for_the_parent(self, tmp_path, monkeypatch):
        """A run child never verifies -- the parent owns verdict construction (KTD5)."""
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path)
        harnessed = build_round_rlm(config)

        outcome = execute_run(harnessed, config.instances[0], model_name="driver-test")

        assert outcome.verdict is None
        assert outcome.completion.response


class TestInterruptedRunAttribution:
    """R16: a lost run is recovered under its own id, never another run's.

    "The first pending run" is only the run that failed while exactly one can
    be in flight. Under concurrency it names a run that may not have started,
    and charging it would attribute one run's spend and failure to another.
    """

    def _config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        return make_round_config(tmp_path)

    def test_the_default_recovers_the_first_pending_run(self, tmp_path, monkeypatch):
        """The sequential path is unchanged: no run id means first-pending."""
        config = self._config(tmp_path, monkeypatch)

        entry = persist_interrupted_run(config, TimeoutError("hung"))

        assert entry is not None
        assert entry["run_id"] == run_id_for(str(config.instances[0]["id"]), 1)
        assert entry["cause"] == VerifierCause.RESOURCE_TERMINATED.value
        assert entry["usage_lower_bound"] is True

    def test_a_named_run_is_recovered_rather_than_the_first_pending(self, tmp_path, monkeypatch):
        config = self._config(tmp_path, monkeypatch)
        target = run_id_for(str(config.instances[2]["id"]), 1)

        entry = persist_interrupted_run(config, TimeoutError("hung"), run_id=target)

        assert entry is not None
        assert entry["run_id"] == target
        # The runs before it are untouched -- still pending, not charged.
        persisted = [e["run_id"] for e in read_manifest(round_dir(config.out_dir, 1))]
        assert persisted == [target]

    def test_recovering_an_already_persisted_run_is_a_no_op(self, tmp_path, monkeypatch):
        config = self._config(tmp_path, monkeypatch)
        target = run_id_for(str(config.instances[0]["id"]), 1)
        persist_interrupted_run(config, TimeoutError("hung"), run_id=target)

        again = persist_interrupted_run(config, TimeoutError("hung again"), run_id=target)

        assert again is None
        persisted = [e["run_id"] for e in read_manifest(round_dir(config.out_dir, 1))]
        assert persisted == [target]  # no duplicate line

    def test_recovering_a_run_the_round_never_configured_is_refused(self, tmp_path, monkeypatch):
        config = self._config(tmp_path, monkeypatch)

        with pytest.raises(RoundPersistenceError) as excinfo:
            persist_interrupted_run(config, TimeoutError("hung"), run_id="no-such-run__a01")

        assert "no-such-run__a01" in str(excinfo.value)


class TestAccountingVersionMarker:
    """A directory may not mix runs priced under different accounting rules.

    The correction gives a terminated run a real cost where it previously had
    none, which moves every aggregate built over it and the promotion rule's
    cost band with them. A resumed out-dir would keep its old lines and add new
    ones, and nothing downstream could tell the two apart -- so the mixture is
    refused at load, where every reader goes through.
    """

    def test_every_line_records_the_accounting_version(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        entries = read_manifest(round_dir(config.out_dir, config.round_index))

        assert [entry[ACCOUNTING_VERSION_KEY] for entry in entries] == [ACCOUNTING_VERSION] * 3

    def test_a_manifest_mixing_versions_is_refused(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        entries = read_manifest(round_path)
        # Simulate resuming a directory written before the correction.
        entries[0].pop(ACCOUNTING_VERSION_KEY)
        (round_path / "runs.jsonl").write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
        )

        with pytest.raises(RoundPersistenceError) as excinfo:
            load_round(config.out_dir, config.round_index)

        assert "accounting" in str(excinfo.value)
        assert "fresh out-dir" in str(excinfo.value)

    def test_a_wholly_pre_correction_manifest_still_loads(self, tmp_path, monkeypatch):
        """Uniformly old is self-consistent: readable, just not mixable."""
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        stripped = [
            {key: value for key, value in entry.items() if key != ACCOUNTING_VERSION_KEY}
            for entry in read_manifest(round_path)
        ]
        (round_path / "runs.jsonl").write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in stripped)
        )

        runs, verdicts, _envelope, entries = load_round(config.out_dir, config.round_index)

        assert len(runs) == 3
        assert all(ACCOUNTING_VERSION_KEY not in entry for entry in entries)


class TestAccountingPreflight:
    """The guard has to fire before money is spent, not after.

    A directory written under older accounting reads back perfectly well on its
    own -- every line agrees with every other. The mixed-version backstop
    therefore cannot see the problem until a new line has already landed beside
    the old ones, by which point a real run has been paid for and the directory
    is unreadable for good. The refusal belongs at the moment work is about to
    execute.
    """

    def _legacy_manifest(self, config) -> Path:
        round_path = round_dir(config.out_dir, config.round_index)
        entries = read_manifest(round_path)
        stripped = [
            {key: value for key, value in entry.items() if key != ACCOUNTING_VERSION_KEY}
            for entry in entries
        ]
        (round_path / "runs.jsonl").write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in stripped)
        )
        return round_path

    def test_resuming_a_legacy_round_is_refused_before_any_run_executes(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path)
        run_round(config, stop_after=1)
        round_path = self._legacy_manifest(config)
        before = (round_path / "runs.jsonl").read_text()

        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        with pytest.raises(RoundPersistenceError) as excinfo:
            run_round(config)

        assert "accounting" in str(excinfo.value)
        assert "fresh out-dir" in str(excinfo.value)
        # Nothing was executed and nothing was appended: the directory is
        # exactly as readable as it was before the attempt.
        assert (round_path / "runs.jsonl").read_text() == before
        assert len(read_manifest(round_path)) == 1

    def test_a_fully_persisted_legacy_round_still_resumes_as_a_no_op(self, tmp_path, monkeypatch):
        """Nothing pending means nothing to mix; the round is complete as it is."""
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path)
        run_round(config)
        self._legacy_manifest(config)

        entries = run_round(config)

        assert len(entries) == 3
        assert all(ACCOUNTING_VERSION_KEY not in entry for entry in entries)

    def test_a_manifest_priced_under_an_unknown_version_is_refused(self, tmp_path, monkeypatch):
        """A build from the future priced these lines; how they differ is unknown."""
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(full_script()))
        config = make_round_config(tmp_path)
        run_round(config)
        round_path = round_dir(config.out_dir, config.round_index)
        entries = read_manifest(round_path)
        for entry in entries:
            entry[ACCOUNTING_VERSION_KEY] = "shrlm-accounting/v9"
        (round_path / "runs.jsonl").write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
        )

        with pytest.raises(RoundPersistenceError) as excinfo:
            load_round(config.out_dir, config.round_index)

        assert "unrecognized" in str(excinfo.value)
        assert "shrlm-accounting/v9" in str(excinfo.value)


class TestManifestBackwardCompatibility:
    def test_old_format_manifest_lines_still_load_through_load_round(self, tmp_path, monkeypatch):
        """Manifest lines persisted before U4 carry no token/time keys; they
        must keep loading (backward compatibility, KTD4 additive-only)."""
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        stripped = [
            {
                key: value
                for key, value in entry.items()
                if key
                not in ("input_tokens", "output_tokens", "execution_time", "usage_lower_bound")
            }
            for entry in read_manifest(round_path)
        ]
        (round_path / "runs.jsonl").write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in stripped)
        )

        runs, verdicts, _envelope, entries = load_round(config.out_dir, config.round_index)
        assert len(runs) == 3
        assert len(verdicts) == 3
        assert all("input_tokens" not in entry for entry in entries)


# ---------------------------------------------------------------------------
# Round identity: harness.json and instances.jsonl
# ---------------------------------------------------------------------------


class TestRoundIdentity:
    def test_harness_json_hash_matches_h0(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        envelope = json.loads(
            (round_dir(config.out_dir, config.round_index) / "harness.json").read_text()
        )
        assert envelope["hash"] == harness_hash(H0)
        assert envelope["format"] == "shrlm-harness/v2"

    def test_instances_jsonl_round_trips(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        lines = (
            (round_dir(config.out_dir, config.round_index) / "instances.jsonl")
            .read_text()
            .splitlines()
        )
        assert [json.loads(line) for line in lines] == make_instances()


# ---------------------------------------------------------------------------
# Resume: completed runs are skipped, never re-run, never rewritten
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_executes_only_the_missing_run(self, tmp_path, monkeypatch):
        first = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", first)
        config = make_round_config(tmp_path)
        run_round(config, stop_after=2)  # simulated crash after two paid runs

        round_path = round_dir(config.out_dir, config.round_index)
        before = (round_path / "runs.jsonl").read_bytes()
        assert len(read_manifest(round_path)) == 2
        assert first.total_calls == 2

        resumed = ClientFactory(full_script()[2:])
        monkeypatch.setattr(rlm_module, "get_client", resumed)
        run_round(make_round_config(tmp_path))

        entries = read_manifest(round_path)
        assert len(entries) == 3
        # Only the missing run touched the model: two iterations of run 3.
        assert resumed.total_calls == 2
        # The pre-existing lines are untouched byte-for-byte.
        assert (round_path / "runs.jsonl").read_bytes()[: len(before)] == before

    def test_fully_complete_round_makes_no_model_calls(self, tmp_path, monkeypatch):
        run_full_round(tmp_path, monkeypatch)
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        entries = run_round(make_round_config(tmp_path))
        assert len(entries) == 3
        assert idle.total_calls == 0

    def test_complete_round_resumes_without_the_backend_credential(self, tmp_path, monkeypatch):
        """A fully persisted round makes zero calls, so it must be resumable
        (as a no-op) on a machine that holds no credential at all."""
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        run_round(make_round_config(tmp_path, backend="openrouter"))

        monkeypatch.delenv("OPENROUTER_API_KEY")
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        entries = run_round(make_round_config(tmp_path, backend="openrouter"))
        assert len(entries) == 3
        assert idle.total_calls == 0

    def test_incomplete_round_still_demands_the_credential_before_any_run(
        self, tmp_path, monkeypatch
    ):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        run_round(make_round_config(tmp_path, backend="openrouter"), stop_after=2)

        monkeypatch.delenv("OPENROUTER_API_KEY")
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            run_round(make_round_config(tmp_path, backend="openrouter"))
        assert idle.total_calls == 0
        # The two paid runs stay persisted, untouched by the refused resume.
        entries = read_manifest(round_dir(tmp_path, 1))
        assert len(entries) == 2

    def test_trace_sha_mismatch_fails_loudly_without_overwrite(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        entry = read_manifest(round_path)[0]
        trace = round_path / entry["trace_path"]
        trace.write_text("tampered")

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        with pytest.raises(RuntimeError, match="sha256"):
            run_round(make_round_config(tmp_path))
        assert trace.read_text() == "tampered"
        assert idle.total_calls == 0


# ---------------------------------------------------------------------------
# Attempts and fail-fast preconditions
# ---------------------------------------------------------------------------


class TestAttemptsAndPreconditions:
    def test_two_attempts_yield_two_distinct_run_ids(self, tmp_path, monkeypatch):
        factory = ClientFactory([final("RIGHT"), final("WRONG")])
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path, instances=[make_instances()[0]], attempts=2)
        run_round(config)

        entries = read_manifest(round_dir(config.out_dir, config.round_index))
        assert len(entries) == 2
        assert len({entry["run_id"] for entry in entries}) == 2
        assert [entry["attempt"] for entry in entries] == [1, 2]
        assert [entry["passed"] for entry in entries] == [True, False]

    def test_missing_openrouter_key_fails_before_any_run(self, tmp_path, monkeypatch):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        config = make_round_config(tmp_path, backend="openrouter")

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            run_round(config)
        assert factory.total_calls == 0
        assert not (round_dir(config.out_dir, config.round_index) / "runs.jsonl").exists()

    def test_azure_backend_with_both_env_vars_missing_names_both(self, tmp_path, monkeypatch):
        """azure_foundry requires two env vars; the fail-fast names each one."""
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
        config = make_round_config(tmp_path, backend="azure_foundry")

        with pytest.raises(RuntimeError) as excinfo:
            run_round(config)
        message = str(excinfo.value)
        assert "AZURE_API_KEY" in message
        assert "AZURE_FOUNDRY_ENDPOINT" in message
        assert factory.total_calls == 0
        assert not (round_dir(config.out_dir, config.round_index) / "runs.jsonl").exists()

    def test_azure_backend_with_one_env_var_missing_names_only_it(self, tmp_path, monkeypatch):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        monkeypatch.setenv("AZURE_API_KEY", "present-key")
        monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
        config = make_round_config(tmp_path, backend="azure_foundry")

        with pytest.raises(RuntimeError) as excinfo:
            run_round(config)
        message = str(excinfo.value)
        assert "AZURE_FOUNDRY_ENDPOINT" in message
        assert "AZURE_API_KEY" not in message
        assert factory.total_calls == 0
        assert not (round_dir(config.out_dir, config.round_index) / "runs.jsonl").exists()

    def test_api_key_material_in_backend_kwargs_is_rejected(self, tmp_path, monkeypatch):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(
            tmp_path, backend_kwargs={"model_name": "m", "api_key": "sk-nope"}
        )
        with pytest.raises(ValueError, match="api_key"):
            run_round(config)
        assert factory.total_calls == 0

    def test_sensitive_top_level_kwarg_is_rejected_while_nested_pricing_passes(
        self, tmp_path, monkeypatch
    ):
        """Regression guard for the nested-pricing choice: the sensitive scan
        matches top-level kwarg names only, so ``pricing`` (whose leaf key
        contains no sensitive fragment anyway) must pass while any top-level
        name carrying a sensitive fragment is still refused."""
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        refused = make_round_config(
            tmp_path / "refused",
            backend_kwargs={"model_name": "m", "access_token": "tok-nope"},
        )
        with pytest.raises(ValueError, match="access_token"):
            run_round(refused)
        assert factory.total_calls == 0

        accepted = make_round_config(
            tmp_path / "accepted",
            backend_kwargs={
                "model_name": "driver-test",
                "pricing": {"input_per_million": 0.60, "output_per_million": 3.00},
            },
        )
        entries = run_round(accepted)
        assert len(entries) == 3

    def test_duplicate_instance_ids_are_rejected(self, tmp_path, monkeypatch):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        instance = make_instances()[0]
        config = make_round_config(tmp_path, instances=[instance, dict(instance)])
        with pytest.raises(ValueError, match="duplicate"):
            run_round(config)
        assert factory.total_calls == 0


# ---------------------------------------------------------------------------
# Credential hygiene: env sentinel values never reach disk or error text
# ---------------------------------------------------------------------------


AZURE_KEY_SENTINEL = "sentinel-azure-key-8b1f2c"
AZURE_ENDPOINT_SENTINEL = "https://sentinel-resource-9d4e.services.ai.azure.com"


class TestCredentialHygiene:
    def test_azure_sentinels_never_reach_persisted_round_bytes(self, tmp_path, monkeypatch):
        """Credentials live in the environment only (R3): with sentinel values
        in both Azure variables, a full round persists no file containing
        either value, and the serialized backend_kwargs carry neither."""
        monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY_SENTINEL)
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", AZURE_ENDPOINT_SENTINEL)
        factory = ClientFactory(full_script(), cost_source="synthesized")
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path, backend="azure_foundry")
        run_round(config)

        assert AZURE_KEY_SENTINEL not in repr(config.backend_kwargs)
        assert AZURE_ENDPOINT_SENTINEL not in repr(config.backend_kwargs)
        scanned = 0
        for path in round_dir(config.out_dir, config.round_index).rglob("*"):
            if not path.is_file():
                continue
            scanned += 1
            content = path.read_text(errors="replace")
            assert AZURE_KEY_SENTINEL not in content, path
            assert AZURE_ENDPOINT_SENTINEL not in content, path
        assert scanned >= 5  # harness.json, instances.jsonl, runs.jsonl, 3 traces

    def test_fail_fast_error_text_carries_no_sentinel_value(self, tmp_path, monkeypatch):
        """The missing-endpoint error names the variable, never the value the
        other (set) variable holds."""
        monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY_SENTINEL)
        monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path, backend="azure_foundry")

        with pytest.raises(RuntimeError) as excinfo:
            run_round(config)
        message = str(excinfo.value)
        assert "AZURE_FOUNDRY_ENDPOINT" in message
        assert AZURE_KEY_SENTINEL not in message
        assert factory.total_calls == 0


# ---------------------------------------------------------------------------
# Mining consumes only persisted artifacts
# ---------------------------------------------------------------------------


class TestMineRound:
    def test_mine_round_uses_persisted_verdicts_only(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        miner = make_miner(BoomVerifier())

        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )

        assert result.bundle.totals.n_runs == 3
        assert result.bundle.totals.n_failures == 2
        causes = {record.verdict.cause for record in result.records}
        assert causes == {VerifierCause.WRONG_VALUE, VerifierCause.RESOURCE_TERMINATED}

    def test_mine_round_defaults_harness_version_to_the_recorded_hash(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        assert result.bundle.config.harness_version == harness_hash(H0)
        assert result.bundle.config.round_index == config.round_index

    def test_terminated_run_is_attributed_from_its_partial_trace(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        by_id = {record.instance_id: record for record in result.records}
        terminated = by_id["inst-term"]
        assert terminated.verdict.cause is VerifierCause.RESOURCE_TERMINATED
        assert not terminated.attribution_failed
        assert terminated.signature is not None

    def test_unparseable_attribution_stays_visible_as_unattributed(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        miner = make_miner(BoomVerifier(), responses=["not a json block"] * 8)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )
        assert result.bundle.totals.n_unattributed == 2

    def test_mine_round_persists_each_digest_content_addressed(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        digests_dir = round_dir(config.out_dir, config.round_index) / "digests"
        assert len(result.records) == 2
        for record in result.records:
            digest_path = digests_dir / f"{record.digest_sha256}.txt"
            assert digest_path.is_file()
            text = digest_path.read_text()
            # Content-addressed: the file's own sha256 is the record's digest sha.
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record.digest_sha256

    def test_mine_round_persists_the_rendered_attributor_prompt(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        miner = make_miner(BoomVerifier())
        mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )
        # Narrow trees with no descendants are grounded NO_RECURSION with or
        # without a sub-verifier: the single grounded, non-aggregate variant,
        # persisted under its content-addressed (sha-suffixed) name.
        sha = miner.attributor.prompt_sha256(grounded=True, no_subcalls=True)
        round_path = round_dir(config.out_dir, config.round_index)
        prompt_path = round_path / f"attributor_prompt_{sha[:16]}.txt"
        assert prompt_path.is_file()
        assert prompt_path.read_text() == miner.attributor.system_prompt(
            grounded=True, no_subcalls=True
        )
        # The legacy unsuffixed name is never written by new mines.
        assert not (round_path / "attributor_prompt.txt").exists()

    def test_every_record_links_to_its_persisted_trace(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        round_path = round_dir(config.out_dir, config.round_index)
        manifest = {entry["run_id"]: entry for entry in read_manifest(round_path)}
        assert len(result.records) == 2
        for record in result.records:
            assert record.run_id in manifest
            assert record.trace_path == manifest[record.run_id]["trace_path"]
            trace = round_path / record.trace_path
            assert trace.is_file()
            assert hashlib.sha256(trace.read_bytes()).hexdigest() == record.trace_sha256

    def test_trace_links_serialize_into_records_jsonl(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        write_bundle(result.bundle, result.records, str(tmp_path / "bundles"))
        lines = (
            (tmp_path / "bundles" / f"round_{config.round_index:02d}" / "records.jsonl")
            .read_text()
            .splitlines()
        )
        for line in lines:
            payload = json.loads(line)
            assert payload["run_id"]
            assert payload["trace_path"]
            assert payload["trace_sha256"]

    def test_bundle_config_carries_round_provenance(self, tmp_path, monkeypatch):
        instances = [
            {"id": "s-fail", "question": "q", "prompt": "ctx", "gold": "RIGHT", "sample_seed": 7},
            {"id": "s-pass", "question": "q", "prompt": "ctx", "gold": "RIGHT", "sample_seed": 7},
        ]
        factory = ClientFactory([final("WRONG"), final("RIGHT")])
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path, instances=instances)
        run_round(config)

        cache_path = tmp_path / "caches" / "attribution.jsonl"
        lm = MockLM(responses=[CANNED_ATTRIBUTION] * 8)
        miner = WeaknessMiner(
            verifier=BoomVerifier(),
            attributor=LLMAttributor(lm, cache=AttributionCache(path=str(cache_path))),
        )
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )
        bundle_config = result.bundle.config
        assert bundle_config.harness_hash == harness_hash(H0)
        # harness_version defaults to the same hash; both stay coherent.
        assert bundle_config.harness_version == bundle_config.harness_hash
        assert bundle_config.sampling_seed == 7
        round_path = round_dir(config.out_dir, config.round_index)
        assert bundle_config.attribution_cache_path == os.path.relpath(cache_path, round_path)
        assert bundle_config.validator_version == VALIDATOR_VERSION

    def test_all_pass_round_stamps_no_path_for_a_never_created_cache_file(
        self, tmp_path, monkeypatch
    ):
        """put() never runs on an all-pass round, so the file-backed cache is
        never created; stamping its path would be a broken audit link."""
        factory = ClientFactory([final("RIGHT"), final("RIGHT")])
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path, instances=make_instances()[:2])
        run_round(config)

        cache_path = tmp_path / "caches" / "attribution.jsonl"
        miner = WeaknessMiner(
            verifier=BoomVerifier(),
            attributor=LLMAttributor(
                MockLM(responses=[]), cache=AttributionCache(path=str(cache_path))
            ),
        )
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )
        assert result.bundle.config.attribution_cache_path is None
        assert not cache_path.exists()

    def test_zero_failure_round_mines_to_an_empty_pattern_list(self, tmp_path, monkeypatch):
        instances = make_instances()[:2]
        factory = ClientFactory([final("RIGHT"), final("RIGHT")])
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path, instances=instances)
        run_round(config)

        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        assert result.bundle.totals.n_failures == 0
        assert result.bundle.patterns == []
        integrity = result.bundle.integrity
        assert integrity.n_unattributed == 0
        assert integrity.n_ungrounded == 0
        assert integrity.n_resource_terminated == 0
        assert integrity.n_transport_errors == 0
        assert [bias.defect_id for bias in integrity.known_substrate_biases] == ["A1", "A2"]

    def test_resource_terminated_run_is_counted_in_integrity(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        assert result.bundle.integrity.n_resource_terminated == 1

    def test_mine_round_fails_on_tampered_trace(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        round_path = round_dir(config.out_dir, config.round_index)
        entry = read_manifest(round_path)[1]
        (round_path / entry["trace_path"]).write_text("tampered")
        with pytest.raises(RuntimeError, match="sha256"):
            mine_round(
                out_dir=config.out_dir,
                round_index=config.round_index,
                miner=make_miner(BoomVerifier()),
                split_id="held_in_v1",
            )


# ---------------------------------------------------------------------------
# Content-addressed prompt persistence: a re-mine never clobbers a variant
# ---------------------------------------------------------------------------


class TestPromptPersistence:
    def test_remine_with_a_different_variant_leaves_both_prompt_files(self, tmp_path, monkeypatch):
        """Mining the same round twice with different prompt variants must
        leave both files on disk, each content-addressed by its prompt sha."""
        config = run_full_round(tmp_path, monkeypatch)
        result_ungrounded = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        # The round's runs have no descendants, so grounding renders the same
        # variant with or without a sub-verifier; the second variant comes from
        # a changed no-sub-calls evidence instruction embedded in the prompt.
        import shrlm.optimization.attribution as attribution_module

        monkeypatch.setattr(
            attribution_module,
            "EVIDENCE_INSTRUCTION_NO_SUBCALLS",
            attribution_module.EVIDENCE_INSTRUCTION_NO_SUBCALLS + " (variant B)",
        )
        result_grounded = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier(), sub_verifier=NoneSubVerifier()),
            split_id="held_in_v1",
        )

        ungrounded_shas = set(result_ungrounded.attributor_prompts)
        grounded_shas = set(result_grounded.attributor_prompts)
        assert ungrounded_shas and grounded_shas
        assert ungrounded_shas.isdisjoint(grounded_shas)

        round_path = round_dir(config.out_dir, config.round_index)
        all_prompts = {**result_ungrounded.attributor_prompts, **result_grounded.attributor_prompts}
        for sha, text in all_prompts.items():
            prompt_path = round_path / f"attributor_prompt_{sha[:16]}.txt"
            assert prompt_path.is_file()
            assert prompt_path.read_text() == text
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == sha

    def test_existing_prompt_file_with_correct_bytes_is_never_rewritten(
        self, tmp_path, monkeypatch
    ):
        """Content-addressed means write-once for intact artifacts: a file
        whose bytes already hash to its address is skipped, not rewritten."""
        config = run_full_round(tmp_path, monkeypatch)
        miner = make_miner(BoomVerifier())
        sha = miner.attributor.prompt_sha256(grounded=True, no_subcalls=True)
        text = miner.attributor.system_prompt(grounded=True, no_subcalls=True)
        round_path = round_dir(config.out_dir, config.round_index)
        prompt_path = round_path / f"attributor_prompt_{sha[:16]}.txt"
        prompt_path.write_text(text)
        mtime_before = os.stat(prompt_path).st_mtime_ns

        mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )

        assert prompt_path.read_text() == text
        assert os.stat(prompt_path).st_mtime_ns == mtime_before

    def test_existing_prompt_file_with_wrong_bytes_is_healed(self, tmp_path, monkeypatch):
        """A file at the sha-keyed name whose bytes do NOT hash to it (a
        truncated crash write, tampering) is healed with the correct bytes:
        trusting it would permanently break the audit's prompt hash link."""
        config = run_full_round(tmp_path, monkeypatch)
        miner = make_miner(BoomVerifier())
        sha = miner.attributor.prompt_sha256(grounded=True, no_subcalls=True)
        round_path = round_dir(config.out_dir, config.round_index)
        prompt_path = round_path / f"attributor_prompt_{sha[:16]}.txt"
        prompt_path.write_text("sentinel: bytes that do not hash to the file's name")

        mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )

        healed = prompt_path.read_text()
        assert healed == miner.attributor.system_prompt(grounded=True, no_subcalls=True)
        assert hashlib.sha256(healed.encode("utf-8")).hexdigest() == sha

    def test_truncated_digest_file_is_healed_on_remine(self, tmp_path, monkeypatch):
        """A digest file truncated after persistence (e.g. a crash mid-write
        under the old non-atomic writer) is restored by the next mining pass,
        so the record's digest_sha256 audit link works again."""
        config = run_full_round(tmp_path, monkeypatch)
        result = mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )
        round_path = round_dir(config.out_dir, config.round_index)
        record = result.records[0]
        digest_path = round_path / "digests" / f"{record.digest_sha256}.txt"
        full_text = digest_path.read_text()
        digest_path.write_text(full_text[: len(full_text) // 2])

        mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=make_miner(BoomVerifier()),
            split_id="held_in_v1",
        )

        assert digest_path.read_text() == full_text
        assert hashlib.sha256(digest_path.read_bytes()).hexdigest() == record.digest_sha256


if __name__ == "__main__":
    pytest.main([__file__])
