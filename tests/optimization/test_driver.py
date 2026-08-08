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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from shrlm.harness_identity import harness_hash
from shrlm.optimization.attribution import VALIDATOR_VERSION, AttributionCache, LLMAttributor
from shrlm.optimization.bundle import write_bundle
from shrlm.optimization.driver import RoundConfig, mine_round, round_dir, run_round
from shrlm.optimization.mining import WeaknessMiner
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
    turns run out is a broken test, not a passing run.
    """

    def __init__(self, script: list[str], calls: list[str], cost_per_call: float):
        super().__init__(model_name="mock-model")
        self._script = script
        self._calls = calls
        self._cost_per_call = cost_per_call

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
        self.calls: list[str] = []
        self.cost_per_call = cost_per_call

    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> ScriptedLM:
        return ScriptedLM(self.script, self.calls, self.cost_per_call)

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
) -> WeaknessMiner:
    lm = MockLM(responses=responses if responses is not None else [CANNED_ATTRIBUTION] * 8)
    return WeaknessMiner(verifier=verifier, attributor=LLMAttributor(lm), sub_verifier=sub_verifier)


def read_manifest(round_path: Path) -> list[dict[str, Any]]:
    lines = (round_path / "runs.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_full_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RoundConfig:
    factory = ClientFactory(full_script())
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_round_config(tmp_path)
    run_round(config)
    return config


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
        # The runtime raised before any usage summary existed for the run.
        assert terminated["cost"] is None

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
# Round identity: harness.json and instances.jsonl
# ---------------------------------------------------------------------------


class TestRoundIdentity:
    def test_harness_json_hash_matches_h0(self, tmp_path, monkeypatch):
        config = run_full_round(tmp_path, monkeypatch)
        envelope = json.loads(
            (round_dir(config.out_dir, config.round_index) / "harness.json").read_text()
        )
        assert envelope["hash"] == harness_hash(H0)
        assert envelope["format"] == "shrlm-harness/v1"

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

    def test_api_key_material_in_backend_kwargs_is_rejected(self, tmp_path, monkeypatch):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(
            tmp_path, backend_kwargs={"model_name": "m", "api_key": "sk-nope"}
        )
        with pytest.raises(ValueError, match="api_key"):
            run_round(config)
        assert factory.total_calls == 0

    def test_duplicate_instance_ids_are_rejected(self, tmp_path, monkeypatch):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        instance = make_instances()[0]
        config = make_round_config(tmp_path, instances=[instance, dict(instance)])
        with pytest.raises(ValueError, match="duplicate"):
            run_round(config)
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
        # No sub-verifier, narrow trees: the single ungrounded, non-aggregate
        # variant, persisted under its content-addressed (sha-suffixed) name.
        sha = miner.attributor.prompt_sha256(grounded=False, no_subcalls=True)
        round_path = round_dir(config.out_dir, config.round_index)
        prompt_path = round_path / f"attributor_prompt_{sha[:16]}.txt"
        assert prompt_path.is_file()
        assert prompt_path.read_text() == miner.attributor.system_prompt(
            grounded=False, no_subcalls=True
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

    def test_existing_prompt_file_is_never_rewritten(self, tmp_path, monkeypatch):
        """Content-addressed means write-once: a file already at the sha-keyed
        name is trusted as-is, never rewritten."""
        config = run_full_round(tmp_path, monkeypatch)
        miner = make_miner(BoomVerifier())
        sha = miner.attributor.prompt_sha256(grounded=False, no_subcalls=True)
        round_path = round_dir(config.out_dir, config.round_index)
        sentinel_path = round_path / f"attributor_prompt_{sha[:16]}.txt"
        sentinel_path.write_text("sentinel: pre-existing bytes")

        mine_round(
            out_dir=config.out_dir,
            round_index=config.round_index,
            miner=miner,
            split_id="held_in_v1",
        )

        assert sentinel_path.read_text() == "sentinel: pre-existing bytes"


if __name__ == "__main__":
    pytest.main([__file__])
