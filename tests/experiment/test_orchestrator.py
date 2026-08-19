"""Tests for the U5 round orchestrator: crash-safe rounds, frozen harness.

The round-trip gate comes first (plan execution note): resume and eval both
stand on ``materialize_harness(envelope["harness"], module_path)``
reconstructing a promoted -- including MERGED -- harness with a matching
``harness_hash``. Only with that gate green does the orchestrator's resume
logic (rebuilding the incumbent from the decision.json/ledger chain) rest on
verified ground.

Everything else runs the real ``run_experiment`` offline: the runtime seam is
the repo-standard ``rlm.core.rlm.get_client`` monkeypatch handing out scripted
clients with per-call costs (``tests.optimization.test_driver``), the
attributor and proposer are injected ``MockLM``s, and splits come from
injected offline loaders. Crashes are simulated the honest way -- a scripted
client whose script runs out raises mid-round, leaving exactly the persisted
state a real crash leaves.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
import shrlm.experiment.analysis_io as analysis_io_module
import shrlm.experiment.incumbent_quality as incumbent_quality_module
import shrlm.experiment.orchestrator as orchestrator_module
from shrlm.experiment.analysis_io import (
    ANALYSIS_DIR,
    PROVENANCE_FILENAME,
    is_published,
    latest_snapshot,
)
from shrlm.experiment.collapse_and_attribution import (
    EVALUATION_FILENAME,
    OPTIMIZATION_FILENAME,
    TOOL_NAME_EVALUATION,
    TOOL_NAME_OPTIMIZATION,
)
from shrlm.experiment.config import ExperimentConfig, load_config
from shrlm.experiment.evaluation import EVAL_DIR
from shrlm.experiment.incumbent_quality import (
    INCUMBENT_QUALITY_CANDIDATES_FILENAME,
    INCUMBENT_QUALITY_FILENAME,
)
from shrlm.experiment.incumbent_quality import TOOL_NAME as TOOL_NAME_INCUMBENT_QUALITY
from shrlm.experiment.orchestrator import (
    CONFIG_FILENAME,
    EVIDENCE_MARKER_FILENAME,
    FROZEN_DIR,
    FROZEN_HARNESS_FILENAME,
    PROPOSALS_MARKER_FILENAME,
    ROUND_MARKER_FILENAME,
    STOP_MAX_ROUNDS,
    STOP_PATIENCE,
    ExperimentPersistenceError,
    IdentityMismatchError,
    MiningBudgetExceededError,
    check_identity,
    experiment_round_dir,
    run_experiment,
)
from shrlm.experiment.pattern_frequency_diff import BUNDLES_FILENAME
from shrlm.experiment.pattern_frequency_diff import TOOL_NAME as TOOL_NAME_DIFF
from shrlm.experiment.splits import (
    MANIFEST_FILE,
    SPLITS_DIR,
    LoaderFn,
    sha256_text,
    split_file_name,
)
from shrlm.experiment.surface_activity import SURFACE_ACTIVITY_FILENAME, UNATTRIBUTED_FILENAME
from shrlm.experiment.surface_activity import TOOL_NAME as TOOL_NAME_SURFACE_ACTIVITY
from shrlm.experiment.usage import STAGE_USAGE_FILE, read_jsonl, read_stage_usage
from shrlm.harness_identity import harness_hash, write_harness_json
from shrlm.optimization.bundle import (
    ATTRIBUTIONS_FILENAME,
    RECORDS_FILENAME,
    round_dir,
    write_bundle,
)
from shrlm.optimization.candidates import materialize_harness
from shrlm.optimization.driver import TRACES_DIR, load_manifest
from shrlm.optimization.proposal import _import_candidate_function
from shrlm.optimization.validation import SPLIT_HELDIN, load_promotion_ledger
from shrlm.rlm_harness import H0
from tests.mock_lm import MockLM
from tests.optimization.test_driver import ClientFactory, GoldVerifier, final

# ---------------------------------------------------------------------------
# The round-trip gate: harness.json envelope -> materialize_harness -> same hash
# ---------------------------------------------------------------------------

S9_SOURCE = (
    "def redirect_empty(answer, repl_inventory):\n"
    "    if not answer.strip():\n"
    "        return AnswerDecision.redirect('aggregate more evidence first')\n"
    "    return AnswerDecision.accept(answer)\n"
)


def round_trip(harness, tmp_path: Path) -> None:
    """write_harness_json -> reload -> materialize_harness -> hash equality."""
    path = tmp_path / "harness.json"
    write_harness_json(harness, path)
    envelope = json.loads(path.read_text())
    rebuilt = materialize_harness(envelope["harness"], tmp_path / "surfaces.py")
    assert harness_hash(rebuilt) == envelope["hash"]
    assert envelope["hash"] == harness_hash(harness)


class TestHarnessRoundTrip:
    def test_single_surface_promoted_harness_round_trips(self, tmp_path):
        promoted = replace(
            H0,
            name="r01-c01-s4",
            verification_instruction=H0.verification_instruction + "\n[gate-edit]",
        )
        round_trip(promoted, tmp_path)

    def test_merged_two_text_surface_harness_round_trips(self, tmp_path):
        merged = replace(
            H0,
            name="H0+r01-c01-s2+r01-c02-s4",
            decomposition_instruction=H0.decomposition_instruction + "\n[merge-s2]",
            verification_instruction=H0.verification_instruction + "\n[merge-s4]",
        )
        round_trip(merged, tmp_path)

    def test_merged_harness_with_generated_code_surface_round_trips(self, tmp_path):
        """A merge whose S9 came from proposer-generated source (the hard case:
        the callable's identity is its getsource text, which must survive a
        write -> materialize -> re-serialize cycle exactly)."""
        middleware = _import_candidate_function(S9_SOURCE, "redirect_empty", tmp_path / "gen", "S9")
        merged = replace(
            H0,
            name="H0+r01-c01-s4+r01-c02-s9",
            verification_instruction=H0.verification_instruction + "\n[merge-s4]",
            answer_middleware=middleware,
        )
        round_trip(merged, tmp_path)


# ---------------------------------------------------------------------------
# Offline experiment fixtures: config template, loaders, canned stage output
# ---------------------------------------------------------------------------

# A complete experiment TOML at test scale: 2 held-in + 2 held-out instances,
# 1 mining and 1 validation repetition, k=2 candidates. gpu_scenarios sits at
# the top because a top-level assignment must precede any table header.
TOML_TEMPLATE = """\
gpu_scenarios = []

[decoding]
temperature = {temperature}
top_p = 0.8
top_k = 20
min_p = 0.0
max_output_tokens = 512

[splits]
n_in = 2
n_ho = 2
test_short = 1
test_long = 1
seed = 0

[loop]
m = 1
v = {v}
k = 2
t = {t}
patience = {patience}

[promotion]
tau_regression = 0.0
tau_improvement = 0.0
cost_band = [0.5, 2.0]

[caps]
attempts = 1
max_iterations = 3
max_depth = 2
max_budget = 0.01
max_timeout = 60.0
candidate_budget = {candidate_budget}

[environments.graphwalks]
dataset_repo = "openai/graphwalks"
dataset_file_short = "short.parquet"
dataset_file_long = "long.parquet"
dataset_revision = "{graphwalks_revision}"
max_chars = 128000
min_chars = 0
problem_types = ["bfs", "parents"]

[environments.oolong_pairs]
dataset_repo = "oolongbench/oolong-synth"
subset = "trec_coarse"
dataset_revision = "rev-oolong"
task_ids = [1, 2]
context_length_short = 8192
context_length_long = 262144
max_scan = 100
n_short = 1
n_long = 1

[backends.runner]
backend = "openai"
model = "runner-test"

[backends.attributor]
backend = "openai"
model = "attributor-test"

[backends.proposer]
backend = "openai"
model = "{proposer_model}"

[backends.openrouter]
provider_order = []
allow_fallbacks = true

[pricing.promo]
input_per_million = 0.08
output_per_million = 0.29

[pricing.list_price]
input_per_million = 0.10
output_per_million = 0.30

[report]
p_merge = 0.5
eval_conditions = 3

[operational]
loader_timeout_seconds = 60.0
attribution_cache_path = ".cache/attribution.jsonl"
proposal_cache_path = ".cache/proposals.jsonl"
eval_repetitions = 1

[smoke]
"""


def make_config(
    tmp_path: Path,
    *,
    temperature: float = 0.7,
    v: int = 1,
    t: int = 1,
    patience: int = 3,
    candidate_budget: float = 1.0,
    graphwalks_revision: str = "rev-gw",
    proposer_model: str = "proposer-test",
) -> ExperimentConfig:
    text = TOML_TEMPLATE.format(
        temperature=temperature,
        v=v,
        t=t,
        patience=patience,
        candidate_budget=candidate_budget,
        graphwalks_revision=graphwalks_revision,
        proposer_model=proposer_model,
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "experiment.toml"
    path.write_text(text)
    return load_config("full", path=path)


def fake_loader(environment: str) -> LoaderFn:
    def load(config: ExperimentConfig, length: str, limit: int, seed: int) -> list[dict[str, Any]]:
        return [
            {
                "id": f"{environment}-{length}-{index}",
                "question": f"{environment} question {index}",
                "prompt": f"{environment} {length} context {index}",
                "gold": "RIGHT",
            }
            for index in range(limit)
        ]

    return load


LOADERS: dict[str, LoaderFn] = {
    "graphwalks": fake_loader("graphwalks"),
    "oolong_pairs": fake_loader("oolong_pairs"),
}


def attribution(mechanism: str) -> str:
    """A canned attributor response (ungrounded variant: failing_level required)."""
    return (
        "```json\n"
        + json.dumps(
            {
                "causal_status": "causal",
                "agent_mechanism": mechanism,
                "failing_level": "root",
                "evidence_node_ids": ["r"],
                "symptom_summary": "the model answered without verifying",
            }
        )
        + "\n```"
    )


def proposer_batch(*edits: tuple[int, str]) -> str:
    """A canned proposer response: one full-replacement text edit per pattern."""
    items = [
        {
            "pattern_index": index,
            "edit": {"kind": "text", "new_text": new_text},
            "predicted_effect": "the root double-checks before answering",
            "regression_risks": ["one extra turn per run"],
        }
        for index, new_text in edits
    ]
    return "```json\n" + json.dumps(items) + "\n```"


TEXT_ROUND_1 = "Verify every claim against the stored evidence before answering. [r1]"
TEXT_ROUND_2 = "Verify every claim against the stored evidence before answering. [r2]"
MERGE_TEXT = "Cover every input chunk and verify before answering. [merge]"

# Script arithmetic at the template's scale (n_in=2, n_ho=2, m=1, v=1):
# mining is 2 runs, each evaluated subject is (2 + 2) * 1 = 4 runs.
MINING_FAIL = [final("WRONG"), final("WRONG")]
MINING_FAIL_V2 = [final("WRONG-2"), final("WRONG-2")]
SUBJECT_FAIL = [final("WRONG")] * 4
SUBJECT_PASS = [final("RIGHT")] * 4


def patch_runner(monkeypatch: pytest.MonkeyPatch, script: list[str]) -> ClientFactory:
    factory = ClientFactory(script)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    return factory


def run(
    config: ExperimentConfig,
    out_dir: Path,
    attributor: MockLM,
    proposer: MockLM,
) -> Any:
    return run_experiment(
        config,
        out_dir,
        verifier=GoldVerifier(),
        attributor_lm=attributor,
        proposer_lm=proposer,
        loaders=LOADERS,
    )


def frozen_envelope(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / FROZEN_DIR / FROZEN_HARNESS_FILENAME).read_text())


def mining_manifest(out_dir: Path, round_index: int) -> list[dict[str, Any]]:
    mining_parent = experiment_round_dir(out_dir, round_index) / "mining"
    return load_manifest(mining_parent, round_index)


def validation_round_path(out_dir: Path, round_index: int) -> Path:
    return round_dir(experiment_round_dir(out_dir, round_index) / "validation", round_index)


# ---------------------------------------------------------------------------
# Full experiment: directory contract, frozen harness, stage usage
# ---------------------------------------------------------------------------


class TestFullExperiment:
    def run_two_promoted_rounds(self, config, out: Path, monkeypatch) -> tuple[Any, ClientFactory]:
        factory = patch_runner(
            monkeypatch,
            MINING_FAIL
            + SUBJECT_FAIL
            + SUBJECT_PASS  # round 1
            + MINING_FAIL_V2
            + SUBJECT_FAIL
            + SUBJECT_PASS,  # round 2
        )
        attributor = MockLM(responses=[attribution("skipped_verification")] * 4)
        proposer = MockLM(
            responses=[
                proposer_batch((0, TEXT_ROUND_1)),
                proposer_batch((0, TEXT_ROUND_2)),
            ]
        )
        return run(config, out, attributor, proposer), factory

    def test_t2_experiment_promotes_twice_and_freezes_the_last_promotion(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"
        result, factory = self.run_two_promoted_rounds(config, out, monkeypatch)

        assert factory.total_calls == 20  # (2 + 4 + 4) per round, nothing re-run
        assert result.stopped == STOP_MAX_ROUNDS
        assert [outcome.promoted for outcome in result.rounds] == [True, True]

        expected = replace(H0, verification_instruction=TEXT_ROUND_2)
        envelope = frozen_envelope(out)
        assert envelope["hash"] == harness_hash(expected)
        assert result.final_harness_hash == harness_hash(expected)
        assert result.rounds[1].promoted_harness_hash == harness_hash(expected)

    def test_directory_contract(self, tmp_path, monkeypatch):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"
        self.run_two_promoted_rounds(config, out, monkeypatch)

        assert (out / CONFIG_FILENAME).exists()
        assert (out / "splits" / "graphwalks_short_held_in.jsonl").exists()
        assert (out / "splits" / "graphwalks_short_held_out.jsonl").exists()
        assert (out / STAGE_USAGE_FILE).exists()
        assert (out / FROZEN_DIR / FROZEN_HARNESS_FILENAME).exists()
        for round_index in (1, 2):
            round_path = experiment_round_dir(out, round_index)
            nested = f"round_{round_index:02d}"
            assert (round_path / "mining" / nested / "bundle.json").exists()
            assert (round_path / "mining" / nested / "runs.jsonl").exists()
            candidate = f"r{round_index:02d}-c01-s4"
            assert (round_path / "proposals" / candidate / "proposal.json").exists()
            assert (round_path / PROPOSALS_MARKER_FILENAME).exists()
            assert (round_path / "validation" / nested / "decision.json").exists()
            assert (round_path / ROUND_MARKER_FILENAME).exists()

    def test_stage_usage_has_one_record_per_stage_per_round_with_nonzero_tokens(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"
        self.run_two_promoted_rounds(config, out, monkeypatch)

        totals = read_stage_usage(out / STAGE_USAGE_FILE)
        expected_ids = {
            f"round_{round_index:02d}/{stage}"
            for round_index in (1, 2)
            for stage in ("mining", "attribution", "proposal", "validation")
        }
        assert set(totals) == expected_ids
        for work_id, usage in totals.items():
            assert usage.input_tokens > 0, work_id
            assert usage.output_tokens > 0, work_id
            assert usage.wall_seconds > 0, work_id

    def test_reinvocation_of_a_finished_experiment_replays_without_any_calls(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"
        first, _ = self.run_two_promoted_rounds(config, out, monkeypatch)

        idle = patch_runner(monkeypatch, [])
        attributor = MockLM(responses=[])
        proposer = MockLM(responses=[])
        second = run(config, out, attributor, proposer)

        assert idle.total_calls == 0
        assert attributor._call_count == 0
        assert proposer._call_count == 0
        assert second.final_harness_hash == first.final_harness_hash
        assert [outcome.promoted for outcome in second.rounds] == [True, True]


# ---------------------------------------------------------------------------
# Patience stopping
# ---------------------------------------------------------------------------


class TestPatience:
    def test_two_consecutive_no_promotion_rounds_stop_before_t(self, tmp_path, monkeypatch):
        config = make_config(tmp_path, t=3, patience=2)
        out = tmp_path / "exp"
        factory = patch_runner(
            monkeypatch,
            MINING_FAIL
            + SUBJECT_FAIL
            + SUBJECT_FAIL  # round 1: candidate no better
            + MINING_FAIL_V2
            + SUBJECT_FAIL
            + SUBJECT_FAIL,  # round 2: same
        )
        attributor = MockLM(responses=[attribution("skipped_verification")] * 4)
        proposer = MockLM(
            responses=[
                proposer_batch((0, TEXT_ROUND_1)),
                proposer_batch((0, TEXT_ROUND_2)),
            ]
        )
        result = run(config, out, attributor, proposer)

        assert result.stopped == STOP_PATIENCE
        assert [outcome.promoted for outcome in result.rounds] == [False, False]
        assert factory.total_calls == 20  # two rounds ran, the third never started
        assert not experiment_round_dir(out, 3).exists()
        assert frozen_envelope(out)["hash"] == harness_hash(H0)


# ---------------------------------------------------------------------------
# Crash-safe resume at every stage boundary
# ---------------------------------------------------------------------------


class TestResumeMidMining:
    def test_killed_mining_resumes_without_rerunning_persisted_runs(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"

        # The script covers only the first mining run: the second run's client
        # call raises, exactly like a crash mid-round.
        factory = patch_runner(monkeypatch, [final("WRONG")])
        with pytest.raises(IndexError):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))
        assert factory.total_calls == 2  # the crashing call itself is counted
        assert len(mining_manifest(out, 1)) == 1  # run 1 persisted before the crash

        # Resume: only the missing mining run plus the validation runs execute.
        resumed = patch_runner(monkeypatch, [final("WRONG")] + SUBJECT_FAIL + SUBJECT_PASS)
        attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_1))])
        result = run(config, out, attributor, proposer)

        assert resumed.total_calls == 9  # 1 mining + 8 validation, run 1 never re-ran
        entries = mining_manifest(out, 1)
        assert [entry["run_id"] for entry in entries] == [
            "graphwalks-short-0__a01",
            "graphwalks-short-1__a01",
        ]
        assert result.rounds[0].promoted


class TestResumeAfterBundle:
    def test_crash_before_proposals_sealed_resumes_without_remining(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"

        def boom(prompt: Any) -> str:
            raise ValueError("proposer crashed")

        factory = patch_runner(monkeypatch, list(MINING_FAIL))
        attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        with pytest.raises(ValueError, match="proposer crashed"):
            run(config, out, attributor, MockLM(response_fn=boom))
        round_path = experiment_round_dir(out, 1)
        assert (round_path / "mining" / "round_01" / "bundle.json").exists()
        assert not (round_path / PROPOSALS_MARKER_FILENAME).exists()
        assert factory.total_calls == 2

        resumed = patch_runner(monkeypatch, SUBJECT_FAIL + SUBJECT_PASS)
        second_attributor = MockLM(responses=[])
        proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_1))])
        result = run(config, out, second_attributor, proposer)

        assert second_attributor._call_count == 0  # bundle.json read back, never re-mined
        assert resumed.total_calls == 8  # validation only
        marker = json.loads((round_path / PROPOSALS_MARKER_FILENAME).read_text())
        assert marker["candidate_ids"] == ["r01-c01-s4"]
        assert result.rounds[0].promoted

    def test_persisted_proposal_cache_replays_the_identical_candidate_set(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        patch_runner(monkeypatch, MINING_FAIL + SUBJECT_FAIL + SUBJECT_PASS)
        attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_1))])
        run(config, out, attributor, proposer)
        round_path = experiment_round_dir(out, 1)
        sealed = json.loads((round_path / PROPOSALS_MARKER_FILENAME).read_text())

        # Simulate a crash after the proposer response was cached but before
        # the proposal stage sealed: everything downstream of the cache is gone.
        (round_path / PROPOSALS_MARKER_FILENAME).unlink()
        (round_path / ROUND_MARKER_FILENAME).unlink()
        import shutil

        shutil.rmtree(round_path / "proposals")
        shutil.rmtree(round_path / "validation")
        (out / FROZEN_DIR / FROZEN_HARNESS_FILENAME).unlink()

        resumed = patch_runner(monkeypatch, SUBJECT_FAIL + SUBJECT_PASS)
        replay_proposer = MockLM(responses=[])  # a cache miss would raise IndexError
        result = run(config, out, MockLM(responses=[]), replay_proposer)

        assert replay_proposer._call_count == 0
        resealed = json.loads((round_path / PROPOSALS_MARKER_FILENAME).read_text())
        assert resealed["candidate_ids"] == sealed["candidate_ids"]
        assert resealed["prompt_sha256"] == sealed["prompt_sha256"]
        assert resumed.total_calls == 8
        assert result.rounds[0].promoted


# ---------------------------------------------------------------------------
# Merged promotion: freeze and resume through the persisted envelope
# ---------------------------------------------------------------------------


class TestMergedPromotion:
    def run_merged_round(self, config, out: Path, monkeypatch) -> tuple[Any, ClientFactory]:
        factory = patch_runner(
            monkeypatch,
            MINING_FAIL  # two failures with distinct mechanisms -> two patterns
            + SUBJECT_FAIL  # baseline
            + SUBJECT_PASS  # candidate 1
            + SUBJECT_PASS  # candidate 2
            + SUBJECT_PASS,  # merged re-evaluation
        )
        attributor = MockLM(
            responses=[attribution("incomplete_coverage"), attribution("skipped_verification")]
        )
        # Both patterns are text surfaces (S2 and S4); the same replacement
        # text on each makes the merged harness independent of pattern order.
        proposer = MockLM(responses=[proposer_batch((0, MERGE_TEXT), (1, MERGE_TEXT))])
        return run(config, out, attributor, proposer), factory

    def test_merged_harness_promotes_and_freezes(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        result, factory = self.run_merged_round(config, out, monkeypatch)

        assert factory.total_calls == 18  # 2 mining + 4 subjects x 4 runs
        assert result.rounds[0].promoted
        _, decision = load_promotion_ledger(validation_round_path(out, 1))
        assert decision["promoted_subject_id"] == "merged"

        expected = replace(
            H0,
            decomposition_instruction=MERGE_TEXT,
            verification_instruction=MERGE_TEXT,
        )
        assert result.final_harness_hash == harness_hash(expected)
        assert frozen_envelope(out)["hash"] == harness_hash(expected)

    def test_resume_rematerializes_the_merged_incumbent_from_disk(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        first, _ = self.run_merged_round(config, out, monkeypatch)

        idle = patch_runner(monkeypatch, [])
        result = run(config, out, MockLM(responses=[]), MockLM(responses=[]))

        assert idle.total_calls == 0
        assert result.final_harness_hash == first.final_harness_hash
        assert harness_hash(result.final_harness) == first.final_harness_hash


# ---------------------------------------------------------------------------
# Identity enforcement (R3)
# ---------------------------------------------------------------------------


class TestIdentityMismatch:
    @pytest.mark.parametrize(
        ("label", "base_kwargs", "changed_kwargs"),
        [
            ("changed_temperature", {}, {"temperature": 0.9}),
            ("lowered_v", {"v": 2}, {"v": 1}),
            ("changed_proposer_model", {}, {"proposer_model": "other-proposer"}),
            ("changed_dataset_revision", {}, {"graphwalks_revision": "rev-gw-2"}),
        ],
    )
    def test_mismatch_raises_before_any_run_executes(
        self, tmp_path, monkeypatch, label, base_kwargs, changed_kwargs
    ):
        base = make_config(tmp_path / "base", **base_kwargs)
        out = tmp_path / "exp"
        check_identity(base, out)

        changed = make_config(tmp_path / "changed", **{**base_kwargs, **changed_kwargs})
        factory = patch_runner(monkeypatch, [])
        with pytest.raises(IdentityMismatchError):
            run(changed, out, MockLM(responses=[]), MockLM(responses=[]))
        assert factory.total_calls == 0

    def test_matching_identity_is_accepted(self, tmp_path):
        base = make_config(tmp_path / "base")
        out = tmp_path / "exp"
        first = check_identity(base, out)
        again = make_config(tmp_path / "again")  # identical values, different file
        assert check_identity(again, out) == first


# ---------------------------------------------------------------------------
# The mining spend breaker (R12)
# ---------------------------------------------------------------------------


class TestMiningSpendBreaker:
    def test_breaker_trips_persists_partial_state_and_stays_resumable(self, tmp_path, monkeypatch):
        # Each run costs 0.001; a candidate budget of 0.0005 trips the breaker
        # after the first mining run, so the second is skipped, never silently
        # dropped.
        config = make_config(tmp_path, candidate_budget=0.0005)
        out = tmp_path / "exp"
        factory = patch_runner(monkeypatch, list(MINING_FAIL))
        with pytest.raises(MiningBudgetExceededError):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))
        assert factory.total_calls == 1
        assert len(mining_manifest(out, 1)) == 1

        # Re-invocation charges the persisted run, trips again, executes nothing.
        idle = patch_runner(monkeypatch, [])
        with pytest.raises(MiningBudgetExceededError) as excinfo:
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))
        assert idle.total_calls == 0
        assert len(mining_manifest(out, 1)) == 1

        # The re-trip is permanent for this directory, so the error must say so
        # rather than reading as a transient "try again" (the budget cannot be
        # raised in place either -- caps are identity keys).
        message = str(excinfo.value)
        assert "cannot be completed" in message
        assert str(out) in message
        assert "candidate_budget" in message
        assert "identity-protected" in message
        assert "FRESH out_dir" in message
        assert "lower the scale" in message


# ---------------------------------------------------------------------------
# The evidence stage's completion marker (bundle.json is not one)
# ---------------------------------------------------------------------------


def mining_round_path(out_dir: Path, round_index: int) -> Path:
    return round_dir(experiment_round_dir(out_dir, round_index) / "mining", round_index)


class TestEvidenceCrashWindow:
    """``write_bundle`` publishes ``bundle.json`` (atomically) BEFORE it writes
    ``records.jsonl`` and ``attributions.jsonl``. A crash in that window leaves
    a readable bundle beside missing audit files, so the bundle alone can never
    be the stage's completion marker."""

    def crash_after_publishing_bundle(self, bundle, records, out_dir, raw_attributions=None):
        write_bundle(bundle, records, out_dir=out_dir, raw_attributions=raw_attributions)
        published = round_dir(out_dir, bundle.config.round_index)
        (published / RECORDS_FILENAME).unlink()
        (published / ATTRIBUTIONS_FILENAME).unlink()
        raise RuntimeError("crashed after publishing bundle.json")

    def truncate_the_records_file(self, bundle, records, out_dir, raw_attributions=None):
        write_bundle(bundle, records, out_dir=out_dir, raw_attributions=raw_attributions)
        published = round_dir(out_dir, bundle.config.round_index)
        first_line = (published / RECORDS_FILENAME).read_text().splitlines(keepends=True)[0]
        (published / RECORDS_FILENAME).write_text(first_line)

    def test_crash_between_bundle_and_audit_files_is_re_mined_on_resume(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        monkeypatch.setattr(orchestrator_module, "write_bundle", self.crash_after_publishing_bundle)
        patch_runner(monkeypatch, list(MINING_FAIL))
        attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        with pytest.raises(RuntimeError, match="crashed after publishing"):
            run(config, out, attributor, MockLM(responses=[]))

        mining_round = mining_round_path(out, 1)
        assert (mining_round / "bundle.json").exists()  # the readable "marker"
        assert not (mining_round / RECORDS_FILENAME).exists()
        assert not (mining_round / EVIDENCE_MARKER_FILENAME).exists()

        monkeypatch.undo()  # the real write_bundle (and a fresh runner script)
        resumed = patch_runner(monkeypatch, SUBJECT_FAIL + SUBJECT_PASS)
        replay_attributor = MockLM(responses=[])  # a cache miss would raise IndexError
        proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_1))])
        result = run(config, out, replay_attributor, proposer)

        assert replay_attributor._call_count == 0  # attributions replay from the cache
        assert resumed.total_calls == 8  # validation only: mining runs stay persisted
        assert len(read_jsonl(mining_round / RECORDS_FILENAME)) == 2
        assert len(read_jsonl(mining_round / ATTRIBUTIONS_FILENAME)) == 2
        marker = json.loads((mining_round / EVIDENCE_MARKER_FILENAME).read_text())
        assert marker["n_records"] == 2
        assert (
            marker["bundle_id"]
            == json.loads((mining_round / "bundle.json").read_text())["bundle_id"]
        )
        assert result.rounds[0].promoted

    def test_a_truncated_audit_file_refuses_to_mark_the_stage_complete(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        monkeypatch.setattr(orchestrator_module, "write_bundle", self.truncate_the_records_file)
        patch_runner(monkeypatch, list(MINING_FAIL))
        attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        with pytest.raises(ExperimentPersistenceError, match="truncated audit file"):
            run(config, out, attributor, MockLM(responses=[]))
        assert not (mining_round_path(out, 1) / EVIDENCE_MARKER_FILENAME).exists()


# ---------------------------------------------------------------------------
# A crashed stage's persisted runs are still metered (R5)
# ---------------------------------------------------------------------------


class TestCrashedStageUsage:
    def test_a_crashed_mining_stage_still_meters_the_runs_it_paid_for(self, tmp_path, monkeypatch):
        """The stage raises after persisting run 1; that run cost real money,
        so it must land in the sidecar rather than being charged to nothing."""
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        patch_runner(monkeypatch, [final("WRONG")])  # the second run's call raises
        with pytest.raises(IndexError):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))

        assert len(mining_manifest(out, 1)) == 1
        usage = read_stage_usage(out / STAGE_USAGE_FILE)["round_01/mining"]
        assert usage.input_tokens == 10  # exactly the persisted run's manifest line
        assert usage.output_tokens == 10
        assert usage.cost == pytest.approx(0.001)
        assert usage.lower_bound is True  # the stage raised: never an exact figure

    def test_a_crashed_validation_stage_still_meters_the_runs_it_paid_for(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        # Mining and attribution complete; validation dies partway through the
        # baseline's four runs.
        patch_runner(monkeypatch, MINING_FAIL + [final("WRONG")] * 2)
        attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_1))])
        with pytest.raises(IndexError):
            run(config, out, attributor, proposer)

        usage = read_stage_usage(out / STAGE_USAGE_FILE)["round_01/validation"]
        assert usage.input_tokens == 20  # the two persisted validation runs
        assert usage.lower_bound is True


# ---------------------------------------------------------------------------
# Persisted state that contradicts itself: the resume guards (R7)
# ---------------------------------------------------------------------------


def complete_one_promoted_round(config, out: Path, monkeypatch) -> Any:
    """One finished, promoting round -- the state the guards below corrupt."""
    patch_runner(monkeypatch, MINING_FAIL + SUBJECT_FAIL + SUBJECT_PASS)
    attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
    proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_1))])
    return run(config, out, attributor, proposer)


def complete_two_promoted_rounds(config, out: Path, monkeypatch) -> Any:
    """Two finished, promoting rounds -- one refresh per executed round."""
    patch_runner(
        monkeypatch,
        MINING_FAIL + SUBJECT_FAIL + SUBJECT_PASS + MINING_FAIL_V2 + SUBJECT_FAIL + SUBJECT_PASS,
    )
    attributor = MockLM(responses=[attribution("skipped_verification")] * 4)
    proposer = MockLM(
        responses=[proposer_batch((0, TEXT_ROUND_1)), proposer_batch((0, TEXT_ROUND_2))]
    )
    return run(config, out, attributor, proposer)


def rewrite_json(path: Path, **changes: Any) -> None:
    payload = json.loads(path.read_text())
    payload.update(changes)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class TestContradictoryPersistedState:
    def test_round_marker_recording_another_round_index_is_refused(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        complete_one_promoted_round(config, out, monkeypatch)
        rewrite_json(experiment_round_dir(out, 1) / ROUND_MARKER_FILENAME, round=7)

        idle = patch_runner(monkeypatch, [])
        with pytest.raises(ExperimentPersistenceError, match="records round 7"):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))
        assert idle.total_calls == 0

    def test_round_marker_of_the_wrong_format_is_refused(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        complete_one_promoted_round(config, out, monkeypatch)
        rewrite_json(experiment_round_dir(out, 1) / ROUND_MARKER_FILENAME, format="something-else")

        patch_runner(monkeypatch, [])
        with pytest.raises(ExperimentPersistenceError, match="shrlm-experiment-round/v1"):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))

    def test_marker_hash_diverging_from_the_ledger_decision_is_refused(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        complete_one_promoted_round(config, out, monkeypatch)
        rewrite_json(
            experiment_round_dir(out, 1) / ROUND_MARKER_FILENAME,
            promoted_harness_hash="0" * 64,
        )

        idle = patch_runner(monkeypatch, [])
        with pytest.raises(ExperimentPersistenceError, match="contradicts itself"):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))
        assert idle.total_calls == 0

    def test_a_promoted_envelope_that_no_longer_round_trips_is_refused(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        complete_one_promoted_round(config, out, monkeypatch)
        validation_path = validation_round_path(out, 1)
        records, _ = load_promotion_ledger(validation_path)
        promoted = [record for record in records if record["decision"] == "promoted"]
        envelope_path = validation_path / str(
            promoted[0]["links"]["splits"][SPLIT_HELDIN]["harness"]
        )
        envelope = json.loads(envelope_path.read_text())
        envelope["harness"]["surfaces"]["S4_verification_instruction"] = "edited after the fact"
        envelope_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")

        patch_runner(monkeypatch, [])
        with pytest.raises(ExperimentPersistenceError, match="does not round-trip"):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))

    def test_a_materialized_split_holding_no_instances_is_refused(self, tmp_path, monkeypatch):
        """Emptying the file alone trips the splits hash guard, so this empties
        it *consistently* -- manifest updated -- to reach the orchestrator's own
        "cannot run a round on it" check."""
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        complete_one_promoted_round(config, out, monkeypatch)
        splits_dir = out / SPLITS_DIR
        file_name = split_file_name("graphwalks", "short", "held_in")
        (splits_dir / file_name).write_text("")
        manifest_path = splits_dir / MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text())
        manifest["environments"]["graphwalks"]["files"][file_name] = {
            "sha256": sha256_text(""),
            "count": 0,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        idle = patch_runner(monkeypatch, [])
        with pytest.raises(ExperimentPersistenceError, match="holds no instances"):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))
        assert idle.total_calls == 0

    def test_a_frozen_harness_diverging_from_the_conclusion_is_refused(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        complete_one_promoted_round(config, out, monkeypatch)
        # Someone froze a different harness into the completed experiment.
        write_harness_json(
            replace(H0, verification_instruction="a different frozen harness"),
            out / FROZEN_DIR / FROZEN_HARNESS_FILENAME,
        )

        patch_runner(monkeypatch, [])
        with pytest.raises(ExperimentPersistenceError, match="refusing to overwrite a frozen"):
            run(config, out, MockLM(responses=[]), MockLM(responses=[]))


# ---------------------------------------------------------------------------
# A round whose proposal stage seals zero candidates
# ---------------------------------------------------------------------------

EMPTY_BATCH = "```json\n[]\n```"


class TestZeroCandidateRound:
    def test_no_candidates_means_no_ledger_no_promotion_and_no_prior_history(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"

        histories: list[int] = []
        real_propose_round = orchestrator_module.propose_round

        def spy(*args: Any, **kwargs: Any) -> Any:
            histories.append(len(kwargs["prior_history"]))
            return real_propose_round(*args, **kwargs)

        monkeypatch.setattr(orchestrator_module, "propose_round", spy)
        factory = patch_runner(
            monkeypatch,
            MINING_FAIL  # round 1: mined, but the proposer offers nothing
            + MINING_FAIL_V2
            + SUBJECT_FAIL
            + SUBJECT_PASS,  # round 2: a real candidate, promoted
        )
        attributor = MockLM(responses=[attribution("skipped_verification")] * 4)
        proposer = MockLM(responses=[EMPTY_BATCH, proposer_batch((0, TEXT_ROUND_2))])
        result = run(config, out, attributor, proposer)

        assert factory.total_calls == 12  # 2 + 2 mining runs, then 8 validation runs
        first, second = result.rounds
        assert (first.has_ledger, first.promoted) == (False, False)
        assert (second.has_ledger, second.promoted) == (True, True)

        marker = json.loads((experiment_round_dir(out, 1) / ROUND_MARKER_FILENAME).read_text())
        assert marker["has_ledger"] is False
        assert marker["promoted"] is False
        assert marker["promoted_harness_hash"] is None
        assert (
            json.loads((experiment_round_dir(out, 1) / PROPOSALS_MARKER_FILENAME).read_text())[
                "candidate_ids"
            ]
            == []
        )
        # A ledger-less round creates no validation round directory at all...
        assert not validation_round_path(out, 1).exists()
        # ...and contributes no prior history to the next round's proposal.
        assert histories == [0, 0]


# ---------------------------------------------------------------------------
# The post-round analysis hook (U5/KTD4): refreshes snapshots, never the
# experiment
# ---------------------------------------------------------------------------

# Every value a persisted artifact takes from the clock (or from a trace hash
# that is itself a function of the clock and of a temp directory name) rather
# than from the experiment's decisions. Two identical experiments agree on
# every other byte -- verified: with these five keys scrubbed and the per-run
# traces compared by presence, two runs of the same config over the same script
# produce identical trees -- so scrubbing exactly these is what makes
# "unchanged by the hook" a checkable claim rather than a hope.
CLOCK_KEYS = frozenset(
    {"created_at", "execution_time", "timestamp", "trace_sha256", "wall_seconds"}
)


def scrub_clock(payload: Any) -> Any:
    """A JSON payload with every clock-derived value removed, recursively."""
    if isinstance(payload, dict):
        return {key: scrub_clock(value) for key, value in payload.items() if key not in CLOCK_KEYS}
    if isinstance(payload, list):
        return [scrub_clock(item) for item in payload]
    return payload


def persisted_tree(out: Path) -> dict[str, Any]:
    """Everything the loop persisted, keyed by relative path.

    The analysis snapshots are excluded because they are precisely what the
    hook is allowed to add; everything else -- ledgers, manifests, markers,
    bundles, proposals, splits, the frozen harness -- is compared. Per-run
    trace files carry the REPL's temp directory names and per-iteration
    timings, so they are compared by presence rather than by content; the
    manifest line and the bundle record that summarize each of them are
    compared in full (minus the clock keys), so a trace that changed identity
    would still show up here.
    """
    tree: dict[str, Any] = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out).as_posix()
        if rel.startswith(f"{ANALYSIS_DIR}/") or "__pycache__" in rel:
            continue
        if f"/{TRACES_DIR}/" in rel:
            tree[rel] = "<trace>"
        elif rel.endswith(".jsonl"):
            tree[rel] = [
                scrub_clock(json.loads(line)) for line in path.read_text().splitlines() if line
            ]
        elif rel.endswith(".json"):
            tree[rel] = scrub_clock(json.loads(path.read_text()))
        else:
            tree[rel] = path.read_bytes()
    return tree


def snapshots(out: Path) -> list[Path]:
    """Every allocated analysis snapshot, oldest first."""
    parent = out / ANALYSIS_DIR
    return sorted(path for path in parent.iterdir() if path.is_dir()) if parent.is_dir() else []


def disable_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-hook loop: what the experiment persisted before U5 existed."""
    monkeypatch.setattr(orchestrator_module, "_run_post_round_analysis", lambda out_dir: None)


class TestPostRoundAnalysisIsolation:
    """The guarantee that comes before any output the hook produces."""

    def test_analysis_that_cannot_even_start_leaves_the_experiment_untouched(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path)

        baseline = tmp_path / "baseline"
        disable_hook(monkeypatch)
        without_hook = complete_one_promoted_round(config, baseline, monkeypatch)

        monkeypatch.undo()

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("analysis exploded")

        monkeypatch.setattr(analysis_io_module, "allocate_snapshot", explode)
        out = tmp_path / "exp"
        with_hook = complete_one_promoted_round(config, out, monkeypatch)

        assert with_hook.stopped == without_hook.stopped
        assert with_hook.final_harness_hash == without_hook.final_harness_hash
        assert [outcome.to_payload() for outcome in with_hook.rounds] == [
            outcome.to_payload() for outcome in without_hook.rounds
        ]
        assert persisted_tree(out) == persisted_tree(baseline)
        assert snapshots(out) == []

    def test_one_failing_aggregation_costs_only_its_own_output(self, tmp_path, monkeypatch, capsys):
        config = make_config(tmp_path)

        baseline = tmp_path / "baseline"
        disable_hook(monkeypatch)
        without_hook = complete_one_promoted_round(config, baseline, monkeypatch)

        monkeypatch.undo()

        def explode(out_dir: Path, snapshot: Any) -> Any:
            raise RuntimeError("quality exploded")

        monkeypatch.setattr(incumbent_quality_module, "run_incumbent_quality", explode)
        out = tmp_path / "exp"
        with_hook = complete_one_promoted_round(config, out, monkeypatch)

        # The experiment is untouched: same result, same persisted bytes.
        assert with_hook.final_harness_hash == without_hook.final_harness_hash
        assert persisted_tree(out) == persisted_tree(baseline)

        # The batch went on: the surviving aggregations wrote their tables...
        (snapshot,) = snapshots(out)
        written = {path.name for path in snapshot.iterdir()}
        assert OPTIMIZATION_FILENAME in written
        assert SURFACE_ACTIVITY_FILENAME in written
        assert UNATTRIBUTED_FILENAME in written
        assert INCUMBENT_QUALITY_FILENAME not in written

        # ...provenance names the one that failed and how...
        provenance = json.loads((snapshot / PROVENANCE_FILENAME).read_text())
        outcomes = {tool["name"]: tool for tool in provenance["tools"]}
        assert outcomes[TOOL_NAME_INCUMBENT_QUALITY]["ok"] is False
        assert "quality exploded" in outcomes[TOOL_NAME_INCUMBENT_QUALITY]["error"]
        assert outcomes[TOOL_NAME_OPTIMIZATION]["ok"] is True
        assert outcomes[TOOL_NAME_SURFACE_ACTIVITY]["ok"] is True

        # ...and the partial batch is never presented as a finished one.
        assert not is_published(snapshot)
        assert latest_snapshot(out) is None
        assert TOOL_NAME_INCUMBENT_QUALITY in capsys.readouterr().err


class TestPostRoundAnalysisSnapshots:
    """What a round's refresh actually produces (KTD2/KTD4)."""

    def test_every_executed_round_publishes_a_snapshot_of_every_aggregation(
        self, tmp_path, monkeypatch
    ):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"
        complete_two_promoted_rounds(config, out, monkeypatch)

        first, second = snapshots(out)  # one per executed round, never shared
        assert is_published(first)
        assert is_published(second)
        assert latest_snapshot(out) == second

        written = {path.name for path in second.iterdir()}
        assert {
            OPTIMIZATION_FILENAME,
            INCUMBENT_QUALITY_FILENAME,
            INCUMBENT_QUALITY_CANDIDATES_FILENAME,
            SURFACE_ACTIVITY_FILENAME,
            UNATTRIBUTED_FILENAME,
            BUNDLES_FILENAME,
        } <= written

        provenance = json.loads((second / PROVENANCE_FILENAME).read_text())
        assert [tool["ok"] for tool in provenance["tools"]] == [True] * 4
        assert {tool["name"] for tool in provenance["tools"]} == {
            TOOL_NAME_OPTIMIZATION,
            TOOL_NAME_INCUMBENT_QUALITY,
            TOOL_NAME_SURFACE_ACTIVITY,
            TOOL_NAME_DIFF,
        }
        assert provenance["rounds"] == [1, 2]
        assert provenance["identity_hash"] == check_identity(config, out)
        assert provenance["sources"]  # every consumed artifact, hashed

    def test_an_optimization_run_says_nothing_about_the_absent_evaluation_phase(
        self, tmp_path, monkeypatch, capsys
    ):
        """``eval/`` is written by the evaluation runner, never by this loop:
        for a whole optimization study it is absent, which is not a fault."""
        config = make_config(tmp_path)
        out = tmp_path / "exp"
        complete_one_promoted_round(config, out, monkeypatch)

        assert not (out / EVAL_DIR).exists()
        (snapshot,) = snapshots(out)
        assert is_published(snapshot)
        assert EVALUATION_FILENAME not in {path.name for path in snapshot.iterdir()}
        provenance = json.loads((snapshot / PROVENANCE_FILENAME).read_text())
        assert TOOL_NAME_EVALUATION not in {tool["name"] for tool in provenance["tools"]}
        assert provenance["eval_sets"] == []
        assert capsys.readouterr().err == ""


class TestPostRoundAnalysisOnResume:
    """A resume refreshes once when it catches up -- not once per replay."""

    def test_a_fully_replayed_resume_refreshes_exactly_once(self, tmp_path, monkeypatch):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"
        first = complete_two_promoted_rounds(config, out, monkeypatch)
        assert len(snapshots(out)) == 2  # one per executed round

        idle = patch_runner(monkeypatch, [])
        second = run(config, out, MockLM(responses=[]), MockLM(responses=[]))

        assert idle.total_calls == 0  # nothing re-executed...
        assert second.final_harness_hash == first.final_harness_hash
        refreshed = snapshots(out)
        assert len(refreshed) == 3  # ...and one refresh, not one per replayed round
        assert is_published(refreshed[-1])
        provenance = json.loads((refreshed[-1] / PROVENANCE_FILENAME).read_text())
        assert provenance["rounds"] == [1, 2]

    def test_a_resume_that_catches_up_then_executes_refreshes_once(self, tmp_path, monkeypatch):
        config = make_config(tmp_path, t=2)
        out = tmp_path / "exp"

        # Round 1 completes; round 2's first mining call raises, exactly like a
        # crash partway through the second round.
        patch_runner(monkeypatch, MINING_FAIL + SUBJECT_FAIL + SUBJECT_PASS)
        attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_1))])
        with pytest.raises(IndexError):
            run(config, out, attributor, proposer)
        assert len(snapshots(out)) == 1  # round 1's refresh, and nothing else

        patch_runner(monkeypatch, MINING_FAIL_V2 + SUBJECT_FAIL + SUBJECT_PASS)
        resumed_attributor = MockLM(responses=[attribution("skipped_verification")] * 2)
        resumed_proposer = MockLM(responses=[proposer_batch((0, TEXT_ROUND_2))])
        result = run(config, out, resumed_attributor, resumed_proposer)

        assert [outcome.promoted for outcome in result.rounds] == [True, True]
        refreshed = snapshots(out)
        # Round 1 was replayed and round 2 executed: the executed round's
        # refresh IS the replay phase's catch-up, so there is one new snapshot.
        assert len(refreshed) == 2
        provenance = json.loads((refreshed[-1] / PROVENANCE_FILENAME).read_text())
        assert provenance["rounds"] == [1, 2]
