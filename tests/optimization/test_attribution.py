"""Tests for attribution auditability and robustness (U6).

Four properties are load-bearing. A cached response replays with zero model
calls, and the cache key covers the validator logic, so a validator change can
never resurrect responses judged under different rules. Every attempt in the
re-ask loop -- accepted or rejected -- is kept with its raw response and named
violation, so an unattributed record still carries a full audit trail. In
per-depth aggregate digest mode the prompt stops demanding node ids the table
cannot show. And a transient LM failure is retried, then checkpointed by the
miner rather than raised through a round.
"""

import json
from typing import Any

import pytest

import shrlm.optimization.attribution as attribution_module
from shrlm.optimization.attribution import (
    AttributionRejection,
    AttributionTransportError,
    AttributorConfig,
    LLMAttributor,
)
from shrlm.optimization.digest import TraceDigest, build_digest
from shrlm.optimization.grounding import GroundingResult
from shrlm.optimization.mining import WeaknessMiner
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import AttributionErrorKind, CallNode, Verdict
from shrlm.optimization.walker import walk
from tests.mock_lm import MockLM
from tests.optimization.fixtures import (
    ROOT_MODEL,
    as_completion,
    code_block,
    completion_dict,
    iteration_entry,
    make_verdict,
    nested_run,
    shallow_run,
    usage,
)


def canned_attribution(evidence: list[str] | None = None) -> str:
    """A valid ungrounded response; ``evidence`` defaults to the root node."""
    payload = {
        "causal_status": "causal",
        "agent_mechanism": "lossy_aggregation",
        "failing_level": "root",
        "evidence_node_ids": ["r"] if evidence is None else evidence,
        "symptom_summary": "the merge step dropped a sub-result",
    }
    return "```json\n" + json.dumps(payload) + "\n```"


OFF_VOCABULARY = (
    "```json\n"
    + json.dumps(
        {
            "causal_status": "definitely_broken",
            "agent_mechanism": "lossy_aggregation",
            "failing_level": "root",
            "evidence_node_ids": ["r"],
            "symptom_summary": "made up a label",
        }
    )
    + "\n```"
)

UNGROUNDED = GroundingResult(failing_level=None, grounded=False, verdicts={})

# No backoff sleeps in tests; retry counts are what is under test.
FAST_CONFIG = AttributorConfig(transport_backoff_seconds=0.0)


class RecordingLM(MockLM):
    """Pops scripted responses (or raises scripted exceptions) and records
    every message list it was handed, so tests can inspect re-ask content."""

    def __init__(self, script: list[Any]):
        super().__init__(model_name="mock-model")
        self.script = list(script)
        self.seen: list[Any] = []

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self._call_count += 1
        self.seen.append(prompt)
        if not self.script:
            raise IndexError("RecordingLM: script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def attribution_inputs(run: dict[str, Any] | None = None) -> tuple[TraceDigest, CallNode, Verdict]:
    root, stats = walk(as_completion(run or nested_run()))
    digest = build_digest(
        instance_id="inst-1",
        question="which letter does the middle slice name?",
        root=root,
        stats=stats,
        verdict=make_verdict(),
    )
    return digest, root, make_verdict()


def wide_run(n_children: int = 41) -> dict[str, Any]:
    """A decomposition wide enough to push the digest into aggregate mode."""
    leaves = [
        {
            "root_model": ROOT_MODEL,
            "prompt": f"summarize piece {i}",
            "response": f"piece {i} says B",
            "usage_summary": usage(),
            "execution_time": 0.1,
        }
        for i in range(n_children)
    ]
    block = code_block(
        code="parts = [rlm_query(p) for p in pieces]",
        stdout="done\n",
        rlm_calls=leaves,
    )
    return completion_dict(
        prompt="what do the pieces say?",
        response="C",
        iterations=[
            iteration_entry(index=1, response="Fan out.\n```repl\n...\n```", code_blocks=[block]),
            iteration_entry(
                index=2,
                response="Committing.\n```repl\n...\n```",
                code_blocks=[code_block(code="answer['ready'] = True", final_answer="C")],
                final_answer="C",
            ),
        ],
        max_depth=2,
    )


class TestCacheReplay:
    def test_second_identical_call_replays_with_zero_lm_calls(self):
        lm = RecordingLM([canned_attribution()])
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs()

        first = attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert lm._call_count == 1

        second = attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert lm._call_count == 1  # replayed from the cache, not the model
        assert second.signature == first.signature
        assert second.attempts[0].cached is True
        assert first.attempts[0].cached is False


class TestReAskAudit:
    def test_re_ask_carries_the_named_violation(self):
        lm = RecordingLM([OFF_VOCABULARY, canned_attribution()])
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs()

        result = attributor.attribute(digest, root, verdict, UNGROUNDED)
        re_ask = lm.seen[1][1]["content"]
        assert "Your previous response was rejected" in re_ask
        assert "causal_status" in re_ask
        assert [attempt.accepted for attempt in result.attempts] == [False, True]

    def test_three_strikes_keeps_every_attempt_with_its_violation(self):
        lm = RecordingLM([OFF_VOCABULARY] * 3)
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs()

        with pytest.raises(AttributionRejection, match="3 attempts") as excinfo:
            attributor.attribute(digest, root, verdict, UNGROUNDED)

        attempts = excinfo.value.attempts
        assert [attempt.attempt for attempt in attempts] == [1, 2, 3]
        for attempt in attempts:
            assert attempt.accepted is False
            assert attempt.raw_response == OFF_VOCABULARY
            assert "causal_status" in attempt.violation

    def test_unattributed_record_gets_a_full_audit_payload(self):
        lm = RecordingLM([OFF_VOCABULARY] * 3)
        miner = WeaknessMiner(
            verifier=_failing_verifier, attributor=LLMAttributor(lm, config=FAST_CONFIG)
        )
        outcome = miner.record_failure(
            {"id": "inst-1", "question": "q"}, as_completion(shallow_run())
        )

        assert outcome.record is not None and outcome.record.attribution_failed
        assert outcome.record.attribution_error_kind is AttributionErrorKind.REJECTION
        assert outcome.raw is not None
        assert outcome.raw["attributed"] is False
        assert outcome.raw["attribution_error_kind"] == AttributionErrorKind.REJECTION.value
        assert "no valid attribution" in outcome.raw["error"]
        assert len(outcome.raw["attempts"]) == 3
        for entry in outcome.raw["attempts"]:
            assert entry["raw_response"] == OFF_VOCABULARY
            assert "causal_status" in entry["violation"]


class TestAggregateModeEvidence:
    def test_wide_tree_digest_is_flagged_aggregated(self):
        digest, _, _ = attribution_inputs(wide_run())
        assert digest.aggregated is True
        assert "aggregated by depth" in digest.text

    def test_narrow_tree_digest_is_not_flagged(self):
        digest, _, _ = attribution_inputs()
        assert digest.aggregated is False

    def test_aggregate_prompt_relaxes_the_evidence_demand(self):
        attributor = LLMAttributor(RecordingLM([]), config=FAST_CONFIG)
        table_prompt = attributor.system_prompt(grounded=False, aggregated=False)
        aggregate_prompt = attributor.system_prompt(grounded=False, aggregated=True)
        assert "must be a node_id that appears in the run" in table_prompt
        assert "aggregated by depth" in aggregate_prompt
        assert "leave evidence_node_ids as an empty list" in aggregate_prompt
        assert table_prompt != aggregate_prompt

    def test_attribute_renders_the_aggregate_instruction_for_a_wide_run(self):
        lm = RecordingLM([canned_attribution(evidence=[])])
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs(wide_run())

        result = attributor.attribute(digest, root, verdict, UNGROUNDED)
        system = lm.seen[0][0]["content"]
        assert "aggregated by depth" in system
        assert result.detail.evidence_node_ids == []  # empty evidence validates

    def test_validate_accepts_an_empty_evidence_list(self):
        attributor = LLMAttributor(RecordingLM([]), config=FAST_CONFIG)
        _, root, _ = attribution_inputs(wide_run())
        payload = json.loads(
            canned_attribution(evidence=[]).removeprefix("```json\n").removesuffix("\n```")
        )
        _, _, _, detail = attributor.validate(payload, root, UNGROUNDED)
        assert detail.evidence_node_ids == []


class TestValidatorVersionInCacheKey:
    def test_validator_version_bump_invalidates_cached_responses(self, monkeypatch):
        lm = RecordingLM([canned_attribution(), canned_attribution()])
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs()

        attributor.attribute(digest, root, verdict, UNGROUNDED)
        attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert lm._call_count == 1  # cached under the current validator

        monkeypatch.setattr(attribution_module, "VALIDATOR_VERSION", "999.0.0")
        attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert lm._call_count == 2  # the bump forced a fresh model call


class TestTransportResilience:
    def test_transient_errors_are_retried_until_success(self):
        lm = RecordingLM([ConnectionError("reset"), ConnectionError("reset"), canned_attribution()])
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs()

        result = attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert lm._call_count == 3
        assert result.signature is not None

    def test_deterministic_client_bug_propagates_immediately_without_retry(self):
        # A TypeError out of the client is a programming error, not a flaky
        # wire: it must surface as itself, after exactly one call.
        lm = RecordingLM([TypeError("completion() got an unexpected keyword argument")])
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs()

        with pytest.raises(TypeError, match="unexpected keyword"):
            attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert lm._call_count == 1

    def test_persistent_failure_raises_a_transport_error_not_a_rejection(self):
        lm = RecordingLM([ConnectionError("reset")] * 3)
        attributor = LLMAttributor(lm, config=FAST_CONFIG)
        digest, root, verdict = attribution_inputs()

        with pytest.raises(AttributionTransportError, match="ConnectionError"):
            attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert lm._call_count == 3

    def test_mine_checkpoints_instead_of_raising_away_the_round(self):
        # Run 1 attributes cleanly; run 2's LM stays down through every retry.
        lm = RecordingLM([canned_attribution()] + [ConnectionError("boom")] * 3)
        miner = WeaknessMiner(
            verifier=_failing_verifier, attributor=LLMAttributor(lm, config=FAST_CONFIG)
        )
        runs = [
            ({"id": "inst-1", "question": "q"}, as_completion(shallow_run())),
            ({"id": "inst-2", "question": "q"}, as_completion(shallow_run())),
        ]

        result = miner.mine(runs, round_index=1, harness_version="H0", split_id="held_in_v1")

        assert len(result.records) == 2  # the completed record survived intact
        attributed, failed = result.records
        assert attributed.instance_id == "inst-1" and attributed.signature is not None
        assert failed.instance_id == "inst-2" and failed.attribution_failed
        assert failed.attribution_error.startswith("transport failure:")
        assert failed.attribution_error_kind is AttributionErrorKind.TRANSPORT
        assert attributed.attribution_error_kind is None
        assert result.errors == [
            {"instance_id": "inst-2", "run_index": 1, "error": result.errors[0]["error"]}
        ]
        assert "ConnectionError" in result.errors[0]["error"]
        assert result.bundle.totals.n_unattributed == 1


class TestAttributorConfigValidation:
    def test_zero_transport_retries_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="transport_retries must be >= 1"):
            AttributorConfig(transport_retries=0)

    def test_negative_transport_backoff_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="transport_backoff_seconds must be >= 0"):
            AttributorConfig(transport_backoff_seconds=-0.5)


class TestMiningAuditSurfaces:
    def test_mining_result_carries_digest_texts_and_prompt(self):
        lm = RecordingLM([canned_attribution()])
        miner = WeaknessMiner(
            verifier=_failing_verifier, attributor=LLMAttributor(lm, config=FAST_CONFIG)
        )
        result = miner.mine(
            [({"id": "inst-1", "question": "q"}, as_completion(shallow_run()))],
            round_index=1,
            harness_version="H0",
            split_id="held_in_v1",
        )
        record = result.records[0]
        assert set(result.digest_texts) == {record.digest_sha256}
        assert "instance_id: inst-1" in result.digest_texts[record.digest_sha256]
        (prompt_sha,) = result.attributor_prompts
        assert result.attributor_prompts[prompt_sha] == miner.attributor.system_prompt(
            False, no_subcalls=True
        )
        assert result.raw_attributions[0]["prompt_sha256"] == prompt_sha
        assert result.errors == []


def _failing_verifier(instance: dict[str, Any], produced: str) -> Verdict:
    return Verdict(
        passed=False, cause=VerifierCause.WRONG_VALUE, gold="expected", produced=produced
    )


if __name__ == "__main__":
    pytest.main([__file__])
