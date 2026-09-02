"""Tests pinning the ``WeaknessMiner`` precomputed-verdict seam.

The default path -- the miner recomputes each verdict from the completion's
response -- must stay byte-identical for existing callers; the new ``verdicts``
argument replaces the verifier's judgment per run and is the only channel
through which a RESOURCE_TERMINATED verdict (which no Verifier can produce)
enters mining.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from shrlm.optimization.attribution import VALIDATOR_VERSION, LLMAttributor
from shrlm.optimization.mining import WeaknessMiner
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from tests.mock_lm import MockLM
from tests.optimization.fixtures import as_completion, shallow_run

CANNED_ATTRIBUTION = (
    "```json\n"
    + json.dumps(
        {
            "causal_status": "causal",
            "agent_mechanism": "premature_termination",
            "failing_level": "no_recursion",
            "evidence_node_ids": ["r"],
            "symptom_summary": "answered without checking anything",
        }
    )
    + "\n```"
)


@dataclass
class CountingVerifier:
    """Fails everything with WRONG_VALUE; records every call it receives."""

    calls: list[str] = field(default_factory=list)

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        self.calls.append(str(instance["id"]))
        return Verdict(
            passed=False,
            cause=VerifierCause.WRONG_VALUE,
            gold="expected",
            produced=produced,
        )


def make_miner(verifier: CountingVerifier) -> WeaknessMiner:
    lm = MockLM(responses=[CANNED_ATTRIBUTION] * 8)
    return WeaknessMiner(verifier=verifier, attributor=LLMAttributor(lm))


def failing_run() -> tuple[dict[str, Any], Any]:
    instance = {"id": "inst-1", "question": "what is 2 + 2?"}
    return instance, as_completion(shallow_run())


class TestBackwardCompatibility:
    def test_mine_without_verdicts_consults_the_verifier(self):
        verifier = CountingVerifier()
        instance, completion = failing_run()
        result = make_miner(verifier).mine(
            [(instance, completion)],
            round_index=1,
            harness_version="H0",
            split_id="held_in_v1",
        )
        assert verifier.calls == ["inst-1"]
        assert result.bundle.totals.n_failures == 1

    def test_record_failure_without_a_verdict_consults_the_verifier(self):
        verifier = CountingVerifier()
        instance, completion = failing_run()
        outcome = make_miner(verifier).record_failure(instance, completion)
        assert verifier.calls == ["inst-1"]
        assert outcome.record is not None
        assert outcome.record.verdict.cause is VerifierCause.WRONG_VALUE


class TestPrecomputedVerdicts:
    def test_a_supplied_verdict_bypasses_the_verifier(self):
        verifier = CountingVerifier()
        instance, completion = failing_run()
        supplied = Verdict(
            passed=False,
            cause=VerifierCause.RESOURCE_TERMINATED,
            gold="",
            produced="",
            detail="BudgetExceededError: spent $0.002000 of $0.001500 budget",
        )
        outcome = make_miner(verifier).record_failure(instance, completion, verdict=supplied)
        assert verifier.calls == []
        assert outcome.record is not None
        assert outcome.record.verdict is supplied

    def test_a_supplied_passing_verdict_yields_no_record(self):
        verifier = CountingVerifier()
        instance, completion = failing_run()
        supplied = Verdict(passed=True, cause=None, gold="4", produced="4")
        outcome = make_miner(verifier).record_failure(instance, completion, verdict=supplied)
        assert (outcome.record, outcome.raw, outcome.coverage) == (None, None, 1.0)
        assert verifier.calls == []

    def test_mine_aligns_verdicts_with_runs_and_allows_none_gaps(self):
        verifier = CountingVerifier()
        instance, completion = failing_run()
        other = {"id": "inst-2", "question": "still 2 + 2?"}
        supplied = Verdict(
            passed=False, cause=VerifierCause.RESOURCE_TERMINATED, gold="", produced=""
        )
        result = make_miner(verifier).mine(
            [(instance, completion), (other, as_completion(shallow_run()))],
            round_index=1,
            harness_version="H0",
            split_id="held_in_v1",
            verdicts=[supplied, None],
        )
        # The None gap fell back to the verifier; the supplied verdict did not.
        assert verifier.calls == ["inst-2"]
        causes = {record.verdict.cause for record in result.records}
        assert causes == {VerifierCause.RESOURCE_TERMINATED, VerifierCause.WRONG_VALUE}

    def test_misaligned_verdicts_are_rejected(self):
        verifier = CountingVerifier()
        instance, completion = failing_run()
        with pytest.raises(ValueError, match="align"):
            make_miner(verifier).mine(
                [(instance, completion)],
                round_index=1,
                harness_version="H0",
                split_id="held_in_v1",
                verdicts=[],
            )
        assert verifier.calls == []


@dataclass
class ConfiguredVerifier(CountingVerifier):
    """A verifier that also exposes its configuration, GraphWalks-style."""

    threshold: float = 1.0

    def config(self) -> dict[str, Any]:
        return {"environment": "toy", "pass_f1_threshold": self.threshold}


class TestConfigProvenance:
    def test_verifier_config_flows_from_a_config_bearing_verifier(self):
        instance, completion = failing_run()
        result = make_miner(ConfiguredVerifier()).mine(
            [(instance, completion)],
            round_index=1,
            harness_version="H0",
            split_id="held_in_v1",
        )
        assert result.bundle.config.verifier_config == {
            "environment": "toy",
            "pass_f1_threshold": 1.0,
        }

    def test_rounds_differing_only_in_verifier_config_have_different_bundle_ids(self):
        def bundle_id(threshold: float) -> str:
            instance, completion = failing_run()
            return (
                make_miner(ConfiguredVerifier(threshold=threshold))
                .mine(
                    [(instance, completion)],
                    round_index=1,
                    harness_version="H0",
                    split_id="held_in_v1",
                )
                .bundle.bundle_id
            )

        assert bundle_id(1.0) != bundle_id(0.5)

    def test_config_less_verifier_yields_an_empty_verifier_config(self):
        instance, completion = failing_run()
        result = make_miner(CountingVerifier()).mine(
            [(instance, completion)],
            round_index=1,
            harness_version="H0",
            split_id="held_in_v1",
        )
        config = result.bundle.config
        assert config.verifier_config == {}
        assert config.sampling_seed is None
        assert config.attribution_cache_path is None
        assert config.harness_hash == ""

    def test_validator_version_is_recorded(self):
        instance, completion = failing_run()
        result = make_miner(CountingVerifier()).mine(
            [(instance, completion)],
            round_index=1,
            harness_version="H0",
            split_id="held_in_v1",
        )
        assert result.bundle.config.validator_version == VALIDATOR_VERSION


class TestVerdictRoundTrip:
    def test_from_dict_inverts_to_dict(self):
        verdict = Verdict(
            passed=False,
            cause=VerifierCause.RESOURCE_TERMINATED,
            gold="",
            produced="partial",
            detail="TimeoutExceededError: 61.0s of 60.0s limit",
        )
        assert Verdict.from_dict(verdict.to_dict()) == verdict

    def test_passing_verdict_round_trips(self):
        verdict = Verdict(passed=True, cause=None, gold="[a, b]", produced="[a, b]")
        assert Verdict.from_dict(verdict.to_dict()) == verdict

    def test_unknown_cause_fails_loudly(self):
        with pytest.raises(ValueError):
            Verdict.from_dict({"passed": False, "cause": "not_a_cause", "gold": "", "produced": ""})


class TestEnvironmentCausedRouting:
    """Environment-owned verdict causes are skipped before the attributor is
    called; time-caused terminations stay attributable (efficiency signal)."""

    @staticmethod
    def _mine(verdict: Verdict):
        from shrlm.optimization.types import AttributionErrorKind

        verifier = CountingVerifier()
        lm = MockLM(responses=[CANNED_ATTRIBUTION] * 8)
        miner = WeaknessMiner(verifier=verifier, attributor=LLMAttributor(lm))
        instance, completion = failing_run()
        outcome = miner.record_failure(instance, completion, verdict=verdict)
        return outcome, lm, AttributionErrorKind

    def test_content_filtered_never_reaches_the_attributor(self):
        verdict = Verdict(
            passed=False,
            cause=VerifierCause.CONTENT_FILTERED,
            gold="",
            produced="",
            detail="BadRequestError: content_filter",
        )
        outcome, lm, kind = self._mine(verdict)
        assert outcome.record.attribution_failed is True
        assert outcome.record.attribution_error_kind is kind.ENVIRONMENT
        assert outcome.record.signature is None
        assert lm._call_count == 0  # the attributor LM was never invoked
        assert outcome.raw["attempts"] == []

    def test_budget_termination_is_attributed(self):
        """A spend-cap termination is an efficiency signal exactly like a
        timeout: the run did too much work for its budget, and the verdict
        detail names the exhausted resource so the proposer can hypothesize
        efficiency edits against it."""
        verdict = Verdict(
            passed=False,
            cause=VerifierCause.RESOURCE_TERMINATED,
            gold="",
            produced="",
            detail="BudgetExceededError: run spent $2.01 of $2.00",
        )
        outcome, lm, kind = self._mine(verdict)
        assert outcome.record.attribution_failed is False
        assert outcome.record.signature is not None
        assert outcome.record.signature.verifier_cause is VerifierCause.RESOURCE_TERMINATED
        assert lm._call_count >= 1

    def test_empty_detail_termination_is_environment_skipped(self):
        """Ambiguous terminations default to environment-owned: feeding them
        to the attributor is how platform noise clusters into mechanisms."""
        verdict = Verdict(
            passed=False,
            cause=VerifierCause.RESOURCE_TERMINATED,
            gold="",
            produced="",
            detail="",
        )
        outcome, lm, kind = self._mine(verdict)
        assert outcome.record.attribution_error_kind is kind.ENVIRONMENT
        assert lm._call_count == 0

    @pytest.mark.parametrize(
        "detail",
        [
            "TimeoutExceededError: Timeout exceeded after iteration 14: 3633.7s of 3600.0s limit",
            "HardDeadlineExceeded: hard deadline of 3600.0s reached",
        ],
    )
    def test_time_termination_is_attributed(self, detail):
        """A timeout IS a minable harness weakness: the run did too much work
        for its wall-clock limit, so it keeps flowing to the attributor."""
        verdict = Verdict(
            passed=False,
            cause=VerifierCause.RESOURCE_TERMINATED,
            gold="",
            produced="",
            detail=detail,
        )
        outcome, lm, kind = self._mine(verdict)
        assert outcome.record.attribution_failed is False
        assert outcome.record.signature is not None
        assert outcome.record.signature.verifier_cause is VerifierCause.RESOURCE_TERMINATED
        assert lm._call_count >= 1

    def test_wrong_value_still_attributed(self):
        verdict = Verdict(
            passed=False,
            cause=VerifierCause.WRONG_VALUE,
            gold="4",
            produced="5",
        )
        outcome, lm, kind = self._mine(verdict)
        assert outcome.record.attribution_failed is False
        assert outcome.record.signature is not None



if __name__ == "__main__":
    pytest.main([__file__])
