"""Tests for the U4 promotion module: the pure offline decision (R6, R7 selection).

Everything here runs on fabricated ``summary.json`` payloads (U3's persisted
aggregate shape) and live in-memory harnesses -- zero model calls, zero
filesystem beyond dataclass ``Path`` placeholders (KTD5). The merged-harness
re-evaluation itself is orchestrated by U6; these tests cover the rule, the
band, selection, merge construction, and the promotes-nothing verdict.
"""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import (
    GATE_SCHEMA,
    SURFACE_SERIALIZATION_KEYS,
    CandidateRejection,
    LoadedCandidate,
)
from shrlm.optimization.costs import OUTCOME_COMPLETED, OUTCOME_OVER_BUDGET
from shrlm.optimization.promotion import (
    DECISION_ACCEPTED,
    DECISION_MERGED_FAILED,
    DECISION_OVER_BUDGET,
    DECISION_PROMOTED,
    DECISION_REJECTED,
    MERGED_SUBJECT_ID,
    PLAN_MERGE,
    PLAN_NONE,
    PLAN_SINGLE,
    SURFACE_HARNESS_FIELDS,
    Band,
    CandidateDecision,
    PromotionConfig,
    apply_merge_verdict,
    assess_round,
    decide_subject,
    merge_harnesses,
    plan_promotion,
    promote_decision,
    score_candidate,
)
from shrlm.optimization.validation import (
    BASELINE_ID,
    SPLIT_HELDIN,
    SPLIT_HELDOUT,
    RoundEvaluation,
    SubjectEvaluation,
)
from shrlm.rlm_harness import H0, Harness

# ---------------------------------------------------------------------------
# Fixtures: fabricated summaries, evaluations, and candidates
# ---------------------------------------------------------------------------

N_RUNS = 4


def split_summary(
    split_id: str,
    pass_count: int,
    *,
    n_runs: int = N_RUNS,
    mean_cost: float = 2.0,
    mean_sub_calls: float = 4.0,
) -> dict[str, Any]:
    return {
        "round_path": f"{split_id}/round_00",
        "n_instances": 2,
        "outcome": OUTCOME_COMPLETED,
        "skipped_run_ids": [],
        "harness_hash": "irrelevant-split-hash",
        "n_runs": n_runs,
        "pass_count": pass_count,
        "pass_rate": pass_count / n_runs if n_runs else None,
        "n_resource_terminated": 0,
        "total_cost": mean_cost * n_runs,
        "mean_cost": mean_cost if n_runs else None,
        "total_sub_calls": int(mean_sub_calls * n_runs),
        "mean_sub_calls": mean_sub_calls if n_runs else None,
    }


def make_summary(
    subject_id: str,
    heldin_pass: int,
    heldout_pass: int,
    *,
    outcome: str = OUTCOME_COMPLETED,
    heldin_kwargs: dict[str, Any] | None = None,
    heldout_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": "shrlm-validation-summary/v1",
        "subject_id": subject_id,
        "harness_hash": f"hash-{subject_id}",
        "repetitions": 2,
        "outcome": outcome,
        "spent": 16.0,
        "splits": {
            SPLIT_HELDIN: split_summary(SPLIT_HELDIN, heldin_pass, **(heldin_kwargs or {})),
            SPLIT_HELDOUT: split_summary(SPLIT_HELDOUT, heldout_pass, **(heldout_kwargs or {})),
        },
    }


BASELINE = make_summary(BASELINE_ID, 2, 2)


def fake_evaluation(summary: dict[str, Any]) -> SubjectEvaluation:
    subject_path = Path("/nonexistent") / summary["subject_id"]
    return SubjectEvaluation(
        subject_id=summary["subject_id"],
        harness_hash=summary["harness_hash"],
        path=subject_path,
        summary_path=subject_path / "summary.json",
        summary=summary,
    )


SURFACE_EDITS = {
    "S2": "decomposition_instruction",
    "S3": "execution_instruction",
    "S4": "verification_instruction",
}


def edited_harness(surface: str, tag: str) -> Harness:
    field_name = SURFACE_EDITS[surface]
    return replace(H0, **{field_name: getattr(H0, field_name) + f"\n[{tag}]"})


def fake_candidate(candidate_id: str, surface: str) -> LoadedCandidate:
    harness = edited_harness(surface, candidate_id)
    return LoadedCandidate(
        candidate_id=candidate_id,
        path=Path("/nonexistent/proposal.json"),
        surface=surface,
        proposal={},
        harness=harness,
        harness_hash=harness_hash(harness),
        module_path=Path("/nonexistent/surfaces.py"),
    )


def score(
    heldin_pass: int, heldout_pass: int, config: PromotionConfig | None = None, **kwargs: Any
) -> CandidateDecision:
    candidate = make_summary("cand-a", heldin_pass, heldout_pass, **kwargs)
    return score_candidate(BASELINE, candidate, config or PromotionConfig())


def accepted_decision(
    candidate_id: str, heldin_pass: int, heldout_pass: int, config: PromotionConfig | None = None
) -> CandidateDecision:
    decision = score_candidate(
        BASELINE, make_summary(candidate_id, heldin_pass, heldout_pass), config or PromotionConfig()
    )
    assert decision.decision == DECISION_ACCEPTED
    return decision


# ---------------------------------------------------------------------------
# The acceptance rule on aggregated pass counts (R6)
# ---------------------------------------------------------------------------


class TestAcceptanceRule:
    def test_improving_both_splits_accepts(self):
        decision = score(3, 4)
        assert decision.decision == DECISION_ACCEPTED
        assert decision.reasons == ()
        assert decision.rule[SPLIT_HELDIN]["delta"] == 1
        assert decision.rule[SPLIT_HELDOUT]["delta"] == 2

    def test_improving_one_split_while_flat_on_the_other_accepts(self):
        assert score(2, 3).decision == DECISION_ACCEPTED

    def test_trading_one_split_for_the_other_rejects(self):
        decision = score(4, 1)
        assert decision.decision == DECISION_REJECTED
        assert any(
            SPLIT_HELDOUT in reason and "tau_regression" in reason for reason in decision.reasons
        )

    def test_flat_on_both_splits_rejects(self):
        decision = score(2, 2)
        assert decision.decision == DECISION_REJECTED
        assert any("tau_improvement" in reason for reason in decision.reasons)

    def test_regressing_both_splits_rejects_with_both_reasons(self):
        decision = score(1, 1)
        assert decision.decision == DECISION_REJECTED
        assert sum("tau_regression" in reason for reason in decision.reasons) == 2

    def test_zero_thresholds_reproduce_the_strict_paper_rule(self):
        # Any single-run regression rejects; any single-run improvement with a
        # flat other split accepts.
        assert score(3, 1).decision == DECISION_REJECTED
        assert score(3, 2).decision == DECISION_ACCEPTED


class TestNoiseMargins:
    def test_single_run_regression_within_tau_reg_still_accepts(self):
        config = PromotionConfig(tau_regression=1)
        decision = score(4, 1, config)
        assert decision.decision == DECISION_ACCEPTED

    def test_regression_boundary_is_inclusive(self):
        # Delta exactly -tau_reg passes the non-regression check (>= -tau).
        config = PromotionConfig(tau_regression=1)
        assert score(4, 1, config).decision == DECISION_ACCEPTED
        assert score(4, 0, config).decision == DECISION_REJECTED

    def test_sub_margin_improvement_rejects(self):
        config = PromotionConfig(tau_improvement=1)
        # Improvement boundary is strict: delta exactly tau_imp is not enough.
        assert score(3, 3, config).decision == DECISION_REJECTED
        assert score(4, 2, config).decision == DECISION_ACCEPTED

    def test_thresholds_are_recorded_on_every_decision(self):
        config = PromotionConfig(tau_regression=1, tau_improvement=2)
        for decision in (score(4, 4, config), score(0, 0, config)):
            assert decision.tau_regression == 1
            assert decision.tau_improvement == 2

    def test_negative_thresholds_are_rejected(self):
        with pytest.raises(ValueError, match="tau_regression"):
            PromotionConfig(tau_regression=-0.5)
        with pytest.raises(ValueError, match="tau_improvement"):
            PromotionConfig(tau_improvement=-1)


# ---------------------------------------------------------------------------
# The preregistered sub-call/cost band (R6, proposal.tex 3.3.3)
# ---------------------------------------------------------------------------


class TestBand:
    def test_cost_outside_the_band_rejects_a_rule_passing_candidate(self):
        config = PromotionConfig(cost_band=Band(0.0, 1.5))
        decision = score(4, 4, config, heldin_kwargs={"mean_cost": 8.0})
        assert decision.decision == DECISION_REJECTED
        assert any("mean_cost" in reason and "band" in reason for reason in decision.reasons)
        assert decision.band["mean_cost"]["within"] is False

    def test_cost_band_aggregates_across_both_splits(self):
        # One split doubles its cost, the other stays flat: the overall mean is
        # 3.0 against a baseline of 2.0, exactly the 1.5x bound -- within.
        config = PromotionConfig(cost_band=Band(0.0, 1.5))
        decision = score(4, 4, config, heldin_kwargs={"mean_cost": 4.0})
        assert decision.band["mean_cost"]["candidate"] == 3.0
        assert decision.decision == DECISION_ACCEPTED

    def test_band_upper_boundary_is_inclusive(self):
        # Baseline overall mean cost 2.0, upper multiplier 1.5: bound exactly 3.0.
        config = PromotionConfig(cost_band=Band(0.0, 1.5))
        per_split = {"mean_cost": 3.0}
        at_bound = score(4, 4, config, heldin_kwargs=per_split, heldout_kwargs=per_split)
        assert at_bound.decision == DECISION_ACCEPTED
        just_above = {"mean_cost": 3.001}
        above = score(4, 4, config, heldin_kwargs=just_above, heldout_kwargs=just_above)
        assert above.decision == DECISION_REJECTED

    def test_band_lower_boundary_is_inclusive(self):
        # Baseline overall mean cost 2.0, lower multiplier 0.5: bound exactly 1.0.
        config = PromotionConfig(cost_band=Band(0.5, 2.0))
        per_split = {"mean_cost": 1.0}
        at_bound = score(4, 4, config, heldin_kwargs=per_split, heldout_kwargs=per_split)
        assert at_bound.decision == DECISION_ACCEPTED
        just_below = {"mean_cost": 0.999}
        below = score(4, 4, config, heldin_kwargs=just_below, heldout_kwargs=just_below)
        assert below.decision == DECISION_REJECTED

    def test_sub_calls_outside_the_band_reject(self):
        config = PromotionConfig(sub_call_band=Band(0.0, 2.0))
        per_split = {"mean_sub_calls": 9.0}
        decision = score(4, 4, config, heldin_kwargs=per_split, heldout_kwargs=per_split)
        assert decision.decision == DECISION_REJECTED
        assert any("mean_sub_calls" in reason for reason in decision.reasons)

    def test_default_band_is_unconstrained(self):
        decision = score(4, 4, heldin_kwargs={"mean_cost": 200.0, "mean_sub_calls": 400.0})
        assert decision.decision == DECISION_ACCEPTED

    def test_zero_baseline_sub_calls_with_a_finite_band_demands_zero(self):
        baseline = make_summary(
            BASELINE_ID,
            2,
            2,
            heldin_kwargs={"mean_sub_calls": 0.0},
            heldout_kwargs={"mean_sub_calls": 0.0},
        )
        config = PromotionConfig(sub_call_band=Band(0.0, 2.0))
        flat = make_summary(
            "cand-a",
            4,
            4,
            heldin_kwargs={"mean_sub_calls": 0.0},
            heldout_kwargs={"mean_sub_calls": 0.0},
        )
        assert score_candidate(baseline, flat, config).decision == DECISION_ACCEPTED
        grew = make_summary(
            "cand-a",
            4,
            4,
            heldin_kwargs={"mean_sub_calls": 1.0},
            heldout_kwargs={"mean_sub_calls": 0.0},
        )
        assert score_candidate(baseline, grew, config).decision == DECISION_REJECTED
        # The default (infinite) band tolerates growth from a zero baseline.
        assert score_candidate(baseline, grew, PromotionConfig()).decision == DECISION_ACCEPTED

    def test_rule_and_band_failures_are_both_reported(self):
        config = PromotionConfig(cost_band=Band(0.0, 1.5))
        decision = score(4, 1, config, heldin_kwargs={"mean_cost": 10.0})
        assert decision.decision == DECISION_REJECTED
        assert any("tau_regression" in reason for reason in decision.reasons)
        assert any("mean_cost" in reason for reason in decision.reasons)

    def test_band_metrics_are_recorded_per_split_metric(self):
        decision = score(3, 3)
        for metric in ("mean_cost", "mean_sub_calls"):
            record = decision.band[metric]
            assert set(record) >= {"baseline", "candidate", "lower", "upper", "within"}
            assert record["within"] is True

    def test_invalid_band_bounds_are_rejected(self):
        with pytest.raises(ValueError, match="lower"):
            Band(-0.1, 1.0)
        with pytest.raises(ValueError, match="upper"):
            Band(2.0, 1.0)


# ---------------------------------------------------------------------------
# Scoring guards: what may never be scored
# ---------------------------------------------------------------------------


class TestScoringGuards:
    def test_over_budget_candidate_is_never_scored(self):
        candidate = make_summary("cand-a", 4, 4, outcome=OUTCOME_OVER_BUDGET)
        with pytest.raises(ValueError, match="over_budget"):
            score_candidate(BASELINE, candidate, PromotionConfig())

    def test_over_budget_baseline_is_refused(self):
        baseline = make_summary(BASELINE_ID, 2, 2, outcome=OUTCOME_OVER_BUDGET)
        with pytest.raises(ValueError, match="baseline"):
            score_candidate(baseline, make_summary("cand-a", 4, 4), PromotionConfig())

    def test_mismatched_run_counts_are_refused(self):
        candidate = make_summary("cand-a", 4, 4, heldin_kwargs={"n_runs": 8})
        with pytest.raises(ValueError, match="n_runs"):
            score_candidate(BASELINE, candidate, PromotionConfig())


# ---------------------------------------------------------------------------
# Assessing a whole round: rejections pass through, never re-scored
# ---------------------------------------------------------------------------


class TestAssessRound:
    def test_every_candidate_gets_exactly_one_decision_in_order(self):
        rejection = CandidateRejection(
            candidate_id="cand-broken",
            gate=GATE_SCHEMA,
            reason="proposal.json unreadable",
            path="/nonexistent/cand-broken/proposal.json",
        )
        evaluation = RoundEvaluation(
            round_path=Path("/nonexistent/round_01"),
            baseline=fake_evaluation(BASELINE),
            candidates=[
                fake_evaluation(make_summary("cand-good", 3, 3)),
                fake_evaluation(make_summary("cand-flat", 2, 2)),
                fake_evaluation(make_summary("cand-burn", 1, 0, outcome=OUTCOME_OVER_BUDGET)),
                rejection,
            ],
        )

        decisions = assess_round(evaluation, PromotionConfig())

        assert [decision.subject_id for decision in decisions] == [
            "cand-good",
            "cand-flat",
            "cand-burn",
            "cand-broken",
        ]
        good, flat, burn, broken = decisions
        assert good.decision == DECISION_ACCEPTED
        assert flat.decision == DECISION_REJECTED
        assert burn.decision == DECISION_OVER_BUDGET
        assert burn.rule is None and burn.band is None
        assert broken.decision == DECISION_REJECTED
        assert broken.upstream == rejection.to_dict()
        assert any(GATE_SCHEMA in reason for reason in broken.reasons)
        # The round-level thresholds land on every record, scored or not.
        assert all(decision.tau_regression == 0 for decision in decisions)

    def test_over_budget_baseline_refuses_assessment(self):
        evaluation = RoundEvaluation(
            round_path=Path("/nonexistent/round_01"),
            baseline=fake_evaluation(make_summary(BASELINE_ID, 2, 2, outcome=OUTCOME_OVER_BUDGET)),
            candidates=[fake_evaluation(make_summary("cand-a", 4, 4))],
        )
        with pytest.raises(ValueError, match="baseline"):
            assess_round(evaluation, PromotionConfig())

    def test_decision_records_serialize_for_the_ledger(self):
        decision = score(3, 3)
        payload = decision.to_dict()
        assert payload["subject_id"] == "cand-a"
        assert payload["decision"] == DECISION_ACCEPTED
        assert payload["reasons"] == []
        assert payload["tau_regression"] == 0.0
        assert payload["tau_improvement"] == 0.0
        assert payload["rule"][SPLIT_HELDIN]["delta"] == 1
        assert payload["band"]["mean_cost"]["within"] is True
        assert payload["harness_hash"] == "hash-cand-a"
        assert payload["upstream"] is None


# ---------------------------------------------------------------------------
# Merge construction and selection (R7, the pure half)
# ---------------------------------------------------------------------------


class TestMergeConstruction:
    def test_surface_field_map_covers_exactly_the_nine_surfaces(self):
        assert set(SURFACE_HARNESS_FIELDS) == set(SURFACE_SERIALIZATION_KEYS)
        for fields in SURFACE_HARNESS_FIELDS.values():
            for field_name in fields:
                assert hasattr(H0, field_name)

    def test_two_disjoint_accepted_candidates_yield_a_merged_plan(self):
        cand_a = fake_candidate("cand-a", "S2")
        cand_b = fake_candidate("cand-b", "S3")
        decisions = [accepted_decision("cand-a", 3, 3), accepted_decision("cand-b", 4, 4)]

        plan = plan_promotion(H0, decisions, [cand_a, cand_b])

        assert plan.kind == PLAN_MERGE
        assert plan.constituent_ids == ("cand-a", "cand-b")
        assert plan.harness.decomposition_instruction == cand_a.harness.decomposition_instruction
        assert plan.harness.execution_instruction == cand_b.harness.execution_instruction
        assert plan.harness.repl_contract == H0.repl_contract
        assert plan.harness_hash == harness_hash(plan.harness)
        assert plan.excluded == {}

    def test_merge_is_order_independent(self):
        cand_a = fake_candidate("cand-a", "S2")
        cand_b = fake_candidate("cand-b", "S3")
        decisions = [accepted_decision("cand-a", 3, 3), accepted_decision("cand-b", 4, 4)]
        forward = plan_promotion(H0, decisions, [cand_a, cand_b])
        backward = plan_promotion(H0, list(reversed(decisions)), [cand_b, cand_a])
        assert forward.harness_hash == backward.harness_hash
        assert forward.constituent_ids == backward.constituent_ids

    def test_same_surface_pair_is_not_merged_higher_heldout_delta_wins(self):
        cand_a = fake_candidate("cand-a", "S2")
        cand_b = fake_candidate("cand-b", "S2")
        decisions = [accepted_decision("cand-a", 4, 3), accepted_decision("cand-b", 3, 4)]

        plan = plan_promotion(H0, decisions, [cand_a, cand_b])

        assert plan.kind == PLAN_SINGLE
        assert plan.constituent_ids == ("cand-b",)
        assert plan.harness is cand_b.harness
        assert plan.harness_hash == cand_b.harness_hash
        assert "cand-a" in plan.excluded
        assert "cand-b" in plan.excluded["cand-a"]

    def test_same_surface_tiebreak_falls_to_heldin_then_candidate_id(self):
        candidates = [fake_candidate("cand-a", "S2"), fake_candidate("cand-b", "S2")]
        # Held-out tied, held-in decides.
        by_heldin = plan_promotion(
            H0,
            [accepted_decision("cand-a", 3, 4), accepted_decision("cand-b", 4, 4)],
            candidates,
        )
        assert by_heldin.constituent_ids == ("cand-b",)
        # Both tied: the lexicographically smaller candidate id wins.
        by_id = plan_promotion(
            H0,
            [accepted_decision("cand-b", 4, 4), accepted_decision("cand-a", 4, 4)],
            candidates,
        )
        assert by_id.constituent_ids == ("cand-a",)

    def test_same_surface_group_still_merges_with_a_disjoint_winner(self):
        candidates = [
            fake_candidate("cand-a", "S2"),
            fake_candidate("cand-b", "S2"),
            fake_candidate("cand-c", "S4"),
        ]
        decisions = [
            accepted_decision("cand-a", 3, 4),
            accepted_decision("cand-b", 3, 3),
            accepted_decision("cand-c", 4, 4),
        ]

        plan = plan_promotion(H0, decisions, candidates)

        assert plan.kind == PLAN_MERGE
        assert plan.constituent_ids == ("cand-a", "cand-c")
        assert "cand-b" in plan.excluded

    def test_single_accepted_candidate_promotes_alone(self):
        cand_a = fake_candidate("cand-a", "S2")
        decisions = [
            accepted_decision("cand-a", 3, 3),
            score_candidate(BASELINE, make_summary("cand-bad", 1, 1), PromotionConfig()),
        ]
        plan = plan_promotion(H0, decisions, [cand_a, fake_candidate("cand-bad", "S3")])
        assert plan.kind == PLAN_SINGLE
        assert plan.constituent_ids == ("cand-a",)
        assert plan.harness_hash == cand_a.harness_hash

    def test_no_accepted_candidates_plans_nothing(self):
        decisions = [score_candidate(BASELINE, make_summary("cand-bad", 1, 1), PromotionConfig())]
        plan = plan_promotion(H0, decisions, [fake_candidate("cand-bad", "S2")])
        assert plan.kind == PLAN_NONE
        assert plan.constituent_ids == ()
        assert plan.harness is None
        assert plan.harness_hash is None

    def test_accepted_decision_without_a_loaded_candidate_is_refused(self):
        with pytest.raises(ValueError, match="cand-a"):
            plan_promotion(H0, [accepted_decision("cand-a", 3, 3)], [])

    def test_merge_harnesses_refuses_overlapping_surfaces(self):
        with pytest.raises(ValueError, match="S2"):
            merge_harnesses(H0, [fake_candidate("cand-a", "S2"), fake_candidate("cand-b", "S2")])


# ---------------------------------------------------------------------------
# The merge verdict: promoted on its own pass, promotes-nothing on failure
# ---------------------------------------------------------------------------


def merged_plan() -> tuple[Any, list[CandidateDecision]]:
    candidates = [fake_candidate("cand-a", "S2"), fake_candidate("cand-b", "S3")]
    decisions = [accepted_decision("cand-a", 3, 3), accepted_decision("cand-b", 4, 4)]
    return plan_promotion(H0, decisions, candidates), decisions


def merge_summary(heldin_pass: int, heldout_pass: int, plan: Any, **kwargs: Any) -> dict[str, Any]:
    summary = make_summary(MERGED_SUBJECT_ID, heldin_pass, heldout_pass, **kwargs)
    summary["harness_hash"] = plan.harness_hash
    return summary


class TestMergeVerdict:
    def test_passing_merge_is_promoted_and_constituents_stay_accepted(self):
        plan, decisions = merged_plan()
        merge_decision = score_candidate(BASELINE, merge_summary(4, 4, plan), PromotionConfig())

        final_merge, final_decisions = apply_merge_verdict(plan, merge_decision, decisions)

        assert final_merge.decision == DECISION_PROMOTED
        assert final_merge.harness_hash == plan.harness_hash
        assert [decision.decision for decision in final_decisions] == [
            DECISION_ACCEPTED,
            DECISION_ACCEPTED,
        ]

    def test_failing_merge_promotes_nothing_and_ledgers_merged_failed(self):
        plan, decisions = merged_plan()
        merge_decision = score_candidate(BASELINE, merge_summary(2, 1, plan), PromotionConfig())
        assert merge_decision.decision == DECISION_REJECTED

        final_merge, final_decisions = apply_merge_verdict(plan, merge_decision, decisions)

        assert final_merge.decision == DECISION_REJECTED
        assert all(decision.decision == DECISION_MERGED_FAILED for decision in final_decisions)
        for decision in final_decisions:
            assert any("promotes nothing" in reason for reason in decision.reasons)
        # The originals are untouched values; nothing was promoted post hoc.
        assert all(decision.decision == DECISION_ACCEPTED for decision in decisions)

    def test_over_budget_merge_also_promotes_nothing(self):
        plan, decisions = merged_plan()
        over_budget = fake_evaluation(merge_summary(0, 0, plan, outcome=OUTCOME_OVER_BUDGET))
        merge_decision = decide_subject(BASELINE, over_budget, PromotionConfig())
        assert merge_decision.decision == DECISION_OVER_BUDGET

        final_merge, final_decisions = apply_merge_verdict(plan, merge_decision, decisions)

        assert final_merge.decision == DECISION_OVER_BUDGET
        assert all(decision.decision == DECISION_MERGED_FAILED for decision in final_decisions)

    def test_non_constituent_decisions_pass_through_untouched(self):
        plan, decisions = merged_plan()
        loser = score_candidate(BASELINE, make_summary("cand-flat", 2, 2), PromotionConfig())
        merge_decision = score_candidate(BASELINE, merge_summary(2, 1, plan), PromotionConfig())

        _, final_decisions = apply_merge_verdict(plan, merge_decision, [*decisions, loser])

        assert final_decisions[-1] is loser

    def test_merge_verdict_demands_a_merge_plan(self):
        cand_a = fake_candidate("cand-a", "S2")
        decisions = [accepted_decision("cand-a", 3, 3)]
        single = plan_promotion(H0, decisions, [cand_a])
        merge_decision = score_candidate(
            BASELINE, make_summary(MERGED_SUBJECT_ID, 4, 4), PromotionConfig()
        )
        with pytest.raises(ValueError, match="merge"):
            apply_merge_verdict(single, merge_decision, decisions)

    def test_merge_verdict_demands_the_planned_harness(self):
        plan, decisions = merged_plan()
        stranger = score_candidate(
            BASELINE, make_summary(MERGED_SUBJECT_ID, 4, 4), PromotionConfig()
        )
        with pytest.raises(ValueError, match="hash"):
            apply_merge_verdict(plan, stranger, decisions)


class TestPromoteDecision:
    def test_accepted_decision_becomes_promoted(self):
        decision = accepted_decision("cand-a", 3, 3)
        promoted = promote_decision(decision)
        assert promoted.decision == DECISION_PROMOTED
        assert promoted.subject_id == "cand-a"

    def test_only_accepted_decisions_may_be_promoted(self):
        rejected = score_candidate(BASELINE, make_summary("cand-bad", 1, 1), PromotionConfig())
        with pytest.raises(ValueError, match="accepted"):
            promote_decision(rejected)


class TestBandMath:
    def test_infinite_upper_bound_never_produces_nan(self):
        band = Band(0.0, math.inf)
        assert band.contains(0.0, 5.0)
        assert band.contains(0.0, 0.0)


if __name__ == "__main__":
    pytest.main([__file__])
