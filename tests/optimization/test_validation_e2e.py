"""End-to-end tests for U6's ``validate_round``: the whole stage as one call.

``validate_round`` composes loader -> evaluation -> promotion -> merged
re-evaluation -> ledger (R9, R7 orchestration) over real ``proposal.json``
artifacts on disk, gated by the real U1 loader (subprocess and all), evaluated
under the MockLM seam, and ledgered by the U5 writer. The scripted round here
is the plan's capstone scenario: four fabricated candidates -- two genuinely
better on disjoint surfaces (so a merge is built and re-evaluated), one
regressing, one over-budget -- plus one loader-rejected proposal, asserting
that the merged harness is what promotes and that the ledger records every
candidate. Variants pin the merge-failure (promotes nothing, constituents
``merged_failed``), the single-winner path (promoted without re-evaluation),
the degenerate rounds (zero loadable candidates -> no model calls, no ledger),
and idempotent re-invocation (same ledger, zero new model calls).
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import GATE_BASE_HASH
from shrlm.optimization.costs import ValidationCaps
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
)
from shrlm.optimization.validation import (
    BASELINE_ID,
    EvaluationConfig,
    ValidationRound,
    load_promotion_ledger,
    validate_round,
)
from shrlm.rlm_harness import H0
from tests.optimization.test_candidates import proposal_payload, write_payload
from tests.optimization.test_validation import (
    ClientFactory,
    GoldVerifier,
    assert_subject_links_resolve,
    final,
    make_splits,
)

# Per-run budget 0.0015 terminates a two-call burn run at cost 0.002; a
# candidate budget of 0.005 lets a four-run subject complete at 0.004 while
# three burn runs (0.006) trip the breaker mid-held-out (same arithmetic the
# U3/U5 tests pin).
CAPS = ValidationCaps(
    max_depth=2,
    max_iterations=3,
    max_budget=0.0015,
    max_timeout=60.0,
    candidate_budget=0.005,
)

BURN = "Scanning the document, no answer yet."

# Per-subject scripts, 2 instances x 2 splits x 1 rep: held-in first.
BASELINE_SCRIPT = [final("RIGHT"), final("WRONG"), final("RIGHT"), final("WRONG")]  # 1/2, 1/2
REGRESS_SCRIPT = [final("WRONG")] * 4  # 0/2, 0/2
S2_SCRIPT = [final("RIGHT"), final("RIGHT"), final("RIGHT"), final("WRONG")]  # 2/2, 1/2
S3_SCRIPT = [final("RIGHT"), final("WRONG"), final("RIGHT"), final("RIGHT")]  # 1/2, 2/2
MERGED_PASSES = [final("RIGHT")] * 4  # 2/2, 2/2
MERGED_FAILS = [final("WRONG")] * 4  # 0/2, 0/2


def make_config(tmp_path: Path, **overrides: Any) -> EvaluationConfig:
    values: dict[str, Any] = {
        "splits": make_splits(),
        "verifier": GoldVerifier(),
        "caps": CAPS,
        "out_dir": tmp_path / "validation",
        "round_index": 0,
        "repetitions": 1,
        "backend": "openai",
        "backend_kwargs": {"model_name": "validation-e2e"},
    }
    values.update(overrides)
    return EvaluationConfig(**values)


def edited(name: str, field: str, tag: str) -> Any:
    """H0 with one string surface extended by a tag -- a one-surface edit."""
    return replace(H0, name=name, **{field: getattr(H0, field) + f"\n[{tag}]"})


def write_candidate(
    proposals_dir: Path, harness: Any, surface: str, candidate_id: str, **overrides: Any
) -> Path:
    payload = proposal_payload(harness, surface, candidate_id=candidate_id, **overrides)
    return write_payload(proposals_dir, payload)


def seed_merge_proposals(proposals_dir: Path) -> None:
    """The capstone roster: one loader-rejected, one burner, one regressor, two winners.

    Directory names sort as cand-aaa-bad < cand-burn < cand-regress < cand-s2
    < cand-s3, which is the loader's (and therefore the evaluation's) order.
    """
    write_candidate(
        proposals_dir,
        edited("cand-aaa-bad", "decomposition_instruction", "bad"),
        "S2",
        "cand-aaa-bad",
        base_harness_hash="0" * 64,  # wrong base: rejected before any code runs
    )
    write_candidate(
        proposals_dir, edited("cand-burn", "recovery_instruction", "burn"), "S5", "cand-burn"
    )
    write_candidate(
        proposals_dir,
        edited("cand-regress", "verification_instruction", "regress"),
        "S4",
        "cand-regress",
    )
    write_candidate(
        proposals_dir, edited("cand-s2", "decomposition_instruction", "s2"), "S2", "cand-s2"
    )
    write_candidate(
        proposals_dir, edited("cand-s3", "execution_instruction", "s3"), "S3", "cand-s3"
    )


def merge_round_script(merged_script: list[str]) -> list[str]:
    """Evaluation order: baseline, cand-burn, cand-regress, cand-s2, cand-s3, merged."""
    return BASELINE_SCRIPT + [BURN] * 6 + REGRESS_SCRIPT + S2_SCRIPT + S3_SCRIPT + merged_script


def run_merge_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, merged_script: list[str]
) -> tuple[EvaluationConfig, ValidationRound, ClientFactory]:
    proposals_dir = tmp_path / "proposals"
    seed_merge_proposals(proposals_dir)
    factory = ClientFactory(merge_round_script(merged_script))
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_config(tmp_path)
    result = validate_round(H0, proposals_dir, config)
    return config, result, factory


# ---------------------------------------------------------------------------
# The capstone: two disjoint winners merge, re-evaluate, and promote
# ---------------------------------------------------------------------------


class TestMergePromotion:
    def test_merged_harness_is_built_reevaluated_and_promoted(self, tmp_path, monkeypatch):
        _config, result, factory = run_merge_round(tmp_path, monkeypatch, MERGED_PASSES)

        # 4 baseline + 6 burn + 4 regress + 4 s2 + 4 s3 + 4 merged re-evaluation.
        assert factory.total_calls == 26
        assert result.plan.kind == PLAN_MERGE
        assert result.plan.constituent_ids == ("cand-s2", "cand-s3")
        assert result.promoted
        assert result.promoted_harness is not None
        # The promotion artifact carries BOTH edits, composed onto H0.
        assert result.promoted_harness.decomposition_instruction.endswith("[s2]")
        assert result.promoted_harness.execution_instruction.endswith("[s3]")
        assert result.promoted_harness_hash == result.plan.harness_hash
        assert harness_hash(result.promoted_harness) == result.plan.harness_hash

        assert result.ledger is not None
        decision = result.ledger.decision
        assert decision["plan"] == PLAN_MERGE
        assert decision["promoted"] is True
        assert decision["promoted_subject_id"] == MERGED_SUBJECT_ID
        assert decision["promoted_harness_hash"] == result.plan.harness_hash
        assert decision["baseline"]["harness_hash"] == harness_hash(H0)

    def test_ledger_records_every_candidate_including_the_never_ran(self, tmp_path, monkeypatch):
        _config, result, _factory = run_merge_round(tmp_path, monkeypatch, MERGED_PASSES)

        records, _decision = load_promotion_ledger(result.round_path)
        assert [record["subject_id"] for record in records] == [
            "cand-aaa-bad",
            "cand-burn",
            "cand-regress",
            "cand-s2",
            "cand-s3",
            MERGED_SUBJECT_ID,
        ]
        by_id = {record["subject_id"]: record for record in records}

        bad = by_id["cand-aaa-bad"]
        assert bad["decision"] == DECISION_REJECTED
        assert bad["upstream"]["gate"] == GATE_BASE_HASH
        assert bad["links"] is None  # never evaluated: nothing on disk to link

        assert by_id["cand-burn"]["decision"] == DECISION_OVER_BUDGET
        regress = by_id["cand-regress"]
        assert regress["decision"] == DECISION_REJECTED
        assert regress["reasons"]

        for constituent_id in ("cand-s2", "cand-s3"):
            record = by_id[constituent_id]
            assert record["decision"] == DECISION_ACCEPTED
            assert record["merge"]["role"] == "constituent"
            assert record["merge"]["constituent_ids"] == ["cand-s2", "cand-s3"]

        merged = by_id[MERGED_SUBJECT_ID]
        assert merged["decision"] == DECISION_PROMOTED
        assert merged["merge"]["role"] == "merged"
        for record in records:
            if record["links"] is not None:
                assert_subject_links_resolve(
                    result.round_path, record["links"], record["harness_hash"]
                )

    def test_failed_merge_promotes_nothing_and_marks_constituents(self, tmp_path, monkeypatch):
        _config, result, _factory = run_merge_round(tmp_path, monkeypatch, MERGED_FAILS)

        assert result.plan.kind == PLAN_MERGE
        assert not result.promoted
        assert result.promoted_harness is None
        assert result.promoted_harness_hash is None

        records, decision = load_promotion_ledger(result.round_path)
        assert decision["promoted"] is False
        assert decision["promoted_subject_id"] is None
        by_id = {record["subject_id"]: record for record in records}
        for constituent_id in ("cand-s2", "cand-s3"):
            record = by_id[constituent_id]
            assert record["decision"] == DECISION_MERGED_FAILED
            assert any("promotes nothing" in reason for reason in record["reasons"])
        assert by_id[MERGED_SUBJECT_ID]["decision"] == DECISION_REJECTED


# ---------------------------------------------------------------------------
# Single winner: promoted directly, never re-evaluated
# ---------------------------------------------------------------------------


class TestSinglePromotion:
    def test_single_winner_promotes_without_reevaluation(self, tmp_path, monkeypatch):
        proposals_dir = tmp_path / "proposals"
        write_candidate(
            proposals_dir, edited("cand-s2", "decomposition_instruction", "s2"), "S2", "cand-s2"
        )
        factory = ClientFactory(BASELINE_SCRIPT + [final("RIGHT")] * 4)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path)

        result = validate_round(H0, proposals_dir, config)

        # Baseline (4) + the candidate (4): a single winner is never re-run.
        assert factory.total_calls == 8
        assert result.plan.kind == PLAN_SINGLE
        assert result.merge_evaluation is None
        assert result.merge_decision is None
        assert result.promoted
        assert result.promoted_harness is not None
        assert result.promoted_harness.decomposition_instruction.endswith("[s2]")
        assert not (result.round_path / MERGED_SUBJECT_ID).exists()

        records, decision = load_promotion_ledger(result.round_path)
        assert [record["subject_id"] for record in records] == ["cand-s2"]
        assert records[0]["decision"] == DECISION_PROMOTED
        assert decision["plan"] == PLAN_SINGLE
        assert decision["promoted_subject_id"] == "cand-s2"


# ---------------------------------------------------------------------------
# Degenerate rounds: nothing loadable means no model calls and no ledger
# ---------------------------------------------------------------------------


class TestDegenerateRounds:
    def test_all_candidates_loader_rejected_makes_no_model_calls(self, tmp_path, monkeypatch):
        proposals_dir = tmp_path / "proposals"
        write_candidate(
            proposals_dir,
            edited("cand-aaa-bad", "decomposition_instruction", "bad"),
            "S2",
            "cand-aaa-bad",
            base_harness_hash="0" * 64,
        )
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        config = make_config(tmp_path)

        result = validate_round(H0, proposals_dir, config)

        assert idle.total_calls == 0  # evaluation (baseline included) never ran
        assert result.evaluation is None
        assert result.ledger is None
        assert not result.round_path.exists()
        assert result.plan.kind == PLAN_NONE
        assert not result.promoted
        assert result.promoted_harness is None
        assert [rejection.candidate_id for rejection in result.loader_rejections] == [
            "cand-aaa-bad"
        ]
        assert [decision.subject_id for decision in result.decisions] == ["cand-aaa-bad"]
        assert result.decisions[0].decision == DECISION_REJECTED

    def test_empty_proposals_dir_is_a_clean_no_promotion_round(self, tmp_path, monkeypatch):
        proposals_dir = tmp_path / "proposals"
        proposals_dir.mkdir()
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)

        result = validate_round(H0, proposals_dir, make_config(tmp_path))

        assert idle.total_calls == 0
        assert result.evaluation is None
        assert result.ledger is None
        assert result.plan.kind == PLAN_NONE
        assert result.loader_rejections == []
        assert result.decisions == []
        assert not result.promoted

    def test_candidate_claiming_the_merged_id_is_refused(self, tmp_path, monkeypatch):
        proposals_dir = tmp_path / "proposals"
        write_candidate(
            proposals_dir,
            edited(MERGED_SUBJECT_ID, "decomposition_instruction", "sly"),
            "S2",
            MERGED_SUBJECT_ID,
        )
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)

        with pytest.raises(ValueError, match=MERGED_SUBJECT_ID):
            validate_round(H0, proposals_dir, make_config(tmp_path))
        assert idle.total_calls == 0

    def test_candidate_claiming_the_baseline_id_is_refused(self, tmp_path, monkeypatch):
        proposals_dir = tmp_path / "proposals"
        write_candidate(
            proposals_dir,
            edited(BASELINE_ID, "decomposition_instruction", "sly"),
            "S2",
            BASELINE_ID,
        )
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)

        with pytest.raises(ValueError, match=BASELINE_ID):
            validate_round(H0, proposals_dir, make_config(tmp_path))
        assert idle.total_calls == 0


# ---------------------------------------------------------------------------
# Idempotency: re-invoking the round replays from disk, byte for byte
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_rerun_replays_the_same_ledger_with_no_new_model_calls(self, tmp_path, monkeypatch):
        _config, first, _factory = run_merge_round(tmp_path, monkeypatch, MERGED_PASSES)
        assert first.ledger is not None
        ledger_bytes = first.ledger.ledger_path.read_bytes()

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        second = validate_round(H0, tmp_path / "proposals", make_config(tmp_path))

        assert idle.total_calls == 0
        assert second.ledger is not None
        assert second.ledger.records == first.ledger.records
        assert second.ledger.decision == first.ledger.decision
        assert second.promoted_harness_hash == first.promoted_harness_hash
        assert first.ledger.ledger_path.read_bytes() == ledger_bytes


if __name__ == "__main__":
    pytest.main([__file__])
