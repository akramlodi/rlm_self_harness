"""Offline proof that an ``H0*``-started loop reaches the model and the gate (U2).

Three things must hold before the live recursion test is worth its spend: the
addendum is in the system prompt the root model actually sees, the candidate
gate accepts one-surface edits against an ``H0*`` incumbent (and still rejects
a flip of the orchestrator scalar), and the proposer's pattern block renders
against ``H0*``'s empty S2-S5 without raising. None of this needs a client.
"""

from dataclasses import replace

from rlm.utils.prompts import ORCHESTRATOR_ADDENDUM
from shrlm.harness_identity import harness_hash, serialize_harness
from shrlm.optimization.candidates import (
    CandidateRejection,
    LoadedCandidate,
    changed_surfaces,
    load_candidate,
)
from shrlm.optimization.proposal import render_prompt
from shrlm.rlm_harness import H0, H0_STAR
from shrlm.runner import effective_system_prompt
from tests.optimization.test_candidates import proposal_payload, write_payload
from tests.optimization.test_proposal import make_pattern
from tests.test_harness_surfaces import STAR_CLAUSES


class TestAddendumReachesThePrompt:
    def test_h0_star_effective_prompt_carries_the_addendum_and_every_star_clause(self):
        prompt = effective_system_prompt(H0_STAR)
        assert ORCHESTRATOR_ADDENDUM in prompt
        for clause in STAR_CLAUSES:
            assert clause in prompt, clause

    def test_h0_effective_prompt_carries_none_of_it(self):
        prompt = effective_system_prompt(H0)
        assert ORCHESTRATOR_ADDENDUM not in prompt
        for clause in STAR_CLAUSES:
            assert clause not in prompt, clause


class TestGateAcceptsAnH0StarIncumbent:
    def test_one_surface_s2_edit_on_h0_star_materializes(self, tmp_path):
        edited = replace(
            H0_STAR,
            name="cand-star-s2",
            decomposition_instruction="Probe `context`; split the edge list by source node.",
        )
        payload = proposal_payload(
            edited, "S2", candidate_id="cand-star-s2", base_harness_hash=harness_hash(H0_STAR)
        )
        result = load_candidate(write_payload(tmp_path, payload), H0_STAR)
        assert isinstance(result, LoadedCandidate), getattr(result, "reason", result)
        assert changed_surfaces(serialize_harness(H0_STAR), serialize_harness(edited)) == ["S2"]

    def test_orchestrator_flip_on_h0_star_is_rejected(self, tmp_path):
        flipped = replace(H0_STAR, name="cand-star-orch", orchestrator=False)
        payload = proposal_payload(
            flipped, "S2", candidate_id="cand-star-orch", base_harness_hash=harness_hash(H0_STAR)
        )
        result = load_candidate(write_payload(tmp_path, payload), H0_STAR)
        assert isinstance(result, CandidateRejection)
        assert result.gate == "surface_diff"
        assert "orchestrator" in result.reason


def test_pattern_block_renders_against_h0_star_empty_surfaces():
    # ``whole_input_subcall_collapse`` routes to S2, which is "" under H0*.
    pattern = make_pattern("whole_input_subcall_collapse")
    rendered, addressable = render_prompt([pattern], serialize_harness(H0_STAR), (), (), k=4)
    assert [index for index, _ in addressable] == [0]
    assert "surface S2" in rendered
    assert '"S2_decomposition_instruction": ""' in rendered
