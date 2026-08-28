"""Offline proof that an ``H0*``-started loop reaches the model and the gate (U2).

Three things must hold before the live recursion test is worth its spend: the
addendum is in the system prompt the root model actually sees, the candidate
gate accepts one-surface edits against an ``H0*`` incumbent (and still rejects
a flip of the orchestrator scalar), and the proposer's pattern block renders
against ``H0*``'s empty S2-S5 without raising. None of this needs a client.
"""

from dataclasses import replace

import pytest

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
    assert "eligible surfaces: S2" in rendered
    assert '"S2_decomposition_instruction": ""' in rendered


# ---------------------------------------------------------------------------
# U3: the fixed-instance loader behind the live recursion test
# ---------------------------------------------------------------------------


class TestRecursionInstanceLoader:
    """``load_recursion_instances`` over a monkeypatched ``fetch_rows`` seam."""

    def _rows(self):
        from tests.environments.test_graphwalks import BFS_PROMPT, PARENTS_PROMPT, make_row

        return [
            make_row(prompt=f"{BFS_PROMPT}\nvariant {index}", problem_type="bfs")
            for index in range(3)
        ] + [make_row(prompt=f"{PARENTS_PROMPT}\nvariant 9", problem_type="parents")]

    def _patch(self, monkeypatch, rows):
        from shrlm.environments import graphwalks

        monkeypatch.setattr(
            graphwalks, "fetch_rows", lambda repo, file, revision: [dict(r) for r in rows]
        )

    def _ids(self, rows):
        from shrlm.environments.graphwalks import row_to_instance

        return tuple(row_to_instance(row, sample_seed=0, sample_index=0)["id"] for row in rows)

    def test_returns_exactly_the_requested_ids_in_order(self, monkeypatch):
        from shrlm.experiment.config import load_config
        from tests.experiment.recursion_instances import load_recursion_instances

        rows = self._rows()
        self._patch(monkeypatch, rows)
        wanted = (self._ids(rows)[3], self._ids(rows)[1])
        instances = load_recursion_instances(load_config(), ids=wanted)
        assert [instance["id"] for instance in instances] == list(wanted)
        assert instances[0]["problem_type"] == "parents"

    def test_missing_id_raises_naming_it(self, monkeypatch):
        from shrlm.experiment.config import load_config
        from tests.experiment.recursion_instances import load_recursion_instances

        rows = self._rows()
        self._patch(monkeypatch, rows)
        with pytest.raises(LookupError, match="bfs-0000000000000000"):
            load_recursion_instances(load_config(), ids=("bfs-0000000000000000",))

    def test_default_ids_are_the_four_largest_held_in_instances(self):
        from tests.experiment.recursion_instances import RECURSION_INSTANCE_IDS

        assert len(RECURSION_INSTANCE_IDS) == 4
        assert all(len(i.split("-", 1)[1]) == 16 for i in RECURSION_INSTANCE_IDS)
