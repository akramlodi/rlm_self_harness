"""Tests for the closed vocabularies in shrlm/optimization/taxonomy.py.

The taxonomy is the clustering key: a mechanism without documentation would be
invisible to the attributor, and one without a surface would be unusable by
the proposal stage. These tests pin the coverage invariants and check that the
generated prompt text cannot drift from the enums it is generated from.
"""

import re

import pytest

from shrlm.optimization.taxonomy import (
    CAUSAL_STATUS_DOCS,
    CAUSAL_WEIGHT,
    FAILING_LEVEL_DOCS,
    MECHANISM_DOCS,
    MECHANISM_SURFACE,
    SURFACE_NAME,
    SURFACE_REACH,
    TAXONOMY_VERSION,
    AgentMechanism,
    CausalStatus,
    EditableSurface,
    FailingLevel,
    SurfaceReach,
    VerifierCause,
    render_failing_level_block,
    render_surface_block,
    render_taxonomy_block,
)
from shrlm.rlm_harness import SURFACES

# Mechanisms whose documented meaning is a child's own behavior rather than the
# root's: INSUFFICIENT_RECURSION ("a sub-call received a piece ... and answered
# it directly") and DEPTH_DEGRADATION (excess recursion happens inside the
# sub-tree, below the root's own decision). An edit fixing them must reach the
# children, so they must never map to a root-only surface.
CHILD_LEVEL_MECHANISMS = {
    AgentMechanism.INSUFFICIENT_RECURSION,
    AgentMechanism.DEPTH_DEGRADATION,
}

# Surface values retired in taxonomy 2.0.0; none may survive in rendered text.
RETIRED_SURFACE_VALUES = {
    "decomposition_guidance",
    "subcall_policy",
    "subcall_metadata",
    "error_policy",
}


def labels_in_block(block: str) -> set[str]:
    """The ``value`` tokens of every ``  - value: doc`` line in a rendered block."""
    return set(re.findall(r"^  - ([a-z_]+):", block, flags=re.MULTILINE))


class TestSurfaceAgreementWithHarness:
    def test_surface_values_are_exactly_the_declared_harness_surface_ids(self):
        assert {member.value for member in EditableSurface} == set(SURFACES)

    def test_member_names_follow_the_declared_builders(self):
        # SURFACE_NAME is derived from the harness's build_<x> naming convention,
        # so a renamed builder or a drifted enum member fails here.
        for surface in EditableSurface:
            assert SURFACE_NAME[surface].startswith(surface.name.lower())

    def test_every_surface_has_a_reach_annotation(self):
        assert set(SURFACE_REACH) == set(EditableSurface)
        assert all(isinstance(reach, SurfaceReach) for reach in SURFACE_REACH.values())

    def test_root_only_surfaces_are_the_c7_seams(self):
        # docs/residual-review-findings/feature-editable_surfaces.md, C7: the
        # S6/S7/S9 seams apply at the root only. S1-S5 travel with the system
        # prompt, which children inherit; S8 propagates via sub_repl_helpers.
        root_only = {s for s, reach in SURFACE_REACH.items() if reach is SurfaceReach.ROOT_ONLY}
        assert root_only == {
            EditableSurface.RUNTIME_POLICY,
            EditableSurface.METADATA,
            EditableSurface.ANSWER_MIDDLEWARE,
        }


class TestCoverageInvariants:
    def test_fifteen_concrete_mechanisms_plus_other(self):
        assert len(AgentMechanism) == 16
        assert AgentMechanism.OTHER in AgentMechanism

    def test_every_mechanism_is_documented(self):
        assert set(MECHANISM_DOCS) == set(AgentMechanism)
        assert all(isinstance(doc, str) and doc for doc in MECHANISM_DOCS.values())

    def test_every_concrete_mechanism_maps_to_exactly_one_surface(self):
        # A dict gives at most one surface per mechanism; coverage gives at least one.
        assert set(MECHANISM_SURFACE) == set(AgentMechanism) - {AgentMechanism.OTHER}
        assert all(isinstance(s, EditableSurface) for s in MECHANISM_SURFACE.values())

    def test_other_has_no_surface_by_design(self):
        assert AgentMechanism.OTHER not in MECHANISM_SURFACE

    def test_reachable_surfaces_are_exactly_the_nine_declared_surfaces(self):
        # Compared against the harness declaration, not the enum, so a shrunken
        # or drifted mapping fails even if the enum drifts with it.
        assert {surface.value for surface in MECHANISM_SURFACE.values()} == set(SURFACES)
        assert set(MECHANISM_SURFACE.values()) == set(EditableSurface)

    def test_child_level_mechanisms_never_map_to_a_root_only_surface(self):
        for mechanism in CHILD_LEVEL_MECHANISMS:
            surface = MECHANISM_SURFACE[mechanism]
            assert SURFACE_REACH[surface] is SurfaceReach.CHILD_REACHABLE

    def test_premature_termination_is_homed_to_pre_submission_verification(self):
        # Taxonomy 2.0.0 decision: committing while incompleteness evidence was
        # available is a failed S4 verification pass, not an S9 middleware issue.
        s4 = MECHANISM_SURFACE[AgentMechanism.PREMATURE_TERMINATION]
        assert s4 is EditableSurface.VERIFICATION_INSTRUCTION
        assert MECHANISM_SURFACE[AgentMechanism.SKIPPED_VERIFICATION] is s4

    def test_every_causal_status_is_documented_and_weighted(self):
        assert set(CAUSAL_STATUS_DOCS) == set(CausalStatus)
        assert set(CAUSAL_WEIGHT) == set(CausalStatus)

    def test_causal_weights_order_matches_the_epistemic_ladder(self):
        assert (
            CAUSAL_WEIGHT[CausalStatus.CAUSAL]
            > CAUSAL_WEIGHT[CausalStatus.CONTRIBUTING]
            > CAUSAL_WEIGHT[CausalStatus.CORRELATED]
            > CAUSAL_WEIGHT[CausalStatus.UNATTRIBUTED]
        )
        assert CAUSAL_WEIGHT[CausalStatus.UNATTRIBUTED] == 0.0

    def test_every_failing_level_is_documented(self):
        assert set(FAILING_LEVEL_DOCS) == set(FailingLevel)

    def test_failing_level_has_no_both_member(self):
        assert "BOTH" not in FailingLevel.__members__

    def test_taxonomy_version_is_a_semver_string(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", TAXONOMY_VERSION)

    def test_taxonomy_version_is_the_nine_surface_contract(self):
        assert TAXONOMY_VERSION == "2.0.0"


class TestRenderedPromptText:
    def test_taxonomy_block_labels_match_the_enums_exactly(self):
        # Every member value appears, and no stale name survives a rename.
        expected = {m.value for m in CausalStatus} | {m.value for m in AgentMechanism}
        assert labels_in_block(render_taxonomy_block()) == expected

    def test_taxonomy_block_shows_all_nine_surfaces_with_reach(self):
        block = render_taxonomy_block()
        for surface in EditableSurface:
            line = f"- {surface.value} {SURFACE_NAME[surface]} [{SURFACE_REACH[surface].value}]:"
            assert line in block
        assert render_surface_block() in block

    def test_taxonomy_block_carries_no_retired_surface_vocabulary(self):
        block = render_taxonomy_block()
        for retired in RETIRED_SURFACE_VALUES:
            assert retired not in block

    @pytest.mark.parametrize("member", list(AgentMechanism))
    def test_each_mechanism_value_appears_in_the_block(self, member: AgentMechanism):
        assert f"- {member.value}:" in render_taxonomy_block()

    def test_verifier_cause_is_absent_from_the_attributor_menu(self):
        # It comes from the verifier; offering it as a choice would invite the
        # model to second-guess a checkable outcome.
        block = render_taxonomy_block()
        assert "verifier_cause" not in block
        own_causes = {m.value for m in VerifierCause} - {m.value for m in AgentMechanism}
        assert not labels_in_block(block) & own_causes

    def test_failing_level_block_labels_match_the_enum_exactly(self):
        assert labels_in_block(render_failing_level_block()) == {m.value for m in FailingLevel}

    def test_failing_levels_are_not_in_the_main_block(self):
        # The failing-level menu is appended only in the ungrounded configuration.
        own_levels = {m.value for m in FailingLevel} - {m.value for m in AgentMechanism}
        assert not labels_in_block(render_taxonomy_block()) & own_levels
