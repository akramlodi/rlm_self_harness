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
    TAXONOMY_VERSION,
    AgentMechanism,
    CausalStatus,
    EditableSurface,
    FailingLevel,
    VerifierCause,
    render_failing_level_block,
    render_taxonomy_block,
)


def labels_in_block(block: str) -> set[str]:
    """The ``value`` tokens of every ``  - value: doc`` line in a rendered block."""
    return set(re.findall(r"^  - ([a-z_]+):", block, flags=re.MULTILINE))


class TestCoverageInvariants:
    def test_thirteen_concrete_mechanisms_plus_other(self):
        assert len(AgentMechanism) == 14
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

    def test_every_editable_surface_is_reachable_from_some_mechanism(self):
        assert set(MECHANISM_SURFACE.values()) == set(EditableSurface)

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


class TestRenderedPromptText:
    def test_taxonomy_block_labels_match_the_enums_exactly(self):
        # Every member value appears, and no stale name survives a rename.
        expected = {m.value for m in CausalStatus} | {m.value for m in AgentMechanism}
        assert labels_in_block(render_taxonomy_block()) == expected

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
