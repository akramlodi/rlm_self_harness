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
    MECHANISM_SURFACES,
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
from shrlm.optimization.types import FailureSignature
from shrlm.rlm_harness import SKILL_LOADER_NAME, SURFACES

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
    def test_sixteen_concrete_mechanisms_plus_other(self):
        assert len(AgentMechanism) == 17
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

    def test_reachable_surfaces_are_exactly_the_ten_declared_surfaces(self):
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

    def test_taxonomy_version_is_the_ten_surface_contract(self):
        # 3.0.0: S10 joined the surface contract (one new mechanism, one new
        # reach entry), so bundles written under 2.0.0 are not comparable.
        # 3.1.0: MECHANISM_SURFACES widened each mechanism to a set of eligible
        # surfaces (primary unchanged) and made OTHER addressable.
        assert TAXONOMY_VERSION == "3.1.0"


class TestSkillsSurface:
    """S10 is declared, reachable, and addressable -- never declared-but-dead.

    The plan's stop condition: a surface no mined mechanism can reach is a
    worse artifact than no surface, because it inflates the denominator of
    every "fraction of surfaces modified" figure while never being proposable.
    """

    def test_s10_is_the_skills_surface(self):
        assert EditableSurface.SKILLS.value == "S10"
        assert SURFACE_NAME[EditableSurface.SKILLS] == "skills"

    def test_s10_is_child_reachable_on_two_legs(self):
        # KTD7: the index rides the system prompt children inherit, and the
        # loader is installed in the child REPL too; neither leg is root-only.
        assert SURFACE_REACH[EditableSurface.SKILLS] is SurfaceReach.CHILD_REACHABLE

    def test_s10_is_the_target_of_the_unconsulted_procedure_mechanism(self):
        assert MECHANISM_SURFACE[AgentMechanism.UNCONSULTED_PROCEDURE] is EditableSurface.SKILLS
        assert EditableSurface.SKILLS in set(MECHANISM_SURFACE.values())

    def test_a_record_carrying_the_mechanism_resolves_to_s10(self):
        signature = FailureSignature(
            verifier_cause=VerifierCause.WRONG_VALUE,
            failing_level=FailingLevel.NO_RECURSION,
            causal_status=CausalStatus.CAUSAL,
            agent_mechanism=AgentMechanism.UNCONSULTED_PROCEDURE,
        )
        assert signature.surface() is EditableSurface.SKILLS

    def test_mechanism_doc_is_defined_against_the_digest_observables(self):
        # The two trace-observable signals, in the words the digest uses, and
        # the no-skills fallback; "loaded and not followed" is excluded by
        # design because adherence is not a trace fact.
        doc = MECHANISM_DOCS[AgentMechanism.UNCONSULTED_PROCEDURE]
        assert "available_skills" in doc
        assert "loaded_skills" in doc
        assert "never loaded" in doc
        assert "already carried out" in doc
        assert "not followed" not in doc

    def test_mechanism_doc_states_the_conservative_precedence_rule(self):
        # S10 claims the terminal signal only when neither S6 budget exhaustion
        # nor S3 depth degradation independently explains it. No code-level
        # precedence exists, so the rule lives in the prompt text and must
        # name both competing mechanisms by their prompt labels.
        doc = MECHANISM_DOCS[AgentMechanism.UNCONSULTED_PROCEDURE]
        assert AgentMechanism.ITERATION_BUDGET_EXHAUSTION.value in doc
        assert AgentMechanism.DEPTH_DEGRADATION.value in doc
        assert "neither" in doc

    def test_surface_block_carries_s10s_governs_contract(self):
        # R14's contract is in the governs text so attributor and proposer
        # both see what separates a legal S10 edit from an S3 rewrite.
        block = render_surface_block()
        s10_line = next(line for line in block.splitlines() if line.startswith("  - S10 "))
        assert "[child_reachable]" in s10_line
        assert "`name`" in s10_line and "`description`" in s10_line and "`body`" in s10_line
        assert "neither description nor body may restate" in s10_line

    def test_surface_block_carries_s8s_loader_exclusion_line(self):
        # S8's governs text names the loader as scaffold that belongs to S10,
        # so a loader NameError is never filed under S8's proposer helpers.
        block = render_surface_block()
        s8_line = next(line for line in block.splitlines() if line.startswith("  - S8 "))
        assert "skill loader" in s8_line and "S10" in s8_line
        # The loader's reserved REPL name never appears as S8 surface content.
        assert SKILL_LOADER_NAME not in s8_line


class TestRenderedPromptText:
    def test_taxonomy_block_labels_match_the_enums_exactly(self):
        # Every member value appears, and no stale name survives a rename.
        expected = {m.value for m in CausalStatus} | {m.value for m in AgentMechanism}
        assert labels_in_block(render_taxonomy_block()) == expected

    def test_taxonomy_block_shows_all_ten_surfaces_with_reach(self):
        block = render_taxonomy_block()
        assert len(EditableSurface) == 10
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


class TestMechanismSurfaces:
    """Taxonomy 3.1.0: every mechanism lists the surfaces it may be addressed
    on, primary first; OTHER may be addressed anywhere."""

    def test_primary_column_is_exactly_mechanism_surface(self):
        for mechanism, primary in MECHANISM_SURFACE.items():
            assert MECHANISM_SURFACES[mechanism][0] is primary

    def test_every_mechanism_including_other_has_eligible_surfaces(self):
        assert set(MECHANISM_SURFACES) == set(AgentMechanism)
        for mechanism, surfaces in MECHANISM_SURFACES.items():
            assert surfaces, mechanism
            assert len(set(surfaces)) == len(surfaces), mechanism
            assert all(isinstance(surface, EditableSurface) for surface in surfaces)

    def test_other_may_be_addressed_on_any_surface(self):
        assert set(MECHANISM_SURFACES[AgentMechanism.OTHER]) == set(EditableSurface)

    def test_answer_middleware_and_skills_are_reachable_without_sub_calls(self):
        # The no-recursion regime of experiment_kimi labelled every failure as
        # one of these four; S9 and S10 must be reachable from at least one.
        no_recursion = (
            AgentMechanism.REPL_CONTRACT_MISUSE,
            AgentMechanism.REPL_EXECUTION_FAULT,
            AgentMechanism.WHOLE_INPUT_SUBCALL_COLLAPSE,
            AgentMechanism.ITERATION_BUDGET_EXHAUSTION,
        )
        reachable = {s for m in no_recursion for s in MECHANISM_SURFACES[m]}
        assert EditableSurface.ANSWER_MIDDLEWARE in reachable
        assert EditableSurface.SKILLS in reachable
        assert EditableSurface.EXECUTION_INSTRUCTION in reachable
