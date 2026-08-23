"""
Closed vocabularies for verifier-grounded failure signatures.

Self-Harness (arXiv:2606.09498, section 3.2) clusters failed runs by *exact
agreement* of a failure signature. Exact-match clustering over labels written
freely by a language model degenerates: nearly every failure receives a unique
signature and every cluster has size one, which defeats the stated purpose of
the stage ("avoid treating failures as isolated anecdotes"). We therefore fix
the vocabulary here. This module is the sole owner of every label literal in
the package; nothing else may hardcode one.

The signature is the four-tuple

    phi(r_i) = (verifier_cause, failing_level, causal_status, agent_mechanism)

extending the paper's three-tuple with ``failing_level``, the level at which
the error signal first appears. Only ``causal_status`` and ``agent_mechanism``
are language-model judgments: ``verifier_cause`` is computed by the verifier
and ``failing_level`` is derived from per-child sub-verifier verdicts whenever
a sub-verifier is supplied.

Version 2.0.0 keys ``EditableSurface`` to the nine surfaces S1-S9 declared in
``shrlm.rlm_harness.SURFACES`` and re-homes ``PREMATURE_TERMINATION`` from the
answer middleware to S4 (pre-submission verification): its documented meaning
-- committing to an answer while evidence of incompleteness was still
available -- is a failure of the verification pass that S4 governs ("what to
check before flipping ``answer['ready']``"), whereas S9 middleware can only
inspect an answer the root has already committed to.

Version 3.0.0 adds the tenth surface, S10 (the skill library), with one
mechanism keyed to it: ``UNCONSULTED_PROCEDURE``, defined against what the
trace digest can show -- the available-skills / loaded-skills pair a run under
a non-empty S10 carries -- with the root re-deriving a procedure it already
carried out as the only signal a pre-S10 trace can show. The surface contract
changed, so bundles written under 2.0.0 are not comparable with 3.0.0 ones;
``shrlm.experiment.pattern_frequency_diff`` excludes them rather than diffing
across the boundary.
"""

from enum import Enum

from shrlm.rlm_harness import SURFACES

# Bumped whenever any enum member, MECHANISM_DOCS entry, or MECHANISM_SURFACE
# mapping changes. Bundles carrying different versions are not comparable, so
# the frequency-before-vs-after analysis excludes bundles written under any
# other version (``pattern_frequency_diff.bundle_completeness``).
TAXONOMY_VERSION = "3.0.0"


class EditableSurface(str, Enum):
    """
    The ten harness surfaces Self-Harness is permitted to edit.

    Values are the surface ids declared in ``shrlm.rlm_harness.SURFACES``, so an
    attribution names a surface stage 2 can actually target; member names follow
    the ``Harness`` field each surface fills. A test asserts the value set equals
    ``SURFACES``' key set and that each member name matches the surface's
    declared builders.

    Fixed by the project design: the prompt-as-variable, programmatic-sub-call,
    and outputs-in-variables invariants of the reference RLM, along with model
    weights, external tools, and the evaluator, are not editable and have no
    member here.
    """

    REPL_CONTRACT = "S1"
    DECOMPOSITION_INSTRUCTION = "S2"
    EXECUTION_INSTRUCTION = "S3"
    VERIFICATION_INSTRUCTION = "S4"
    RECOVERY_INSTRUCTION = "S5"
    RUNTIME_POLICY = "S6"
    METADATA = "S7"
    REPL_HELPERS = "S8"
    ANSWER_MIDDLEWARE = "S9"
    SKILLS = "S10"


# Human-readable surface names, derived from the harness's declared builders
# (``build_<x>`` fills the field ``<x>``) rather than duplicated as literals.
# S8 has two builders and renders as "repl_helpers+sub_repl_helpers".
SURFACE_NAME: dict[EditableSurface, str] = {
    surface: "+".join(
        builder.__name__.removeprefix("build_") for builder in SURFACES[surface.value].builders
    )
    for surface in EditableSurface
}


class SurfaceReach(str, Enum):
    """
    Whether an edit to a surface reaches child RLMs or only the root.

    An attribution that blames a child's own behavior on a root-only surface
    proposes an edit that cannot fix the failure, so the attributor is shown
    this annotation alongside each surface.
    """

    ROOT_ONLY = "root_only"
    CHILD_REACHABLE = "child_reachable"


# Grounded in the code and the residual review record, not in intent:
# - S1-S5 are prompt surfaces concatenated into the system prompt, and a child
#   RLM is constructed with ``custom_system_prompt=self.system_prompt``
#   (``rlm/core/rlm.py``, child spawn in ``_handle_subcall``), so prompt edits
#   reach every level of the tree.
# - The S6/S7/S9 seams are applied at the root only; children do not receive
#   them (docs/residual-review-findings/feature-editable_surfaces.md, C7).
# - S8 is child-reachable through its second builder: ``sub_repl_helpers``
#   becomes ``custom_sub_tools``, which the child spawn propagates as the
#   child's own ``custom_tools``.
# - S10 is child-reachable on two legs (plan KTD7): its name/description index
#   rides the system prompt children inherit, and the runner installs the
#   skill loader in both ``custom_tools`` and ``custom_sub_tools``, so an
#   ``rlm_query`` child below max_depth can load any body itself; the root may
#   additionally forward a loaded body into a sub-call prompt.
SURFACE_REACH: dict[EditableSurface, SurfaceReach] = {
    EditableSurface.REPL_CONTRACT: SurfaceReach.CHILD_REACHABLE,
    EditableSurface.DECOMPOSITION_INSTRUCTION: SurfaceReach.CHILD_REACHABLE,
    EditableSurface.EXECUTION_INSTRUCTION: SurfaceReach.CHILD_REACHABLE,
    EditableSurface.VERIFICATION_INSTRUCTION: SurfaceReach.CHILD_REACHABLE,
    EditableSurface.RECOVERY_INSTRUCTION: SurfaceReach.CHILD_REACHABLE,
    EditableSurface.RUNTIME_POLICY: SurfaceReach.ROOT_ONLY,
    EditableSurface.METADATA: SurfaceReach.ROOT_ONLY,
    EditableSurface.REPL_HELPERS: SurfaceReach.CHILD_REACHABLE,
    EditableSurface.ANSWER_MIDDLEWARE: SurfaceReach.ROOT_ONLY,
    EditableSurface.SKILLS: SurfaceReach.CHILD_REACHABLE,
}


class VerifierCause(str, Enum):
    """
    The terminal verifier-level cause of rejection.

    Produced by the verifier, not by the attributor. The paper names this the
    terminal *verifier-level* cause, and computing it rather than asking for it
    removes one degree of freedom from the language model.

    The three-way split of set-valued errors into INCOMPLETE / SPURIOUS /
    MIXED_SET_ERROR is deliberate and costs nothing (it is arithmetic on two
    sets). INCOMPLETE is the verifier-side shadow of dropped slices; SPURIOUS
    is the shadow of redundant decomposition and of hallucinated aggregation.
    Collapsing them would let one cluster hold failures needing opposite edits.
    """

    NO_ANSWER = "no_answer"
    WRONG_FORMAT = "wrong_format"
    INCOMPLETE = "incomplete"
    SPURIOUS = "spurious"
    MIXED_SET_ERROR = "mixed_set_error"
    WRONG_VALUE = "wrong_value"
    RESOURCE_TERMINATED = "resource_terminated"
    OTHER = "other"


class FailingLevel(str, Enum):
    """
    The level at which the error signal first appears.

    "First appears" resolves the mixed case deterministically: if any child is
    wrong, the signal first appears at the child regardless of whether the
    root's aggregation was also faulty. There is deliberately no BOTH member --
    it would not be derivable from sub-verdicts alone, so the grounded and
    ungrounded configurations would have different label spaces and the
    sub-verification ablation would be comparing incomparable signatures.

    UNDETERMINED serves as the escape member; the space is otherwise
    exhaustive, so a separate OTHER would be dead.
    """

    NO_RECURSION = "no_recursion"  # no sub-calls of any kind were issued
    CHILD = "child"
    ROOT = "root"
    UNDETERMINED = "undetermined"


class CausalStatus(str, Enum):
    """
    How firmly the identified agent behavior is tied to the rejection.

    Kept to four members on purpose. This is the component least anchored in
    observables, so a finer epistemic ladder would buy cluster fragmentation
    rather than resolution. It is load-bearing in exactly one place: the
    actionability score.
    """

    CAUSAL = "causal"
    CONTRIBUTING = "contributing"
    CORRELATED = "correlated"
    UNATTRIBUTED = "unattributed"


class AgentMechanism(str, Enum):
    """
    The reusable, harness-addressable behavior exposed by the trace.

    Cardinality is fixed by one rule: each mechanism maps to exactly one
    EditableSurface. The proposal stage requires every edit to modify a single
    declared surface, so a mechanism spanning two surfaces would be unusable
    downstream. Ten surfaces at one to four mechanisms each gives sixteen
    plus OTHER, and every surface is reachable from at least one mechanism.

    Three pairs are opposite in direction and must never be merged, since a
    cluster holding both members would demand contradictory edits:
    INCOMPLETE_COVERAGE vs REDUNDANT_DECOMPOSITION, DEPTH_DEGRADATION vs
    INSUFFICIENT_RECURSION, and PREMATURE_TERMINATION vs
    ITERATION_BUDGET_EXHAUSTION.
    """

    # S1 -- REPL contract
    REPL_CONTRACT_MISUSE = "repl_contract_misuse"

    # S2 -- decomposition instruction
    WHOLE_INPUT_SUBCALL_COLLAPSE = "whole_input_subcall_collapse"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    REDUNDANT_DECOMPOSITION = "redundant_decomposition"
    MISALIGNED_UNIT = "misaligned_unit"

    # S3 -- execution instruction (what to print, when to offload, depth)
    DEPTH_DEGRADATION = "depth_degradation"
    INSUFFICIENT_RECURSION = "insufficient_recursion"
    LLM_FOR_RLM_SUBSTITUTION = "llm_for_rlm_substitution"

    # S4 -- pre-submission verification
    PREMATURE_TERMINATION = "premature_termination"
    SKIPPED_VERIFICATION = "skipped_verification"

    # S5 -- sub-call failure recovery
    SWALLOWED_SUBCALL_ERROR = "swallowed_subcall_error"

    # S6 -- runtime policy (numbers and switches)
    ITERATION_BUDGET_EXHAUSTION = "iteration_budget_exhaustion"

    # S7 -- metadata / sub-call return structure
    UNPARSED_CHILD_OUTPUT = "unparsed_child_output"

    # S8 -- REPL helpers
    REPL_EXECUTION_FAULT = "repl_execution_fault"

    # S9 -- answer middleware
    LOSSY_AGGREGATION = "lossy_aggregation"

    # S10 -- skill library (reusable procedures, available across turns)
    UNCONSULTED_PROCEDURE = "unconsulted_procedure"

    OTHER = "other"


# One-line definitions rendered verbatim into the attributor prompt. Keeping
# them here rather than in the prompt string means the prompt cannot drift from
# the code, and a parametrized test asserts every member has an entry.
MECHANISM_DOCS: dict[AgentMechanism, str] = {
    AgentMechanism.REPL_CONTRACT_MISUSE: (
        "The root violated or misread the documented REPL contract -- for example attempting to "
        "read the prompt variable into context wholesale, or ignoring the documented variable "
        "protocol."
    ),
    AgentMechanism.WHOLE_INPUT_SUBCALL_COLLAPSE: (
        "The root delegated nearly the entire context to a single sub-call, or performed no "
        "meaningful decomposition at all."
    ),
    AgentMechanism.INCOMPLETE_COVERAGE: (
        "The union of the sub-call inputs did not cover the input; some spans were never examined."
    ),
    AgentMechanism.REDUNDANT_DECOMPOSITION: (
        "Sub-call inputs overlapped or duplicated each other, inflating cost and double-counting "
        "results at merge time."
    ),
    AgentMechanism.MISALIGNED_UNIT: (
        "The decomposition unit did not match the task's unit of solvability, so no sub-call was "
        "in a position to answer its piece correctly."
    ),
    AgentMechanism.DEPTH_DEGRADATION: (
        "Recursion went deeper than the task required; accuracy degraded and cost inflated with "
        "no corresponding gain."
    ),
    AgentMechanism.INSUFFICIENT_RECURSION: (
        "A sub-call received a piece that was still too large and answered it directly instead of "
        "decomposing further."
    ),
    AgentMechanism.LLM_FOR_RLM_SUBSTITUTION: (
        "A plain llm_query was used where a recursive rlm_query was required, or the reverse."
    ),
    AgentMechanism.LOSSY_AGGREGATION: (
        "Sub-calls returned correct local results, but the root's combine step dropped, truncated, "
        "or mis-merged them."
    ),
    AgentMechanism.UNPARSED_CHILD_OUTPUT: (
        "The root could not reliably parse free-text sub-call returns because no return structure "
        "was imposed on them."
    ),
    AgentMechanism.PREMATURE_TERMINATION: (
        "The root committed to an answer while evidence of incompleteness was still available, "
        "such as unvisited slices or an unresolved frontier."
    ),
    AgentMechanism.SKIPPED_VERIFICATION: (
        "The root submitted an answer without any verification pass over the accumulated "
        "results; nothing was checked before the answer was marked ready."
    ),
    AgentMechanism.ITERATION_BUDGET_EXHAUSTION: (
        "The root never committed to an answer; the iteration budget ran out and the answer was "
        "synthesized by the fallback."
    ),
    AgentMechanism.SWALLOWED_SUBCALL_ERROR: (
        "A sub-call failed and returned an error string, which the root consumed as if it were a "
        "valid result."
    ),
    AgentMechanism.REPL_EXECUTION_FAULT: (
        "The failure originated in the root's own code -- a traceback, an off-by-one slice, a bad "
        "index -- rather than in any model judgment."
    ),
    # Defined against the digest's observables only. The digest's Run header
    # carries an available_skills / loaded_skills pair whenever a skill loader
    # was installed (a non-empty S10); an unloaded skill leaves no other trace,
    # and adherence to a loaded body is not a trace fact at all, so "loaded and
    # not followed" is deliberately absent. The precedence clause is the only
    # tie-break there is: no code-level rule exists, so the attributor and the
    # analysis share this text.
    AgentMechanism.UNCONSULTED_PROCEDURE: (
        "A reusable procedure the run needed was never consulted. With a non-empty skill index "
        "(the digest's available_skills line): a skill whose description names the failing step "
        "was available and the root never loaded it (absent from loaded_skills) before failing. "
        "With no available_skills line (no skill was available): the fallback signal is the root "
        "re-deriving, in a later turn or for a later sub-call, a procedure it had already carried "
        "out in this run. Precedence: choose this mechanism only when neither "
        "iteration_budget_exhaustion (S6) nor depth_degradation (S3) independently explains the "
        "terminal failure; when either does, choose that one."
    ),
    AgentMechanism.OTHER: (
        "A recurring mechanism not covered above. Requires a free-text detail describing it."
    ),
}


# Each mechanism maps to exactly one editable surface. OTHER is deliberately
# absent: an unclassified mechanism has no known surface, and
# `surface_addressable` in the actionability score reads that absence.
MECHANISM_SURFACE: dict[AgentMechanism, EditableSurface] = {
    AgentMechanism.REPL_CONTRACT_MISUSE: EditableSurface.REPL_CONTRACT,
    AgentMechanism.WHOLE_INPUT_SUBCALL_COLLAPSE: EditableSurface.DECOMPOSITION_INSTRUCTION,
    AgentMechanism.INCOMPLETE_COVERAGE: EditableSurface.DECOMPOSITION_INSTRUCTION,
    AgentMechanism.REDUNDANT_DECOMPOSITION: EditableSurface.DECOMPOSITION_INSTRUCTION,
    AgentMechanism.MISALIGNED_UNIT: EditableSurface.DECOMPOSITION_INSTRUCTION,
    AgentMechanism.DEPTH_DEGRADATION: EditableSurface.EXECUTION_INSTRUCTION,
    AgentMechanism.INSUFFICIENT_RECURSION: EditableSurface.EXECUTION_INSTRUCTION,
    AgentMechanism.LLM_FOR_RLM_SUBSTITUTION: EditableSurface.EXECUTION_INSTRUCTION,
    AgentMechanism.PREMATURE_TERMINATION: EditableSurface.VERIFICATION_INSTRUCTION,
    AgentMechanism.SKIPPED_VERIFICATION: EditableSurface.VERIFICATION_INSTRUCTION,
    AgentMechanism.SWALLOWED_SUBCALL_ERROR: EditableSurface.RECOVERY_INSTRUCTION,
    AgentMechanism.ITERATION_BUDGET_EXHAUSTION: EditableSurface.RUNTIME_POLICY,
    AgentMechanism.UNPARSED_CHILD_OUTPUT: EditableSurface.METADATA,
    AgentMechanism.REPL_EXECUTION_FAULT: EditableSurface.REPL_HELPERS,
    AgentMechanism.LOSSY_AGGREGATION: EditableSurface.ANSWER_MIDDLEWARE,
    AgentMechanism.UNCONSULTED_PROCEDURE: EditableSurface.SKILLS,
}


CAUSAL_STATUS_DOCS: dict[CausalStatus, str] = {
    CausalStatus.CAUSAL: ("An evidence chain in the trace links the behavior to the wrong output."),
    CausalStatus.CONTRIBUTING: (
        "One of several factors; removing it would plausibly but not certainly repair the run."
    ),
    CausalStatus.CORRELATED: (
        "The behavior co-occurs with the failure but no evidence chain connects them."
    ),
    CausalStatus.UNATTRIBUTED: (
        "The trace evidence is insufficient to place the behavior. Requires a free-text detail."
    ),
}


FAILING_LEVEL_DOCS: dict[FailingLevel, str] = {
    FailingLevel.NO_RECURSION: "No sub-call of any kind was made; decomposition never happened.",
    FailingLevel.CHILD: "At least one sub-call returned a wrong local result.",
    FailingLevel.ROOT: (
        "Every sub-call returned a correct local result; the root combined them into a wrong "
        "answer."
    ),
    FailingLevel.UNDETERMINED: (
        "Sub-calls exist but the trace does not establish whether the error arose at the root or "
        "in a child. Requires a free-text detail."
    ),
}


# Weight assigned to each causal status in the actionability score. Treating
# the attributor's causal judgment as calibrated is an assumption, and it is
# precisely what the sub-verification ablation tests.
CAUSAL_WEIGHT: dict[CausalStatus, float] = {
    CausalStatus.CAUSAL: 1.0,
    CausalStatus.CONTRIBUTING: 0.6,
    CausalStatus.CORRELATED: 0.2,
    CausalStatus.UNATTRIBUTED: 0.0,
}


def render_enum_block(title: str, docs: dict[Enum, str]) -> str:
    """Render one closed vocabulary as prompt text: value, then definition."""
    lines = [f"{title}:"]
    for member, doc in docs.items():
        lines.append(f"  - {member.value}: {doc}")
    return "\n".join(lines)


def render_surface_block() -> str:
    """
    Render the ten editable surfaces with their reach annotations.

    Reach matters to attribution: a mechanism describing a child's own behavior
    must not be pinned on a root-only surface, whose edits children never see.
    """
    lines = ["editable_surfaces (each mechanism implicates exactly one):"]
    for surface in EditableSurface:
        declared = SURFACES[surface.value]
        lines.append(
            f"  - {surface.value} {SURFACE_NAME[surface]}"
            f" [{SURFACE_REACH[surface].value}]: {declared.governs}"
        )
    return "\n".join(lines)


def render_taxonomy_block() -> str:
    """
    Render the vocabularies the attributor may choose from.

    verifier_cause is absent by design: it comes from the verifier, and showing
    it as a choice would invite the model to second-guess a checkable outcome.
    """
    return "\n\n".join(
        [
            render_surface_block(),
            render_enum_block("causal_status (choose exactly one)", CAUSAL_STATUS_DOCS),
            render_enum_block("agent_mechanism (choose exactly one)", MECHANISM_DOCS),
        ]
    )


def render_failing_level_block() -> str:
    """
    Render the failing-level vocabulary.

    Appended to the prompt only in the ungrounded configuration, where no
    sub-verifier is available to derive the level from checkable child
    outcomes.
    """
    return render_enum_block("failing_level (choose exactly one)", FAILING_LEVEL_DOCS)
