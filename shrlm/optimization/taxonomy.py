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
"""

from enum import Enum

# Bumped whenever any enum member, MECHANISM_DOCS entry, or MECHANISM_SURFACE
# mapping changes. Bundles carrying different versions are not comparable, so
# the frequency-before-vs-after analysis must assert on this.
TAXONOMY_VERSION = "1.0.0"


class EditableSurface(str, Enum):
    """
    The harness components Self-Harness is permitted to edit.

    Fixed by the project design: the prompt-as-variable, programmatic-sub-call,
    and outputs-in-variables invariants of the reference RLM, along with model
    weights, external tools, and the evaluator, are not editable and have no
    member here.
    """

    DECOMPOSITION_GUIDANCE = "decomposition_guidance"
    SUBCALL_POLICY = "subcall_policy"
    SUBCALL_METADATA = "subcall_metadata"
    ANSWER_MIDDLEWARE = "answer_middleware"
    ERROR_POLICY = "error_policy"
    REPL_HELPERS = "repl_helpers"


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
    downstream. Six surfaces at roughly two mechanisms each gives thirteen
    plus OTHER.

    Three pairs are opposite in direction and must never be merged, since a
    cluster holding both members would demand contradictory edits:
    INCOMPLETE_COVERAGE vs REDUNDANT_DECOMPOSITION, DEPTH_DEGRADATION vs
    INSUFFICIENT_RECURSION, and PREMATURE_TERMINATION vs
    ITERATION_BUDGET_EXHAUSTION.
    """

    # Decomposition guidance
    WHOLE_INPUT_SUBCALL_COLLAPSE = "whole_input_subcall_collapse"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    REDUNDANT_DECOMPOSITION = "redundant_decomposition"
    MISALIGNED_UNIT = "misaligned_unit"

    # Sub-call / batching policy and depth
    DEPTH_DEGRADATION = "depth_degradation"
    INSUFFICIENT_RECURSION = "insufficient_recursion"
    LLM_FOR_RLM_SUBSTITUTION = "llm_for_rlm_substitution"

    # Answer protocol and sub-call return structure
    LOSSY_AGGREGATION = "lossy_aggregation"
    UNPARSED_CHILD_OUTPUT = "unparsed_child_output"

    # Termination policy
    PREMATURE_TERMINATION = "premature_termination"
    ITERATION_BUDGET_EXHAUSTION = "iteration_budget_exhaustion"

    # Sub-call error handling and REPL helpers
    SWALLOWED_SUBCALL_ERROR = "swallowed_subcall_error"
    REPL_EXECUTION_FAULT = "repl_execution_fault"

    OTHER = "other"


# One-line definitions rendered verbatim into the attributor prompt. Keeping
# them here rather than in the prompt string means the prompt cannot drift from
# the code, and a parametrized test asserts every member has an entry.
MECHANISM_DOCS: dict[AgentMechanism, str] = {
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
    AgentMechanism.OTHER: (
        "A recurring mechanism not covered above. Requires a free-text detail describing it."
    ),
}


# Each mechanism maps to exactly one editable surface. OTHER is deliberately
# absent: an unclassified mechanism has no known surface, and
# `surface_addressable` in the actionability score reads that absence.
MECHANISM_SURFACE: dict[AgentMechanism, EditableSurface] = {
    AgentMechanism.WHOLE_INPUT_SUBCALL_COLLAPSE: EditableSurface.DECOMPOSITION_GUIDANCE,
    AgentMechanism.INCOMPLETE_COVERAGE: EditableSurface.DECOMPOSITION_GUIDANCE,
    AgentMechanism.REDUNDANT_DECOMPOSITION: EditableSurface.DECOMPOSITION_GUIDANCE,
    AgentMechanism.MISALIGNED_UNIT: EditableSurface.DECOMPOSITION_GUIDANCE,
    AgentMechanism.DEPTH_DEGRADATION: EditableSurface.SUBCALL_POLICY,
    AgentMechanism.INSUFFICIENT_RECURSION: EditableSurface.SUBCALL_POLICY,
    AgentMechanism.LLM_FOR_RLM_SUBSTITUTION: EditableSurface.SUBCALL_POLICY,
    AgentMechanism.LOSSY_AGGREGATION: EditableSurface.ANSWER_MIDDLEWARE,
    AgentMechanism.UNPARSED_CHILD_OUTPUT: EditableSurface.SUBCALL_METADATA,
    AgentMechanism.PREMATURE_TERMINATION: EditableSurface.ANSWER_MIDDLEWARE,
    AgentMechanism.ITERATION_BUDGET_EXHAUSTION: EditableSurface.SUBCALL_POLICY,
    AgentMechanism.SWALLOWED_SUBCALL_ERROR: EditableSurface.ERROR_POLICY,
    AgentMechanism.REPL_EXECUTION_FAULT: EditableSurface.REPL_HELPERS,
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


def render_taxonomy_block() -> str:
    """
    Render the vocabularies the attributor may choose from.

    verifier_cause is absent by design: it comes from the verifier, and showing
    it as a choice would invite the model to second-guess a checkable outcome.
    """
    return "\n\n".join(
        [
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
