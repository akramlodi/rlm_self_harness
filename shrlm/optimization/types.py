"""
Data structures for weakness mining.

The call tree (CallNode and friends) is a reconstruction of a completed RLM run
from its logged trajectory. FailureRecord is the mining-stage analogue of the
paper's trace record r_i = (x_i, tau_i, y_i, z_i); FailurePattern is a cluster
C_phi together with the evidence the proposer is shown; EvidenceBundle is B_t.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from shrlm.optimization.taxonomy import (
    AgentMechanism,
    CausalStatus,
    EditableSurface,
    FailingLevel,
    VerifierCause,
)


class NodeKind(str, Enum):
    """
    What a node in the reconstructed call tree is.

    INDETERMINATE is not a failure of the walker. At maximum depth an
    ``rlm_query`` and an ``llm_query`` produce byte-identical records, so the
    distinction genuinely is not present in the trace; asserting one would
    fabricate evidence.
    """

    ROOT = "root"
    RLM_CHILD = "rlm_child"
    LLM_LEAF = "llm_leaf"
    ERRORED = "errored"
    INDETERMINATE = "indeterminate"


class TraceIntegrity(str, Enum):
    """Whether the reconstructed tree is known to be missing nodes."""

    COMPLETE = "complete"
    DEGRADED = "degraded"


class AttributionErrorKind(str, Enum):
    """Why an attribution failed: the model's response was unusable, or the
    model was never reached at all. The distinction matters downstream --
    transport failures are exempt from the attempts-audit demand and are
    counted separately in the integrity report."""

    REJECTION = "rejection"
    TRANSPORT = "transport"


@dataclass
class CodeBlockNode:
    """One ```repl block executed within an iteration."""

    code: str
    stdout: str
    stderr: str
    final_answer: str | None
    calls: list["CallNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "final_answer": self.final_answer,
            "calls": [call.to_dict() for call in self.calls],
        }


@dataclass
class IterationNode:
    """One root-model turn: a response, plus any code it executed."""

    index: int
    response: str
    code_blocks: list[CodeBlockNode]
    final_answer: str | None
    terminated_by_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "response": self.response,
            "code_blocks": [block.to_dict() for block in self.code_blocks],
            "final_answer": self.final_answer,
            "terminated_by_fallback": self.terminated_by_fallback,
        }


@dataclass
class CallNode:
    """
    One model call in the reconstructed tree.

    ``node_id`` is positional and stable -- "r" for the root, then
    ``{parent}/i{iteration}/b{block}/c{call}``. It is the join key between the
    tree, the sub-verifier verdicts, and the excerpts cited in the bundle, so
    it must not depend on anything that varies between runs.

    The REPL ``locals`` dict is deliberately never carried here: it is lossy
    once serialized (rlm/core/types.py::_serialize_value reprs anything
    non-primitive) and it is large.
    """

    node_id: str
    parent_id: str | None
    kind: NodeKind
    depth: int
    model: str
    prompt: str | dict[str, Any]
    response: str
    prompt_chars: int
    response_chars: int
    execution_time: float | None
    error_kind: str | None = None
    ambiguous: bool = False
    iterations: list[IterationNode] = field(default_factory=list)
    children: list["CallNode"] = field(default_factory=list)
    sub_verdict: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "kind": self.kind.value,
            "depth": self.depth,
            "model": self.model,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "execution_time": self.execution_time,
            "error_kind": self.error_kind,
            "ambiguous": self.ambiguous,
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "children": [child.to_dict() for child in self.children],
            "sub_verdict": self.sub_verdict,
        }


@dataclass
class TreeStats:
    """
    Mechanical statistics over a reconstructed tree.

    Every field here is computed without a language model, which is what lets
    cluster-level "shared symptoms" stay a deterministic function of the
    records rather than a second sampling step.
    """

    n_nodes: int
    n_rlm_children: int
    n_llm_leaves: int
    n_errored: int
    n_indeterminate: int
    max_observed_depth: int
    n_iterations: int
    root_context_chars: int
    max_child_prompt_chars: int
    collapse_ratio: float
    terminated_by_fallback: bool
    recursion_available: bool
    block_attribution_reliable: bool
    suspected_lost_subcalls: int
    trace_integrity: TraceIntegrity

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_rlm_children": self.n_rlm_children,
            "n_llm_leaves": self.n_llm_leaves,
            "n_errored": self.n_errored,
            "n_indeterminate": self.n_indeterminate,
            "max_observed_depth": self.max_observed_depth,
            "n_iterations": self.n_iterations,
            "root_context_chars": self.root_context_chars,
            "max_child_prompt_chars": self.max_child_prompt_chars,
            "collapse_ratio": self.collapse_ratio,
            "terminated_by_fallback": self.terminated_by_fallback,
            "recursion_available": self.recursion_available,
            "block_attribution_reliable": self.block_attribution_reliable,
            "suspected_lost_subcalls": self.suspected_lost_subcalls,
            "trace_integrity": self.trace_integrity.value,
        }


@dataclass
class Verdict:
    """
    A verifier's judgment on a completed run.

    Returning a cause rather than a bare bool is what keeps the terminal
    verifier-level cause out of the language model's hands. Environment
    verifiers stay unmodified: a thin adapter derives the cause from the
    pass/fail outcome plus the gold and produced answers.
    """

    passed: bool
    cause: VerifierCause | None
    gold: str
    produced: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.passed and self.cause is not None:
            raise ValueError("A passing verdict must not carry a failure cause")
        if not self.passed and self.cause is None:
            raise ValueError("A failing verdict must carry a failure cause")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cause": self.cause.value if self.cause else None,
            "gold": self.gold,
            "produced": self.produced,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Verdict":
        """Rehydrate a persisted verdict; the inverse of ``to_dict``.

        An unknown cause string raises ``ValueError`` through the enum
        constructor rather than being coerced, since a persisted verdict from a
        different taxonomy version is not comparable evidence.
        """
        cause = data.get("cause")
        return cls(
            passed=bool(data["passed"]),
            cause=VerifierCause(cause) if cause is not None else None,
            gold=str(data.get("gold", "")),
            produced=str(data.get("produced", "")),
            detail=str(data.get("detail", "")),
        )


class Verifier(Protocol):
    """Deterministic outcome for a whole run. Supplied per environment."""

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict: ...


class SubVerifier(Protocol):
    """
    Deterministic outcome for a single sub-call.

    Returns None when the node's sub-problem is not checkable in isolation --
    which is the normal case for a plain llm_query leaf, and the open question
    for grandchildren, whose sub-problems are defined by their parent's
    arbitrary decomposition rather than by the environment.
    """

    def __call__(self, instance: dict[str, Any], node: CallNode) -> bool | None: ...


@dataclass(frozen=True)
class FailureSignature:
    """
    phi(r_i). The clustering key.

    The ``*_detail`` free-text fields live on AttributionDetail, not here, so
    that two records differing only in wording cluster together. That exclusion
    is the whole reason the closed taxonomy exists.
    """

    verifier_cause: VerifierCause
    failing_level: FailingLevel
    causal_status: CausalStatus
    agent_mechanism: AgentMechanism

    def key(self) -> tuple[str, str, str, str]:
        return (
            self.verifier_cause.value,
            self.failing_level.value,
            self.causal_status.value,
            self.agent_mechanism.value,
        )

    def surface(self) -> EditableSurface | None:
        from shrlm.optimization.taxonomy import MECHANISM_SURFACE

        return MECHANISM_SURFACE.get(self.agent_mechanism)

    def to_dict(self) -> dict[str, str]:
        return {
            "verifier_cause": self.verifier_cause.value,
            "failing_level": self.failing_level.value,
            "causal_status": self.causal_status.value,
            "agent_mechanism": self.agent_mechanism.value,
        }


@dataclass
class AttributionDetail:
    """
    Free-text context accompanying a signature. Never part of the cluster key.
    """

    symptom_summary: str
    evidence_node_ids: list[str]
    failing_level_detail: str = ""
    causal_status_detail: str = ""
    agent_mechanism_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symptom_summary": self.symptom_summary,
            "evidence_node_ids": list(self.evidence_node_ids),
            "failing_level_detail": self.failing_level_detail,
            "causal_status_detail": self.causal_status_detail,
            "agent_mechanism_detail": self.agent_mechanism_detail,
        }


@dataclass(frozen=True)
class RunTraceLink:
    """Where one run's persisted trace lives: the join from record to disk.

    ``trace_path`` is relative to the round directory and ``trace_sha256`` is
    over the trace file's bytes, exactly as ``runs.jsonl`` records them, so a
    bundle reader can resolve and verify every record's raw evidence.
    """

    run_id: str
    trace_path: str
    trace_sha256: str


@dataclass
class FailureRecord:
    """
    One failed run, attributed. The mining-stage analogue of r_i.

    ``level_grounded`` is the single ablation switch: True when failing_level
    came from checkable sub-verdicts, False when the attributor inferred it.
    Appendix B's sub-verification ablation is exactly a comparison across this
    flag, so nothing else may vary with it.

    ``run_id``, ``trace_path``, and ``trace_sha256`` link the record back to
    its persisted trace (see ``RunTraceLink``). They are None for legacy
    callers that mine in-memory completions with no persisted round behind
    them; ``mine_round`` always populates them from ``runs.jsonl``.

    ``attribution_error_kind`` holds an ``AttributionErrorKind`` value when
    ``attribution_failed`` is set by mining; it is None on records that
    attributed cleanly and on legacy records, whose readers fall back to the
    ``transport failure`` prefix of ``attribution_error``.
    """

    instance_id: str
    verdict: Verdict
    stats: TreeStats
    signature: FailureSignature | None
    detail: AttributionDetail | None
    level_grounded: bool
    sub_verdicts: dict[str, bool | None] = field(default_factory=dict)
    digest_sha256: str = ""
    attribution_failed: bool = False
    attribution_error: str = ""
    attribution_error_kind: AttributionErrorKind | None = None
    run_id: str | None = None
    trace_path: str | None = None
    trace_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "verdict": self.verdict.to_dict(),
            "stats": self.stats.to_dict(),
            "signature": self.signature.to_dict() if self.signature else None,
            "detail": self.detail.to_dict() if self.detail else None,
            "level_grounded": self.level_grounded,
            "sub_verdicts": dict(self.sub_verdicts),
            "digest_sha256": self.digest_sha256,
            "attribution_failed": self.attribution_failed,
            "attribution_error": self.attribution_error,
            "attribution_error_kind": (
                self.attribution_error_kind.value if self.attribution_error_kind else None
            ),
            "run_id": self.run_id,
            "trace_path": self.trace_path,
            "trace_sha256": self.trace_sha256,
        }


@dataclass
class FailurePattern:
    """
    A cluster C_phi with the evidence shown to the proposer.

    There is deliberately no field that could hold a proposed edit.
    ``surface`` is a taxonomy fact -- which editable surface this mechanism
    lives on -- and not a recommendation to edit it.

    ``support`` counts member *runs*; ``instance_support`` counts distinct
    instance ids (KTD7/R2). Repeated failing attempts of one instance are real
    evidence of the failure but not evidence of breadth, so cluster ordering
    uses instance_support and ``support`` stays the run count.

    ``actionability`` orders the proposer's reading order only. It is not a
    prediction of edit success, and no promotion decision may consume it.
    """

    signature: FailureSignature
    support: int
    instance_support: int
    instance_ids: list[str]
    representatives: list[str]
    shared_symptoms: list[str]
    verifier_evidence: list[str]
    grounded_fraction: float
    surface: EditableSurface | None
    actionability: float
    below_support_floor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature.to_dict(),
            "support": self.support,
            "instance_support": self.instance_support,
            "instance_ids": list(self.instance_ids),
            "representatives": list(self.representatives),
            "shared_symptoms": list(self.shared_symptoms),
            "verifier_evidence": list(self.verifier_evidence),
            "grounded_fraction": self.grounded_fraction,
            "surface": self.surface.value if self.surface else None,
            "actionability": self.actionability,
            "below_support_floor": self.below_support_floor,
        }


@dataclass
class MiningTotals:
    """Round-level counts. Makes excluded records visible rather than silent."""

    n_runs: int
    n_failures: int
    n_attributed: int
    n_unattributed: int
    n_grounded: int
    n_degraded_trees: int

    def to_dict(self) -> dict[str, int]:
        return {
            "n_runs": self.n_runs,
            "n_failures": self.n_failures,
            "n_attributed": self.n_attributed,
            "n_unattributed": self.n_unattributed,
            "n_grounded": self.n_grounded,
            "n_degraded_trees": self.n_degraded_trees,
        }


@dataclass(frozen=True)
class SubstrateBias:
    """A known, code-level measurement bias in the run substrate.

    Named by its residual-review defect id (docs/residual-review-findings/)
    so a bundle reader can find the full analysis. These are static facts
    about the substrate the round ran on, not observations about the round;
    they belong in the bundle because they bias exactly the statistics the
    bundle reports.
    """

    defect_id: str
    summary: str
    effect: str

    def to_dict(self) -> dict[str, str]:
        return {
            "defect_id": self.defect_id,
            "summary": self.summary,
            "effect": self.effect,
        }


# The substrate biases every bundle must disclose, from the residual review of
# feature/editable_surfaces. Kept static: they describe the code the runs
# executed under, and they stay listed until the defects are fixed.
KNOWN_SUBSTRATE_BIASES: tuple[SubstrateBias, ...] = (
    SubstrateBias(
        defect_id="A1",
        summary="batched sub-calls always record syntax_error=False",
        effect=(
            "syntax-error counts are a lower bound: the batched execution paths never "
            "classify, so syntax-error rate reads artificially low as batching increases"
        ),
    ),
    SubstrateBias(
        defect_id="A2",
        summary="retried sub-calls are billed as one call",
        effect=(
            "per-run cost is a lower bound: only the last attempt of a retried sub-call "
            "is recorded, under-pricing retry-heavy behavior"
        ),
    ),
)


@dataclass
class IntegrityReport:
    """
    Known limits on what the traces could show.

    ``total_suspected_lost_subcalls`` is a lower bound: a swallowed sub-call
    error that was never printed leaves no trace at all. It matters because
    missing child nodes bias failing_level toward ROOT, which is the very
    distinction the sub-verifier grounding exists to establish.

    The ``n_unattributed`` / ``n_ungrounded`` / ``n_resource_terminated`` /
    ``n_transport_errors`` counts make the round's excluded or weakened
    evidence visible in the bundle itself, and ``known_substrate_biases``
    names the code-level defects (by review id) that bias what the traces
    could record in the first place.
    """

    total_suspected_lost_subcalls: int
    n_records_with_lost_subcalls: int
    n_records_unreliable_block_attribution: int
    n_indeterminate_nodes: int
    mean_digest_coverage: float
    n_unattributed: int
    n_ungrounded: int
    n_resource_terminated: int
    n_transport_errors: int
    known_substrate_biases: list[SubstrateBias] = field(
        default_factory=lambda: list(KNOWN_SUBSTRATE_BIASES)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_suspected_lost_subcalls": self.total_suspected_lost_subcalls,
            "n_records_with_lost_subcalls": self.n_records_with_lost_subcalls,
            "n_records_unreliable_block_attribution": (self.n_records_unreliable_block_attribution),
            "n_indeterminate_nodes": self.n_indeterminate_nodes,
            "mean_digest_coverage": self.mean_digest_coverage,
            "n_unattributed": self.n_unattributed,
            "n_ungrounded": self.n_ungrounded,
            "n_resource_terminated": self.n_resource_terminated,
            "n_transport_errors": self.n_transport_errors,
            "known_substrate_biases": [bias.to_dict() for bias in self.known_substrate_biases],
        }


@dataclass
class MiningConfig:
    """
    Everything needed to reproduce a bundle from saved configuration.

    Serialized into the bundle and hashed into its id, so a re-run that differs
    in any of these fields is visibly a different bundle rather than a silently
    different one.

    Provenance fields:

    * ``harness_version`` is the caller-facing harness identifier and
      ``harness_hash`` is the content hash recorded in the round's
      ``harness.json``. ``mine_round`` sets harness_version to that same hash
      by default, so the two normally coincide; they stay separate fields
      because a caller may label the harness ("H0") while the hash remains the
      checkable identity. ``harness_hash`` is empty only for legacy in-memory
      callers with no persisted round.
    * ``verifier_config`` holds the environment verifier's own facts (its
      ``config()`` payload, e.g. pass threshold and extraction rule) so two
      rounds judged under different verifier settings can never share a
      bundle id.
    * ``sampling_seed`` is the dataset sampling seed carried by the round's
      instances as provenance; None when the instances carry none.
    * ``prompt_sha256`` pins the attributor system-prompt *template*: the
      sha256 of the raw ``ATTRIBUTOR_SYSTEM_PROMPT`` constant, before any
      per-variant rendering. A round renders per-record prompt variants, so
      the rendered shas live on each ``attributions.jsonl`` entry (audited
      against the persisted prompt files), not here.
    * ``validator_version`` pins the attribution response validator, and
      ``attribution_cache_path`` (round-dir-relative when set by
      ``mine_round``) names the cache whose replayed responses made the round
      reproducible; it is None when the cache file does not exist at mining
      time (e.g. an all-pass round that never sampled an attribution).
    """

    round_index: int
    harness_version: str
    split_id: str
    taxonomy_version: str
    prompt_version: str
    digest_version: str
    prompt_sha256: str
    attributor_model: str
    attributor_sampling_args: dict[str, Any]
    digest_char_budget: int
    digest_focus_k: int
    max_attempts: int
    sub_verifier_enabled: bool
    min_support: int
    actionability_weights: dict[str, float]
    verifier_config: dict[str, Any] = field(default_factory=dict)
    sampling_seed: int | None = None
    validator_version: str = ""
    attribution_cache_path: str | None = None
    harness_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "harness_version": self.harness_version,
            "split_id": self.split_id,
            "taxonomy_version": self.taxonomy_version,
            "prompt_version": self.prompt_version,
            "digest_version": self.digest_version,
            "prompt_sha256": self.prompt_sha256,
            "attributor_model": self.attributor_model,
            "attributor_sampling_args": dict(self.attributor_sampling_args),
            "digest_char_budget": self.digest_char_budget,
            "digest_focus_k": self.digest_focus_k,
            "max_attempts": self.max_attempts,
            "sub_verifier_enabled": self.sub_verifier_enabled,
            "min_support": self.min_support,
            "actionability_weights": dict(self.actionability_weights),
            "verifier_config": dict(self.verifier_config),
            "sampling_seed": self.sampling_seed,
            "validator_version": self.validator_version,
            "attribution_cache_path": self.attribution_cache_path,
            "harness_hash": self.harness_hash,
        }


@dataclass
class EvidenceBundle:
    """
    B_t. The output of weakness mining and the input to harness proposal.

    Per the paper, the bundle does not prescribe a harness edit: it separates
    verifier-level failure from agent-level mechanism so the proposer can
    target a reusable weakness rather than patch a coarse outcome. That
    property is enforced structurally (no field can hold an edit) and by the
    lint in bundle.py.
    """

    bundle_id: str
    created_at: str
    config: MiningConfig
    totals: MiningTotals
    patterns: list[FailurePattern]
    marginals: dict[str, dict[str, int]]
    integrity: IntegrityReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "config": self.config.to_dict(),
            "totals": self.totals.to_dict(),
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "marginals": {k: dict(v) for k, v in self.marginals.items()},
            "integrity": self.integrity.to_dict(),
        }


def iter_nodes(root: CallNode) -> Iterator[CallNode]:
    """Depth-first walk over the tree in trajectory order, root first."""
    yield root
    for child in root.children:
        yield from iter_nodes(child)
