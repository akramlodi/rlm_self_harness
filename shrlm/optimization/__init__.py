"""
Self-Harness weakness mining for recursive language models.

Converts verifier-grounded execution traces into an evidence bundle of
recurring, actionable failure patterns. This is the first of the three
Self-Harness stages; it deliberately stops short of proposing harness edits.
"""

from shrlm.optimization.taxonomy import (
    TAXONOMY_VERSION,
    AgentMechanism,
    CausalStatus,
    EditableSurface,
    FailingLevel,
    VerifierCause,
)
from shrlm.optimization.types import (
    CallNode,
    EvidenceBundle,
    FailurePattern,
    FailureRecord,
    FailureSignature,
    NodeKind,
    SubVerifier,
    Verdict,
    Verifier,
)
from shrlm.optimization.walker import build_call_tree, walk

__all__ = [
    "TAXONOMY_VERSION",
    "AgentMechanism",
    "CallNode",
    "CausalStatus",
    "EditableSurface",
    "EvidenceBundle",
    "FailingLevel",
    "FailurePattern",
    "FailureRecord",
    "FailureSignature",
    "NodeKind",
    "SubVerifier",
    "Verdict",
    "Verifier",
    "VerifierCause",
    "build_call_tree",
    "walk",
]
