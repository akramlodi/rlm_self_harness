"""
Derive the failing level from checkable per-sub-call outcomes.

This module is the sub-verification ablation switch. With a SubVerifier the
failing level is a checkable fact; without one it is a language-model judgment,
and the record says which. Nothing else in the pipeline may vary with that
choice, or the ablation would be comparing two things at once.

The distinction being established is the one the project rests on: a root
failure means correct sub-results were aggregated into a wrong answer, while a
child failure means a sub-call returned a wrong local result that the root
faithfully combined. The two implicate different editable surfaces.
"""

from dataclasses import dataclass, field
from typing import Any

from shrlm.optimization.taxonomy import FailingLevel
from shrlm.optimization.types import CallNode, SubVerifier, iter_nodes


@dataclass
class GroundingResult:
    """
    Outcome of the sub-verification pass.

    ``verdicts`` maps node id to True, False, or None for "not checkable in
    isolation". ``failing_level`` is None when no sub-verifier ran, in which
    case the attributor is asked for it instead. ``grounded`` is True only
    when the level is a checkable fact: a sub-verifier ran and either the tree
    had no descendants (NO_RECURSION) or at least one verdict was non-None.
    An all-None pass leaves ``failing_level`` UNDETERMINED with ``grounded``
    False, and the attributor is asked for the level there too.
    """

    failing_level: FailingLevel | None
    grounded: bool
    verdicts: dict[str, bool | None] = field(default_factory=dict)


def derive_failing_level(root: CallNode, verdicts: dict[str, bool | None]) -> FailingLevel:
    """
    Locate where the error signal first appears.

    "First appears" makes the mixed case deterministic: any wrong sub-call
    means the signal appeared at the child, whether or not the root's
    aggregation was independently faulty. That is why there is no BOTH member,
    and why this function is total over sub-verdicts alone.
    """
    descendants = [node for node in iter_nodes(root) if node.node_id != root.node_id]
    if not descendants:
        return FailingLevel.NO_RECURSION

    checkable = [verdict for verdict in verdicts.values() if verdict is not None]
    if not checkable:
        return FailingLevel.UNDETERMINED
    if False in checkable:
        return FailingLevel.CHILD
    return FailingLevel.ROOT


def apply_sub_verifier(
    instance: dict[str, Any], root: CallNode, sub_verifier: SubVerifier | None
) -> GroundingResult:
    """
    Score every sub-call and derive the failing level, when a sub-verifier exists.

    Verdicts are written onto the nodes as well as returned, so the digest can
    show the attributor which sub-calls were independently wrong without
    threading a second structure through it.
    """
    if sub_verifier is None:
        return GroundingResult(failing_level=None, grounded=False, verdicts={})

    verdicts: dict[str, bool | None] = {}
    for node in iter_nodes(root):
        if node.node_id == root.node_id:
            continue
        verdict = sub_verifier(instance, node)
        node.sub_verdict = verdict
        verdicts[node.node_id] = verdict

    failing_level = derive_failing_level(root, verdicts)

    # "Ran" must not count as "checked": see the GroundingResult docstring.
    return GroundingResult(
        failing_level=failing_level,
        grounded=failing_level is not FailingLevel.UNDETERMINED,
        verdicts=verdicts,
    )
