"""
Group failed runs by exact signature agreement and order the resulting patterns.

Two runs are clustered together only when they agree on what the verifier
rejected, where the error signal first appeared, how the agent behavior
contributed, and which reusable mechanism was involved. The goal is not latent
semantic similarity between traces; it is aggregating failures that plausibly
admit the same harness-level intervention.

Cluster-level evidence is computed from mechanical tree statistics rather than
by re-querying a model, so a bundle stays a deterministic function of its
records.
"""

from collections import Counter
from dataclasses import dataclass

from shrlm.optimization.taxonomy import CAUSAL_WEIGHT, AgentMechanism
from shrlm.optimization.types import FailurePattern, FailureRecord

# Patterns with support below this are flagged, never dropped. A mechanism seen
# once may still be real, and silently discarding it would misrepresent the
# round as having found nothing there.
DEFAULT_MIN_SUPPORT = 2

DEFAULT_REPRESENTATIVES = 3

# Weights of the actionability terms. Recorded in the bundle config so a change
# is visible in the artifact rather than only in the code.
ACTIONABILITY_WEIGHTS: dict[str, float] = {
    "causal": 0.4,
    "grounded": 0.3,
    "surface": 0.2,
    "homogeneity": 0.1,
}


@dataclass(frozen=True)
class ClusteringConfig:
    min_support: int = DEFAULT_MIN_SUPPORT
    n_representatives: int = DEFAULT_REPRESENTATIVES


def attributed(records: list[FailureRecord]) -> list[FailureRecord]:
    """Records that carry a signature. Unattributed ones stay in the totals."""
    return [
        record
        for record in records
        if record.signature is not None and not record.attribution_failed
    ]


def grounded_fraction(records: list[FailureRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if record.level_grounded) / len(records)


def homogeneity(records: list[FailureRecord]) -> float:
    """
    How consistently the cluster's members describe the same thing.

    Members share a signature by construction, so divergent free-text detail is
    a signal that the enum member is being stretched across distinct behaviors.
    """
    if not records:
        return 0.0
    details = {
        (record.detail.agent_mechanism_detail.strip().lower() if record.detail else "")
        for record in records
    }
    return 1.0 - (len(details) - 1) / len(records)


def actionability(records: list[FailureRecord]) -> float:
    """
    How likely this cluster is to map onto a single harness surface.

    Orders the proposer's reading order only. It is not a prediction of edit
    success, and no promotion decision may consume it.
    """
    if not records:
        return 0.0

    signature = records[0].signature
    assert signature is not None  # attributed() filtered these

    causal = sum(CAUSAL_WEIGHT[record.signature.causal_status] for record in records) / len(records)
    surface = 0.0 if signature.agent_mechanism is AgentMechanism.OTHER else 1.0

    return (
        ACTIONABILITY_WEIGHTS["causal"] * causal
        + ACTIONABILITY_WEIGHTS["grounded"] * grounded_fraction(records)
        + ACTIONABILITY_WEIGHTS["surface"] * surface
        + ACTIONABILITY_WEIGHTS["homogeneity"] * homogeneity(records)
    )


def shared_symptoms(records: list[FailureRecord]) -> list[str]:
    """
    Mechanical statements about what the cluster's runs have in common.

    Every line here is arithmetic over tree statistics. Nothing consults a
    model, so the bundle is reproducible from the records alone.
    """
    n = len(records)
    stats = [record.stats for record in records]

    collapsed = sum(1 for s in stats if s.collapse_ratio > 0.8)
    fallback = sum(1 for s in stats if s.terminated_by_fallback)
    degraded = sum(1 for s in stats if s.suspected_lost_subcalls > 0)
    no_children = sum(1 for s in stats if s.n_nodes <= 1)
    errored = sum(1 for s in stats if s.n_errored > 0)

    median_children = sorted(s.n_nodes - 1 for s in stats)[n // 2]
    median_iterations = sorted(s.n_iterations for s in stats)[n // 2]
    median_depth = sorted(s.max_observed_depth for s in stats)[n // 2]

    lines = [
        f"median sub-calls per run: {median_children}",
        f"median iterations: {median_iterations}",
        f"median observed depth: {median_depth}",
    ]
    if collapsed:
        lines.append(f"largest sub-call took >80% of the root context in {collapsed}/{n} runs")
    if no_children:
        lines.append(f"no sub-calls issued at all in {no_children}/{n} runs")
    if fallback:
        lines.append(f"answer synthesized by the iteration fallback in {fallback}/{n} runs")
    if errored:
        lines.append(f"at least one sub-call errored in {errored}/{n} runs")
    if degraded:
        lines.append(
            f"trace known to be missing sub-calls in {degraded}/{n} runs "
            "(lower bound; child-level attribution is understated here)"
        )
    return lines


def verifier_evidence(records: list[FailureRecord], limit: int = 3) -> list[str]:
    """Concrete gold-versus-produced pairs, so the cause is inspectable."""
    lines = []
    for record in sorted(records, key=lambda r: r.instance_id)[:limit]:
        lines.append(
            f"{record.instance_id}: expected {record.verdict.gold!r}, "
            f"produced {record.verdict.produced!r}"
        )
    return lines


def select_representatives(records: list[FailureRecord], count: int) -> list[str]:
    """
    A fixed selection rule, so ordering is stable under input permutation.

    Grounded records first, since their failing level is a checkable fact
    rather than a judgment.
    """
    ranked = sorted(
        records,
        key=lambda r: (not r.level_grounded, len(r.digest_sha256), r.instance_id),
    )
    return [record.instance_id for record in ranked[:count]]


def cluster_failures(
    records: list[FailureRecord], config: ClusteringConfig | None = None
) -> list[FailurePattern]:
    """Group by exact signature agreement, then order by support and actionability."""
    config = config or ClusteringConfig()

    groups: dict[tuple[str, str, str, str], list[FailureRecord]] = {}
    for record in attributed(records):
        assert record.signature is not None
        groups.setdefault(record.signature.key(), []).append(record)

    patterns = []
    for members in groups.values():
        signature = members[0].signature
        assert signature is not None
        patterns.append(
            FailurePattern(
                signature=signature,
                support=len(members),
                instance_ids=sorted(record.instance_id for record in members),
                representatives=select_representatives(members, config.n_representatives),
                shared_symptoms=shared_symptoms(members),
                verifier_evidence=verifier_evidence(members),
                grounded_fraction=grounded_fraction(members),
                surface=signature.surface(),
                actionability=actionability(members),
                below_support_floor=len(members) < config.min_support,
            )
        )

    return rank_patterns(patterns)


def rank_patterns(patterns: list[FailurePattern]) -> list[FailurePattern]:
    """
    Order by support, then actionability, then signature.

    Lexicographic rather than a weighted blend of the two: the paper orders by
    support *and* estimated actionability, and inventing an exchange rate
    between frequency and tractability would be a modeling choice with no
    grounding in the evidence.
    """
    return sorted(
        patterns,
        key=lambda p: (-p.support, -p.actionability, p.signature.key()),
    )


def compute_marginals(records: list[FailureRecord]) -> dict[str, dict[str, int]]:
    """
    Per-component counts across all attributed records.

    The full four-tuple space is large relative to a round's failure count, so
    most clusters will be small. These marginals are the principled backoff:
    the by-surface view in particular has nine buckets and is directly
    consumable by a proposal stage that must target one surface anyway.
    """
    usable = attributed(records)

    causes: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    mechanisms: Counter[str] = Counter()
    surfaces: Counter[str] = Counter()

    for record in usable:
        signature = record.signature
        assert signature is not None
        causes[signature.verifier_cause.value] += 1
        levels[signature.failing_level.value] += 1
        statuses[signature.causal_status.value] += 1
        mechanisms[signature.agent_mechanism.value] += 1
        surface = signature.surface()
        surfaces[surface.value if surface else "unmapped"] += 1

    return {
        "by_cause": dict(sorted(causes.items())),
        "by_level": dict(sorted(levels.items())),
        "by_causal_status": dict(sorted(statuses.items())),
        "by_mechanism": dict(sorted(mechanisms.items())),
        "by_surface": dict(sorted(surfaces.items())),
    }
