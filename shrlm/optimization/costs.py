"""The cost governor: experiment-owned caps and the per-candidate circuit breaker.

Cost governance is experiment-owned (KTD3), and deliberately not delegated to
the harness's own S6 runtime policy: that policy is an editable surface, so a
candidate edit could lift its own limits. The caps here bind where no surface
can reach them -- at the RLM constructor, through the round config -- and a
candidate's enabled S6 values may *tighten* but never exceed them (R3).

Three pieces, in the order the validation driver (U3) uses them:

1. ``ValidationCaps`` -- the experiment's limits: ``max_depth``,
   ``max_iterations``, per-run ``max_budget`` (USD), per-run ``max_timeout``
   (seconds), and the per-candidate cumulative ``candidate_budget`` (USD).
2. ``governed_limits`` -- the tighten-only merge. It validates a candidate's
   S6 policy against the caps with U1's ``cap_violations`` comparison (a
   violation comes back as a ``CandidateRejection`` at the ``caps`` gate, a
   value, never an exception) and returns the RLM-constructor kwargs for
   ``RoundConfig``. The forwarding rule honors the runner's S6-ownership
   guard: when an enabled policy declares ``max_depth``, the (validated,
   therefore tighter-or-equal) policy value binds via the runner and the
   constructor value is omitted -- forwarding both would be the runner's
   double-declaration ValueError; when the policy is silent, the cap forwards.
3. ``CandidateSpendBreaker`` and ``run_governed_round`` -- the circuit breaker
   (R4). Driver-level accounting over *persisted* run costs: the governed
   round executes ``run_round`` one run at a time and charges each newly
   persisted manifest line to the breaker; once cumulative spend crosses the
   candidate budget, the remaining runs are skipped. Completed runs stand on
   disk untouched (persist-first, KTD5), the resume path prices them again
   from the manifest alone, and the candidate comes back marked
   ``over_budget`` -- rejected, never silently dropped.

Pricing one persisted run (``breaker_run_cost``) has one policy worth pinning:
a budget-terminated run carries the limit exception's actual ``spent`` figure
as its persisted cost (see ``driver._partial_completion``) and is charged
verbatim, even above the per-run cap; a termination that genuinely persisted no
cost (a timeout on a cost-less backend) is charged at the per-run budget
ceiling -- the worst a run is allowed to spend, never zero, so a candidate
cannot burn wall-clock for free; and a *non*-terminated run with no cost fails
loudly, because a spend breaker fed cost-less runs is a run counter wearing a
costume (the same stance ``runner.acceptance_inputs`` takes).
"""

from dataclasses import dataclass, field
from typing import Any

from shrlm.optimization.bundle import round_dir
from shrlm.optimization.candidates import GATE_CAPS, CandidateRejection, cap_violations
from shrlm.optimization.driver import RoundConfig, run_id_for, run_round
from shrlm.optimization.taxonomy import VerifierCause

# Candidate outcomes the promotion ledger records for a governed round.
OUTCOME_COMPLETED = "completed"
OUTCOME_OVER_BUDGET = "over_budget"


@dataclass(frozen=True)
class ValidationCaps:
    """The experiment-owned evaluation limits (R3; KTD3).

    ``max_budget`` and ``max_timeout`` are per run; ``candidate_budget`` is the
    cumulative circuit-breaker budget across all of one candidate's runs. All
    five are mandatory: a validation stage without a binding cap is exactly the
    unbounded spend KTD3 exists to prevent.
    """

    max_depth: int
    max_iterations: int
    max_budget: float
    max_timeout: float
    candidate_budget: float

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("max_budget", "max_timeout", "candidate_budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")

    def s6_caps(self) -> dict[str, int | float]:
        """The caps an S6 policy can express, as ``cap_violations`` input.

        The runner already forbids S6 from declaring budget/timeout caps
        (``EXPERIMENT_OWNED_KEYS``), so ``max_depth`` is the entire overlap
        between the policy surface and these caps.
        """
        return {"max_depth": self.max_depth}


def governed_limits(
    candidate_id: str, policy: dict[str, Any], caps: ValidationCaps
) -> dict[str, Any] | CandidateRejection:
    """The tighten-only merge: RLM-constructor kwargs for one harness's round.

    Args:
        candidate_id: Who to name in a rejection (the incumbent baseline works
            too; its disabled policy always passes).
        policy: The harness's S6 runtime policy dict.
        caps: The experiment-owned limits.

    Returns:
        ``RoundConfig`` limit kwargs -- ``max_iterations``, ``max_budget``,
        ``max_timeout``, plus ``max_depth`` unless an enabled policy declares
        its own (then the runner binds the policy's value and the constructor
        value must stay unset) -- or a ``CandidateRejection`` at the ``caps``
        gate when an enabled policy value exceeds its cap.
    """
    violations = cap_violations(policy, caps.s6_caps())
    if violations:
        return CandidateRejection(
            candidate_id=candidate_id, gate=GATE_CAPS, reason="; ".join(violations)
        )
    limits: dict[str, Any] = {
        "max_iterations": caps.max_iterations,
        "max_budget": caps.max_budget,
        "max_timeout": caps.max_timeout,
    }
    if not (policy.get("enabled") and policy.get("max_depth") is not None):
        limits["max_depth"] = caps.max_depth
    return limits


def breaker_run_cost(entry: dict[str, Any], caps: ValidationCaps) -> float:
    """What one persisted manifest line charges the breaker (see module docstring).

    Raises:
        ValueError: If a non-terminated run carries no cost -- the backend is
            not reporting spend, and a blind breaker must refuse rather than
            undercount.
    """
    cost = entry.get("cost")
    if cost is not None:
        return float(cost)
    if entry.get("cause") == VerifierCause.RESOURCE_TERMINATED.value:
        # A cost-less termination (e.g. timeout on a backend without cost
        # tracking) is priced at the worst a run may spend, never zero.
        return caps.max_budget
    raise ValueError(
        f"run {entry.get('run_id')!r} persisted no cost and was not resource-terminated; "
        "the spend breaker needs a cost-reporting backend (or stubbed costs in tests) "
        "and refuses to count a paid run as free."
    )


@dataclass
class CandidateSpendBreaker:
    """Cumulative spend accounting for one candidate, across its rounds (R4).

    Charge every persisted manifest line (``charge`` is idempotent per
    ``(namespace, run_id)``, so resumes and re-reads never double-bill; the
    namespace -- the round directory in ``run_governed_round`` -- keeps
    identical run ids from different splits distinct). ``tripped`` is strictly
    greater-than: spending exactly the budget is within it, matching the
    runtime's own ``max_budget`` semantics.
    """

    caps: ValidationCaps
    spent: float = 0.0
    _charged: set[str] = field(default_factory=set, repr=False)

    def charge(self, entry: dict[str, Any], *, namespace: str = "") -> float:
        """Add one persisted run's cost; returns the amount newly charged."""
        key = f"{namespace}::{entry['run_id']}"
        if key in self._charged:
            return 0.0
        cost = breaker_run_cost(entry, self.caps)
        self._charged.add(key)
        self.spent += cost
        return cost

    @property
    def tripped(self) -> bool:
        return self.spent > self.caps.candidate_budget


@dataclass(frozen=True)
class GovernedRoundResult:
    """One governed round's outcome: what ran, what it cost, what was refused.

    ``entries`` are the round's manifest lines (pre-existing and new, file
    order); ``spent`` is the breaker's cumulative figure *including* runs
    charged before this round; ``skipped_run_ids`` are the configured runs the
    breaker refused, in the execution order they would have had.
    """

    entries: list[dict[str, Any]]
    outcome: str
    spent: float
    skipped_run_ids: list[str]

    @property
    def over_budget(self) -> bool:
        return self.outcome == OUTCOME_OVER_BUDGET


def run_governed_round(config: RoundConfig, breaker: CandidateSpendBreaker) -> GovernedRoundResult:
    """Run one round under the breaker: persist-first, one run at a time.

    Pre-existing manifest lines are charged before anything executes, so a
    resumed round that is already over budget makes zero model calls and a
    crash between runs loses nothing (each ``run_round`` slice re-verifies the
    persisted state it resumes over). Every newly persisted run is charged the
    moment its manifest line lands; once the breaker trips, the remaining runs
    are skipped and reported, never silently dropped.

    Args:
        config: The round to run, its limits already merged by
            ``governed_limits``.
        breaker: The candidate's breaker, shared across that candidate's
            rounds (splits, repetitions) so the budget is truly cumulative.

    Returns:
        The manifest entries, the ``completed``/``over_budget`` outcome, the
        breaker's cumulative spend, and the run ids the breaker skipped.
    """
    namespace = str(round_dir(config.out_dir, config.round_index))
    entries = run_round(config, stop_after=0)
    for entry in entries:
        breaker.charge(entry, namespace=namespace)

    while not breaker.tripped:
        known = len(entries)
        entries = run_round(config, stop_after=1)
        if len(entries) == known:
            break  # nothing pending: the round is complete
        for entry in entries[known:]:
            breaker.charge(entry, namespace=namespace)

    done = {str(entry["run_id"]) for entry in entries}
    skipped = [
        run_id
        for instance in config.instances
        for attempt in range(1, config.attempts + 1)
        if (run_id := run_id_for(str(instance["id"]), attempt)) not in done
    ]
    return GovernedRoundResult(
        entries=entries,
        outcome=OUTCOME_OVER_BUDGET if breaker.tripped else OUTCOME_COMPLETED,
        spent=breaker.spent,
        skipped_run_ids=skipped,
    )


__all__ = [
    "OUTCOME_COMPLETED",
    "OUTCOME_OVER_BUDGET",
    "CandidateSpendBreaker",
    "GovernedRoundResult",
    "ValidationCaps",
    "breaker_run_cost",
    "governed_limits",
    "run_governed_round",
]
