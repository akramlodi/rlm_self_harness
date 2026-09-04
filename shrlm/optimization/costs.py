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

One more experiment-owned containment, for the same KTD3 reason the caps are:
the *hard wall-clock backstop*. Candidate callables (the S7/S8/S9 surfaces)
run synchronously inside the host loop, and the runtime's own ``max_timeout``
is only checked between iterations -- so a candidate call that never returns
(``while True`` in a middleware, a hang in harness construction) would block
the host forever: nothing persisted, nothing charged, ``validate_round`` never
returning. The harness cannot be trusted to bound itself (its runtime policy
is an editable surface), so ``run_governed_round`` arms a SIGALRM timer around
each single-run slice at ``hard_deadline_seconds`` -- the per-run
``max_timeout`` times ``HARD_DEADLINE_FACTOR`` plus
``HARD_DEADLINE_GRACE_SECONDS``. The factor leaves room for a final iteration
that started just under the limit (the runtime's between-iteration check
remains the primary, precise mechanism); the fixed grace absorbs the
per-slice overhead outside the RLM loop (harness build, verification,
persistence). When the alarm fires, ``HardDeadlineExceeded`` is raised at the
hung bytecode; as a ``TimeoutExceededError`` subclass it flows through the
driver's per-run limit handler and persists as an ordinary
RESOURCE_TERMINATED run (persist-first, KTD5), which the breaker then charges
at the per-run ceiling. If the interrupt escapes ``run_round`` entirely (a
hang outside the per-run try, e.g. in harness construction), the governed
round synthesizes and persists the in-flight run's terminated manifest line
itself (``driver.persist_interrupted_run``) before continuing. The backstop
is skipped -- documented, not silently -- where SIGALRM cannot bind (no
SIGALRM on the platform, or not the main thread); it closes the in-host hang
hole at the v1 level, and a full subprocess-per-run boundary is the v2
escalation. A candidate that swallows ``BaseException`` inside its hang can
still defeat a signal-raised exception -- that adversarial case is also v2's.

Pricing one persisted run (``breaker_run_cost``) has one policy worth pinning:
a terminated run carries whatever the runtime recorded for it (see
``driver._partial_completion``, which persists the total the completion context
published) and is charged verbatim, even above the per-run cap;
``BudgetExceededError.spent`` remains the fallback when nothing was published.
Only a termination that recorded no calls at all -- a request that hung before
it was counted, or a run that never reached the runtime -- is charged at the
per-run budget ceiling: the worst a run is allowed to spend, never zero, so a candidate
cannot burn wall-clock for free; and a *non*-terminated run with no cost fails
loudly, because a spend breaker fed cost-less runs is a run counter wearing a
costume (the same stance ``runner.acceptance_inputs`` takes).
"""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from rlm.utils.exceptions import HardDeadlineExceeded, HardDeadlineSignal, TimeoutExceededError
from shrlm.harness_identity import harness_hash, serialize_harness
from shrlm.optimization.bundle import round_dir
from shrlm.optimization.candidates import (
    GATE_CAPS,
    CandidateRejection,
    cap_violations,
    write_surface_module,
)
from shrlm.optimization.driver import (
    RoundConfig,
    append_child_run,
    load_manifest,
    persist_interrupted_run,
    prepare_round,
    read_child_trace,
    require_backend_credential,
    run_id_for,
    run_round,
    trace_path_for,
)
from shrlm.optimization.run_worker import LOG_FILENAME as RUN_LOG_FILENAME
from shrlm.optimization.run_worker import REQUEST_FILENAME as RUN_REQUEST_FILENAME
from shrlm.optimization.run_worker import RESULT_FILENAME as RUN_RESULT_FILENAME
from shrlm.optimization.run_worker import RunWorkerError
from shrlm.optimization.run_worker import build_request as build_run_request
from shrlm.optimization.run_worker import read_result as read_run_result
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict

# Candidate outcomes the promotion ledger records for a governed round.
OUTCOME_COMPLETED = "completed"
OUTCOME_OVER_BUDGET = "over_budget"

# Where a run child's own artifacts live: its request, its log, its result.
# Everything shared within the split stays outside this directory and is
# written by the parent alone.
RUN_REQUESTS_DIRNAME = "run_workers"
# The live run child's pid, written by the parent after spawn and removed on
# reap. It is what lets a replacement parent tell "this run was abandoned" from
# "this run is still being paid for by someone else's child".
RUN_PID_FILENAME = "worker.pid"

# How long the parent gives a run child to exit after SIGTERM before SIGKILL,
# so an interrupt never hangs on a child that ignores the polite signal.
TERMINATE_GRACE_SECONDS = 10.0

# How often the parent polls its run children. Runs take seconds to minutes,
# so a coarse poll costs nothing measurable. ``_sleep`` is the poll's own seam
# so a test can drive the loop without touching the stdlib sleep that
# ``Popen.wait(timeout=...)`` relies on during cleanup.
POLL_SECONDS = 0.2
_sleep = time.sleep

# The hard wall-clock deadline for one run slice, derived from the per-run
# ``max_timeout`` cap: ``max_timeout * HARD_DEADLINE_FACTOR +
# HARD_DEADLINE_GRACE_SECONDS`` (see the module docstring for the rationale).
HARD_DEADLINE_FACTOR = 1.5
HARD_DEADLINE_GRACE_SECONDS = 30.0

_T = TypeVar("_T")


# ``HardDeadlineExceeded`` now lives beside ``HardDeadlineSignal`` in
# ``rlm.utils.exceptions`` (the driver raises it too); re-exported here so
# ``costs.HardDeadlineExceeded`` keeps working for existing imports.


def hard_deadline_seconds(max_timeout: float | None) -> float | None:
    """The backstop deadline for one run slice; ``None`` disables the backstop.

    ``None`` in means ``None`` out: governed rounds always carry a
    ``max_timeout`` (``ValidationCaps`` makes it mandatory), so an unbounded
    slice only occurs for a hand-built config that opted out of the cap.
    """
    if max_timeout is None:
        return None
    return max_timeout * HARD_DEADLINE_FACTOR + HARD_DEADLINE_GRACE_SECONDS


def _alarm_available() -> bool:
    """Whether SIGALRM can bind here: POSIX only, main thread only."""
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


def _call_with_hard_deadline(fn: Callable[[], _T], deadline: float | None) -> _T:
    """Run ``fn`` under a SIGALRM hard deadline, restoring alarm state after.

    With no deadline, or where SIGALRM cannot bind (``_alarm_available``), the
    call runs unguarded -- the documented no-backstop mode; the runtime's own
    between-iteration timeout remains in force either way.
    """
    if deadline is None or not _alarm_available():
        return fn()

    def _on_alarm(signum: int, frame: Any) -> None:
        # A BaseException so the REPL's and sub-call wrappers' ``except
        # Exception`` cannot swallow it; ``execute_run`` persists it as a
        # terminated run, and anything that escapes ``run_round`` is converted
        # below into the ``HardDeadlineExceeded`` the slice handler expects.
        raise HardDeadlineSignal(deadline)

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, deadline)
    try:
        return fn()
    except HardDeadlineSignal as signal_:
        raise HardDeadlineExceeded(signal_.deadline) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


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


def _run_slice(
    config: RoundConfig, *, stop_after: int, deadline: float | None, known: int
) -> list[dict[str, Any]]:
    """One ``run_round`` slice under the hard-deadline backstop.

    The common hang -- inside a candidate call during ``harnessed.completion``
    -- never reaches this function's handler: the driver's per-run limit
    handler catches ``HardDeadlineExceeded`` (a ``TimeoutExceededError``) and
    persists the terminated run itself, so the slice returns normally. What is
    handled here is the interrupt that *escaped* ``run_round`` -- a hang in
    harness construction, or an alarm landing between a run's completion and
    its manifest append. ``known`` (the manifest lines already accounted for)
    decides recovery: if nothing new persisted, the in-flight run's terminated
    line is synthesized (``persist_interrupted_run``); if the run did persist
    before the alarm landed, the reloaded manifest is simply returned.
    """
    try:
        return _call_with_hard_deadline(lambda: run_round(config, stop_after=stop_after), deadline)
    except HardDeadlineExceeded as error:
        persisted = load_manifest(config.out_dir, config.round_index)
        if len(persisted) == known:
            entry = persist_interrupted_run(config, error)
            if entry is not None:
                persisted.append(entry)
        return persisted


# The claim a governed round holds on its split for as long as it is dispatching
# into it. A directory rather than a file: exclusive creation is the atomic
# primitive here, and the pid inside names the holder for the refusal message.
CLAIM_DIRNAME = ".claim"
CLAIM_PID_FILENAME = "pid"
# How long to keep re-examining a claim that resolves to neither a live owner
# nor an evictable one. Only a process killed between staging and rename can
# produce that state, and the next loop turn sees the finished claim, so this
# bound exists to fail loudly rather than spin.
CLAIM_ACQUIRE_SECONDS = 10.0


class SplitClaimedError(RuntimeError):
    """Another live process is already dispatching runs into this split."""


def _live_run_children(round_path: Path) -> list[int]:
    """Pids of run children from a previous parent that are still alive.

    A child notices its parent died only on its next watchdog poll, so a
    replacement parent can win the split claim while the old children are still
    executing -- and re-dispatch the very runs they are still paying for. Each
    child records its pid beside its request; a pid still alive here means that
    run is in flight no matter which parent started it.
    """
    requests_dir = round_path / RUN_REQUESTS_DIRNAME
    if not requests_dir.exists():
        return []
    alive: list[int] = []
    for pid_path in requests_dir.glob(f"*/{RUN_PID_FILENAME}"):
        try:
            pid = int(pid_path.read_text().strip())
        except (OSError, ValueError):
            continue
        if pid != os.getpid() and _pid_alive(pid):
            alive.append(pid)
    return alive


def _claim_holder(claim_dir: Path) -> int | None:
    """The pid recorded inside a claim, or None when it records no readable one."""
    try:
        return int((claim_dir / CLAIM_PID_FILENAME).read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def claim_split(path: Path) -> Iterator[None]:
    """Hold this split exclusively for the duration of the block.

    Two invocations dispatching into one split would interleave manifest
    appends and double-charge runs, and neither would know the other existed.
    The guard is not gated on the worker count: a second sequential invocation
    of the same split is exactly as damaging as a concurrent one, so the claim
    is taken on every governed round. Same-pid is refused too -- nothing here
    is legitimately re-entrant.

    Ownership changes hands atomically. The claim is staged as a private
    directory that already contains its owner's pid and then moved into place
    with a single rename, so a claim is never observable without an owner: a
    reader that finds the claim directory always finds a pid inside it. Only
    that rename decides who holds the split, which is what makes the three
    races here unwinnable -- a holder descheduled between creating the
    directory and naming itself, two processes both reclaiming one dead
    holder, and a reader mistaking a half-built claim for an abandoned one.

    A claim whose owner is gone is evicted rather than honoured, since a
    crashed round must not lock its split forever; eviction is itself a rename,
    so only one of several racing processes can perform it and the losers
    re-examine what the winner left behind.
    """
    claim_dir = path / CLAIM_DIRNAME
    staging = path / f"{CLAIM_DIRNAME}.staging-{os.getpid()}"
    deadline = time.monotonic() + CLAIM_ACQUIRE_SECONDS
    while True:
        staging.mkdir(parents=True, exist_ok=True)
        (staging / CLAIM_PID_FILENAME).write_text(f"{os.getpid()}\n")
        try:
            # Atomic: fails when the claim already exists, and publishes a
            # claim that already names its owner when it succeeds.
            os.rename(staging, claim_dir)
            break
        except OSError:
            pass

        holder = _claim_holder(claim_dir)
        if holder is not None and _pid_alive(holder):
            shutil.rmtree(staging, ignore_errors=True)
            raise SplitClaimedError(
                f"{path} is already claimed by live process {holder}; refusing to "
                "dispatch a second round into one split."
            ) from None

        # The owner is gone, or the claim has not finished being published.
        # Evict by rename so exactly one racer wins the right to clear it; a
        # loser simply loops and re-reads whatever the winner left.
        if time.monotonic() >= deadline:
            shutil.rmtree(staging, ignore_errors=True)
            raise SplitClaimedError(
                f"{path} holds a claim that never resolved to a live or dead owner "
                f"within {CLAIM_ACQUIRE_SECONDS:.0f}s; refusing to dispatch. Remove "
                f"{claim_dir} by hand once no round is running against this split."
            )
        evicted = path / f"{CLAIM_DIRNAME}.evicted-{os.getpid()}"
        try:
            os.rename(claim_dir, evicted)
        except OSError:
            continue  # another racer got there first; re-examine
        shutil.rmtree(evicted, ignore_errors=True)

    try:
        yield
    finally:
        # Only tear down a claim this process still owns. A claim evicted as
        # stale and retaken by someone else must not be deleted from under
        # them, which is what an unconditional release would do.
        if _claim_holder(claim_dir) == os.getpid():
            shutil.rmtree(claim_dir, ignore_errors=True)


# What one in-flight run may charge beyond the per-run cap before it is reaped.
# A run is not capped at ``max_budget``: the runtime checks the budget only
# between iterations, and a budget termination is charged its exception's figure
# verbatim. The live data has a run charged $0.866 against a $0.50 cap, so
# reserving the cap alone would under-reserve by 73%. Dispatch reserves this
# multiple of the cap per in-flight run and the realised worst case is
# ``in-flight x (worst per-run charge - reservation)``, not zero (KTD9).
RUN_RESERVATION_FACTOR = 2.0


def _run_reservation(caps: ValidationCaps) -> float:
    """What dispatch sets aside for one run it has not yet reaped."""
    return caps.max_budget * RUN_RESERVATION_FACTOR


# Mirror of ``rlm.clients.openai._CONTENT_FILTER_MARKERS``. Duplicated rather
# than imported: this module is imported by every run child on spawn, and
# pulling the OpenAI SDK into that path costs ~0.2s per child.
_CONTENT_FILTER_MARKERS = ("content_filter", "responsibleai", "content management policy")


def _error_verdict(detail: str, produced: str = "") -> Verdict:
    """The failing verdict for a child trace that carries an ``error``.

    A child runs ``execute_run`` and records a provider content-filter refusal
    as a CONTENT_FILTERED verdict, but only the trace (not the verdict) crosses
    the process boundary, and the parent used to relabel every ``error`` as
    RESOURCE_TERMINATED. The error string is the refusal text itself, so the
    label is recovered from it here.
    """
    lowered = detail.lower()
    if any(marker in lowered for marker in _CONTENT_FILTER_MARKERS):
        return Verdict(
            passed=False,
            cause=VerifierCause.CONTENT_FILTERED,
            gold="",
            produced=produced,
            detail=detail,
        )
    return _terminated_verdict(detail, produced)


def _terminated_verdict(detail: str, produced: str = "") -> Verdict:
    """The failing verdict a terminated run carries.

    ``produced`` is the partial answer the run had reached, which the
    sequential path records verbatim. Dropping it here would make the two paths
    persist different verdicts for the same termination.
    """
    return Verdict(
        passed=False,
        cause=VerifierCause.RESOURCE_TERMINATED,
        gold="",
        produced=produced,
        detail=detail,
    )


def _adopt_orphan_traces(
    path: Path,
    pending: list[tuple[dict[str, Any], int]],
    config: RoundConfig,
    breaker: CandidateSpendBreaker,
    namespace: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Record runs whose traces exist but whose manifest lines never landed.

    A parent killed between a child publishing its trace and the parent
    appending its line leaves exactly this. The run was made and the money was
    spent, so re-dispatching it would pay twice; the trace is complete, so it
    can be verified and recorded now (R15).
    """
    adopted: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for instance, attempt in pending:
        instance_id = str(instance["id"])
        run_id = run_id_for(instance_id, attempt)
        completion = read_child_trace(trace_path_for(path, run_id))
        if completion is None:
            continue
        verdict = (
            _error_verdict(str(completion.error), completion.response)
            if completion.error
            else config.verifier(instance, completion.response)
        )
        entry = append_child_run(
            path,
            run_id,
            instance_id,
            attempt,
            completion,
            verdict,
            usage_lower_bound=bool(completion.error),
        )
        breaker.charge(entry, namespace=namespace)
        adopted.append(entry)
        claimed.add(run_id)
    return adopted, claimed


def _dispatch_runs_concurrently(
    config: RoundConfig,
    breaker: CandidateSpendBreaker,
    path: Path,
    entries: list[dict[str, Any]],
    pending: list[tuple[dict[str, Any], int]],
) -> GovernedRoundResult:
    """Execute this round's pending runs in bounded, concurrent child processes.

    The parent owns every byte that is shared within the split: it writes the
    one surface module every child imports, verifies persisted traces once,
    and appends every manifest line itself on reap (KTD4, KTD6). A child's
    only footprint is its own per-run directory. Common preparation and orphan
    adoption happen before this branch is selected.

    Dispatch order is the pending list's order -- instance-major, attempt-minor
    -- and the reservation gate below is the only thing that stops it, so a
    round that stops early stops on a contiguous tail, the same shape the
    sequential path produces (R7).
    """
    if not pending:
        return _governed_result(config, breaker, entries)

    namespace = str(path)
    require_backend_credential(config)

    # One surface module, written once by the parent. Letting each child
    # rematerialize to a shared path would reintroduce the write race that
    # keeping shared files parent-owned exists to remove.
    serialization = serialize_harness(config.harness)
    expected_hash = harness_hash(config.harness)
    module_path = path / f"run_module_{expected_hash[:16]}.py"
    write_surface_module(serialization, module_path)

    reservation = _run_reservation(breaker.caps)
    if reservation > breaker.caps.candidate_budget:
        # The gate reserves one reservation per in-flight run, so a budget
        # smaller than a single reservation admits nothing at all: every run
        # would be reported skipped and the subject would carry an empty
        # sample that merely looks like a budget stop. That is a
        # misconfiguration, not a result, so it is refused rather than
        # measured.
        raise ValueError(
            f"candidate_budget ${breaker.caps.candidate_budget:.6f} cannot reserve even one "
            f"concurrent run (reservation ${reservation:.6f} = {RUN_RESERVATION_FACTOR:g} x "
            f"max_budget ${breaker.caps.max_budget:.6f}). Raise candidate_budget, lower "
            "max_budget, or set the applicable run-worker setting to 1 to run this "
            "subject sequentially."
        )

    queue = deque(pending)

    limits = {
        name: getattr(config, name)
        for name in ("max_iterations", "max_depth", "max_budget", "max_timeout")
        if getattr(config, name) is not None
    }
    deadline = hard_deadline_seconds(config.max_timeout)
    factory_args = config.client_factory

    running: dict[str, dict[str, Any]] = {}
    stopped = False
    pending_dir = path / RUN_REQUESTS_DIRNAME
    try:
        while queue or running:
            while (
                queue
                and not stopped
                and len(running) < config.run_workers
                and breaker.spent + reservation * (len(running) + 1)
                <= breaker.caps.candidate_budget
            ):
                instance, attempt = queue.popleft()
                instance_id = str(instance["id"])
                run_id = run_id_for(instance_id, attempt)
                run_path = pending_dir / run_id
                run_path.mkdir(parents=True, exist_ok=True)
                request = build_run_request(
                    run_id=run_id,
                    instance=instance,
                    attempt=attempt,
                    harness_serialization=serialization,
                    expected_hash=expected_hash,
                    module_path=module_path,
                    backend=config.backend,
                    backend_kwargs=dict(config.backend_kwargs),
                    limits=limits,
                    trace_path=trace_path_for(path, run_id),
                    deadline_seconds=deadline,
                    parent_pid=os.getpid(),
                    client_factory=factory_args,
                )
                request_path = run_path / RUN_REQUEST_FILENAME
                request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
                # A result document left by an earlier attempt at this run would
                # otherwise be read back as this attempt's outcome.
                (run_path / RUN_RESULT_FILENAME).unlink(missing_ok=True)
                log = open(run_path / RUN_LOG_FILENAME, "ab")  # noqa: SIM115 - closed on reap
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "shrlm.optimization.run_worker", str(request_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                # Register before the pid marker write: once Popen succeeds,
                # every later failure must flow through the cleanup handler.
                running[run_id] = {
                    "process": process,
                    "log": log,
                    "path": run_path,
                    "instance": instance,
                    "attempt": attempt,
                    # The parent's own bound, independent of the child's alarm:
                    # a child that ignores or swallows SIGALRM is still reaped.
                    "expires": (time.monotonic() + deadline * 2) if deadline else None,
                }
                (run_path / RUN_PID_FILENAME).write_text(f"{process.pid}\n")

            if not running:
                # Nothing in flight and the fill loop above placed nothing.
                # Either the breaker tripped, or the reservation gate cannot
                # admit even one more run within the remaining budget. No
                # further progress is possible, so whatever is still queued is
                # skipped -- and because dispatch order is the pending order,
                # what is skipped is a contiguous tail.
                break

            finished = [
                run_id for run_id, live in running.items() if live["process"].poll() is not None
            ]
            overdue = [
                run_id
                for run_id, live in running.items()
                if run_id not in finished
                and live["expires"] is not None
                and time.monotonic() >= live["expires"]
            ]
            if overdue:
                # Individually, never by group: run children are deliberately
                # not session leaders, so their pid is not a process-group id
                # and a group-directed signal would hit an unrelated group.
                # Signal them all first, then wait once -- terminating them one
                # at a time would spend the grace period per child while the
                # dispatcher polls nothing.
                _terminate_all([running[run_id]["process"] for run_id in overdue])
                finished.extend(overdue)

            if not finished:
                _sleep(POLL_SECONDS)
                continue

            for run_id in finished:
                live = running.pop(run_id)
                live["log"].close()
                (live["path"] / RUN_PID_FILENAME).unlink(missing_ok=True)
                entry = _reap_run(
                    path, run_id, live, config, timed_out=run_id in overdue, breaker=breaker
                )
                entries.append(entry)
                breaker.charge(entry, namespace=namespace)
                if breaker.tripped:
                    stopped = True

            if stopped and not running:
                break
    except BaseException:
        _terminate_all([live["process"] for live in running.values()])
        for live in running.values():
            live["log"].close()
            # The child is dead; its pid marker must not outlive it, or the
            # next invocation would refuse to start on a run nobody is running.
            (live["path"] / RUN_PID_FILENAME).unlink(missing_ok=True)
        raise

    return _governed_result(config, breaker, entries)


def _reap_run(
    path: Path,
    run_id: str,
    live: dict[str, Any],
    config: RoundConfig,
    *,
    timed_out: bool,
    breaker: CandidateSpendBreaker,
) -> dict[str, Any]:
    """Record one finished child's run. The parent verifies and appends (KTD5).

    A child that produced a usable trace is verified and recorded from it. One
    that did not is recorded as terminated under its own run id and charged --
    never re-dispatched (KTD11): retrying would leave the crashed attempt's
    spend uncharged and punch a hole in the contiguous tail.
    """
    instance = live["instance"]
    instance_id = str(instance["id"])
    completion = read_child_trace(trace_path_for(path, run_id))
    if completion is not None:
        verdict = (
            _error_verdict(str(completion.error), completion.response)
            if completion.error
            else config.verifier(instance, completion.response)
        )
        return append_child_run(
            path,
            run_id,
            instance_id,
            attempt=live["attempt"],
            completion=completion,
            verdict=verdict,
            usage_lower_bound=bool(completion.error),
        )

    # No usable trace. The run may still have spent money, so it is recorded as
    # terminated under its own id and charged rather than retried: re-dispatching
    # would leave the lost attempt's spend uncharged and punch a hole in the
    # contiguous tail (KTD11). The error names what actually happened -- a child
    # killed at its deadline is a timeout, a child that died is not.
    error: Exception
    if timed_out:
        elapsed = hard_deadline_seconds(config.max_timeout) or 0.0
        error = TimeoutExceededError(
            elapsed=elapsed,
            timeout=elapsed,
            message=(
                f"run worker for {run_id} ignored its deadline and was terminated by "
                f"the parent after {elapsed:.1f}s"
            ),
        )
    else:
        result = read_run_result(live["path"])
        detail = (result or {}).get("error") or (
            f"run worker exited {live['process'].returncode} without a usable trace"
        )
        error = RunWorkerError(str(detail))
    entry = persist_interrupted_run(config, error, run_id=run_id)
    if entry is None:  # pragma: no cover - the run cannot already be persisted here
        raise RunWorkerError(f"run {run_id!r} vanished from the round while being reaped")
    return entry


def _signal_children(processes: list[subprocess.Popen[bytes]]) -> None:
    """Ask every live child to stop, without waiting for any of them.

    Signalling is separated from waiting so several children stop in parallel.
    Terminating one at a time would spend the full grace period on each in
    turn, and the dispatcher polls nothing while it blocks.
    """
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.terminate()


def _reap_children(processes: list[subprocess.Popen[bytes]]) -> None:
    """Wait out the grace period once, then kill whatever is still alive."""
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    for process in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.kill()
    for process in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=TERMINATE_GRACE_SECONDS)


def _terminate_all(processes: list[subprocess.Popen[bytes]]) -> None:
    """Stop every child: politely first, all at once, then not."""
    _signal_children(processes)
    _reap_children(processes)


def _governed_result(
    config: RoundConfig, breaker: CandidateSpendBreaker, entries: list[dict[str, Any]]
) -> GovernedRoundResult:
    done = {str(entry["run_id"]) for entry in entries}
    skipped = [
        run_id
        for instance in config.instances
        for attempt in range(1, config.attempts + 1)
        if (run_id := run_id_for(str(instance["id"]), attempt)) not in done
    ]
    return GovernedRoundResult(
        entries=entries,
        outcome=OUTCOME_OVER_BUDGET if (breaker.tripped or skipped) else OUTCOME_COMPLETED,
        spent=breaker.spent,
        skipped_run_ids=skipped,
    )


def run_governed_round(config: RoundConfig, breaker: CandidateSpendBreaker) -> GovernedRoundResult:
    """Run one round under the breaker: persist-first, one run at a time.

    Pre-existing manifest lines are charged before anything executes, so a
    resumed round that is already over budget makes zero model calls and a
    crash between runs loses nothing (each ``run_round`` slice re-verifies the
    persisted state it resumes over). Every newly persisted run is charged the
    moment its manifest line lands; once the breaker trips, the remaining runs
    are skipped and reported, never silently dropped.

    Each slice runs under the hard wall-clock backstop (module docstring): a
    candidate call that hangs is interrupted at ``hard_deadline_seconds``,
    persisted as a RESOURCE_TERMINATED run, charged (at the per-run ceiling
    when the termination persisted no cost), and the round continues -- the
    host never blocks indefinitely on candidate code.

    Args:
        config: The round to run, its limits already merged by
            ``governed_limits``.
        breaker: The candidate's breaker, shared across that candidate's
            rounds (splits, repetitions) so the budget is truly cumulative.

    Returns:
        The manifest entries, the ``completed``/``over_budget`` outcome, the
        breaker's cumulative spend, and the run ids the breaker skipped.
    """
    round_path = round_dir(config.out_dir, config.round_index)
    round_path.mkdir(parents=True, exist_ok=True)
    with claim_split(round_path):
        return _run_governed_round_claimed(config, breaker, round_path)


def _run_governed_round_claimed(
    config: RoundConfig, breaker: CandidateSpendBreaker, round_path: Path
) -> GovernedRoundResult:
    """``run_governed_round``'s body, with this split's claim already held."""
    orphaned = _live_run_children(round_path)
    if orphaned:
        # The split claim names only the parent. A crashed parent's children
        # can outlive it, so every execution branch must refuse them before
        # prepare_round rewrites the execution sidecar or any run is repeated.
        raise SplitClaimedError(
            f"{round_path} still has {len(orphaned)} run worker(s) alive from an earlier "
            f"invocation (pids {sorted(orphaned)}); they are still paying for runs this "
            "round would repeat. Wait for them to exit, or terminate them, then re-run."
        )

    path, existing, pending = prepare_round(config)
    namespace = str(path)
    for entry in existing:
        breaker.charge(entry, namespace=namespace)
    adopted, claimed = _adopt_orphan_traces(path, pending, config, breaker, namespace)
    entries = [*existing, *adopted]
    pending = [
        (instance, attempt)
        for instance, attempt in pending
        if run_id_for(str(instance["id"]), attempt) not in claimed
    ]

    if config.run_workers > 1:
        return _dispatch_runs_concurrently(config, breaker, path, entries, pending)

    deadline = hard_deadline_seconds(config.max_timeout)
    # The stop_after=0 slice executes no runs but does build the harness when
    # runs are pending -- candidate code that can itself hang, hence the guard.
    known = len(entries)
    entries = _run_slice(
        config,
        stop_after=0,
        deadline=deadline,
        known=known,
    )
    for entry in entries[known:]:
        breaker.charge(entry, namespace=namespace)

    while not breaker.tripped:
        known = len(entries)
        entries = _run_slice(config, stop_after=1, deadline=deadline, known=known)
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
    "HARD_DEADLINE_FACTOR",
    "HARD_DEADLINE_GRACE_SECONDS",
    "OUTCOME_COMPLETED",
    "OUTCOME_OVER_BUDGET",
    "CandidateSpendBreaker",
    "GovernedRoundResult",
    "HardDeadlineExceeded",
    "SplitClaimedError",
    "ValidationCaps",
    "breaker_run_cost",
    "governed_limits",
    "hard_deadline_seconds",
    "run_governed_round",
]
