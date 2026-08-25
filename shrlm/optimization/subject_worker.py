"""The validation subject worker: one subject's evaluation in a child process.

``evaluate_validation_round`` runs its subjects (the baseline plus every
loaded candidate) concurrently when ``EvaluationConfig.workers > 1``. Each
subject is one child process::

    python -m shrlm.optimization.subject_worker <subject_dir>/worker_request.json

The child is a plain main-thread Python process, so everything the sequential
path relies on still holds inside it: the SIGALRM hard deadline binds
(``shrlm.optimization.costs``), the subject's ``CandidateSpendBreaker`` is its
own, and ``evaluate_subject`` persists first and resumes from its manifests.
Nothing about *how* a subject is evaluated changes -- only *where*.

The request file is self-contained JSON (KTD3): the subject id, the harness
envelope with the hash the parent expects, the split instances, the caps,
repetitions, backend name and kwargs, the round's ``out_dir`` and index, the
dotted path of a zero-argument verifier factory, and an optional test-only
client factory. Live harness objects never cross the process boundary: the
child rebuilds the harness from the envelope with ``materialize_harness`` and
refuses to run when the rebuilt hash differs from the expected one (R7) --
the same round trip the candidate gate and the orchestrator's resume path
already prove.

The child's outcome lands in ``<subject_dir>/worker_result.json`` (one JSON
document, ``ok`` plus either ``summary_path`` or ``error``) and is echoed to
stdout, which the parent redirects into ``<subject_dir>/worker.log`` together
with stderr. The parent never parses stdout: it reads the result file, then
rebuilds the ``SubjectEvaluation`` from the persisted ``summary.json`` --
the same bytes the sequential path returns (KTD5).

Test seam (KTD9): when the request names a ``client_factory``, the child
resolves that dotted path, calls it with the request's per-subject args, and
installs the returned callable on ``rlm.core.rlm.get_client`` -- the seam the
in-process tests monkeypatch -- before evaluating. On exit it writes
``<subject_dir>/client_calls.json`` with the factory's ``total_calls`` (when
the factory exposes one) so tests can assert call counts across the process
boundary.
"""

import json
import os
import pkgutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from shrlm.harness_identity import harness_hash, serialize_harness
from shrlm.optimization.candidates import CandidateRejection, materialize_harness
from shrlm.optimization.costs import ValidationCaps, governed_limits
from shrlm.rlm_harness import Harness

if TYPE_CHECKING:  # validation imports this module; its types stay annotations here
    from shrlm.optimization.validation import EvaluationConfig, SubjectEvaluation

REQUEST_FILENAME = "worker_request.json"
RESULT_FILENAME = "worker_result.json"
LOG_FILENAME = "worker.log"
CLIENT_CALLS_FILENAME = "client_calls.json"
# The live child's pid, written by the parent after spawn and removed when the
# child exits. A resume that finds a live pid refuses to spawn a second child
# on the same subject directory (the parent-death case: an orphaned child is
# still paying for runs the new child would repeat).
PID_FILENAME = "worker.pid"

REQUEST_FORMAT = "shrlm-subject-worker-request/v1"
RESULT_FORMAT = "shrlm-subject-worker-result/v1"

# The child materializes its harness into a subject-local module file, named
# by the expected hash so two subjects' modules never collide.
MODULE_PREFIX = "subject_module_"

# How often the parent polls its children. Runs take seconds to minutes, so a
# coarse poll costs nothing measurable. ``_sleep`` is the poll's own seam so a
# test can interrupt the loop without touching the stdlib ``time.sleep`` that
# ``Popen.wait(timeout=...)`` relies on during cleanup.
POLL_SECONDS = 0.2
_sleep = time.sleep

# How long the parent gives a child to exit after SIGTERM before SIGKILL, so an
# interrupt never hangs on a child that ignores the polite signal.
TERMINATE_GRACE_SECONDS = 10.0

# How often a child checks that its parent is still alive. A parent that was
# SIGKILLed cannot terminate its children, so each child watches for itself and
# exits rather than keep spending on runs a resumed parent will redo.
PARENT_WATCH_SECONDS = 1.0

BASELINE_REJECTED_MESSAGE = (
    "the incumbent violates the experiment-owned caps ({reason}); a baseline that "
    "cannot run under the validation limits is a misconfigured experiment, not a "
    "rejectable candidate"
)


class SubjectWorkerError(RuntimeError):
    """One or more subject workers failed; every completed sibling is persisted.

    Raised by the parent after *all* workers have exited (the sibling-completing
    policy): re-invoking the same command resumes each subject from its
    manifests, re-paying only the failed subject's missing runs.
    """

    def __init__(self, failures: list[dict[str, Any]]):
        self.failures = failures
        lines = [
            f"  {failure['subject_id']}: {failure['error']} (log: {failure['log_path']})"
            for failure in failures
        ]
        super().__init__(
            f"{len(failures)} validation subject worker(s) failed; every other subject's "
            "runs are persisted, so re-running resumes only the missing runs:\n" + "\n".join(lines)
        )


class SubjectWorkerBusyError(RuntimeError):
    """A subject directory still has a live worker; refusing to spawn a second one."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_worker_pid(subject_path: Path) -> int | None:
    """The pid recorded in ``worker.pid`` when that process is still alive.

    A stale file (its process is gone) is removed and reads as ``None``.
    """
    path = subject_path / PID_FILENAME
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    if _pid_alive(pid):
        return pid
    path.unlink(missing_ok=True)
    return None


def _watch_parent(parent_pid: int, subject_path: Path, subject_id: str) -> None:
    """Child-side watchdog: exit when the parent that spawned us is gone."""
    while True:
        time.sleep(PARENT_WATCH_SECONDS)
        if os.getppid() != parent_pid:
            try:
                (subject_path / RESULT_FILENAME).write_text(
                    json.dumps(
                        {
                            "format": RESULT_FORMAT,
                            "ok": False,
                            "subject_id": subject_id,
                            "error": f"parent process {parent_pid} died; worker exited",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            finally:
                os._exit(3)


def resolve_dotted(path: str) -> Any:
    """Resolve ``pkg.mod:attr`` or ``pkg.mod.attr`` to the named object."""
    return pkgutil.resolve_name(path)


def build_request(
    *,
    subject_id: str,
    harness_serialization: dict[str, Any],
    expected_hash: str,
    splits: dict[str, list[dict[str, Any]]],
    caps: ValidationCaps,
    repetitions: int,
    backend: str,
    backend_kwargs: dict[str, Any],
    run_workers: int,
    out_dir: Path | str,
    round_index: int,
    verifier_factory: str,
    client_factory: tuple[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """The request document one child evaluates (KTD3)."""
    return {
        "format": REQUEST_FORMAT,
        "subject_id": subject_id,
        "harness": {"harness": harness_serialization, "hash": expected_hash},
        "splits": splits,
        "caps": asdict(caps),
        "repetitions": repetitions,
        "backend": backend,
        "backend_kwargs": backend_kwargs,
        # A child rebuilds its EvaluationConfig field by field, so a knob that
        # is absent here silently defaults to 1 in every child -- the subject
        # would fan out at the parent's setting but run its own runs one at a
        # time, and nothing would say so.
        "run_workers": run_workers,
        "out_dir": str(out_dir),
        "round_index": round_index,
        "verifier_factory": verifier_factory,
        "client_factory": (
            [client_factory[0], client_factory[1]] if client_factory is not None else None
        ),
    }


def write_request(subject_path: Path, request: dict[str, Any]) -> Path:
    """Persist the request under the subject directory, overwriting a stale one.

    The request is regenerated on every invocation from the live config, so
    it is deliberately *not* non-clobbering: a resume must never be refused
    because an ephemeral request document drifted by a byte.
    """
    subject_path.mkdir(parents=True, exist_ok=True)
    # A stale result from an earlier invocation must never be read as this
    # child's outcome: the child rewrites it, and a child that dies before
    # doing so leaves none.
    (subject_path / RESULT_FILENAME).unlink(missing_ok=True)
    path = subject_path / REQUEST_FILENAME
    path.write_text(json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def read_result(subject_path: Path) -> dict[str, Any] | None:
    """The child's result document, or ``None`` when it never wrote one."""
    path = subject_path / RESULT_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != RESULT_FORMAT:
        return None
    return payload


def _install_client_factory(spec: list[Any] | None) -> Any:
    """Resolve and install the test-only client factory on the runtime seam."""
    if spec is None:
        return None
    dotted, args = spec
    factory = resolve_dotted(str(dotted))(dict(args))
    import rlm.core.rlm as rlm_module

    rlm_module.get_client = factory
    return factory


def run_subject_worker(request_path: str | Path) -> dict[str, Any]:
    """Evaluate one subject from its request file. Child-side.

    Every failure -- a bad request, a harness that does not round-trip, an
    unresolvable factory, an exception out of the evaluation -- is a result
    dict with ``ok: false`` and the error text, never an uncaught exception,
    so the child always leaves a result document behind.
    """
    request_path = Path(request_path)
    subject_path = request_path.parent
    subject_id = "<unknown>"
    factory: Any = None
    try:
        request = json.loads(request_path.read_text())
        if request.get("format") != REQUEST_FORMAT:
            raise ValueError(f"{request_path} is not a {REQUEST_FORMAT} document")
        subject_id = str(request["subject_id"])

        # Deferred: importing the validation module pulls the whole runtime.
        from shrlm.optimization.validation import (
            EvaluationConfig,
            ValidationSplits,
            evaluate_subject,
        )

        verifier = resolve_dotted(str(request["verifier_factory"]))()
        factory = _install_client_factory(request.get("client_factory"))

        envelope = request["harness"]
        expected = str(envelope["hash"])
        module_path = subject_path / f"{MODULE_PREFIX}{expected[:16]}.py"
        harness = materialize_harness(envelope["harness"], module_path)
        rebuilt = harness_hash(harness)
        if rebuilt != expected:
            raise ValueError(
                f"the request's harness envelope rematerializes to hash {rebuilt}, not the "
                f"expected {expected}; refusing to evaluate a harness whose identity cannot "
                "be verified"
            )

        config = EvaluationConfig(
            splits=ValidationSplits(
                heldin=list(request["splits"]["heldin"]),
                heldout=list(request["splits"]["heldout"]),
            ),
            verifier=verifier,
            caps=ValidationCaps(**request["caps"]),
            out_dir=Path(request["out_dir"]),
            round_index=int(request["round_index"]),
            repetitions=int(request["repetitions"]),
            backend=str(request["backend"]),
            backend_kwargs=dict(request["backend_kwargs"]),
            run_workers=int(request.get("run_workers", 1)),
        )
        outcome = evaluate_subject(subject_id, harness, config)
        if isinstance(outcome, CandidateRejection):
            # The parent gates caps before spawning (KTD4), so a rejection here
            # is a contradiction between parent and child, not an expected path.
            raise RuntimeError(
                f"the caps gate rejected {subject_id!r} inside the worker ({outcome.reason}) "
                "although the parent already admitted it"
            )
        result = {
            "format": RESULT_FORMAT,
            "ok": True,
            "subject_id": subject_id,
            "summary_path": str(outcome.summary_path),
        }
    except BaseException as error:  # noqa: BLE001 - the child reports every failure as data
        result = {
            "format": RESULT_FORMAT,
            "ok": False,
            "subject_id": subject_id,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if factory is not None:
            calls = getattr(factory, "total_calls", None)
            (subject_path / CLIENT_CALLS_FILENAME).write_text(
                json.dumps({"subject_id": subject_id, "calls": calls}, sort_keys=True) + "\n"
            )
    subject_path.mkdir(parents=True, exist_ok=True)
    (subject_path / RESULT_FILENAME).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
    """Signal the child's whole process group (it was spawned as a session leader)."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _terminate_all(children: list[subprocess.Popen[bytes]]) -> None:
    """SIGTERM every child's group, then SIGKILL any that outlives the grace period."""
    for process in children:
        _signal_group(process, signal.SIGTERM)
    for process in children:
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_group(process, signal.SIGKILL)
            process.wait()


def evaluate_subjects_in_processes(
    subjects: list[tuple[str, Harness, str]], config: "EvaluationConfig"
) -> list["SubjectEvaluation | CandidateRejection"]:
    """Evaluate ``subjects`` concurrently, one child process each (KTD1). Parent-side.

    The caps gate runs here in the parent (KTD4), so a subject whose enabled
    S6 policy exceeds the caps comes back as its ``CandidateRejection`` without
    a child ever starting; a rejected *baseline* raises before any spawn. Every
    admitted subject gets a request file and a child under
    ``<subject_dir>/worker.log``; at most ``config.workers`` children are alive
    at once, in ``subjects`` order. Results land by index (KTD8), rebuilt from
    each child's persisted ``summary.json`` (KTD5). A child that exits non-zero
    or leaves no ``ok`` result is recorded as a failure; the remaining children
    still run to completion, then ``SubjectWorkerError`` names every failure
    with its log path (R6). A ``KeyboardInterrupt`` (or any other escape)
    terminates every live child's process group -- SIGTERM, then SIGKILL after
    ``TERMINATE_GRACE_SECONDS`` -- before propagating (R8).

    Orphan guard: every admitted subject's ``worker.pid`` is checked before
    the first spawn, and a live pid raises ``SubjectWorkerBusyError`` -- a
    parent that was SIGKILLed left a child still paying for that subject, and
    a second child would redo its runs. Each child also exits on its own when
    its parent disappears (``_watch_parent``).
    """
    # Deferred: validation imports this module at load time.
    from shrlm.optimization.validation import (
        BASELINE_ID,
        SUMMARY_FILENAME,
        SubjectEvaluation,
        load_summary,
        subject_dir,
    )

    assert config.verifier_factory, "EvaluationConfig admits workers > 1 only with a factory"
    results: list[SubjectEvaluation | CandidateRejection | None] = [None] * len(subjects)
    queue: deque[tuple[int, str, Harness, str]] = deque()
    for index, (subject_id, harness, expected_hash) in enumerate(subjects):
        limits = governed_limits(subject_id, harness.runtime_policy, config.caps)
        if isinstance(limits, CandidateRejection):
            if subject_id == BASELINE_ID:
                raise ValueError(BASELINE_REJECTED_MESSAGE.format(reason=limits.reason))
            results[index] = limits
            continue
        queue.append((index, subject_id, harness, expected_hash))

    busy = [
        (subject_id, pid)
        for _index, subject_id, _harness, _hash in queue
        if (pid := live_worker_pid(subject_dir(config.out_dir, config.round_index, subject_id)))
        is not None
    ]
    if busy:
        named = ", ".join(f"{subject_id} (pid {pid})" for subject_id, pid in busy)
        raise SubjectWorkerBusyError(
            f"a worker is still running for subject(s) {named}; a previous invocation's "
            "parent died without terminating its children. Wait for or kill those "
            "processes, then re-run to resume."
        )

    split_instances = {split_id: instances for split_id, instances in config.splits.items()}
    factory_args = config.client_factory[1] if config.client_factory is not None else {}
    running: dict[int, tuple[subprocess.Popen[bytes], Path, IO[bytes], str]] = {}
    failures: list[dict[str, Any]] = []
    try:
        while queue or running:
            while queue and len(running) < config.workers:
                index, subject_id, harness, expected_hash = queue.popleft()
                subject_path = subject_dir(config.out_dir, config.round_index, subject_id)
                request = build_request(
                    subject_id=subject_id,
                    harness_serialization=serialize_harness(harness),
                    expected_hash=expected_hash,
                    splits=split_instances,
                    caps=config.caps,
                    repetitions=config.repetitions,
                    backend=config.backend,
                    backend_kwargs=dict(config.backend_kwargs),
                    run_workers=config.run_workers,
                    out_dir=config.out_dir,
                    round_index=config.round_index,
                    verifier_factory=config.verifier_factory,
                    client_factory=(
                        (config.client_factory[0], dict(factory_args.get(subject_id, {})))
                        if config.client_factory is not None
                        else None
                    ),
                )
                request_path = write_request(subject_path, request)
                log = open(subject_path / LOG_FILENAME, "ab")  # noqa: SIM115 - closed on exit
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-m", __name__, str(request_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                (subject_path / PID_FILENAME).write_text(f"{process.pid}\n")
                running[index] = (process, subject_path, log, subject_id)

            finished = [
                index for index, (process, *_) in running.items() if process.poll() is not None
            ]
            if not finished:
                _sleep(POLL_SECONDS)
                continue
            for index in finished:
                process, subject_path, log, subject_id = running.pop(index)
                log.close()
                (subject_path / PID_FILENAME).unlink(missing_ok=True)
                result = read_result(subject_path)
                if process.returncode == 0 and result is not None and result.get("ok"):
                    results[index] = SubjectEvaluation(
                        subject_id=subject_id,
                        path=subject_path,
                        summary_path=subject_path / SUMMARY_FILENAME,
                        summary=load_summary(subject_path),
                    )
                    continue
                error = (result or {}).get("error") or (
                    f"worker exited {process.returncode} without a result document"
                )
                failures.append(
                    {
                        "subject_id": subject_id,
                        "error": str(error),
                        "log_path": str(subject_path / LOG_FILENAME),
                    }
                )
    except BaseException:
        _terminate_all([process for process, *_ in running.values()])
        for _process, subject_path, log, _subject_id in running.values():
            log.close()
            (subject_path / PID_FILENAME).unlink(missing_ok=True)
        raise
    if failures:
        raise SubjectWorkerError(failures)
    return [outcome for outcome in results if outcome is not None]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            f"usage: python -m shrlm.optimization.subject_worker <{REQUEST_FILENAME}>",
            file=sys.stderr,
        )
        return 2
    request_path = Path(argv[1])
    subject_id = "<unknown>"
    try:
        subject_id = str(json.loads(request_path.read_text()).get("subject_id", subject_id))
    except (OSError, ValueError):
        pass
    threading.Thread(
        target=_watch_parent,
        args=(os.getppid(), request_path.parent, subject_id),
        daemon=True,
        name="parent-watchdog",
    ).start()
    result = run_subject_worker(argv[1])
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}))
    if not result["ok"]:
        print(result.get("traceback", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
