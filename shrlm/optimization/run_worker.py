"""One validation run in a child process.

A validation subject is 256 runs executed one at a time, and the long tail
dominates: on the live qwen experiment the mean run is twice the median. This
module is the unit the parent fans those runs out across::

    python -m shrlm.optimization.run_worker <split_dir>/run_workers/<run_id>/request.json

The child executes exactly one ``(instance, attempt)`` pair through the same
``driver.execute_run`` the sequential loop uses, so the timing window, the
limit-exception handling, and the lower-bound flag cannot drift between the two
paths (KTD14).

**The child writes nothing the parent owns.** Not ``runs.jsonl``, not
``harness.json``, not ``instances.jsonl``, not the materialized surface module.
A validation manifest line reaches 48 KB and is appended through a buffered
handle, so a single line becomes several ``write()`` calls and two concurrent
appenders interleave into a torn line that ``_load_manifest`` then refuses
(KTD4). The parent appends every line itself, on reap. The child's whole
footprint is its own per-run directory: the trace, and one result document.

The trace is written through a temporary file and an atomic rename, so the
parent either sees a complete trace or no trace at all -- never a half-written
one it might hash and record (R13).

The child does not verify. Verdict construction stays in the parent, which
already holds the live verifier callable and would otherwise have to thread a
factory down another process boundary (KTD5). A complete trace from a child
that died before reporting can therefore still be adopted and verified rather
than re-paid for.

Two things bound a child's life. It arms its own ``SIGALRM`` hard deadline --
the reason child processes rather than threads are the unit at all, since that
alarm binds only on a main thread -- and it watches for its parent
disappearing, because a parent that was ``SIGKILL``ed cannot terminate anything
and an orphan would keep spending on a run the resumed parent will redo (R14,
R18).

Every failure is reported as data, never as an uncaught exception: a bad
request, a harness that does not round-trip, an unresolvable factory, a limit
that ended the run. The child always leaves a result document behind, because a
child that exits silently is indistinguishable to the parent from one that
never started.
"""

import json
import os
import pkgutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from rlm.utils.exceptions import HardDeadlineSignal
from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import assemble_harness
from shrlm.optimization.types import Verdict

REQUEST_FILENAME = "request.json"
RESULT_FILENAME = "result.json"
LOG_FILENAME = "run.log"

REQUEST_FORMAT = "shrlm-run-worker-request/v1"
RESULT_FORMAT = "shrlm-run-worker-result/v1"

# How often a child checks that its parent is still alive.
PARENT_WATCH_SECONDS = 1.0


class RunWorkerError(RuntimeError):
    """A run worker could not be dispatched or its result could not be read."""


def build_request(
    *,
    run_id: str,
    instance: dict[str, Any],
    attempt: int,
    harness_serialization: dict[str, Any],
    expected_hash: str,
    module_path: Path | str,
    backend: str,
    backend_kwargs: dict[str, Any],
    limits: dict[str, Any],
    trace_path: Path | str,
    deadline_seconds: float | None,
    parent_pid: int,
    client_factory: tuple[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The self-contained document one child needs to execute one run.

    Live objects never cross the process boundary. The harness travels as its
    serialization plus the hash the parent expects, and the child refuses to
    run when its rebuilt harness hashes to anything else -- the same round trip
    the candidate gate and the resume path already prove. ``module_path`` names
    the surface module the *parent* already wrote; the child imports it and
    never writes it (KTD6).
    """
    request: dict[str, Any] = {
        "format": REQUEST_FORMAT,
        "run_id": run_id,
        "instance": instance,
        "attempt": attempt,
        "harness": {"harness": harness_serialization, "hash": expected_hash},
        "module_path": str(module_path),
        "backend": backend,
        "backend_kwargs": dict(backend_kwargs),
        "limits": dict(limits),
        "trace_path": str(trace_path),
        "deadline_seconds": deadline_seconds,
        "parent_pid": parent_pid,
    }
    if client_factory is not None:
        dotted, args = client_factory
        request["client_factory"] = [str(dotted), dict(args)]
    return request


def _install_client_factory(spec: list[Any] | None) -> Any:
    """Resolve and install the test-only client factory on the runtime seam."""
    if spec is None:
        return None
    dotted, args = spec
    # Resolved here rather than imported from the subject worker: costs.py
    # imports this module, and the subject worker imports costs, so reaching
    # across for a one-line helper would close an import cycle.
    factory = pkgutil.resolve_name(str(dotted))(dict(args))
    import rlm.core.rlm as rlm_module

    rlm_module.get_client = factory
    return factory


def _arm_deadline(seconds: float | None) -> None:
    """Arm this child's own hard wall-clock deadline.

    The alarm binds cleanly here because a child is a plain main-thread
    process. It is the child's own backstop; the parent holds an independent
    deadline for the case where a child ignores or swallows the signal.

    It raises ``HardDeadlineSignal`` -- a ``BaseException`` -- rather than the
    runtime's ``TimeoutExceededError``. The alarm usually lands while the main
    thread is inside model code or a sub-call wrapper, both of which catch
    ``Exception`` and turn it into an in-band error string; a plain exception
    was swallowed there and the run continued past its deadline (observed
    2026-08-29: 3790s and 5011s against an 1800s cap). ``execute_run`` catches
    the signal explicitly and persists the run as a timeout with the partial
    completion and recorded usage, so the parent charges what was spent rather
    than the flat per-run ceiling.
    """
    if seconds is None or seconds <= 0:
        return
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - platform without SIGALRM
        return

    def _fire(_signum: int, _frame: Any) -> None:
        # A BaseException: ``except Exception`` scopes inside the REPL and the
        # sub-call wrappers would otherwise swallow the alarm as an in-band
        # error string and the run would continue past its deadline. The
        # driver's limit handler turns it into the usual TimeoutExceededError
        # persistence (partial completion, recorded usage).
        raise HardDeadlineSignal(
            seconds,
            message=(
                f"run worker exceeded its {seconds:.1f}s hard deadline; candidate code "
                "likely hung inside a live call"
            ),
        )

    signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def _disarm_deadline() -> None:
    if hasattr(signal, "SIGALRM"):
        signal.setitimer(signal.ITIMER_REAL, 0)


def _watch_parent(parent_pid: int, run_path: Path, run_id: str) -> None:
    """Child-side watchdog: exit when the parent that spawned us is gone."""
    while True:
        time.sleep(PARENT_WATCH_SECONDS)
        if os.getppid() != parent_pid:
            try:
                _write_result(
                    run_path,
                    {
                        "format": RESULT_FORMAT,
                        "ok": False,
                        "run_id": run_id,
                        "error": f"parent process {parent_pid} died; run worker exited",
                    },
                )
            finally:
                os._exit(3)


def _write_result(run_path: Path, result: dict[str, Any]) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / RESULT_FILENAME).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _write_trace_atomically(trace_path: Path, payload: str) -> None:
    """Publish the trace in one step.

    The parent hashes this file after the child exits and records that hash in
    the manifest. A partially written file at the final path could be hashed as
    if it were the whole run, so the bytes are staged beside it and renamed.
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    staged = trace_path.with_name(f"{trace_path.name}.partial-{os.getpid()}")
    staged.write_text(payload)
    os.replace(staged, trace_path)


def run_run_worker(request_path: str | Path) -> dict[str, Any]:
    """Execute one run from its request file. Child-side.

    Every failure -- a bad request, a harness that does not round-trip, an
    unresolvable factory, a limit that ended the run, an exception out of the
    runtime -- is a result document with ``ok`` and the error text, never an
    uncaught exception, so the parent always has something to read.
    """
    request_path = Path(request_path)
    run_path = request_path.parent
    run_id = "<unknown>"
    factory: Any = None
    try:
        request = json.loads(request_path.read_text())
        if request.get("format") != REQUEST_FORMAT:
            raise ValueError(f"{request_path} is not a {REQUEST_FORMAT} document")
        run_id = str(request["run_id"])

        # Deferred: importing the driver pulls the whole runtime.
        from shrlm.optimization.driver import RoundConfig, build_round_rlm, execute_run

        factory = _install_client_factory(request.get("client_factory"))

        envelope = request["harness"]
        expected = str(envelope["hash"])
        harness = assemble_harness(envelope["harness"], Path(request["module_path"]))
        rebuilt = harness_hash(harness)
        if rebuilt != expected:
            raise ValueError(
                f"the request's harness envelope rematerializes to hash {rebuilt}, not the "
                f"expected {expected}; refusing to execute a run whose harness identity "
                "cannot be verified"
            )

        instance = dict(request["instance"])
        model_name = str(request["backend_kwargs"].get("model_name", "unknown"))
        # A RoundConfig with a single instance: build_round_rlm reads only the
        # harness, backend, and limits from it, and no round directory is
        # prepared here -- the parent owns every shared file (KTD6).
        config = RoundConfig(
            round_index=0,
            harness=harness,
            instances=[instance],
            verifier=_never_called_verifier,
            out_dir=run_path,
            backend=str(request["backend"]),
            backend_kwargs=dict(request["backend_kwargs"]),
            **dict(request["limits"]),
        )

        _arm_deadline(request.get("deadline_seconds"))
        try:
            harnessed = build_round_rlm(config)
            # No verifier: the parent verifies (KTD5).
            outcome = execute_run(harnessed, instance, model_name=model_name)
        finally:
            _disarm_deadline()

        _write_trace_atomically(
            Path(request["trace_path"]),
            json.dumps(outcome.completion.to_dict(), sort_keys=True) + "\n",
        )
        result = {
            "format": RESULT_FORMAT,
            "ok": True,
            "run_id": run_id,
            "usage_lower_bound": outcome.usage_lower_bound,
            "terminated": outcome.verdict is not None,
            "detail": outcome.verdict.detail if outcome.verdict is not None else None,
        }
    except BaseException as error:  # noqa: BLE001 - the child reports every failure as data
        result = {
            "format": RESULT_FORMAT,
            "ok": False,
            "run_id": run_id,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if factory is not None:
            calls = getattr(factory, "total_calls", None)
            (run_path / "client_calls.json").write_text(
                json.dumps({"run_id": run_id, "calls": calls}, sort_keys=True) + "\n"
            )
    _write_result(run_path, result)
    return result


def _never_called_verifier(instance: dict[str, Any], produced: str) -> Verdict:
    """Placeholder for the config field a child never uses.

    Annotated ``-> Verdict`` rather than ``-> NoReturn`` even though it always
    raises: it is passed where a ``Verifier`` is expected, and the protocol is
    what this signature has to satisfy.

    ``RoundConfig`` requires a verifier, but the child passes ``None`` into
    ``execute_run`` and the parent constructs every verdict. Reaching this
    would mean a child had started verifying, which is a contract break worth
    a loud failure rather than a quiet wrong verdict.
    """
    raise AssertionError("a run worker must never verify; the parent owns verdict construction")


def read_result(run_path: Path) -> dict[str, Any] | None:
    """The child's result document, or None when it is absent or unreadable."""
    path = Path(run_path) / RESULT_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != RESULT_FORMAT:
        return None
    return payload


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            f"usage: python -m shrlm.optimization.run_worker <{REQUEST_FILENAME}>",
            file=sys.stderr,
        )
        return 2
    request_path = Path(argv[1])
    run_id = "<unknown>"
    parent_pid = os.getppid()
    try:
        payload = json.loads(request_path.read_text())
        run_id = str(payload.get("run_id", run_id))
        parent_pid = int(payload.get("parent_pid", parent_pid))
    except (OSError, ValueError):
        pass
    threading.Thread(
        target=_watch_parent,
        args=(parent_pid, request_path.parent, run_id),
        daemon=True,
        name="parent-watchdog",
    ).start()
    result = run_run_worker(argv[1])
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}))
    if not result["ok"]:
        print(result.get("traceback", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
