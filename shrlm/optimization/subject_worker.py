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
import pkgutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import CandidateRejection, materialize_harness
from shrlm.optimization.costs import ValidationCaps

REQUEST_FILENAME = "worker_request.json"
RESULT_FILENAME = "worker_result.json"
LOG_FILENAME = "worker.log"
CLIENT_CALLS_FILENAME = "client_calls.json"

REQUEST_FORMAT = "shrlm-subject-worker-request/v1"
RESULT_FORMAT = "shrlm-subject-worker-result/v1"

# The child materializes its harness into a subject-local module file, named
# by the expected hash so two subjects' modules never collide.
MODULE_PREFIX = "subject_module_"


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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            f"usage: python -m shrlm.optimization.subject_worker <{REQUEST_FILENAME}>",
            file=sys.stderr,
        )
        return 2
    result = run_subject_worker(argv[1])
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}))
    if not result["ok"]:
        print(result.get("traceback", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
