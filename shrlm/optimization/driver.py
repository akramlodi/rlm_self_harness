"""The mining driver: run a round under one harness, persist everything, then mine.

Persist-first is the design center (KTD5). A round's runs cost real money, so
the moment a run completes -- pass, fail, or resource termination -- its full
trace is written to disk and one line is appended to the round manifest. A
crash between runs loses nothing already paid for, a re-invocation with the
same ``out_dir`` skips every run the manifest already records (after verifying
its trace file still matches its recorded sha256), and the round's pass set is
auditable from the manifest alone, with no process state required.

Round layout under ``out_dir``::

    round_NN/
        harness.json      # write_harness_json envelope: name, hash, all surfaces
        instances.jsonl   # the instances verbatim, one JSON object per line
        runs.jsonl        # the manifest: one line per completed run
        runs/<run_id>.json  # RLMChatCompletion.to_dict() of each run's trace
        digests/<digest_sha256>.txt  # after mine_round: each digest verbatim
        attributor_prompt_<sha16>.txt  # after mine_round: each rendered prompt variant

Each ``runs.jsonl`` line carries: ``run_id``, ``instance_id``, ``attempt``,
``passed``, ``cause`` (denormalized from the verdict for scanability),
``verdict`` (the full ``Verdict.to_dict()``), ``trace_path`` (relative to the
round directory), ``trace_sha256`` (over the trace file's bytes), ``cost``
(``usage_summary.total_cost`` when the backend reported one), the U4 usage
keys ``input_tokens``, ``output_tokens``, ``execution_time``, and
``usage_lower_bound`` (additive, KTD4; the flag marks terminated runs whose
persisted figures are lower bounds on true usage), ``cost_source`` (additive,
R7: "provider" when the cost was provider-reported, "synthesized" when the
client computed it from token counts x configured pricing; omitted entirely
when the client reports no source), and ``timestamp`` (UTC ISO-8601). Lines
persisted before U4 lack the usage keys; readers must treat them (and
``cost_source``) as optional.

``cost`` and the token keys cover the whole run, including spend by recursive
sub-calls. A child RLM runs its own LM handler, so its usage reaches the parent
only through the completion it returns; the runtime folds that in before the
completion is persisted (``UsageSummary.merged_with``). Manifest lines written
before that fix carry the root's own calls only -- on the live qwen round that
was a median of 45% of a decomposing run's true cost, so a directory whose
lines predate it must not be compared against one whose lines do not.

Resource terminations are runs, not crashes. The four root-level limit
exceptions (budget, timeout, tokens, error threshold) are caught per run and
recorded as a failing ``Verdict`` with cause RESOURCE_TERMINATED -- the
environment ``Verifier`` is never handed an exception. The trace for such a run
is a partial ``RLMChatCompletion`` rebuilt from the attached logger's in-memory
trajectory. One documented gap: the runtime checks budget/token/error limits
after executing an iteration but *before* logging it (``rlm/core/rlm.py``), so
the terminating iteration itself is missing from the partial trajectory; the
timeout check runs before the iteration, so timeouts lose nothing. A run
terminated on its very first iteration therefore persists an empty iteration
list.

Mining consumes only what was persisted. ``mine_round`` reads the manifest and
trace files back from disk, verifies every sha, rehydrates completions through
``RLMChatCompletion.from_dict`` and verdicts through ``Verdict.from_dict``, and
feeds the ``WeaknessMiner`` those persisted verdicts -- it never re-runs the
verifier, so the mined round is exactly the round the manifest describes.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm.utils.exceptions import (
    BudgetExceededError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from shrlm.harness_identity import harness_hash, write_harness_json
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN, round_dir
from shrlm.optimization.mining import MiningResult, WeaknessMiner
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import RunTraceLink, Verdict, Verifier
from shrlm.rlm_harness import Harness
from shrlm.runner import HarnessedRLM, build_harnessed_rlm

# The root-level limit exceptions ``RLM.completion`` can raise. Each one is a
# paid, recorded run that the verifier never sees; CancellationError (Ctrl+C)
# is deliberately absent -- an interrupt is a crash to resume from, not a run.
ROOT_LIMIT_EXCEPTIONS: tuple[type[Exception], ...] = (
    BudgetExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
    ErrorThresholdExceededError,
)

HARNESS_FILE = "harness.json"
INSTANCES_FILE = "instances.jsonl"
MANIFEST_FILE = "runs.jsonl"
TRACES_DIR = "runs"
# Facts about how a round executed, as opposed to what it produced. A sidecar
# rather than a per-line manifest field: stamping it on every line would break
# the concurrent-equivalence check that a sequential and a fanned-out manifest
# differ only in line order.
EXECUTION_FILE = "execution.json"
EXECUTION_FORMAT = "shrlm-round-execution/v1"

# How a manifest line's usage figures were arrived at. Stamped on every line
# written from here on, and absent on every line written before.
#
# v1 (implicit, absent key): a resource-terminated run persisted an empty usage
# summary; only a budget termination salvaged a cost, from the exception's own
# ``spent``. Every other termination read back as ``cost: null`` and was priced
# at the per-run ceiling.
# v2: the runtime publishes the total it recorded (``RLM.last_completion_usage``)
# and a terminated run persists that verbatim, so it carries real calls, tokens
# and cost. Recursive sub-call spend is folded into the root's total on every
# run, terminated or not.
#
# The two are not comparable: the same run costs more under v2 than v1, which
# moves ``total_cost``, ``mean_cost``, and the promotion rule's cost band. A
# baseline and the candidates scored against it must be priced under one
# version, so a directory that mixes them is refused rather than averaged.
ACCOUNTING_VERSION = "shrlm-accounting/v2"
ACCOUNTING_VERSION_KEY = "accounting_version"
LEGACY_ACCOUNTING_VERSION = "shrlm-accounting/v1"
# Every version this build knows how to reason about. A line stamped with
# anything else was written by a build from the future: its figures cannot be
# compared with these, and guessing which way they differ is worse than
# refusing, so an unrecognized version is rejected by name.
KNOWN_ACCOUNTING_VERSIONS = frozenset({LEGACY_ACCOUNTING_VERSION, ACCOUNTING_VERSION})
DIGESTS_DIR = "digests"

# Credentials must come from the environment, never from backend_kwargs: the
# kwargs are serialized into every trajectory's run_metadata, and the traces
# are persisted verbatim.
_SENSITIVE_KWARG_FRAGMENTS = ("key", "token", "secret", "password", "authorization")

# Backends whose client reads its credentials from these environment variables.
# Every listed variable must be non-empty before a paid run starts; a backend
# may require more than one (azure_foundry needs both the key and the endpoint).
_BACKEND_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "azure_foundry": ("AZURE_API_KEY", "AZURE_FOUNDRY_ENDPOINT"),
}


class RoundPersistenceError(RuntimeError):
    """Persisted round state contradicts itself or the caller's configuration."""


@dataclass(frozen=True)
class RoundConfig:
    """Everything one round needs: what to run, under what, and where to persist.

    ``max_iterations``, ``max_depth``, ``max_budget``, and ``max_timeout`` are
    experiment-owned RLM limits, forwarded to ``build_harnessed_rlm``; leave
    ``max_depth`` unset when the harness's S6 policy declares it, since the
    runner rejects a double declaration.
    """

    round_index: int
    harness: Harness
    instances: list[dict[str, Any]]
    verifier: Verifier
    out_dir: Path | str
    backend: str = "openrouter"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    max_iterations: int = 30
    max_depth: int | None = None
    max_budget: float | None = None
    max_timeout: float | None = None
    # How many of this round's runs may execute concurrently. 1 is the
    # sequential path. Recorded in the execution sidecar so a reader of the
    # aggregate can tell what conditions produced it -- a subject evaluated
    # while siblings compete for one API key is not under the same conditions
    # as one that ran alone, and that is a confound on measured cost.
    run_workers: int = 1
    # Test-only seam for run children: a dotted path to a factory plus its
    # arguments. A child resolves it and installs the result on the runtime's
    # client seam, which is what in-process tests monkeypatch directly. Live
    # runs leave it None.
    client_factory: tuple[str, dict[str, Any]] | None = None


def run_id_for(instance_id: str, attempt: int) -> str:
    """The stable per-run identifier: ``<instance_id>__aNN``, 1-based attempts."""
    return f"{instance_id}__a{attempt:02d}"


def instance_lines(instances: list[dict[str, Any]]) -> str:
    """The canonical byte content of ``instances.jsonl`` for these instances.

    Public because ``shrlm.experiment.splits`` renders persisted split files
    through it: the driver byte-compares a resumed round's ``instances.jsonl``
    against exactly these bytes, so the split writer and the round writer must
    never be able to drift apart.
    """
    return "".join(json.dumps(instance, sort_keys=True) + "\n" for instance in instances)


def sha256_file(path: Path) -> str:
    """Sha256 hex digest over the file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Fail-fast validation: everything that can be wrong before a run is paid for
# ---------------------------------------------------------------------------


def _validate_config(config: RoundConfig) -> None:
    """Reject a misconfigured round before any model call is made."""
    if config.attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {config.attempts}")
    if not config.instances:
        raise ValueError("a round needs at least one instance")

    seen: set[str] = set()
    for instance in config.instances:
        instance_id = str(instance["id"])
        if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance_id):
            raise ValueError(
                f"instance id {instance_id!r} is not filesystem-safe; ids become trace "
                f"file names and must match {FILESYSTEM_SAFE_ID_PATTERN.pattern}"
            )
        if instance_id in seen:
            raise ValueError(
                f"duplicate instance id {instance_id!r}: run ids are derived from "
                "(instance id, attempt), so repeats would collide in the manifest. "
                "Use attempts to run an instance more than once."
            )
        seen.add(instance_id)
        if "prompt" not in instance:
            raise ValueError(f"instance {instance_id!r} carries no 'prompt' field")

    reject_sensitive_backend_kwargs(config.backend_kwargs)


def reject_sensitive_backend_kwargs(backend_kwargs: dict[str, Any]) -> None:
    """Refuse kwargs that look like credentials: they are persisted verbatim.

    Shared by the round driver and by every caller that writes kwargs to disk
    before a round runs (the validation subject worker's request file).
    """
    for name in backend_kwargs:
        lowered = name.lower()
        if any(fragment in lowered for fragment in _SENSITIVE_KWARG_FRAGMENTS):
            raise ValueError(
                f"backend_kwargs may not carry credential material ({name!r}, e.g. api_key): "
                "kwargs are serialized into every persisted trajectory. Supply credentials "
                "through the environment instead."
            )


def require_backend_credential(config: RoundConfig) -> None:
    """Demand every one of the backend's environment credentials.

    Called only once at least one run remains to execute: a fully persisted
    round must be resumable as a no-op on a machine without the credentials,
    while a round with pending runs still fails fast before any run is paid
    for. The error names each missing variable (and only its name -- never a
    value) so a partially configured environment is diagnosed in one read.
    """
    missing = [
        env_key
        for env_key in _BACKEND_ENV_KEYS.get(config.backend, ())
        if not os.environ.get(env_key)
    ]
    if missing:
        raise RuntimeError(
            f"backend {config.backend!r} requires the {', '.join(missing)} environment "
            f"variable{'s' if len(missing) > 1 else ''}; refusing to start a paid round "
            "that would fail on its first call."
        )


def _write_execution_sidecar(path: Path, run_workers: int) -> None:
    """Record the effective run-worker concurrency this round executed under.

    Written on every round including the sequential one, where the value is 1,
    so a reader never has to distinguish "ran alone" from "written before this
    was recorded". A resume that raises the worker count rewrites it: the
    sidecar describes execution conditions, which legitimately differ between
    the invocation that ran the first half and the one that ran the rest, and
    the aggregate reports the last count rather than pretending to a single
    figure it does not have.
    """
    payload = (
        json.dumps(
            {"format": EXECUTION_FORMAT, "run_workers": int(run_workers)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    # Written through a temp file and renamed: a crash mid-write would
    # otherwise leave truncated JSON that turns the read-only aggregate path
    # into a parse error on a round whose runs are all perfectly fine.
    staged = path / f"{EXECUTION_FILE}.tmp-{os.getpid()}"
    staged.write_text(payload)
    os.replace(staged, path / EXECUTION_FILE)


def read_run_workers(path: Path | str) -> int:
    """The run-worker concurrency a round executed under; 1 when unrecorded.

    Rounds persisted before the sidecar existed were strictly sequential, so
    their absent value and a recorded 1 mean the same thing.
    """
    sidecar = Path(path) / EXECUTION_FILE
    if not sidecar.exists():
        # Written before the sidecar existed, which means strictly sequential.
        # This is the only case where 1 is an inference rather than a reading.
        return 1
    try:
        payload = json.loads(sidecar.read_text())
        recorded = payload["run_workers"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RoundPersistenceError(
            f"{sidecar} exists but cannot be read ({error}); refusing to report this round "
            "as sequential when the conditions it ran under are unknown"
        ) from error
    if payload.get("format") != EXECUTION_FORMAT:
        raise RoundPersistenceError(
            f"{sidecar} is not a {EXECUTION_FORMAT} document; refusing to guess what "
            "concurrency it describes"
        )
    if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 1:
        raise RoundPersistenceError(
            f"{sidecar} records run_workers={recorded!r}, which is not a positive integer"
        )
    return recorded


def _prepare_round_dir(config: RoundConfig) -> Path:
    """Create (or verify) the round directory's identity artifacts.

    On a fresh round this writes ``harness.json`` and ``instances.jsonl``. On a
    resume it verifies both against the caller's configuration: a re-invocation
    under a different harness or instance set would silently mix two rounds'
    evidence, which is exactly what the persisted identity exists to prevent.
    """
    path = round_dir(config.out_dir, config.round_index)
    path.mkdir(parents=True, exist_ok=True)
    (path / TRACES_DIR).mkdir(exist_ok=True)
    _write_execution_sidecar(path, config.run_workers)

    harness_path = path / HARNESS_FILE
    expected_hash = harness_hash(config.harness)
    if harness_path.exists():
        recorded = json.loads(harness_path.read_text())["hash"]
        if recorded != expected_hash:
            raise RoundPersistenceError(
                f"{harness_path} records harness hash {recorded}, but the configured "
                f"harness hashes to {expected_hash}; refusing to mix two harnesses in "
                "one round."
            )
    else:
        write_harness_json(config.harness, harness_path)

    instances_path = path / INSTANCES_FILE
    expected_lines = instance_lines(config.instances)
    if instances_path.exists():
        if instances_path.read_text() != expected_lines:
            raise RoundPersistenceError(
                f"{instances_path} does not match the configured instances; resuming a "
                "round requires the identical instance list, verbatim."
            )
    else:
        instances_path.write_text(expected_lines)

    return path


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    """Parse ``runs.jsonl``, rejecting duplicate run ids."""
    manifest_path = path / MANIFEST_FILE
    if not manifest_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in manifest_path.read_text().splitlines():
        if not raw.strip():
            continue
        entry = json.loads(raw)
        run_id = str(entry["run_id"])
        if run_id in seen:
            raise RoundPersistenceError(f"{manifest_path} lists run id {run_id!r} twice")
        seen.add(run_id)
        entries.append(entry)
    _reject_mixed_accounting(manifest_path, entries)
    return entries


def _accounting_version_of(entry: dict[str, Any]) -> str:
    """The accounting rules one manifest line was written under.

    Lines predating the marker carry no key; they are v1 by construction, and
    saying so explicitly is what lets the mixed-version check compare them.
    """
    return str(entry.get(ACCOUNTING_VERSION_KEY, LEGACY_ACCOUNTING_VERSION))


def manifest_accounting_version(entries: list[dict[str, Any]]) -> str:
    """The single accounting version every one of these lines was priced under.

    An empty manifest is this build's own version: nothing has been priced yet,
    so the first run that lands sets it.
    """
    versions = {_accounting_version_of(entry) for entry in entries}
    return next(iter(versions)) if len(versions) == 1 else ACCOUNTING_VERSION


def _reject_mixed_accounting(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    """Refuse a manifest whose lines cannot be read under one accounting rule.

    Two ways that happens. Lines priced under *different* rules cannot be
    compared with each other at all, and deleting summaries does not fix it
    because the manifest lines are the problem. Lines priced under a rule this
    build does not know are equally unusable -- it cannot say how they differ.

    This is the backstop, not the gate: it fires when the damage already
    exists. ``prepare_round`` refuses a resume that *would* mix versions before
    any run is paid for, which is the check that actually protects money.
    """
    versions = {_accounting_version_of(entry) for entry in entries}
    if len(versions) > 1:
        raise RoundPersistenceError(
            f"{manifest_path} mixes cost-accounting versions {sorted(versions)}. "
            "Runs priced under different accounting rules are not comparable; "
            "use a fresh out-dir rather than resuming this one."
        )
    unknown = versions - KNOWN_ACCOUNTING_VERSIONS
    if unknown:
        raise RoundPersistenceError(
            f"{manifest_path} was priced under unrecognized cost-accounting version "
            f"{sorted(unknown)[0]!r}; this build knows "
            f"{sorted(KNOWN_ACCOUNTING_VERSIONS)}. Refusing to read figures whose "
            "accounting rules are unknown."
        )


def _verify_trace(path: Path, entry: dict[str, Any]) -> Path:
    """Check one manifest entry's trace file exists and matches its sha256."""
    trace_path = path / str(entry["trace_path"])
    if not trace_path.exists():
        raise RoundPersistenceError(
            f"manifest entry {entry['run_id']!r} points at missing trace {trace_path}"
        )
    actual = sha256_file(trace_path)
    if actual != entry["trace_sha256"]:
        raise RoundPersistenceError(
            f"trace {trace_path} sha256 {actual} does not match the manifest's "
            f"{entry['trace_sha256']} for run {entry['run_id']!r}; the file was modified "
            "after it was recorded. Refusing to overwrite -- resolve the discrepancy "
            "before resuming."
        )
    return trace_path


# ---------------------------------------------------------------------------
# The run phase
# ---------------------------------------------------------------------------


def _partial_completion(
    prompt: str | dict[str, Any],
    trajectory: dict[str, Any] | None,
    model_name: str,
    error: Exception,
    elapsed_seconds: float,
    published_usage: UsageSummary | None = None,
) -> RLMChatCompletion:
    """A trace for a run the runtime terminated at a resource limit.

    The trajectory is whatever the in-memory logger held when the root raised.
    Budget/token/error limits are checked before the terminating iteration is
    logged, so that iteration is absent; see the module docstring.

    ``published_usage`` is the total the runtime recorded for the run, handed
    over by ``RLM.last_completion_usage`` before the handler holding it was
    stopped. It is the accurate figure and covers every termination path,
    including a limit raised inside a client, so it is persisted verbatim when
    it records at least one call.

    A published summary with no calls is not evidence of a free run -- a run
    that hung before its request was recorded, or one that never reached the
    runtime at all, produces exactly that -- so it is treated as nothing
    published. In that case ``BudgetExceededError`` still carries the figure it
    tripped on, and that ``spent`` amount is persisted so the validation
    stage's circuit breaker never undercounts a paid termination. The other
    limit exceptions carry no cost of their own, so usage stays empty, token
    counts stay zero, and the breaker prices the run at its per-run ceiling.

    ``usage_lower_bound`` (see ``_persist_run``) stays true for a terminated
    run either way: a request killed in flight, a response with a deficient
    body, and a route that reports no cost are all still genuinely unrecorded.
    The published figure removes the large systematic loss; it does not make
    the number exact, and the flag is what says so.
    ``elapsed_seconds`` is the driver-observed wall clock around the
    terminated completion call (zero when nothing was timed, as in
    ``persist_interrupted_run``). Some limit exceptions carry a partial
    answer; persisting it keeps the trace honest about what the run had
    produced.
    """
    partial_answer = getattr(error, "partial_answer", None)
    spent = getattr(error, "spent", None)
    usage = UsageSummary(model_usage_summaries={})
    if published_usage is not None and published_usage.total_calls > 0:
        usage = published_usage
    elif isinstance(spent, int | float) and not isinstance(spent, bool):
        usage = UsageSummary(
            model_usage_summaries={
                model_name: ModelUsageSummary(
                    total_calls=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_cost=float(spent),
                )
            }
        )
    return RLMChatCompletion(
        root_model=model_name,
        prompt=prompt,
        response=partial_answer or "",
        usage_summary=usage,
        execution_time=elapsed_seconds,
        metadata=trajectory,
        error=f"{type(error).__name__}: {error}",
    )


def _persist_run(
    path: Path,
    run_id: str,
    instance_id: str,
    attempt: int,
    completion: RLMChatCompletion,
    verdict: Verdict,
    *,
    usage_lower_bound: bool = False,
) -> dict[str, Any]:
    """Write the trace file, then append the manifest line. Order matters:
    a crash between the two leaves an orphan trace that the next invocation
    simply overwrites, whereas the reverse order could record a sha for bytes
    that never hit the disk.

    ``usage_lower_bound`` marks a terminated run whose persisted usage figures
    (tokens, time, cost) are lower bounds on what was actually consumed --
    observed values are persisted as-is, and only genuinely unknown fields
    stay zero -- so the most expensive runs are never read back as free (R5).
    """
    trace_rel = f"{TRACES_DIR}/{run_id}.json"
    trace_path = path / trace_rel
    trace_path.write_text(json.dumps(completion.to_dict(), sort_keys=True) + "\n")

    entry = {
        "run_id": run_id,
        "instance_id": instance_id,
        "attempt": attempt,
        "passed": verdict.passed,
        "cause": verdict.cause.value if verdict.cause is not None else None,
        "verdict": verdict.to_dict(),
        "trace_path": trace_rel,
        "trace_sha256": sha256_file(trace_path),
        "cost": completion.usage_summary.total_cost,
        "input_tokens": completion.usage_summary.total_input_tokens,
        "output_tokens": completion.usage_summary.total_output_tokens,
        "execution_time": completion.execution_time,
        "usage_lower_bound": usage_lower_bound,
        ACCOUNTING_VERSION_KEY: ACCOUNTING_VERSION,
        "timestamp": _utc_now(),
    }
    # Cost provenance (R7): additive-only. A client that reports where its cost
    # came from ("provider" vs "synthesized" from token counts x pricing) has
    # that recorded beside the cost; a client that reports nothing changes no
    # persisted byte -- lines without the key keep their exact prior shape.
    cost_source = completion.usage_summary.cost_source
    if cost_source is not None:
        entry["cost_source"] = cost_source
    with open(path / MANIFEST_FILE, "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_child_trace(trace_path: Path) -> RLMChatCompletion | None:
    """Rehydrate a trace a run child wrote, or None when it is unusable.

    A child publishes its trace by atomic rename, so a file that exists is
    complete. Unusable here therefore means a child that died before publishing
    at all, or wrote something that is not a trace -- either way the parent has
    no evidence of what the run produced and must treat it as a lost run rather
    than guess.
    """
    try:
        payload = json.loads(trace_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "response" not in payload:
        return None
    return RLMChatCompletion.from_dict(payload)


def append_child_run(
    path: Path,
    run_id: str,
    instance_id: str,
    attempt: int,
    completion: RLMChatCompletion,
    verdict: Verdict,
    *,
    usage_lower_bound: bool = False,
) -> dict[str, Any]:
    """Record a run whose trace a child already wrote, without rewriting it.

    The sibling of ``_persist_run`` for the concurrent path. The bytes on disk
    are the child's; the parent hashes exactly those bytes rather than
    re-serializing the rehydrated completion, so the recorded sha256 describes
    the file a reader will actually load. Re-serializing could differ in
    whitespace or key order and turn every later verification into a mismatch.

    Only the parent appends to the manifest (KTD4), so this is called on the
    reaping side, after the child has exited.
    """
    trace_rel = f"{TRACES_DIR}/{run_id}.json"
    trace_path = path / trace_rel
    if not trace_path.exists():
        raise RoundPersistenceError(
            f"cannot record run {run_id!r}: its child left no trace at {trace_path}"
        )
    entry = {
        "run_id": run_id,
        "instance_id": instance_id,
        "attempt": attempt,
        "passed": verdict.passed,
        "cause": verdict.cause.value if verdict.cause is not None else None,
        "verdict": verdict.to_dict(),
        "trace_path": trace_rel,
        "trace_sha256": sha256_file(trace_path),
        "cost": completion.usage_summary.total_cost,
        "input_tokens": completion.usage_summary.total_input_tokens,
        "output_tokens": completion.usage_summary.total_output_tokens,
        "execution_time": completion.execution_time,
        "usage_lower_bound": usage_lower_bound,
        ACCOUNTING_VERSION_KEY: ACCOUNTING_VERSION,
        "timestamp": _utc_now(),
    }
    cost_source = completion.usage_summary.cost_source
    if cost_source is not None:
        entry["cost_source"] = cost_source
    with open(path / MANIFEST_FILE, "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def trace_path_for(path: Path, run_id: str) -> Path:
    """Where a run's trace lives inside a prepared round directory."""
    return path / TRACES_DIR / f"{run_id}.json"


def load_manifest(out_dir: Path | str, round_index: int) -> list[dict[str, Any]]:
    """The round's persisted manifest lines, in file order; ``[]`` before any run.

    A read-only view for callers that must inspect persisted state without
    executing anything -- in particular without ``run_round``'s harness build,
    which runs candidate code and is therefore not safe to invoke while
    recovering from a hung candidate call.
    """
    return _load_manifest(round_dir(out_dir, round_index))


def persist_interrupted_run(
    config: RoundConfig, error: Exception, *, run_id: str | None = None
) -> dict[str, Any] | None:
    """Persist an interrupted run as a RESOURCE_TERMINATED run.

    The recovery path for a limit exception that escaped ``run_round``'s
    per-run handler -- e.g. a hard wall-clock interrupt (see
    ``costs.HardDeadlineExceeded``) that fired during harness construction or
    between a run's completion and its persistence. Nothing was recorded for
    the in-flight run, so its terminated manifest line is synthesized here:
    the same ``_partial_completion`` trace and failing verdict a caught limit
    exception persists, except with no trajectory (the in-memory logger was
    lost when the exception escaped).

    Args:
        run_id: The run to recover. With more than one run in flight, "the
            first pending run" names a run that may never have executed, and
            charging it would attribute one run's failure to another (R16), so
            a concurrent caller always says which run it lost. Omitted, the
            first pending run is recovered -- correct when only one run can be
            in flight, which is the sequential path.

    Returns:
        The newly appended manifest entry, or ``None`` when the named run (or,
        by default, every configured run) is already persisted -- so recovering
        a run that landed before the interrupt is a no-op, never a duplicate
        line.
    """
    _validate_config(config)
    path = _prepare_round_dir(config)
    done = {str(entry["run_id"]) for entry in _load_manifest(path)}
    model_name = str(config.backend_kwargs.get("model_name", "unknown"))
    for instance in config.instances:
        for attempt in range(1, config.attempts + 1):
            instance_id = str(instance["id"])
            candidate_run_id = run_id_for(instance_id, attempt)
            if run_id is not None:
                if candidate_run_id != run_id:
                    continue
                if candidate_run_id in done:
                    return None
            elif candidate_run_id in done:
                continue
            completion = _partial_completion(
                prompt=instance["prompt"],
                trajectory=None,
                model_name=model_name,
                error=error,
                elapsed_seconds=0.0,
            )
            verdict = Verdict(
                passed=False,
                cause=VerifierCause.RESOURCE_TERMINATED,
                gold="",
                produced=completion.response,
                detail=f"{type(error).__name__}: {error}",
            )
            return _persist_run(
                path,
                candidate_run_id,
                instance_id,
                attempt,
                completion,
                verdict,
                usage_lower_bound=True,
            )
    if run_id is not None:
        raise RoundPersistenceError(
            f"cannot recover run {run_id!r}: it is not one of this round's configured runs"
        )
    return None


@dataclass(frozen=True)
class RunOutcome:
    """What executing one run produced, before anything is persisted.

    ``verdict`` is None only when the caller supplied no verifier -- a run
    child, which never verifies because the parent owns verdict construction
    (KTD5). A terminated run always carries its own failing verdict, since that
    one is derived from the exception rather than from the answer.
    """

    completion: RLMChatCompletion
    verdict: Verdict | None
    usage_lower_bound: bool


def prepare_round(
    config: RoundConfig,
) -> tuple[Path, list[dict[str, Any]], list[tuple[dict[str, Any], int]]]:
    """Validate the round, verify what is already persisted, and list what is left.

    Extracted from ``run_round`` so the sequential loop and a concurrent
    dispatcher derive their pending work identically -- the pending list is
    instance-major, attempt-minor, and a round that stops early must stop on a
    contiguous tail of it (R7).

    Returns:
        The round directory, every already-persisted manifest entry (each with
        its trace re-verified against the recorded sha256), and the pending
        ``(instance, attempt)`` pairs in dispatch order.
    """
    _validate_config(config)
    path = _prepare_round_dir(config)

    existing = _load_manifest(path)
    for entry in existing:
        _verify_trace(path, entry)
    done = {str(entry["run_id"]) for entry in existing}

    pending = [
        (instance, attempt)
        for instance in config.instances
        for attempt in range(1, config.attempts + 1)
        if run_id_for(str(instance["id"]), attempt) not in done
    ]

    # Refuse before the first run is paid for, not after. A directory written
    # under older accounting reads back fine on its own -- every line agrees --
    # so the mixed-version backstop cannot see the problem until a new line has
    # already landed beside the old ones. By then the operator has paid for a
    # run and the directory is unreadable for good. The moment there is work to
    # execute is the moment to refuse.
    if pending:
        persisted_version = manifest_accounting_version(existing)
        if existing and persisted_version != ACCOUNTING_VERSION:
            raise RoundPersistenceError(
                f"{path} holds runs priced under cost-accounting version "
                f"{persisted_version!r}, but this build prices runs as "
                f"{ACCOUNTING_VERSION!r}. Executing the {len(pending)} pending run(s) "
                "here would leave the round priced two ways and unreadable. Use a "
                "fresh out-dir; the existing directory stays readable as it is."
            )

    return path, existing, pending


def build_round_rlm(config: RoundConfig) -> HarnessedRLM:
    """The harnessed RLM one round's runs execute under.

    The credential is demanded here rather than at validation, so a fully
    persisted round stays resumable on a machine without the key: callers only
    reach this once at least one run must actually execute.
    """
    require_backend_credential(config)

    rlm_kwargs: dict[str, Any] = {"max_iterations": config.max_iterations}
    for name in ("max_depth", "max_budget", "max_timeout"):
        value = getattr(config, name)
        if value is not None:
            rlm_kwargs[name] = value

    harnessed = build_harnessed_rlm(
        config.harness,
        backend=config.backend,
        backend_kwargs=dict(config.backend_kwargs),
        **rlm_kwargs,
    )
    if harnessed.logger is None:  # pragma: no cover - build_harnessed_rlm always attaches one
        raise RuntimeError("harnessed RLM carries no logger; partial traces would be lost")
    return harnessed


def execute_run(
    harnessed: HarnessedRLM,
    instance: dict[str, Any],
    *,
    model_name: str,
    verifier: Verifier | None = None,
) -> RunOutcome:
    """Execute one run and describe its outcome, persisting nothing.

    The single implementation of the timing window, the limit-exception
    handling, the partial-completion path, and the lower-bound flag (KTD14):
    the sequential loop and a run child both call this, so the two cannot drift
    apart and the accounting correction lives in one place.

    ``verifier`` is passed by the in-process caller so verification happens
    *inside* the limit handler -- a hard deadline landing during verification
    is then caught here and persisted as a terminated run, rather than escaping
    the round. A run child omits it: the parent verifies (KTD5).
    """
    prompt = instance["prompt"]
    run_started = time.perf_counter()
    try:
        run = harnessed.completion(prompt)
        completion = run.completion
        verdict = verifier(instance, completion.response) if verifier is not None else None
        return RunOutcome(completion=completion, verdict=verdict, usage_lower_bound=False)
    except ROOT_LIMIT_EXCEPTIONS as error:
        completion = _partial_completion(
            prompt=prompt,
            trajectory=harnessed.logger.get_trajectory(),
            model_name=model_name,
            error=error,
            elapsed_seconds=time.perf_counter() - run_started,
            published_usage=harnessed.rlm.last_completion_usage,
        )
        return RunOutcome(
            completion=completion,
            verdict=Verdict(
                passed=False,
                cause=VerifierCause.RESOURCE_TERMINATED,
                gold="",
                produced=completion.response,
                detail=f"{type(error).__name__}: {error}",
            ),
            usage_lower_bound=True,
        )
    except Exception as error:
        # A content filter that survived the client's CONTENT_FILTER_ATTEMPTS
        # retries is deterministic for this prompt, so re-running it will never
        # succeed. Containing it here makes mining behave like validation, where
        # the same provider refusal has always been a recorded run failure
        # rather than a fatal error: a mining run in the ORCHESTRATOR's own
        # process previously took the whole experiment down with it (observed
        # 2026-08-26, round 3, 47/48 runs in). Any other 400 is a real bug and
        # still propagates.
        # Imported HERE, not at module scope: run_worker children import this
        # module on every spawn, and pulling the OpenAI SDK into that path cost
        # 0.06s -> 0.23s per child (measured 2026-08-26) -- enough to make a
        # child miss a tight deadline. By the time this handler runs the SDK is
        # already imported by the client that raised, so the cost here is nil.
        import openai

        from rlm.clients.openai import is_content_filter_error

        if not isinstance(error, openai.BadRequestError) or not is_content_filter_error(error):
            raise
        completion = _partial_completion(
            prompt=prompt,
            trajectory=harnessed.logger.get_trajectory(),
            model_name=model_name,
            error=error,
            elapsed_seconds=time.perf_counter() - run_started,
            published_usage=harnessed.rlm.last_completion_usage,
        )
        return RunOutcome(
            completion=completion,
            verdict=Verdict(
                passed=False,
                cause=VerifierCause.CONTENT_FILTERED,
                gold="",
                produced=completion.response,
                detail=f"{type(error).__name__}: {error}",
            ),
            usage_lower_bound=True,
        )


def run_round(config: RoundConfig, *, stop_after: int | None = None) -> list[dict[str, Any]]:
    """Execute a round's runs under the configured harness, persist-first.

    Idempotent over ``out_dir``: run ids already in the manifest are skipped
    after their trace files are re-verified against the recorded sha256s, so a
    crashed round resumes by re-invoking with the same configuration and only
    the missing runs are paid for again.

    Args:
        config: The round to run.
        stop_after: Execute at most this many *new* runs, then return. A
            deliberate kill switch for budget-capped sessions (and for tests
            that simulate a crash); the runs already executed stay persisted.

    Returns:
        Every manifest entry for the round, pre-existing and new, in file order.

    Raises:
        ValueError: On a misconfigured round (bad attempts, duplicate or unsafe
            instance ids, credential material in backend_kwargs).
        RuntimeError: When any of the backend's environment credentials is
            missing and at least one run remains to execute; the message names
            each missing variable. A fully persisted round needs no
            credential: it resumes as a no-op straight from the manifest.
        RoundPersistenceError: When persisted state contradicts the
            configuration or a recorded trace no longer matches its sha256.
    """
    path, existing, pending = prepare_round(config)
    entries = list(existing)
    if not pending:
        return entries

    harnessed = build_round_rlm(config)
    model_name = str(config.backend_kwargs.get("model_name", "unknown"))

    executed = 0
    for instance, attempt in pending:
        if stop_after is not None and executed >= stop_after:
            return entries
        instance_id = str(instance["id"])
        outcome = execute_run(
            harnessed,
            instance,
            model_name=model_name,
            verifier=config.verifier,
        )
        # ``config.verifier`` is mandatory, so the in-process path always gets a
        # verdict back; only a run child omits the verifier, and it never
        # persists. Narrow explicitly rather than leave the guarantee implicit.
        assert outcome.verdict is not None
        entries.append(
            _persist_run(
                path,
                run_id_for(instance_id, attempt),
                instance_id,
                attempt,
                outcome.completion,
                outcome.verdict,
                usage_lower_bound=outcome.usage_lower_bound,
            )
        )
        executed += 1

    return entries


# ---------------------------------------------------------------------------
# The mining phase: disk in, evidence bundle out
# ---------------------------------------------------------------------------


def load_round(
    out_dir: Path | str, round_index: int
) -> tuple[
    list[tuple[dict[str, Any], RLMChatCompletion]],
    list[Verdict],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Read a persisted round back: (instance, completion) pairs, aligned
    verdicts, the harness envelope, and the aligned manifest entries (the
    source of each run's run_id / trace_path / trace_sha256). Every trace is
    sha-verified before it is trusted, so mining cannot silently consume a
    modified file."""
    path = round_dir(out_dir, round_index)
    envelope = json.loads((path / HARNESS_FILE).read_text())
    instances = {
        str(instance["id"]): instance
        for line in (path / INSTANCES_FILE).read_text().splitlines()
        if line.strip()
        for instance in [json.loads(line)]
    }

    runs: list[tuple[dict[str, Any], RLMChatCompletion]] = []
    verdicts: list[Verdict] = []
    entries: list[dict[str, Any]] = []
    for entry in _load_manifest(path):
        trace_path = _verify_trace(path, entry)
        instance_id = str(entry["instance_id"])
        if instance_id not in instances:
            raise RoundPersistenceError(
                f"manifest run {entry['run_id']!r} references instance {instance_id!r}, "
                f"which {INSTANCES_FILE} does not contain"
            )
        completion = RLMChatCompletion.from_dict(json.loads(trace_path.read_text()))
        runs.append((instances[instance_id], completion))
        verdicts.append(Verdict.from_dict(entry["verdict"]))
        entries.append(entry)
    return runs, verdicts, envelope, entries


def _sampling_seed(runs: list[tuple[dict[str, Any], RLMChatCompletion]]) -> int | None:
    """The dataset sampling seed the instances carry as provenance.

    Environment loaders (see ``graphwalks.row_to_instance``) stamp each
    sampled instance with its ``sample_seed``. One round samples once, so a
    mix of seeds means the instance list was assembled from two samplings --
    that is a provenance contradiction, not something to average away.
    """
    seeds = {int(instance["sample_seed"]) for instance, _ in runs if "sample_seed" in instance}
    if not seeds:
        return None
    if len(seeds) > 1:
        raise RoundPersistenceError(
            f"instances carry conflicting sample_seed values {sorted(seeds)}; a round's "
            "instances must come from a single sampling."
        )
    return seeds.pop()


def mine_round(
    out_dir: Path | str,
    round_index: int,
    miner: WeaknessMiner,
    split_id: str,
    harness_version: str | None = None,
    created_at: str | None = None,
) -> MiningResult:
    """Mine a persisted round from disk alone.

    Args:
        out_dir: The experiment directory ``run_round`` persisted into.
        round_index: Which round to mine.
        miner: The configured ``WeaknessMiner``. Its verifier is not consulted:
            the persisted verdicts are replayed, which is how RESOURCE_TERMINATED
            runs flow into mining without the Verifier protocol ever seeing an
            exception.
        split_id: The evaluation split identifier recorded in the bundle.
        harness_version: Bundle-facing harness identifier; defaults to the
            content hash recorded in the round's ``harness.json`` (which is
            always recorded separately as ``MiningConfig.harness_hash``).
        created_at: Optional bundle timestamp override.

    Returns:
        The ``MiningResult`` over exactly the runs the manifest records.

    Besides mining, this persists what the attributor saw: each failure
    record's digest text lands in ``digests/<digest_sha256>.txt`` (content
    addressed, so the file's own sha256 is the record's ``digest_sha256``),
    and every rendered attributor system prompt variant lands in
    ``attributor_prompt_<sha16>.txt`` (content addressed by the rendered
    prompt's sha256, resolvable through the ``prompt_sha256`` each
    attributions.jsonl entry carries; existing files with matching bytes are
    never rewritten, so a later mining pass cannot invalidate an earlier
    bundle's prompt link, while mismatched bytes are healed atomically).

    Provenance flows from the round's artifacts into the bundle: every
    failure record is stamped with its manifest run_id / trace_path /
    trace_sha256, and the MiningConfig carries the harness hash, the
    instances' sampling seed, and the attribution cache path relative to the
    round directory.
    """
    runs, verdicts, envelope, entries = load_round(out_dir, round_index)
    path = round_dir(out_dir, round_index)
    trace_links = [
        RunTraceLink(
            run_id=str(entry["run_id"]),
            trace_path=str(entry["trace_path"]),
            trace_sha256=str(entry["trace_sha256"]),
        )
        for entry in entries
    ]
    cache_path = miner.attributor.cache.path
    result = miner.mine(
        runs,
        round_index=round_index,
        harness_version=harness_version or str(envelope["hash"]),
        split_id=split_id,
        created_at=created_at,
        verdicts=verdicts,
        trace_links=trace_links,
        harness_hash=str(envelope["hash"]),
        sampling_seed=_sampling_seed(runs),
        attribution_cache_path=(
            os.path.relpath(cache_path, path) if cache_path is not None else None
        ),
    )
    _persist_mining_artifacts(path, result)
    return result


def _write_content_addressed(target: Path, text: str) -> None:
    """Write ``text`` to its content-addressed name: atomic and self-healing.

    ``text`` is the file's whole identity -- the name is (derived from) the
    sha256 of exactly these bytes -- so an existing file is trusted only after
    its bytes are compared: matching bytes are left untouched (write-once for
    intact artifacts, so an earlier bundle's audit link is never invalidated),
    while anything else at the name (a truncated crash write, external
    tampering) is healed by rewriting the known-correct bytes. The write goes
    through a same-directory tmp file and ``os.replace``, so a crash mid-write
    can never leave partial bytes at a content-addressed name.
    """
    data = text.encode("utf-8")
    if target.exists() and target.read_bytes() == data:
        return
    tmp_path = target.with_name(target.name + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, target)


def _persist_mining_artifacts(path: Path, result: MiningResult) -> None:
    """Write the digests and rendered prompt(s) a mining pass was built on.

    Both artifacts are persisted content-addressed: each digest lands as
    ``digests/<digest_sha256>.txt`` and every rendered prompt variant as
    ``attributor_prompt_<sha16>.txt`` (the first 16 hex digits of the prompt
    sha256 it is keyed by). A file whose bytes already match its address is
    never rewritten -- which is what lets one round be mined more than once,
    e.g. the sub-verification ablation's grounded and ablated passes, with
    each pass adding its own variants and no pass ever invalidating the
    ``attribution_prompt`` audit link of a bundle persisted earlier. A file
    whose bytes do NOT match its address is healed rather than trusted (see
    ``_write_content_addressed``): trusting it would leave the audit's hash
    link permanently broken. Legacy rounds that hold only the unsuffixed
    ``attributor_prompt.txt`` keep resolving through the audit's fallback
    (see ``audit._resolve_prompt_file``).
    """
    if result.digest_texts:
        digests_dir = path / DIGESTS_DIR
        digests_dir.mkdir(parents=True, exist_ok=True)
        for sha, text in result.digest_texts.items():
            _write_content_addressed(digests_dir / f"{sha}.txt", text)

    for sha, text in result.attributor_prompts.items():
        _write_content_addressed(path / f"attributor_prompt_{sha[:16]}.txt", text)


__all__ = [
    "ROOT_LIMIT_EXCEPTIONS",
    "RoundConfig",
    "RoundPersistenceError",
    "instance_lines",
    "load_manifest",
    "load_round",
    "mine_round",
    "persist_interrupted_run",
    "round_dir",
    "run_id_for",
    "run_round",
    "sha256_file",
]
