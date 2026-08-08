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
(``usage_summary.total_cost`` when the backend reported one), and
``timestamp`` (UTC ISO-8601).

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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlm.core.types import RLMChatCompletion, UsageSummary
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
from shrlm.runner import build_harnessed_rlm

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
DIGESTS_DIR = "digests"

# Credentials must come from the environment, never from backend_kwargs: the
# kwargs are serialized into every trajectory's run_metadata, and the traces
# are persisted verbatim.
_SENSITIVE_KWARG_FRAGMENTS = ("key", "token", "secret", "password", "authorization")

# Backends whose client reads its credential from this environment variable.
_BACKEND_ENV_KEYS: dict[str, str] = {"openrouter": "OPENROUTER_API_KEY"}


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


def run_id_for(instance_id: str, attempt: int) -> str:
    """The stable per-run identifier: ``<instance_id>__aNN``, 1-based attempts."""
    return f"{instance_id}__a{attempt:02d}"


def _instance_lines(instances: list[dict[str, Any]]) -> str:
    """The canonical byte content of ``instances.jsonl`` for these instances."""
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

    for name in config.backend_kwargs:
        lowered = name.lower()
        if any(fragment in lowered for fragment in _SENSITIVE_KWARG_FRAGMENTS):
            raise ValueError(
                f"backend_kwargs may not carry credential material ({name!r}, e.g. api_key): "
                "kwargs are serialized into every persisted trajectory. Supply credentials "
                "through the environment instead."
            )


def _require_backend_credential(config: RoundConfig) -> None:
    """Demand the backend's environment credential.

    Called only once at least one run remains to execute: a fully persisted
    round must be resumable as a no-op on a machine without the credential,
    while a round with pending runs still fails fast before any run is paid
    for.
    """
    env_key = _BACKEND_ENV_KEYS.get(config.backend)
    if env_key is not None and not os.environ.get(env_key):
        raise RuntimeError(
            f"backend {config.backend!r} requires the {env_key} environment variable; "
            "refusing to start a paid round that would fail on its first call."
        )


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
    expected_lines = _instance_lines(config.instances)
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
    return entries


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
) -> RLMChatCompletion:
    """A trace for a run the runtime terminated at a resource limit.

    The trajectory is whatever the in-memory logger held when the root raised.
    Budget/token/error limits are checked before the terminating iteration is
    logged, so that iteration is absent; see the module docstring. Usage is
    empty because the per-completion handler that held it is gone by the time
    the exception surfaces. Some limit exceptions carry a partial answer;
    persisting it keeps the trace honest about what the run had produced.
    """
    partial_answer = getattr(error, "partial_answer", None)
    return RLMChatCompletion(
        root_model=model_name,
        prompt=prompt,
        response=partial_answer or "",
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.0,
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
) -> dict[str, Any]:
    """Write the trace file, then append the manifest line. Order matters:
    a crash between the two leaves an orphan trace that the next invocation
    simply overwrites, whereas the reverse order could record a sha for bytes
    that never hit the disk."""
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
        "timestamp": _utc_now(),
    }
    with open(path / MANIFEST_FILE, "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


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
        RuntimeError: When the backend's environment credential is missing and
            at least one run remains to execute. A fully persisted round needs
            no credential: it resumes as a no-op straight from the manifest.
        RoundPersistenceError: When persisted state contradicts the
            configuration or a recorded trace no longer matches its sha256.
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
    entries = list(existing)
    if not pending:
        return entries

    # The credential is demanded only now that a run must actually execute,
    # so a complete round stays resumable on a machine without the key.
    _require_backend_credential(config)

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
    model_name = str(config.backend_kwargs.get("model_name", "unknown"))

    executed = 0
    for instance, attempt in pending:
        if stop_after is not None and executed >= stop_after:
            return entries
        instance_id = str(instance["id"])
        run_id = run_id_for(instance_id, attempt)

        prompt = instance["prompt"]
        try:
            run = harnessed.completion(prompt)
            completion = run.completion
            verdict = config.verifier(instance, completion.response)
        except ROOT_LIMIT_EXCEPTIONS as error:
            completion = _partial_completion(
                prompt=prompt,
                trajectory=harnessed.logger.get_trajectory(),
                model_name=model_name,
                error=error,
            )
            verdict = Verdict(
                passed=False,
                cause=VerifierCause.RESOURCE_TERMINATED,
                gold="",
                produced=completion.response,
                detail=f"{type(error).__name__}: {error}",
            )

        entries.append(_persist_run(path, run_id, instance_id, attempt, completion, verdict))
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
    "load_round",
    "mine_round",
    "round_dir",
    "run_id_for",
    "run_round",
    "sha256_file",
]
