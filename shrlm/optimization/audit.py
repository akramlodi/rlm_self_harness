"""Audit walk over a completed round: prove the evidence chain link by link.

A mined round claims a chain of custody: every pattern in ``bundle.json``
names failure records, every record names a persisted run, every run names a
trace file whose bytes hash to the recorded sha256, every attribution names
the digest and the rendered prompt the attributor actually saw, and the whole
round names the exact harness it measured. ``audit_round`` walks that chain
from disk alone and verifies each link independently, so deleting or
corrupting any single artifact breaks a *named* link rather than silently
weakening the evidence.

The links, in walk order:

* ``harness_file`` -- ``harness.json`` exists and parses as an identity
  envelope.
* ``harness_hash`` -- the envelope's ``hash`` recomputes from its *stored*
  ``harness`` serialization (sha256 over canonical JSON minus the
  informational ``name``, mirroring ``shrlm.harness_identity.harness_hash``
  without needing a live ``Harness`` object).
* ``instances_file`` -- ``instances.jsonl`` exists and parses.
* ``manifest_file`` -- ``runs.jsonl`` exists, parses, and repeats no run id;
  the pass rate is recomputable from its ``passed`` fields alone.
* ``manifest_trace`` -- every manifest line's trace file exists and its bytes
  hash to the recorded ``trace_sha256`` (passing runs included: they have no
  failure record, so this is their only trace check).
* ``bundle_file`` -- ``bundle.json`` exists and parses.
* ``bundle_id`` -- the recorded id recomputes through the same
  ``compute_bundle_id`` from the bundle's own config plus the instance ids in
  ``records.jsonl``.
* ``bundle_harness_hash`` -- ``config.harness_hash`` equals the recomputed
  envelope hash.
* ``attribution_cache`` -- ``config.attribution_cache_path``, when set,
  resolves (relative to the round directory) to an existing file.
* ``records_file`` -- ``records.jsonl`` exists and parses.
* ``record_run`` / ``record_trace`` / ``record_instance`` /
  ``record_digest`` -- every failure record joins to a manifest run, a
  sha-verified trace file (matching both the record's and the manifest's
  sha256), an instance in ``instances.jsonl``, and a content-addressed
  ``digests/<sha256>.txt`` whose bytes hash to its name.
* ``pattern_instances`` -- every pattern's ``instance_ids`` and
  ``representatives`` resolve to records in ``records.jsonl``.
* ``attributions_file`` / ``attribution_digest`` / ``attribution_prompt`` /
  ``attribution_attempts`` -- every ``attributions.jsonl`` entry resolves its
  digest and its rendered prompt (content-addressed
  ``attributor_prompt_<sha16>.txt``, or the plain ``attributor_prompt.txt``
  a legacy round persisted; either way the file's bytes must hash to the
  entry's ``prompt_sha256``), and every unattributed entry carries a non-empty
  per-attempt audit trail whose rejected attempts name their violations.
  Entries whose ``attribution_error_kind`` is ``transport``,
  ``content_filtered``, or ``token_limit`` (or, for legacy entries, an
  ``error`` prefixed ``transport failure:``) are exempt from the non-empty
  demand: none produced a response to record.

Grounding coverage, reported in the summary stats, is derived from
``records.jsonl`` alone: the fraction of failure records whose
``sub_verdicts`` map holds at least one non-null value -- i.e. at least one
child whose sub-problem a sub-verifier actually checked. This is narrower
than ``level_grounded`` on purpose: ``level_grounded`` is also True for
NO_RECURSION trees, where the level is read off the tree structure with no
child verdict involved.

``run_audited_round`` composes the whole pipeline -- ``run_round`` ->
``mine_round`` -> ``write_bundle`` -> ``audit_round`` -- into one call so a
live capstone script stays trivial.

One round can hold more than one bundle. The default layout keeps the triplet
(``bundle.json``, ``records.jsonl``, ``attributions.jsonl``) at the round
root; a ``bundle_label`` puts a second mode's triplet under
``round_NN/bundles/<label>/`` (see ``bundle_dir_for``), which is what lets
the sub-verification ablation persist its grounded and ablated bundles side
by side, each auditable against the same shared round artifacts.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shrlm.harness_identity import hash_of_serialization
from shrlm.optimization.bundle import (
    ATTRIBUTIONS_FILENAME,
    BUNDLE_FILENAME,
    BUNDLES_DIR,
    RECORDS_FILENAME,
    bundle_dir_for,
    compute_bundle_id,
    write_bundle,
)
from shrlm.optimization.driver import (
    DIGESTS_DIR,
    HARNESS_FILE,
    INSTANCES_FILE,
    MANIFEST_FILE,
    RoundConfig,
    mine_round,
    round_dir,
    run_round,
    sha256_file,
)
from shrlm.optimization.mining import MiningResult, WeaknessMiner
from shrlm.optimization.types import AttributionErrorKind, FailureRecord, MiningConfig

# Every link the walk can report, in walk order. The report always lists all
# of them, so a link that checked nothing is visibly "0 checked" rather than
# silently absent.
LINK_ORDER: tuple[str, ...] = (
    "harness_file",
    "harness_hash",
    "instances_file",
    "manifest_file",
    "manifest_trace",
    "bundle_file",
    "bundle_id",
    "bundle_harness_hash",
    "attribution_cache",
    "records_file",
    "record_run",
    "record_trace",
    "record_instance",
    "record_digest",
    "pattern_instances",
    "attributions_file",
    "attribution_digest",
    "attribution_prompt",
    "attribution_attempts",
)

_TRANSPORT_ERROR_PREFIX = "transport failure"


@dataclass(frozen=True)
class BrokenLink:
    """One verified-and-failed join: which link, for what, and why."""

    link: str
    subject: str
    message: str


@dataclass(frozen=True)
class LinkResult:
    """How one named link fared across everything it was checked against."""

    link: str
    checked: int
    failures: int


@dataclass(frozen=True)
class AuditStats:
    """Round-level numbers recomputed from the persisted artifacts alone.

    ``pass_rate`` is None when the manifest holds no runs; ``grounding_coverage``
    is None when there are no failure records (see the module docstring for its
    derivation from ``sub_verdicts``).

    ``n_unattributed`` and ``n_transport_errors`` are derived from
    ``records.jsonl`` (the typed ``attribution_error_kind`` field, with the
    legacy ``transport failure`` prefix fallback). Unattributed records are
    legitimate evidence and never flip the audit to broken -- but a round
    mined with the attributor LM completely down would otherwise present as a
    clean zero-signature bundle, and these counts make that visible.
    """

    n_runs: int
    n_passed: int
    pass_rate: float | None
    n_records: int
    n_unattributed: int
    n_transport_errors: int
    n_patterns: int
    grounding_coverage: float | None


@dataclass(frozen=True)
class AuditReport:
    """The outcome of one audit walk: per-link results plus summary stats."""

    round_path: str
    ok: bool
    links: tuple[LinkResult, ...]
    broken: tuple[BrokenLink, ...]
    stats: AuditStats

    def broken_link_names(self) -> set[str]:
        return {broken.link for broken in self.broken}


@dataclass(frozen=True)
class _InstanceRef:
    """The one field ``compute_bundle_id`` reads from a record."""

    instance_id: str


class _Walk:
    """Mutable collector for one audit walk: counts per link, failures overall."""

    def __init__(self) -> None:
        self.checked: dict[str, int] = {link: 0 for link in LINK_ORDER}
        self.failures: dict[str, int] = {link: 0 for link in LINK_ORDER}
        self.broken: list[BrokenLink] = []

    def check(self, link: str) -> None:
        self.checked[link] += 1

    def fail(self, link: str, subject: str, message: str) -> None:
        self.failures[link] += 1
        self.broken.append(BrokenLink(link=link, subject=subject, message=message))

    def results(self) -> tuple[LinkResult, ...]:
        return tuple(
            LinkResult(link=link, checked=self.checked[link], failures=self.failures[link])
            for link in LINK_ORDER
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _is_transport_record(record: dict[str, Any]) -> bool:
    """Whether a persisted record's attribution failed at the transport level.

    Reads the typed ``attribution_error_kind`` field, falling back to the
    ``transport failure`` prefix of ``attribution_error`` for legacy records
    that predate the field (mirroring ``bundle._is_transport_error``).
    """
    kind = record.get("attribution_error_kind")
    if kind is not None:
        return kind == AttributionErrorKind.TRANSPORT.value
    return str(record.get("attribution_error", "")).startswith(_TRANSPORT_ERROR_PREFIX)


def stored_harness_hash(envelope: dict[str, Any]) -> str:
    """Recompute the harness hash from the envelope's *stored* serialization.

    The audit-side entrypoint for hashing a persisted envelope dict: it needs
    no live ``Harness`` object. Delegates to
    ``shrlm.harness_identity.hash_of_serialization`` (canonical JSON sha256
    over the serialization minus the informational ``name``), so the audit can
    never drift from the hash that minted the envelope.
    """
    return hash_of_serialization(envelope["harness"])


# ---------------------------------------------------------------------------
# Per-file loaders: each marks its own "<file>" link, returning None on a break
# ---------------------------------------------------------------------------


def _load_json_file(path: Path, link: str, walk: _Walk) -> dict[str, Any] | None:
    walk.check(link)
    if not path.is_file():
        walk.fail(link, path.name, f"{path} is missing")
        return None
    try:
        return cast("dict[str, Any]", json.loads(path.read_text()))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        walk.fail(link, path.name, f"{path} does not parse: {error}")
        return None


def _load_jsonl_file(path: Path, link: str, walk: _Walk) -> list[dict[str, Any]] | None:
    walk.check(link)
    if not path.is_file():
        walk.fail(link, path.name, f"{path} is missing")
        return None
    try:
        return _read_jsonl(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        walk.fail(link, path.name, f"{path} does not parse: {error}")
        return None


def _audit_harness(path: Path, walk: _Walk) -> str | None:
    """Verify the harness envelope; return its recomputed hash, or None."""
    envelope = _load_json_file(path / HARNESS_FILE, "harness_file", walk)
    if envelope is None:
        return None
    walk.check("harness_hash")
    try:
        recomputed = stored_harness_hash(envelope)
    except (KeyError, TypeError) as error:
        walk.fail("harness_hash", HARNESS_FILE, f"envelope carries no harness dict: {error}")
        return None
    recorded = str(envelope.get("hash", ""))
    if recomputed != recorded:
        walk.fail(
            "harness_hash",
            HARNESS_FILE,
            f"stored serialization hashes to {recomputed}, envelope records {recorded}",
        )
        return None
    return recomputed


def _audit_manifest(path: Path, walk: _Walk) -> list[dict[str, Any]]:
    """Verify the manifest parses, repeats no run id, and every trace matches.

    A line missing a required key is a broken link naming the line, never an
    exception: the audit's job is to report a damaged chain, not crash on it.
    """
    entries = _load_jsonl_file(path / MANIFEST_FILE, "manifest_file", walk)
    if entries is None:
        return []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        raw_run_id = entry.get("run_id")
        subject = str(raw_run_id) if raw_run_id is not None else f"line {index + 1}"
        if raw_run_id is None:
            walk.fail(
                "manifest_file", subject, f"{MANIFEST_FILE} line {index + 1} carries no run_id"
            )
        else:
            run_id = str(raw_run_id)
            if run_id in seen:
                walk.fail("manifest_file", run_id, f"{MANIFEST_FILE} lists run id {run_id!r} twice")
            seen.add(run_id)

        walk.check("manifest_trace")
        trace_rel = entry.get("trace_path")
        if trace_rel is None:
            walk.fail(
                "manifest_trace", subject, f"{MANIFEST_FILE} line {index + 1} carries no trace_path"
            )
            continue
        trace_path = path / str(trace_rel)
        if not trace_path.is_file():
            walk.fail("manifest_trace", subject, f"trace file {trace_rel} is missing")
            continue
        actual = sha256_file(trace_path)
        recorded_sha = entry.get("trace_sha256")
        if recorded_sha is None:
            walk.fail(
                "manifest_trace",
                subject,
                f"{MANIFEST_FILE} line {index + 1} carries no trace_sha256",
            )
        elif actual != str(recorded_sha):
            walk.fail(
                "manifest_trace",
                subject,
                f"trace {trace_rel} hashes to {actual}, manifest records {recorded_sha}",
            )
    return entries


def _check_digest(path: Path, sha: str, link: str, subject: str, walk: _Walk) -> None:
    """One content-addressed digest join: ``digests/<sha>.txt`` hashes to sha."""
    walk.check(link)
    if not sha:
        walk.fail(link, subject, "carries no digest_sha256")
        return
    digest_path = path / DIGESTS_DIR / f"{sha}.txt"
    if not digest_path.is_file():
        walk.fail(link, subject, f"digest file {DIGESTS_DIR}/{sha}.txt is missing")
        return
    actual = sha256_file(digest_path)
    if actual != sha:
        walk.fail(link, subject, f"digest file {DIGESTS_DIR}/{sha}.txt hashes to {actual}")


def _audit_records(
    path: Path,
    records: list[dict[str, Any]],
    manifest_by_run_id: dict[str, dict[str, Any]],
    instance_ids: set[str],
    walk: _Walk,
) -> None:
    for record in records:
        subject = str(record.get("instance_id", "?"))

        walk.check("record_run")
        run_id = record.get("run_id")
        if run_id is None:
            walk.fail("record_run", subject, "record carries no run_id (pre-provenance record)")
        elif str(run_id) not in manifest_by_run_id:
            walk.fail("record_run", subject, f"run_id {run_id!r} is not in {MANIFEST_FILE}")

        walk.check("record_trace")
        trace_rel = record.get("trace_path")
        if trace_rel is None:
            subject_id = str(run_id) if run_id is not None else subject
            walk.fail("record_trace", subject_id, "record carries no trace_path")
        else:
            trace_path = path / str(trace_rel)
            if not trace_path.is_file():
                walk.fail("record_trace", str(run_id), f"trace file {trace_rel} is missing")
            else:
                actual = sha256_file(trace_path)
                if actual != str(record.get("trace_sha256")):
                    walk.fail(
                        "record_trace",
                        str(run_id),
                        f"trace {trace_rel} hashes to {actual}, record says "
                        f"{record.get('trace_sha256')}",
                    )
                manifest_entry = manifest_by_run_id.get(str(run_id))
                manifest_sha = (
                    manifest_entry.get("trace_sha256") if manifest_entry is not None else None
                )
                if manifest_sha is not None and actual != str(manifest_sha):
                    walk.fail(
                        "record_trace",
                        str(run_id),
                        f"trace {trace_rel} hashes to {actual}, manifest says {manifest_sha}",
                    )

        walk.check("record_instance")
        if subject not in instance_ids:
            walk.fail("record_instance", subject, f"instance is not in {INSTANCES_FILE}")

        _check_digest(path, str(record.get("digest_sha256", "")), "record_digest", subject, walk)


def _audit_bundle(
    path: Path,
    bundle: dict[str, Any],
    records: list[dict[str, Any]] | None,
    envelope_hash: str | None,
    walk: _Walk,
) -> None:
    bundle_id = str(bundle.get("bundle_id", "?"))

    # bundle_id recomputes from the bundle's own config plus the record
    # instance ids -- through the same compute_bundle_id that minted it. The
    # check needs records.jsonl; when that file is broken, records_file has
    # already named the break and this link stays unchecked.
    if records is not None:
        walk.check("bundle_id")
        try:
            config = MiningConfig(**bundle["config"])
        except (KeyError, TypeError) as error:
            walk.fail("bundle_id", bundle_id, f"config does not reconstruct: {error}")
        else:
            # A record missing instance_id contributes "" here: the recompute
            # then visibly diverges (and record_instance names the record)
            # instead of the walk crashing on the malformed line.
            refs = [_InstanceRef(str(record.get("instance_id", ""))) for record in records]
            recomputed = compute_bundle_id(config, cast("list[FailureRecord]", refs))
            if recomputed != bundle_id:
                walk.fail(
                    "bundle_id",
                    bundle_id,
                    f"recomputes to {recomputed} from config + {RECORDS_FILENAME} instance ids",
                )

    if envelope_hash is not None:
        walk.check("bundle_harness_hash")
        recorded = str(bundle.get("config", {}).get("harness_hash", ""))
        if recorded != envelope_hash:
            walk.fail(
                "bundle_harness_hash",
                bundle_id,
                f"config.harness_hash {recorded!r} does not match the envelope hash "
                f"{envelope_hash}",
            )

    cache_rel = bundle.get("config", {}).get("attribution_cache_path")
    if cache_rel is not None:
        walk.check("attribution_cache")
        if not (path / str(cache_rel)).is_file():
            walk.fail(
                "attribution_cache",
                bundle_id,
                f"config.attribution_cache_path {cache_rel!r} resolves to no file",
            )

    if records is not None:
        record_ids = {str(record.get("instance_id", "")) for record in records}
        for pattern in bundle.get("patterns", []):
            walk.check("pattern_instances")
            signature = json.dumps(pattern.get("signature", {}), sort_keys=True)
            referenced = list(pattern.get("instance_ids", [])) + list(
                pattern.get("representatives", [])
            )
            for instance_id in referenced:
                if str(instance_id) not in record_ids:
                    walk.fail(
                        "pattern_instances",
                        signature,
                        f"pattern references instance {instance_id!r}, which has no record "
                        f"in {RECORDS_FILENAME}",
                    )


# The legacy prompt file name. New mines persist only content-addressed
# ``attributor_prompt_<sha16>.txt`` files; this name survives for the
# fallback in ``_resolve_prompt_file`` over rounds mined before prompt
# persistence was content-addressed.
ATTRIBUTOR_PROMPT_FILE = "attributor_prompt.txt"


def _resolve_prompt_file(path: Path, prompt_sha: str) -> Path | None:
    """Find the persisted prompt file whose bytes hash to ``prompt_sha``.

    ``mine_round`` persists every rendered variant content-addressed as
    ``attributor_prompt_<sha16>.txt``; legacy rounds mined before that hold a
    single plain ``attributor_prompt.txt``, which stays acceptable as the
    fallback. Either name resolves as long as the content hashes to the
    entry's recorded sha.
    """
    candidates = (
        path / f"attributor_prompt_{prompt_sha[:16]}.txt",
        path / ATTRIBUTOR_PROMPT_FILE,
    )
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == prompt_sha:
            return candidate
    return None


def _audit_attributions(path: Path, entries: list[dict[str, Any]], walk: _Walk) -> None:
    for entry in entries:
        subject = str(entry.get("instance_id", "?"))

        _check_digest(
            path, str(entry.get("digest_sha256", "")), "attribution_digest", subject, walk
        )

        walk.check("attribution_prompt")
        prompt_sha = str(entry.get("prompt_sha256", ""))
        if not prompt_sha or _resolve_prompt_file(path, prompt_sha) is None:
            walk.fail(
                "attribution_prompt",
                subject,
                f"no persisted attributor prompt hashes to {prompt_sha or '<empty>'} "
                f"(looked for attributor_prompt_{prompt_sha[:16]}.txt and "
                f"{ATTRIBUTOR_PROMPT_FILE})",
            )

        # Unattributed entries must carry the per-attempt audit trail, each
        # rejected attempt naming its violation. Transport failures,
        # content-filter blocks, and reasoning-exhausted output budgets are
        # exempt from the non-empty demand: in none of these cases did the
        # model produce a response to record.
        if entry.get("attributed") is False:
            walk.check("attribution_attempts")
            attempts = entry.get("attempts") or []
            kind = entry.get("attribution_error_kind")
            no_response = (
                kind
                in (
                    AttributionErrorKind.TRANSPORT.value,
                    AttributionErrorKind.CONTENT_FILTERED.value,
                    AttributionErrorKind.TOKEN_LIMIT.value,
                    AttributionErrorKind.ENVIRONMENT.value,
                )
                if kind is not None
                else str(entry.get("error", "")).startswith(_TRANSPORT_ERROR_PREFIX)
            )
            if not attempts and not no_response:
                walk.fail(
                    "attribution_attempts",
                    subject,
                    "unattributed entry carries no attempts audit trail",
                )
            for attempt in attempts:
                if not attempt.get("accepted") and not str(attempt.get("violation", "")).strip():
                    walk.fail(
                        "attribution_attempts",
                        subject,
                        f"rejected attempt {attempt.get('attempt')} names no violation",
                    )


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def audit_round(
    out_dir: Path | str, round_index: int, bundle_label: str | None = None
) -> AuditReport:
    """Walk one completed round's directory and verify every audit link.

    All failures are collected -- the walk never stops at the first broken
    link -- and the report names each one by link type, offending id, and
    message. Dependent checks are skipped rather than double-reported when
    their prerequisite file is already broken (e.g. ``bundle_id`` is not
    checked when ``records.jsonl`` is missing, because ``records_file`` has
    already named that break).

    Args:
        out_dir: The experiment directory the round was persisted into.
        round_index: Which round to audit.
        bundle_label: Audit the labeled secondary bundle under
            ``round_NN/bundles/<label>/`` instead of the round-root bundle.
            Only the triplet (``bundle.json``, ``records.jsonl``,
            ``attributions.jsonl``) resolves against that directory; every
            shared round artifact -- the harness envelope, instances, the
            manifest and traces, the content-addressed digests, the persisted
            attributor prompts, and ``config.attribution_cache_path`` (which
            stays round-root-relative in both modes) -- resolves against the
            round root exactly as in the default walk, under the same link
            names.

    Returns:
        An ``AuditReport`` with the per-link results, the broken links, and
        summary stats recomputed from the artifacts alone.

    Raises:
        FileNotFoundError: If the round directory itself does not exist --
            there is no evidence chain to walk at all. A missing *labeled
            bundle directory* is not this case: the round exists, so its
            absent triplet is reported as broken links instead.
    """
    path = round_dir(out_dir, round_index)
    if not path.is_dir():
        raise FileNotFoundError(f"round directory {path} does not exist")
    triplet_dir = bundle_dir_for(path, bundle_label)

    walk = _Walk()

    envelope_hash = _audit_harness(path, walk)

    instances = _load_jsonl_file(path / INSTANCES_FILE, "instances_file", walk)
    instance_ids: set[str] = set()
    if instances is not None:
        for index, instance in enumerate(instances):
            if "id" in instance:
                instance_ids.add(str(instance["id"]))
            else:
                walk.fail(
                    "instances_file",
                    f"line {index + 1}",
                    f"{INSTANCES_FILE} line {index + 1} carries no 'id'",
                )

    manifest = _audit_manifest(path, walk)
    manifest_by_run_id = {
        str(entry["run_id"]): entry for entry in manifest if entry.get("run_id") is not None
    }

    # The triplet lives in the bundle directory; every join it asserts is
    # still checked against the shared artifacts at the round root.
    records = _load_jsonl_file(triplet_dir / RECORDS_FILENAME, "records_file", walk)
    if records is not None:
        _audit_records(path, records, manifest_by_run_id, instance_ids, walk)

    bundle = _load_json_file(triplet_dir / BUNDLE_FILENAME, "bundle_file", walk)
    if bundle is not None:
        _audit_bundle(path, bundle, records, envelope_hash, walk)

    attributions = _load_jsonl_file(triplet_dir / ATTRIBUTIONS_FILENAME, "attributions_file", walk)
    if attributions is not None:
        _audit_attributions(path, attributions, walk)

    n_runs = len(manifest)
    n_passed = sum(1 for entry in manifest if entry.get("passed"))
    n_records = len(records) if records is not None else 0
    n_unattributed = 0
    n_transport_errors = 0
    grounding_coverage: float | None = None
    if records:
        grounded = sum(
            1
            for record in records
            if any(v is not None for v in (record.get("sub_verdicts") or {}).values())
        )
        grounding_coverage = grounded / len(records)
        n_unattributed = sum(1 for record in records if record.get("attribution_failed"))
        n_transport_errors = sum(
            1
            for record in records
            if record.get("attribution_failed") and _is_transport_record(record)
        )

    stats = AuditStats(
        n_runs=n_runs,
        n_passed=n_passed,
        pass_rate=n_passed / n_runs if n_runs else None,
        n_records=n_records,
        n_unattributed=n_unattributed,
        n_transport_errors=n_transport_errors,
        n_patterns=len(bundle.get("patterns", [])) if bundle is not None else 0,
        grounding_coverage=grounding_coverage,
    )

    return AuditReport(
        round_path=str(path),
        ok=not walk.broken,
        links=walk.results(),
        broken=tuple(walk.broken),
        stats=stats,
    )


def run_audited_round(
    config: RoundConfig,
    miner: WeaknessMiner,
    split_id: str,
    harness_version: str | None = None,
    created_at: str | None = None,
    overwrite_bundle: bool = False,
    bundle_label: str | None = None,
) -> tuple[MiningResult, AuditReport]:
    """Run, mine, bundle, and audit one round: the whole pipeline in one call.

    The bundle triplet is written into the same ``round_NN`` directory the
    driver persisted into -- or, when ``bundle_label`` is given, into
    ``round_NN/bundles/<label>/`` -- so one round directory holds the round's
    complete, auditable evidence chain (and can hold a second mode's bundle
    beside the first, as the sub-verification ablation requires). The audit
    result is returned rather than raised on: the caller (the live capstone
    script) decides what a broken link means for the session.

    Re-invocation over the same ``out_dir`` is idempotent per destination:
    completed runs are resumed from the manifest, and when the destination's
    own ``bundle.json`` already exists its recorded ``created_at`` is reused
    (unless the caller supplies one) -- never another destination's, so each
    mode reproduces its own persisted bundle byte for byte and the no-clobber
    guard passes.

    A clean audit is not proof the mining signal exists: a round mined with
    the attributor LM unreachable completes with every record unattributed
    and zero signatures, and still audits ok because unattributed records are
    legitimate evidence. Callers must check ``MiningResult.errors`` (the runs
    whose attribution failed at the transport level) and the report's
    ``n_unattributed`` / ``n_transport_errors`` stats before treating the
    bundle as usable signal.

    Args:
        config: The round to run (see ``RoundConfig``).
        miner: The configured ``WeaknessMiner`` for the mining phase.
        split_id: The evaluation split identifier recorded in the bundle.
        harness_version: Optional bundle-facing harness label; defaults to the
            recorded harness hash (see ``mine_round``).
        created_at: Optional bundle timestamp override, for reproducing a
            previously persisted bundle byte-for-byte; defaults to the
            timestamp already recorded in ``round_NN/bundle.json`` when that
            file exists and parses.
        overwrite_bundle: Explicitly replace a divergent ``bundle.json``
            (see ``write_bundle``). This deliberately replaces evidence that
            may already have been audited -- the escape hatch for a poisoned
            round (e.g. a re-mine after an attributor outage), never a
            default.
        bundle_label: Write and audit the bundle triplet under
            ``round_NN/bundles/<label>/`` instead of the round root (see
            ``bundle_dir_for`` / ``audit_round``); shared round artifacts stay
            at the round root either way. Default None keeps today's layout.

    Returns:
        The ``MiningResult`` and the ``AuditReport`` over the finished round.
    """
    destination = bundle_dir_for(round_dir(config.out_dir, config.round_index), bundle_label)
    if created_at is None:
        # created_at reuse keys on the bundle being rewritten: read only the
        # destination's own bundle.json, never another destination's, so each
        # mode is byte-idempotent on re-invocation.
        bundle_path = destination / BUNDLE_FILENAME
        if bundle_path.is_file():
            try:
                existing = json.loads(bundle_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Corrupt bundle.json: leave created_at fresh and let
                # write_bundle raise its clear corruption error (or replace
                # the file when overwrite_bundle is set).
                existing = None
            if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
                created_at = existing["created_at"]

    run_round(config)
    result = mine_round(
        out_dir=config.out_dir,
        round_index=config.round_index,
        miner=miner,
        split_id=split_id,
        harness_version=harness_version,
        created_at=created_at,
    )
    write_bundle(
        result.bundle,
        result.records,
        str(config.out_dir),
        raw_attributions=result.raw_attributions,
        overwrite=overwrite_bundle,
        bundle_dir=str(destination),
    )
    report = audit_round(config.out_dir, config.round_index, bundle_label=bundle_label)
    return result, report


# ---------------------------------------------------------------------------
# CLI: python -m shrlm.optimization.audit <out_dir> <round_index>
# ---------------------------------------------------------------------------


def _format_report(report: AuditReport) -> str:
    lines = [f"audit walk over {report.round_path}"]
    for link in report.links:
        status = "FAIL" if link.failures else "  ok"
        lines.append(f"  {status}  {link.link:<22} {link.checked} checked, {link.failures} broken")
    if report.broken:
        lines.append("broken links:")
        for broken in report.broken:
            lines.append(f"  {broken.link} [{broken.subject}]: {broken.message}")
    stats = report.stats
    pass_rate = f"{stats.pass_rate:.3f}" if stats.pass_rate is not None else "n/a"
    coverage = f"{stats.grounding_coverage:.3f}" if stats.grounding_coverage is not None else "n/a"
    lines.append(
        f"stats: n_runs={stats.n_runs} n_passed={stats.n_passed} pass_rate={pass_rate} "
        f"n_records={stats.n_records} n_unattributed={stats.n_unattributed} "
        f"n_transport_errors={stats.n_transport_errors} n_patterns={stats.n_patterns} "
        f"grounding_coverage={coverage}"
    )
    if stats.n_records and stats.n_unattributed == stats.n_records:
        lines.append(
            "warning: every failure record is unattributed -- the bundle carries no "
            "signatures (was the attributor LM reachable?)"
        )
    verdict = (
        "OK: every audit link verified"
        if report.ok
        else (f"BROKEN: {len(report.broken)} link failure(s)")
    )
    lines.append(verdict)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print a human-readable audit walk; exit 1 when any link is broken."""
    parser = argparse.ArgumentParser(
        description="Audit one completed round's evidence chain link by link."
    )
    parser.add_argument("out_dir", type=Path, help="the experiment directory holding round_NN/")
    parser.add_argument("round_index", type=int, help="which round to audit")
    parser.add_argument(
        "--bundle-label",
        default=None,
        help=(
            "audit the labeled secondary bundle under round_NN/bundles/<label>/ "
            "instead of the round-root bundle (shared artifacts still resolve "
            "against the round root)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = audit_round(args.out_dir, args.round_index, bundle_label=args.bundle_label)
    except ValueError as error:
        # A malformed --bundle-label (bundle_dir_for's validation) is an
        # operator error, not a broken evidence chain: one line, exit 2.
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(_format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLES_DIR",
    "AuditReport",
    "AuditStats",
    "BrokenLink",
    "LinkResult",
    "audit_round",
    "bundle_dir_for",
    "main",
    "run_audited_round",
    "stored_harness_hash",
]
