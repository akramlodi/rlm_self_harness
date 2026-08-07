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
  digest and its rendered prompt (``attributor_prompt.txt`` for the usual
  single-variant round, ``attributor_prompt_<sha16>.txt`` when the round
  mixed variants; either way the file's bytes must hash to the entry's
  ``prompt_sha256``), and every unattributed entry carries a non-empty
  per-attempt audit trail whose rejected attempts name their violations.
  Transport-failed entries (``error`` prefixed ``transport failure:``) are
  exempt from the non-empty demand: the model may never have produced a
  response to record.

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
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shrlm.optimization.bundle import (
    ATTRIBUTIONS_FILENAME,
    BUNDLE_FILENAME,
    RECORDS_FILENAME,
    compute_bundle_id,
    write_bundle,
)
from shrlm.optimization.driver import (
    ATTRIBUTOR_PROMPT_FILE,
    DIGESTS_DIR,
    HARNESS_FILE,
    INSTANCES_FILE,
    MANIFEST_FILE,
    RoundConfig,
    mine_round,
    round_dir,
    run_round,
)
from shrlm.optimization.mining import MiningResult, WeaknessMiner
from shrlm.optimization.types import FailureRecord, MiningConfig

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
    """

    n_runs: int
    n_passed: int
    pass_rate: float | None
    n_records: int
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stored_harness_hash(envelope: dict[str, Any]) -> str:
    """Recompute the harness hash from the envelope's *stored* serialization.

    Mirrors ``shrlm.harness_identity.harness_hash`` -- sha256 over the
    canonical JSON (sorted keys, compact separators, ASCII) of the
    serialization minus the informational ``name`` -- but operates on the
    persisted dict, so the audit needs no live ``Harness`` object.
    """
    material = dict(envelope["harness"])
    material.pop("name", None)
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    """Verify the manifest parses, repeats no run id, and every trace matches."""
    entries = _load_jsonl_file(path / MANIFEST_FILE, "manifest_file", walk)
    if entries is None:
        return []
    seen: set[str] = set()
    for entry in entries:
        run_id = str(entry["run_id"])
        if run_id in seen:
            walk.fail("manifest_file", run_id, f"{MANIFEST_FILE} lists run id {run_id!r} twice")
        seen.add(run_id)

        walk.check("manifest_trace")
        trace_path = path / str(entry["trace_path"])
        if not trace_path.is_file():
            walk.fail("manifest_trace", run_id, f"trace file {entry['trace_path']} is missing")
            continue
        actual = _sha256_file(trace_path)
        if actual != str(entry["trace_sha256"]):
            walk.fail(
                "manifest_trace",
                run_id,
                f"trace {entry['trace_path']} hashes to {actual}, manifest records "
                f"{entry['trace_sha256']}",
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
    actual = _sha256_file(digest_path)
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
                actual = _sha256_file(trace_path)
                if actual != str(record.get("trace_sha256")):
                    walk.fail(
                        "record_trace",
                        str(run_id),
                        f"trace {trace_rel} hashes to {actual}, record says "
                        f"{record.get('trace_sha256')}",
                    )
                manifest_entry = manifest_by_run_id.get(str(run_id))
                if manifest_entry is not None and actual != str(manifest_entry["trace_sha256"]):
                    walk.fail(
                        "record_trace",
                        str(run_id),
                        f"trace {trace_rel} hashes to {actual}, manifest says "
                        f"{manifest_entry['trace_sha256']}",
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
            refs = [_InstanceRef(str(record["instance_id"])) for record in records]
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
        record_ids = {str(record["instance_id"]) for record in records}
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


def _resolve_prompt_file(path: Path, prompt_sha: str) -> Path | None:
    """Find the persisted prompt file whose bytes hash to ``prompt_sha``.

    ``mine_round`` writes ``attributor_prompt.txt`` for the usual
    single-variant round and ``attributor_prompt_<sha16>.txt`` per variant
    when a round mixed prompt variants; either name is acceptable as long as
    the content hashes to the entry's recorded sha.
    """
    candidates = (
        path / f"attributor_prompt_{prompt_sha[:16]}.txt",
        path / ATTRIBUTOR_PROMPT_FILE,
    )
    for candidate in candidates:
        if candidate.is_file() and _sha256_file(candidate) == prompt_sha:
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
        # rejected attempt naming its violation. Transport failures are exempt
        # from the non-empty demand: the model may never have responded.
        if entry.get("attributed") is False:
            walk.check("attribution_attempts")
            attempts = entry.get("attempts") or []
            transport = str(entry.get("error", "")).startswith(_TRANSPORT_ERROR_PREFIX)
            if not attempts and not transport:
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


def audit_round(out_dir: Path | str, round_index: int) -> AuditReport:
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

    Returns:
        An ``AuditReport`` with the per-link results, the broken links, and
        summary stats recomputed from the artifacts alone.

    Raises:
        FileNotFoundError: If the round directory itself does not exist --
            there is no evidence chain to walk at all.
    """
    path = round_dir(out_dir, round_index)
    if not path.is_dir():
        raise FileNotFoundError(f"round directory {path} does not exist")

    walk = _Walk()

    envelope_hash = _audit_harness(path, walk)

    instances = _load_jsonl_file(path / INSTANCES_FILE, "instances_file", walk)
    instance_ids = {str(instance["id"]) for instance in instances} if instances else set()

    manifest = _audit_manifest(path, walk)
    manifest_by_run_id = {str(entry["run_id"]): entry for entry in manifest}

    records = _load_jsonl_file(path / RECORDS_FILENAME, "records_file", walk)
    if records is not None:
        _audit_records(path, records, manifest_by_run_id, instance_ids, walk)

    bundle = _load_json_file(path / BUNDLE_FILENAME, "bundle_file", walk)
    if bundle is not None:
        _audit_bundle(path, bundle, records, envelope_hash, walk)

    attributions = _load_jsonl_file(path / ATTRIBUTIONS_FILENAME, "attributions_file", walk)
    if attributions is not None:
        _audit_attributions(path, attributions, walk)

    n_runs = len(manifest)
    n_passed = sum(1 for entry in manifest if entry.get("passed"))
    n_records = len(records) if records is not None else 0
    grounding_coverage: float | None = None
    if records:
        grounded = sum(
            1
            for record in records
            if any(v is not None for v in (record.get("sub_verdicts") or {}).values())
        )
        grounding_coverage = grounded / len(records)

    stats = AuditStats(
        n_runs=n_runs,
        n_passed=n_passed,
        pass_rate=n_passed / n_runs if n_runs else None,
        n_records=n_records,
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
) -> tuple[MiningResult, AuditReport]:
    """Run, mine, bundle, and audit one round: the whole pipeline in one call.

    The bundle triplet is written into the same ``round_NN`` directory the
    driver persisted into, so a single directory holds the round's complete,
    auditable evidence chain. The audit result is returned rather than raised
    on: the caller (the live capstone script) decides what a broken link
    means for the session.

    Args:
        config: The round to run (see ``RoundConfig``).
        miner: The configured ``WeaknessMiner`` for the mining phase.
        split_id: The evaluation split identifier recorded in the bundle.
        harness_version: Optional bundle-facing harness label; defaults to the
            recorded harness hash (see ``mine_round``).
        created_at: Optional bundle timestamp override, for reproducing a
            previously persisted bundle byte-for-byte.

    Returns:
        The ``MiningResult`` and the ``AuditReport`` over the finished round.
    """
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
    )
    report = audit_round(config.out_dir, config.round_index)
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
        f"n_records={stats.n_records} n_patterns={stats.n_patterns} "
        f"grounding_coverage={coverage}"
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
    args = parser.parse_args(argv)

    report = audit_round(args.out_dir, args.round_index)
    print(_format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuditReport",
    "AuditStats",
    "BrokenLink",
    "LinkResult",
    "audit_round",
    "main",
    "run_audited_round",
    "stored_harness_hash",
]
