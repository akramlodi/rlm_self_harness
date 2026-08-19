"""The one output layer every analysis writes through (KTD2, KTD8).

Before this module each aggregation wrote its CSVs straight into the
experiment directory under a fixed name. Two consequences, both fatal to the
reproducibility claim the study rests on: a rerun silently destroyed the
previous run's numbers, and a CSV on disk carried nothing saying which
experiment it came from, when it was produced, what code produced it, or which
bytes it was derived from. A reader holding ``surface_activity.csv`` could not
tell a figure built from a finished round from one built from the same round
re-mined after a crash.

So every batch of analyses writes into a fresh, UTC-stamped, provenance-stamped
snapshot directory::

    <out_dir>/analysis/20260818T164800Z/
        surface_activity.csv
        incumbent_quality.csv
        ...
        provenance.json      # identity, revision, tools, source hashes
        published.json       # written LAST; absent means "do not trust this"
        plots/               # deliberately re-renderable (see below)

One snapshot per invocation batch, not per tool
    The caller -- a CLI run, or the orchestrator's post-round hook -- allocates
    ONE snapshot and passes it to every aggregation it runs, so the CSVs a
    reader compares against each other are guaranteed to have been derived from
    one pass over one tree. A tool that allocated its own would leave four
    directories per invocation, each a different instant, and nothing saying
    they belonged together.

Allocation is a race, and is treated as one
    ``allocate_snapshot`` creates the stamped directory with
    ``mkdir(exist_ok=False)``: the exclusive create IS the lock. A stamp that
    already exists -- two invocations inside the same UTC second, which the
    post-round hook makes entirely reachable -- retries with a monotonic suffix
    (``-2``, ``-3``, ...). Checking ``exists()`` first and then creating would
    leave exactly the window in which two batches decide the same directory is
    free and then write over each other.

Published means finished
    The completion marker is written LAST, after every tool in the batch
    finished and provenance is on disk. ``latest_snapshot`` only ever returns a
    published directory, so a crashed or partially-failed batch can never be
    picked up as "the latest results" by a plotting run or a reader -- it stays
    on disk, inspectable, but never selected. A batch with a failed tool
    writes its provenance (that record is the audit trail for the failure) and
    stays unpublished.

Never-overwrite, and the one deliberate exception
    ``write_csv`` / ``write_json`` refuse to replace an existing file with
    different bytes, in ``orchestrator._persist_once``'s established style: a
    byte-identical rewrite is a no-op, a diverging one raises. ``plots/`` is
    exempt by construction -- it is written through neither function -- because
    a frozen snapshot's figures must stay re-renderable by improved plotting
    code without the frozen data underneath them ever moving (KTD2).

Provenance, and why the source hashes are the load-bearing part
    Experiment identity plus round number does not distinguish two snapshots
    built from a re-mined or resumed round; the content hashes of the consumed
    artifacts do. ``sources`` is therefore a manifest of every artifact an
    analysis read, each as a path relative to the experiment directory (or to
    the repository, for a bundle passed in from outside) plus its sha256 -- an
    absolute path would be machine-specific and would make the same tree hash
    differently on two machines. A directory of per-run traces is recorded as
    ONE entry whose hash covers every file inside it, so a snapshot's
    provenance does not grow to thousands of lines while still detecting any
    byte that moved.

    ``identity_hash`` is read from ``<out_dir>/config.json`` -- the value the
    orchestrator recorded, taken verbatim rather than re-derived, because the
    question provenance answers is "which experiment produced this", and that
    is what the experiment directory itself claims. A pre-orchestrator tree has
    no such file; rather than failing the analysis (an old tree is exactly the
    thing worth analyzing), the identity is derived deterministically from the
    source hashes and flagged ``identity_source: derived`` /
    ``legacy_derived: true``, so a reader can always tell a real experiment
    identity from a stand-in.

    ``analysis_revision`` is the repository commit the analysis ran from,
    resolved best-effort with ``git rev-parse HEAD``. No git, no repository, or
    a git failure records ``null``: an unknown revision is worth recording as
    unknown, and is never worth failing an analysis over.
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from shrlm.experiment.errors import ExperimentError
from shrlm.experiment.orchestrator import CONFIG_FILENAME, CONFIG_FORMAT
from shrlm.experiment.rounds import ExperimentInventory, discover_rounds
from shrlm.optimization.validation import DECISION_FILENAME, PROMOTIONS_FILENAME

ANALYSIS_DIR = "analysis"
PLOTS_DIR = "plots"

PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_FORMAT = "shrlm-analysis-provenance/v1"
PUBLISHED_FILENAME = "published.json"
PUBLISHED_FORMAT = "shrlm-analysis-snapshot/v1"

# The directory name of a snapshot: a UTC instant, sortable as a string, with
# no characters a filesystem or a shell argues about.
SNAPSHOT_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

IDENTITY_SOURCE_CONFIG = "config"
IDENTITY_SOURCE_DERIVED = "derived"

# How a KTD9 three-valued completeness flag reaches a CSV cell. Spelled out as
# words rather than left to Python's own rendering because ``None`` writes as
# an EMPTY cell, and an empty cell in a completeness column is indistinguishable
# from a column the writer forgot -- which is precisely the reading KTD9
# forbids: "unknown is always its own value and never collapses into false", so
# an unmeasurable round must never be presented as a failed one.
TRISTATE_TRUE = "true"
TRISTATE_FALSE = "false"
TRISTATE_UNKNOWN = "unknown"

SOURCE_KIND_FILE = "file"
SOURCE_KIND_DIRECTORY = "directory"

# Ceiling on the collision retry. Reaching it means something other than a
# same-second collision is wrong (a read-only parent, a name held by a file),
# and spinning forever would hide it.
MAX_STAMP_COLLISIONS = 1000

_REPO_ROOT = Path(__file__).resolve().parents[2]


class AnalysisOutputError(ExperimentError):
    """A published analysis artifact would have been overwritten or replaced.

    Its own class, per the package's convention, because the recovery differs
    from every other persistence failure: nothing about the experiment is
    wrong, the analysis simply tried to write twice into one snapshot.
    """


class CsvRow(Protocol):
    """Anything ``write_csv`` can write: a row that renders itself to a dict.

    Every analysis row dataclass in this package already implements exactly
    this, which is what lets one writer serve all four aggregations without
    any of them sharing a base class -- and what removes the
    ``# type: ignore[attr-defined]`` the duplicated writers needed to call
    ``to_dict()`` on a bare ``object``.
    """

    def to_dict(self) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# Non-clobbering writes
# ---------------------------------------------------------------------------


def _write_once(path: Path, text: str) -> None:
    """Write text exactly once, in ``orchestrator._persist_once``'s style.

    A byte-identical rewrite is a no-op (a re-run of the same batch over the
    same bytes is not a conflict); anything else raises rather than destroying
    a published number.

    Bytes rather than text on both sides of the comparison: the CSV writer
    emits CRLF line endings (``csv``'s own default, which the per-module
    writers this replaces also produced), and text-mode reads translate those
    back to LF -- so a text comparison would call a byte-identical file
    different and raise on a legitimate no-op rewrite.
    """
    data = text.encode()
    if path.exists():
        if path.read_bytes() == data:
            return
        raise AnalysisOutputError(
            f"{path} already exists with different content; a published analysis "
            "artifact is never overwritten -- allocate a new snapshot instead"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_csv(path: Path | str, rows: Sequence[CsvRow], *, fieldnames: Sequence[str]) -> None:
    """The repository's one CSV writer (KTD8).

    ``fieldnames`` is required and explicit rather than read off ``rows[0]``:
    an empty result set has no first row, and an empty result must still write
    a header-only CSV -- a downstream reader has to be able to tell "this
    analysis found nothing" from "this analysis never ran", and a zero-byte
    file says the second.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row.to_dict()))
    _write_once(Path(path), buffer.getvalue())


def tristate(value: bool | None) -> str:
    """Serialize one three-valued completeness flag for a CSV cell (KTD9).

    Every analysis in this package writes its completeness columns through
    this one function, so ``unknown`` reads identically in every table and no
    module can accidentally render it as ``False`` -- the single translation
    KTD9 exists to forbid. Numeric completeness columns (an expected or missing
    run count) stay numeric and write an empty cell when unknown; the flag
    beside them is what says which kind of blank it is.
    """
    if value is None:
        return TRISTATE_UNKNOWN
    return TRISTATE_TRUE if value else TRISTATE_FALSE


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write one analysis JSON artifact, non-clobbering, in the repo's shape."""
    text = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_once(Path(path), text)


# ---------------------------------------------------------------------------
# Identity and revision
# ---------------------------------------------------------------------------


def recorded_identity(out_dir: Path | str) -> str | None:
    """The identity hash ``<out_dir>/config.json`` records, or None.

    Taken verbatim from the marker rather than re-derived from the TOML (which
    is what ``rounds.resolve_config`` does, for a different question): the
    directory's own claim about which experiment it is stays true even if the
    profile in ``configs/experiment.toml`` has since drifted, and provenance
    wants exactly that claim.
    """
    path = Path(out_dir) / CONFIG_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != CONFIG_FORMAT:
        return None
    recorded = payload.get("identity_hash")
    return None if recorded is None else str(recorded)


def analysis_revision() -> str | None:
    """The repository commit this analysis ran from, or None when unresolvable.

    Best-effort by design: an analysis run from a tarball, from a machine with
    no git, or inside a repository in a state ``rev-parse`` dislikes still has
    to produce its numbers. Recording ``null`` says "unknown revision", which
    is honest; failing the run would say nothing at all.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    return revision or None


# ---------------------------------------------------------------------------
# Source hashing
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    """One artifact's sha256, read in chunks rather than into memory.

    ``hashlib.file_digest`` is the standard library's own chunked reader (3.11+,
    which this package already requires), so the loop this used to spell by
    hand is the same loop with one fewer place to get the buffer size wrong.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _relative_source_path(path: Path, out_dir: Path) -> str:
    """A machine-independent name for a consumed artifact.

    Experiment-relative first (the ordinary case, and the one that makes two
    provenance records of the same tree comparable), repository-relative next
    (a bundle passed in from a checked-in fixture), absolute only when the
    artifact lives outside both -- at which point the absolute path is the
    only true thing left to say about it.
    """
    resolved = path.resolve()
    for root in (out_dir.resolve(), _REPO_ROOT):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------


class Snapshot:
    """One invocation batch's output directory, and its provenance in progress.

    Mutable on purpose: a batch accumulates sources, analyzed rounds, and
    per-tool outcomes as its tools run, and only at ``publish`` does that
    accumulation become the immutable ``provenance.json``.
    """

    def __init__(self, path: Path, out_dir: Path, created_at: datetime) -> None:
        self.path = path
        self.out_dir = Path(out_dir)
        self.created_at = created_at.isoformat()
        self.stamp = path.name
        self._sources: dict[str, dict[str, Any]] = {}
        self._rounds: set[int] = set()
        self._eval_sets: set[str] = set()
        self._tools: list[dict[str, Any]] = []
        self._identity: tuple[str, str] | None = None
        self._digests: dict[Path, str] = {}

    # -- outputs ----------------------------------------------------------

    @property
    def plots_dir(self) -> Path:
        """``<snapshot>/plots``, created on demand and deliberately mutable.

        The one part of a published snapshot that may be rewritten: figures
        are re-renderable from frozen data (KTD2), so nothing here goes through
        the non-clobbering writers.
        """
        plots = self.path / PLOTS_DIR
        plots.mkdir(parents=True, exist_ok=True)
        return plots

    @property
    def is_published(self) -> bool:
        return (self.path / PUBLISHED_FILENAME).exists()

    def stamp_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """A JSON output's payload with the experiment identity and time added.

        ``eval_summary.json`` already stamps ``identity_hash`` without a
        timestamp; this extends that precedent rather than inventing a second
        convention, adding ``created_at`` so a JSON lifted out of its snapshot
        directory still says when it was produced.
        """
        return {
            **dict(payload),
            "identity_hash": self.identity_hash,
            "created_at": self.created_at,
        }

    # -- provenance accumulation -----------------------------------------

    def _digest(self, path: Path) -> str:
        """One artifact's sha256, hashed at most once per snapshot.

        The batch is a set of independent readings over ONE tree that nothing
        rewrites while it runs, so two tools consuming the same artifact --
        which the ledger-reading pair do for every ``promotions.jsonl`` and
        ``decision.json`` in the experiment -- must agree on its hash, and a
        second full read to reach that same answer is pure cost. Keyed on the
        resolved path, so the same file reached by two spellings is hashed
        once and recorded once.
        """
        resolved = path.resolve()
        digest = self._digests.get(resolved)
        if digest is None:
            digest = _hash_file(path)
            self._digests[resolved] = digest
        return digest

    def record_source(self, path: Path | str) -> None:
        """Record one consumed artifact's content hash.

        An artifact that is not on disk is recorded with a ``null`` hash rather
        than omitted: "this analysis looked for it and it was not there" is
        part of how the numbers came out, and dropping the entry would make the
        absence invisible.
        """
        resolved = Path(path)
        name = _relative_source_path(resolved, self.out_dir)
        self._sources[name] = {
            "path": name,
            "kind": SOURCE_KIND_FILE,
            "sha256": self._digest(resolved) if resolved.is_file() else None,
        }

    def record_sources(self, paths: Iterable[Path | str]) -> None:
        for path in paths:
            self.record_source(path)

    def record_source_dir(self, path: Path | str, *, pattern: str = "*") -> None:
        """Record a directory of artifacts (per-run traces) as ONE entry.

        The hash covers every matching file's name and content, so any byte
        that moved anywhere in the directory moves this entry -- without
        writing one provenance line per run, which for a real experiment would
        be thousands.
        """
        resolved = Path(path)
        name = _relative_source_path(resolved, self.out_dir)
        if not resolved.is_dir():
            self._sources[name] = {
                "path": name,
                "kind": SOURCE_KIND_DIRECTORY,
                "sha256": None,
                "n_files": 0,
            }
            return
        digest = hashlib.sha256()
        n_files = 0
        for child in sorted(resolved.glob(pattern)):
            if not child.is_file():
                continue
            n_files += 1
            digest.update(child.name.encode())
            digest.update(self._digest(child).encode())
        self._sources[name] = {
            "path": name,
            "kind": SOURCE_KIND_DIRECTORY,
            "sha256": digest.hexdigest(),
            "n_files": n_files,
        }

    def record_rounds(self, round_indices: Iterable[int]) -> None:
        """Record which optimization rounds this batch actually analyzed."""
        self._rounds.update(int(index) for index in round_indices)

    def record_eval_sets(self, set_ids: Iterable[str]) -> None:
        """Record which ``<condition_id>/<set_id>`` pairs this batch analyzed."""
        self._eval_sets.update(str(set_id) for set_id in set_ids)

    def record_tool(self, name: str, *, ok: bool, error: str | None = None) -> None:
        self._tools.append({"name": name, "ok": ok, "error": error})

    def run_tool(self, name: str, call: Callable[[], Any]) -> bool:
        """Run one aggregation under its own guard; never propagate its failure.

        The batch is a set of independent readings over the same tree, so one
        tool's exception must not cost the others their output -- it is
        recorded against that tool's name in provenance and the batch carries
        on. The snapshot then stays unpublished (see ``publish``), which is how
        a partial batch is prevented from being read as a complete one.
        """
        try:
            call()
        except Exception as error:  # noqa: BLE001 -- isolation is the point
            self.record_tool(name, ok=False, error=f"{type(error).__name__}: {error}")
            return False
        self.record_tool(name, ok=True)
        return True

    def failure_message(self, name: str) -> str:
        """The one line a CLI writes to stderr when its tool failed.

        One wording, owned beside ``publish`` -- the provenance record it points
        at is the only place the failure itself is written down, so every
        aggregation CLI has to say the same thing about where to look.
        """
        return f"{name} failed; see {self.path / PROVENANCE_FILENAME}\n"

    # -- identity ---------------------------------------------------------

    @property
    def identity_hash(self) -> str:
        return self._resolve_identity()[0]

    @property
    def identity_source(self) -> str:
        return self._resolve_identity()[1]

    def _resolve_identity(self) -> tuple[str, str]:
        """The experiment identity, memoized on first use.

        Memoized rather than recomputed because the derived (no ``config.json``)
        branch is a function of the sources recorded so far, and a value that
        drifted between a stamped JSON output and ``provenance.json`` would
        make the two disagree about which experiment they describe. The
        contract that follows -- a tool records its sources before it writes
        its outputs -- is what every aggregation in this package does.
        """
        if self._identity is None:
            recorded = recorded_identity(self.out_dir)
            if recorded is not None:
                self._identity = (recorded, IDENTITY_SOURCE_CONFIG)
            else:
                self._identity = (self._derived_identity(), IDENTITY_SOURCE_DERIVED)
        return self._identity

    def _derived_identity(self) -> str:
        """A stand-in identity for a tree that never recorded one.

        Deterministic over the source manifest, so two analyses of an
        unchanged legacy tree agree on what to call it, and an analysis of a
        changed one does not.
        """
        canonical = json.dumps(self._source_entries(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _source_entries(self) -> list[dict[str, Any]]:
        return [self._sources[name] for name in sorted(self._sources)]

    # -- publication ------------------------------------------------------

    def provenance_payload(self) -> dict[str, Any]:
        identity, source = self._resolve_identity()
        return {
            "format": PROVENANCE_FORMAT,
            "identity_hash": identity,
            "identity_source": source,
            "legacy_derived": source == IDENTITY_SOURCE_DERIVED,
            "created_at": self.created_at,
            "analysis_revision": analysis_revision(),
            "rounds": sorted(self._rounds),
            "eval_sets": sorted(self._eval_sets),
            "tools": list(self._tools),
            "sources": self._source_entries(),
        }

    def publish(self) -> bool:
        """Write provenance, then the completion marker if every tool finished.

        Order matters and is the whole guarantee: provenance lands first (a
        failed batch still leaves its audit trail), and the marker -- the only
        thing ``latest_snapshot`` looks at -- lands last and only when nothing
        in the batch failed. A snapshot missing its marker is inspectable but
        never selected as anybody's latest results.

        Returns:
            Whether the snapshot was published.
        """
        write_json(self.path / PROVENANCE_FILENAME, self.provenance_payload())
        if not all(tool["ok"] for tool in self._tools):
            return False
        write_json(
            self.path / PUBLISHED_FILENAME,
            {
                "format": PUBLISHED_FORMAT,
                "identity_hash": self.identity_hash,
                "created_at": self.created_at,
                "published_at": datetime.now(UTC).isoformat(),
                "n_tools": len(self._tools),
            },
        )
        return True


def snapshot_parent(out_dir: Path | str, parent: Path | str | None = None) -> Path:
    """Where snapshots live: ``<out_dir>/analysis`` unless ``--out`` overrode it."""
    return Path(parent) if parent is not None else Path(out_dir) / ANALYSIS_DIR


def add_snapshot_parent_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--out`` option every aggregation CLI takes, one definition.

    ``--out`` is the ``snapshot_parent`` question, which this module owns, so
    the flag that asks it is defined here rather than copied into each CLI: a
    help text that drifted in one copy would describe an option the other three
    do not have.
    """
    parser.add_argument(
        "--out",
        dest="snapshot_parent",
        metavar="DIR",
        default=None,
        help=(
            "parent directory for the timestamped analysis snapshot (default: <out_dir>/analysis)"
        ),
    )


def allocate_snapshot(
    out_dir: Path | str,
    *,
    parent: Path | str | None = None,
    now: datetime | None = None,
) -> Snapshot:
    """Claim one fresh snapshot directory for this batch (KTD2).

    The claim is the ``mkdir(exist_ok=False)`` itself -- an atomic exclusive
    create, so two batches in the same UTC second cannot both believe they own
    the directory. The loser retries with the next monotonic suffix rather than
    waiting a second, because the post-round hook can fire two batches back to
    back and neither may write over the other's numbers.

    Args:
        out_dir: The experiment directory being analyzed. It anchors both the
            default snapshot parent and the identity lookup.
        parent: Override for the snapshot parent (the CLIs' ``--out``). The
            timestamped directory is still created inside it -- ``--out`` moves
            where snapshots accumulate, never whether they are stamped.
        now: The batch's timestamp. Injected by tests to force the same-second
            collision deterministically; production passes nothing.

    Returns:
        The allocated (unpublished) snapshot.
    """
    created_at = now if now is not None else datetime.now(UTC)
    root = snapshot_parent(out_dir, parent)
    root.mkdir(parents=True, exist_ok=True)
    stamp = created_at.strftime(SNAPSHOT_STAMP_FORMAT)
    for suffix in range(1, MAX_STAMP_COLLISIONS + 1):
        name = stamp if suffix == 1 else f"{stamp}-{suffix}"
        path = root / name
        try:
            path.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return Snapshot(path=path, out_dir=Path(out_dir), created_at=created_at)
    raise AnalysisOutputError(
        f"could not allocate a snapshot under {root}: {MAX_STAMP_COLLISIONS} names "
        f"starting at {stamp} were already taken"
    )


def is_published(path: Path | str) -> bool:
    """Whether a snapshot directory carries its completion marker."""
    return (Path(path) / PUBLISHED_FILENAME).exists()


def latest_snapshot(out_dir: Path | str, *, parent: Path | str | None = None) -> Path | None:
    """The newest PUBLISHED snapshot, or None when there is not one.

    Unpublished directories are skipped, never merely sorted below: a crashed
    batch that happens to be newest must not become the thing a plotting run or
    a reader picks up. The stamp names sort lexicographically in time order,
    and a collision suffix sorts after the bare stamp it collided with, so name
    order is chronological order.
    """
    root = snapshot_parent(out_dir, parent)
    if not root.is_dir():
        return None
    published = [path for path in sorted(root.iterdir()) if path.is_dir() and is_published(path)]
    return published[-1] if published else None


# ---------------------------------------------------------------------------
# Shared source recording
# ---------------------------------------------------------------------------


def record_ledger_sources(
    snapshot: Snapshot, out_dir: Path | str, *, inventory: ExperimentInventory | None = None
) -> list[int]:
    """Record every promotion ledger the inventory holds; return its rounds.

    Shared by the two ledger-reading aggregations, which consume exactly the
    same artifacts (``promotions.jsonl`` plus the round's ``decision.json``)
    through the same discovery. Recording them here rather than in each CLI
    keeps the provenance manifest and the reader in agreement by construction.

    Args:
        snapshot: The batch's snapshot; the sources land in its provenance.
        out_dir: The experiment directory, used when no inventory is passed.
        inventory: The discovery the caller already ran over ``out_dir``.
            Discovery does real per-round IO and re-reads the experiment TOML,
            and nothing rewrites the tree mid-batch, so a caller that already
            holds one passes it rather than paying for an identical second
            pass. Omitted, it is discovered here.
    """
    resolved = inventory if inventory is not None else discover_rounds(out_dir)
    analyzed: list[int] = []
    for record in resolved.rounds:
        if not record.has_ledger:
            continue
        analyzed.append(record.round_index)
        snapshot.record_source(record.validation_round_path / PROMOTIONS_FILENAME)
        snapshot.record_source(record.validation_round_path / DECISION_FILENAME)
    snapshot.record_rounds(analyzed)
    return analyzed


__all__ = [
    "ANALYSIS_DIR",
    "IDENTITY_SOURCE_CONFIG",
    "IDENTITY_SOURCE_DERIVED",
    "PLOTS_DIR",
    "PROVENANCE_FILENAME",
    "PROVENANCE_FORMAT",
    "PUBLISHED_FILENAME",
    "PUBLISHED_FORMAT",
    "SNAPSHOT_STAMP_FORMAT",
    "TRISTATE_FALSE",
    "TRISTATE_TRUE",
    "TRISTATE_UNKNOWN",
    "AnalysisOutputError",
    "CsvRow",
    "Snapshot",
    "add_snapshot_parent_argument",
    "allocate_snapshot",
    "analysis_revision",
    "is_published",
    "latest_snapshot",
    "record_ledger_sources",
    "recorded_identity",
    "snapshot_parent",
    "tristate",
    "write_csv",
    "write_json",
]
