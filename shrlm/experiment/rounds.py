"""The one inventory of what an experiment directory holds (KTD1).

Every analysis in this package -- and the report -- must read exactly what the
loop wrote. Before this module there were two independent walkers over the
same tree (one for mining rounds, one for promotion ledgers), each
reconstructing ``opt/round_NN/{mining,validation}/round_NN/`` from its own
string arithmetic; a layout change would have moved the loop and left the
analyses reading a tree that no longer exists, silently and without failing.
So path construction here goes exclusively through the loop's own layout code
-- ``experiment_round_dir`` and the ``OPT_DIR`` / ``MINING_DIR`` /
``VALIDATION_DIR`` / ``PROPOSALS_DIR`` constants from ``orchestrator``,
``round_dir`` from ``shrlm.optimization.bundle``, ``EVAL_DIR`` from
``evaluation`` -- and enumerating ``round_*`` directories happens in exactly
one function (``_opt_round_indices``), which is the whole point of the module.

Named ``rounds`` rather than ``walker`` because ``shrlm.optimization.walker``
already means the trace-tree walker. The import direction is strictly
``shrlm.experiment`` -> ``shrlm.optimization``, never the reverse.

Discovery answers three questions, and keeps them apart:

*What exists.* Directory presence, not payload validity. A round directory the
loop created is discovered even when nothing inside it finished -- including
the case the review flagged as invisible, a completed no-op round (marker
present, no ledger, because the round's proposal stage produced no candidate).
That is a real outcome, not missing data, so it appears with ``has_ledger``
false rather than vanishing from the round accounting.

*How complete it is.* One truth table (KTD9), so no analysis re-decides what
"complete" means:

===================  ==========================================================
``round_complete``   ``round.json`` present, well-formed, and recording this
                     round's index. Never unknown -- marker presence is always
                     knowable.
``evidence_complete````evidence_complete.json`` present, naming this round and
                     the bundle beside it, and its recorded record counts
                     matching the PARSED ``records.jsonl`` /
                     ``attributions.jsonl``. Never unknown. The re-check
                     matters because ``write_bundle`` publishes ``bundle.json``
                     before those two files, so a marker written over a
                     truncated audit file is exactly the crash window the
                     orchestrator's own marker payload guards.
``runs_complete``    ``n_present == n_expected`` when the expected count is
                     knowable; ``None`` (``count_unknown``) when it is not.
eval set ``complete``the set's outcome is ``completed``, no skipped runs are
                     recorded, AND the set's persisted runs reach the count it
                     planned; ``None`` when the set is absent from
                     ``eval_summary.json`` or its planned count is unknowable.
===================  ==========================================================

``unknown`` is always its own value and never collapses into false: an
unmeasurable round must never be presented as a failed one.

*Which surface a ledger row touched.* ``resolve_surface`` (KTD3), the third
question, because it is the same "read what the loop wrote" problem one level
down: a row's ``surface`` is not always in the ledger, but the proposal
artifact that named it is still on disk beside it. The ledger value wins when
there is one; a missing OR null one is recovered by joining ``subject_id`` to
the round's persisted proposals; the merged harness's record short-circuits to
its own category, because a null surface is correct for a record that spans
several (KTD7). Every answer carries the source it came from, so a recovered
surface is never presented as a recorded one.

Expected run counts, and why nothing is inferred from the runs (KTD6)
    ``n_expected = len(instances.jsonl) x repetitions``, with ``repetitions``
    read from the experiment config PER STAGE -- ``loop.m`` for mining,
    ``loop.v`` for validation, ``operational.eval_repetitions`` for evaluation
    -- because the config assigns those separately and one fixed multiplier
    would misreport whichever stage it does not match. When the config cannot
    be resolved, ``n_expected`` is ``None`` and the row is flagged
    ``count_unknown``. There is deliberately NO fallback that infers the
    attempt count from the observed run ids: the manifest records only
    completed runs, so a round truncated after attempt 2 of 3 would infer an
    expected count of 2 and report itself complete -- manufacturing exactly the
    false completeness this module exists to remove.

    Resolving the config from disk means reading the profile from
    ``<out_dir>/config.json`` and re-loading that profile from
    ``configs/experiment.toml``, then checking the reloaded config still hashes
    to the identity the directory recorded. A drifted TOML therefore yields
    "unknown" rather than counts from a configuration this experiment never
    ran under. A caller that already holds the config (the loop itself) passes
    it instead.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrlm.experiment.config import (
    CONFIG_PATH,
    ExperimentConfig,
    identity_hash,
    load_config,
)
from shrlm.experiment.evaluation import EVAL_DIR, EVAL_SUMMARY_FILENAME
from shrlm.experiment.orchestrator import (
    CONFIG_FILENAME,
    CONFIG_FORMAT,
    EVIDENCE_MARKER_FILENAME,
    EVIDENCE_MARKER_FORMAT,
    MINING_DIR,
    OPT_DIR,
    PROPOSALS_DIR,
    PROPOSALS_MARKER_FILENAME,
    ROUND_MARKER_FILENAME,
    ROUND_MARKER_FORMAT,
    VALIDATION_DIR,
    experiment_round_dir,
)
from shrlm.experiment.usage import read_jsonl
from shrlm.optimization.bundle import (
    ATTRIBUTIONS_FILENAME,
    BUNDLE_FILENAME,
    RECORDS_FILENAME,
    round_dir,
)
from shrlm.optimization.costs import OUTCOME_COMPLETED
from shrlm.optimization.driver import INSTANCES_FILE, MANIFEST_FILE, load_manifest
from shrlm.optimization.promotion import MERGED_SUBJECT_ID
from shrlm.optimization.proposal import PROPOSAL_FILENAME, PROPOSAL_FORMAT
from shrlm.optimization.validation import (
    EVAL_ROUND_INDEX,
    PROMOTIONS_FILENAME,
    load_promotion_ledger,
)

# The ONE place this repository enumerates round directories. The verification
# gate (`grep -rn 'round_\*' shrlm/ --include='*.py'`) is expected to match this
# file and no other; a second walker holding the pattern in a constant of its
# own is caught by that grep.
ROUND_DIR_PATTERN = "round_*"
ROUND_DIR_PREFIX = "round_"

# The three stages that repeat runs per instance, each with its own configured
# repetition count (KTD6).
STAGE_MINING = "mining"
STAGE_VALIDATION = "validation"
STAGE_EVALUATION = "evaluation"

# Where a ledger row's surface attribution came from (KTD3). Every analysis
# that counts surfaces reports one of these beside the count, so a reader can
# always tell a recorded surface from a reconstructed one.
SURFACE_SOURCE_LEDGER = "ledger"
SURFACE_SOURCE_BACKFILLED = "backfilled"
SURFACE_SOURCE_MERGED = "merged"
SURFACE_SOURCE_NONE = "none"

# The tally the merged harness's own re-evaluation record belongs in: a
# category beside the nine surfaces, never one of them (KTD7).
MERGED_CATEGORY = MERGED_SUBJECT_ID


# ---------------------------------------------------------------------------
# Per-stage repetition counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRepetitions:
    """How many runs each stage plans per instance, from the experiment config.

    Three separate numbers rather than one: ``loop.m`` and ``loop.v`` are the
    optimization loop's mining and validation repetitions, and
    ``operational.eval_repetitions`` is the evaluation stage's -- the config
    sets them independently, and the smoke profile may shrink each of them on
    its own.
    """

    mining: int
    validation: int
    evaluation: int

    def for_stage(self, stage: str) -> int:
        """The repetition count for one stage name; unknown names are a bug."""
        counts = {
            STAGE_MINING: self.mining,
            STAGE_VALIDATION: self.validation,
            STAGE_EVALUATION: self.evaluation,
        }
        if stage not in counts:
            raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(counts)}")
        return counts[stage]


def stage_repetitions(config: ExperimentConfig) -> StageRepetitions:
    """The per-stage repetition counts a config assigns (KTD6)."""
    return StageRepetitions(
        mining=config.loop.m,
        validation=config.loop.v,
        evaluation=config.operational.eval_repetitions,
    )


def resolve_config(out_dir: Path | str, path: Path | str = CONFIG_PATH) -> ExperimentConfig | None:
    """The configuration this experiment ran under, or None when unresolvable.

    The experiment directory persists only its profile and identity hash, so
    the counts come from re-loading that profile out of the TOML and are
    trusted only when the reload still hashes to the recorded identity. A
    missing, unparseable, or drifted config resolves to ``None``, which every
    caller renders as "unknown" rather than substituting a plausible number.
    """
    marker_path = Path(out_dir) / CONFIG_FILENAME
    if not marker_path.exists():
        return None
    try:
        payload = json.loads(marker_path.read_text())
    except (OSError, ValueError):
        return None
    if payload.get("format") != CONFIG_FORMAT:
        return None
    profile = payload.get("profile")
    if not isinstance(profile, str):
        return None
    try:
        config = load_config(profile, path)
    except (OSError, ValueError):
        return None
    if identity_hash(config) != str(payload.get("identity_hash")):
        return None
    return config


# ---------------------------------------------------------------------------
# Run counts (KTD9 row 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunCounts:
    """How many runs a stage persisted against how many it planned.

    ``n_expected`` is ``None`` exactly when the plan is not knowable from
    persisted state -- no resolvable config, or no ``instances.jsonl`` to count
    -- and every derived field then reports unknown rather than guessing.
    """

    n_present: int
    n_expected: int | None

    @property
    def count_unknown(self) -> bool:
        """Whether the planned count is unknowable (the KTD6 fallback case)."""
        return self.n_expected is None

    @property
    def n_missing(self) -> int | None:
        """Planned runs with nothing on disk; ``None`` when the plan is unknown."""
        if self.n_expected is None:
            return None
        return max(0, self.n_expected - self.n_present)

    @property
    def runs_complete(self) -> bool | None:
        """True / False / None per KTD9 -- never False merely for being unknown.

        A surplus (more persisted runs than planned) counts as complete: the
        driver's ``<instance>__aNN`` run ids make it unreachable in practice,
        and reporting it as missing data would be false in the other direction.
        """
        if self.n_expected is None:
            return None
        return self.n_present >= self.n_expected


def _count_lines(path: Path) -> int | None:
    """Non-blank lines in a JSON-lines file; ``None`` when the file is absent."""
    if not path.exists():
        return None
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _run_counts(
    round_path: Path, parent: Path, round_index: int, repetitions: int | None
) -> RunCounts:
    """One round directory's persisted-vs-planned run counts (KTD6).

    ``round_path`` is where the driver persisted ``instances.jsonl``;
    ``parent`` is ``load_manifest``'s own argument (it appends ``round_NN``
    itself), which is why both are passed rather than derived here.
    """
    n_present = len(load_manifest(parent, round_index))
    n_instances = _count_lines(round_path / INSTANCES_FILE)
    if repetitions is None or n_instances is None:
        return RunCounts(n_present=n_present, n_expected=None)
    return RunCounts(n_present=n_present, n_expected=n_instances * repetitions)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundRecord:
    """One optimization round as it exists on disk, with its completeness.

    Every path is the loop's own; an analysis reads these instead of building
    its own, so a layout change cannot leave the two views disagreeing.
    """

    round_index: int
    round_path: Path
    mining_parent: Path
    mining_round_path: Path
    validation_parent: Path
    validation_round_path: Path
    proposals_dir: Path
    has_round_marker: bool
    has_evidence_marker: bool
    has_proposals_marker: bool
    has_manifest: bool
    has_ledger: bool
    round_complete: bool
    evidence_complete: bool
    mining_runs: RunCounts


@dataclass(frozen=True)
class EvalSetRecord:
    """One evaluation condition x test set, with the summary's verdict on it.

    ``in_summary`` false means the set has run directories but no entry in
    ``eval_summary.json`` -- an evaluation that crashed before it aggregated,
    or a summary written by an earlier invocation. Its ``complete`` is then
    ``None``: unmeasured, not failed.

    ``complete`` reads BOTH the summary's verdict and the set's own run counts,
    because the summary is a self-report: ``evaluation.py`` writes its outcome
    from the state it held in memory, so a set truncated after the aggregate
    landed -- or one whose run directory was never fully persisted -- would
    otherwise publish itself complete while ``runs`` says runs are missing. The
    optimization rounds have always carried ``runs_complete`` next to their
    marker flags; this is the same evidence on the evaluation side.
    """

    condition_id: str
    set_id: str
    set_path: Path
    round_path: Path
    environment: str | None
    length: str | None
    outcome: str | None
    skipped_run_ids: tuple[str, ...]
    in_summary: bool
    runs: RunCounts

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_run_ids)

    @property
    def summary_complete(self) -> bool | None:
        """The evaluation summary's own verdict on the set, on its own.

        ``None`` when the set never reached ``eval_summary.json``. Published
        beside ``complete`` through the ``outcome`` / ``n_skipped`` columns, so
        a reader can always see which half of the verdict moved.
        """
        if not self.in_summary:
            return None
        return self.outcome == OUTCOME_COMPLETED and not self.skipped_run_ids

    @property
    def complete(self) -> bool | None:
        """True / False / None per KTD9's evaluation row, over both signals.

        The summary's verdict and ``runs.runs_complete`` are combined
        three-valued (KTD9): either one being ``False`` makes the set
        incomplete, an unknown planned count leaves the set unknown, and only
        two trues make it complete. Unknown never collapses into false in
        either direction -- a set the config cannot size is unmeasurable, not
        failed, and a set the summary never recorded stays unmeasured even when
        its runs are all on disk.
        """
        summary = self.summary_complete
        if summary is None:
            # KTD9's unknown row: a set absent from the summary is unmeasured.
            # The run counts still travel beside it, so a reader sees whatever
            # its directory does say.
            return None
        runs = self.runs.runs_complete
        if summary is False or runs is False:
            return False
        if runs is None:
            return None
        return True


@dataclass(frozen=True)
class ExperimentInventory:
    """Everything one experiment directory holds, as one read-only view.

    ``repetitions`` is ``None`` when the config could not be resolved; every
    ``RunCounts`` in the inventory then reports ``count_unknown``, which is the
    single place a reader has to look to know why.
    """

    out_dir: Path
    rounds: tuple[RoundRecord, ...]
    eval_sets: tuple[EvalSetRecord, ...]
    repetitions: StageRepetitions | None


# ---------------------------------------------------------------------------
# Marker reading: presence and payload agreement, never an exception
# ---------------------------------------------------------------------------


def _read_marker(path: Path, expected_format: str) -> dict[str, Any] | None:
    """A stage marker's payload, or ``None`` when absent or not that format.

    Discovery describes a directory; it never raises over one. A marker that
    cannot be read is reported as incompleteness by the caller, which is the
    honest reading -- the loop writes markers atomically and only after the
    stage's artifacts, so an unreadable one means the stage did not finish.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != expected_format:
        return None
    return payload


def _marker_round_agrees(payload: dict[str, Any] | None, round_index: int) -> bool:
    """Whether a marker exists and records the round it was found under.

    Both stage markers record a ``round``, and both are checked against the
    directory they were read from (KTD9 rows 1 and 2). A marker naming another
    round is a marker that travelled -- copied, restored from another tree, or
    written by a resume against a different index -- and it certifies nothing
    about the files beside it.
    """
    if payload is None:
        return False
    try:
        return int(payload["round"]) == round_index
    except (KeyError, TypeError, ValueError):
        return False


def _jsonl_record_count(path: Path) -> int | None:
    """Parsed records in a JSON-lines file; ``None`` when absent or unparseable.

    Parsed, not line-counted, because the writer parses: a half-written final
    line still counts as a line, and counting it would let a torn audit file
    match a marker that describes an intact one.
    """
    if not path.exists():
        return None
    try:
        return len(read_jsonl(path))
    except (OSError, ValueError):
        return None


def _bundle_id_agrees(payload: dict[str, Any], mining_round_path: Path) -> bool:
    """Whether the marker names the bundle sitting in the same directory.

    The writer records the id it read out of ``bundle.json`` so the marker
    names the evidence it certifies; a marker carrying no id at all did not
    come from that writer and is not trusted. The bundle's own id is the one
    resolvable-or-not half: a ``bundle.json`` that is absent, unreadable, or
    carries no ``bundle_id`` leaves nothing to compare against, and the round
    and record-count checks decide alone rather than this returning a verdict
    it has no evidence for. A bundle that DOES name itself and names another
    bundle is a positive disagreement -- the marker describes evidence that is
    no longer the evidence on disk.
    """
    recorded = payload.get("bundle_id")
    if recorded is None:
        return False
    bundle_path = mining_round_path / BUNDLE_FILENAME
    if not bundle_path.exists():
        return True
    try:
        bundle = json.loads(bundle_path.read_text())
    except (OSError, ValueError):
        return True
    if not isinstance(bundle, dict) or bundle.get("bundle_id") is None:
        return True
    return str(recorded) == str(bundle["bundle_id"])


def _evidence_is_complete(
    payload: dict[str, Any] | None, mining_round_path: Path, round_index: int
) -> bool:
    """Whether the evidence marker actually certifies the files beside it.

    ``orchestrator._evidence_marker_payload`` is the writer, and this is the
    same check made again by a process that did not write the tree, so it has
    to match: the marker names its round and its bundle, and BOTH audit files
    parse to the record counts it recorded. ``write_bundle`` publishes
    ``bundle.json`` before ``records.jsonl`` and ``attributions.jsonl``, so the
    re-read of the two JSON-lines files is what proves the whole triplet
    landed.

    Every failure direction is honest: an absent, foreign, or unparseable
    marker -- and an audit file that cannot be parsed at all -- yields false,
    never true, because "complete" here is a positive claim that has to be
    demonstrated. (KTD9 gives this row no unknown value: the marker is either
    on disk and checkable or it is not there.)
    """
    if payload is None:
        return False
    if not _marker_round_agrees(payload, round_index):
        return False
    if not _bundle_id_agrees(payload, mining_round_path):
        return False
    for file_name, key in (
        (RECORDS_FILENAME, "n_records"),
        (ATTRIBUTIONS_FILENAME, "n_attributions"),
    ):
        actual = _jsonl_record_count(mining_round_path / file_name)
        if actual is None:
            return False
        try:
            if int(payload[key]) != actual:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _opt_round_indices(opt_root: Path) -> list[int]:
    """Every round index with a directory under ``<out_dir>/opt``, in order.

    The single enumeration point (KTD1), and it yields indices rather than
    paths on purpose: the paths every caller sees are rebuilt by the loop's own
    ``experiment_round_dir``, so discovery cannot report a path the loop would
    not have written. A name that is not an integer suffix -- a stray
    ``round_x`` -- or one whose index does not round-trip through the loop's
    zero-padded naming -- ``round_1`` -- is skipped rather than crashing the
    whole inventory: one malformed directory is not a reason to blind every
    analysis of the rounds beside it.
    """
    if not opt_root.is_dir():
        return []
    found: set[int] = set()
    for path in sorted(opt_root.glob(ROUND_DIR_PATTERN)):
        if not path.is_dir():
            continue
        try:
            round_index = int(path.name.removeprefix(ROUND_DIR_PREFIX))
        except ValueError:
            continue
        if round_dir(opt_root, round_index) == path:
            found.add(round_index)
    return sorted(found)


def _round_record(
    out_dir: Path, round_index: int, repetitions: StageRepetitions | None
) -> RoundRecord:
    """Describe one round directory: its paths, its markers, its counts."""
    round_path = experiment_round_dir(out_dir, round_index)
    mining_parent = round_path / MINING_DIR
    mining_round_path = round_dir(mining_parent, round_index)
    validation_parent = round_path / VALIDATION_DIR
    validation_round_path = round_dir(validation_parent, round_index)

    round_marker = _read_marker(round_path / ROUND_MARKER_FILENAME, ROUND_MARKER_FORMAT)
    evidence_marker = _read_marker(
        mining_round_path / EVIDENCE_MARKER_FILENAME, EVIDENCE_MARKER_FORMAT
    )
    return RoundRecord(
        round_index=round_index,
        round_path=round_path,
        mining_parent=mining_parent,
        mining_round_path=mining_round_path,
        validation_parent=validation_parent,
        validation_round_path=validation_round_path,
        proposals_dir=round_path / PROPOSALS_DIR,
        has_round_marker=(round_path / ROUND_MARKER_FILENAME).exists(),
        has_evidence_marker=(mining_round_path / EVIDENCE_MARKER_FILENAME).exists(),
        has_proposals_marker=(round_path / PROPOSALS_MARKER_FILENAME).exists(),
        has_manifest=(mining_round_path / MANIFEST_FILE).exists(),
        has_ledger=(validation_round_path / PROMOTIONS_FILENAME).exists(),
        round_complete=_marker_round_agrees(round_marker, round_index),
        evidence_complete=_evidence_is_complete(evidence_marker, mining_round_path, round_index),
        mining_runs=_run_counts(
            mining_round_path,
            mining_parent,
            round_index,
            None if repetitions is None else repetitions.mining,
        ),
    )


def _summary_sets(out_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Every ``(condition_id, set_id)`` aggregate ``eval_summary.json`` records."""
    summary_path = out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return {}
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for condition_id, condition in dict(summary.get("conditions", {})).items():
        for set_id, aggregate in dict(condition.get("test_sets", {})).items():
            aggregates[(str(condition_id), str(set_id))] = aggregate
    return aggregates


def _disk_sets(out_dir: Path) -> set[tuple[str, str]]:
    """Every condition x set that has a round directory on disk.

    A test set is a condition subdirectory holding ``round_00`` (the evaluation
    round index), which is also what distinguishes it from the condition's
    ``work/`` scratch directory without hardcoding that name here.
    """
    eval_root = out_dir / EVAL_DIR
    if not eval_root.is_dir():
        return set()
    found: set[tuple[str, str]] = set()
    for condition_path in sorted(eval_root.iterdir()):
        if not condition_path.is_dir():
            continue
        for set_path in sorted(condition_path.iterdir()):
            if set_path.is_dir() and round_dir(set_path, EVAL_ROUND_INDEX).is_dir():
                found.add((condition_path.name, set_path.name))
    return found


def _eval_set_record(
    out_dir: Path,
    condition_id: str,
    set_id: str,
    aggregate: dict[str, Any] | None,
    repetitions: StageRepetitions | None,
) -> EvalSetRecord:
    set_path = out_dir / EVAL_DIR / condition_id / set_id
    skipped = (
        ()
        if aggregate is None
        else tuple(str(run_id) for run_id in aggregate.get("skipped_run_ids", ()))
    )
    outcome = None if aggregate is None else aggregate.get("outcome")
    return EvalSetRecord(
        condition_id=condition_id,
        set_id=set_id,
        set_path=set_path,
        round_path=round_dir(set_path, EVAL_ROUND_INDEX),
        environment=None if aggregate is None else _optional_str(aggregate.get("environment")),
        length=None if aggregate is None else _optional_str(aggregate.get("length")),
        outcome=_optional_str(outcome),
        skipped_run_ids=skipped,
        in_summary=aggregate is not None,
        runs=_run_counts(
            round_dir(set_path, EVAL_ROUND_INDEX),
            set_path,
            EVAL_ROUND_INDEX,
            None if repetitions is None else repetitions.evaluation,
        ),
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def discover_rounds(
    out_dir: Path | str, *, config: ExperimentConfig | None = None
) -> ExperimentInventory:
    """Everything one experiment directory holds: rounds and evaluation sets.

    The single inventory every analysis (and the report) reads. Nothing is
    dropped for being incomplete -- partial rounds and partial test sets come
    back flagged (KTD5), because refusing to report them would hide exactly the
    data a reader needs to discount.

    Args:
        out_dir: The experiment directory the loop wrote.
        config: The experiment's configuration, when the caller already holds
            it (the loop does). Omitted, it is resolved from
            ``<out_dir>/config.json`` and verified against its identity hash;
            an unresolvable config leaves every expected run count unknown
            rather than guessed (KTD6).

    Returns:
        The inventory: round records in ascending round order, evaluation set
        records sorted by ``(condition_id, set_id)``, and the per-stage
        repetition counts (``None`` when the config was unresolvable).
    """
    out = Path(out_dir)
    resolved = config if config is not None else resolve_config(out)
    repetitions = stage_repetitions(resolved) if resolved is not None else None

    rounds = tuple(
        _round_record(out, round_index, repetitions)
        for round_index in _opt_round_indices(out / OPT_DIR)
    )

    aggregates = _summary_sets(out)
    keys = sorted(set(aggregates) | _disk_sets(out))
    eval_sets = tuple(
        _eval_set_record(
            out, condition_id, set_id, aggregates.get((condition_id, set_id)), repetitions
        )
        for condition_id, set_id in keys
    )
    return ExperimentInventory(
        out_dir=out, rounds=rounds, eval_sets=eval_sets, repetitions=repetitions
    )


# ---------------------------------------------------------------------------
# Surface attribution: the ledger first, the round's proposals second (KTD3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceAttribution:
    """Which tally one ledger row belongs in, and where that came from.

    ``category`` is one of the nine surface ids, the ``merged`` category, or
    ``None`` when nothing on disk can place the row; ``source`` is the
    ``SURFACE_SOURCE_*`` value naming the evidence. The two travel together on
    purpose: a count of S3 touches means something different when the S3 came
    out of a proposal artifact than when the ledger recorded it, and an output
    that carried only the category would let the difference disappear.
    """

    category: str | None
    source: str

    @property
    def is_merged(self) -> bool:
        """Whether this is the merged harness's own record (never a surface)."""
        return self.source == SURFACE_SOURCE_MERGED

    @property
    def unresolved(self) -> bool:
        """Whether no tally could be found for the row -- the honest failure."""
        return self.category is None


def _proposal_surface(proposals_dir: Path, candidate_id: str) -> str | None:
    """The surface the round's proposal for ``candidate_id`` declared, if any.

    The join key is the ledger's ``subject_id`` against the proposal
    directory's ``candidate_id`` -- ``write_proposal``'s own layout, read back
    through the path discovery reports for the round rather than one rebuilt
    here. A proposal that is absent, unreadable, of another format, or carrying
    no surface resolves to ``None``: the row stays honestly unattributed rather
    than borrowing a value from a file this analysis cannot vouch for.
    """
    if not candidate_id:
        return None
    payload = _read_marker(proposals_dir / candidate_id / PROPOSAL_FILENAME, PROPOSAL_FORMAT)
    if payload is None:
        return None
    surface = payload.get("surface")
    return None if surface is None else str(surface)


def resolve_surface(record: Mapping[str, Any], round_record: RoundRecord) -> SurfaceAttribution:
    """One ledger row's surface tally, backfilled from its proposal when needed.

    Three steps, in this order (KTD3, KTD7):

    1. ``MERGED_SUBJECT_ID`` short-circuits to the ``merged`` category. The
       merged harness spans several surfaces by construction, so its null
       surface is correct rather than missing -- backfilling it would invent a
       single-surface claim the experiment never made.
    2. A non-null ledger value is taken verbatim. The ledger is the evidence;
       a proposal never overrides it.
    3. A ``surface`` that is missing OR present-and-null triggers the proposal
       join. Both spellings matter, and the second is the load-bearing one:
       ``CandidateDecision.to_dict`` always emits the key, so every loader
       rejection the current loop writes serializes as ``surface: null``, and a
       backfill keyed on an absent key would never fire for anything but a
       pre-``surface`` ledger.

    Args:
        record: One ``promotions.jsonl`` line.
        round_record: That ledger's round, from discovery -- it carries the
            proposals directory the join reads.

    Returns:
        The attribution: a surface id, the ``merged`` category, or ``None``
        with ``SURFACE_SOURCE_NONE`` when no proposal survives to recover from.
    """
    subject_id = str(record.get("subject_id", ""))
    if subject_id == MERGED_SUBJECT_ID:
        return SurfaceAttribution(MERGED_CATEGORY, SURFACE_SOURCE_MERGED)
    surface = record.get("surface")
    if surface is not None:
        return SurfaceAttribution(str(surface), SURFACE_SOURCE_LEDGER)
    backfilled = _proposal_surface(round_record.proposals_dir, subject_id)
    if backfilled is not None:
        return SurfaceAttribution(backfilled, SURFACE_SOURCE_BACKFILLED)
    return SurfaceAttribution(None, SURFACE_SOURCE_NONE)


def iter_promotion_rounds(
    out_dir: Path | str, *, inventory: ExperimentInventory | None = None
) -> Iterator[tuple[int, list[dict[str, Any]], dict[str, Any]]]:
    """``(round_index, records, decision)`` for every ledgered round, in order.

    ``records`` are the ``promotions.jsonl`` lines (one per candidate, plus the
    merged harness's own record when the round's plan merged); ``decision`` is
    the round's ``decision.json`` payload. Rounds with no ledger -- the
    completed no-op round, or one interrupted before validation -- are skipped
    here rather than yielded empty, so callers must not assume the indices are
    contiguous from 1.

    ``round_index`` is not a field inside any ``promotions.jsonl`` record (see
    ``shrlm.optimization.validation._ledger_record``); it is recoverable only
    from the directory the ledger sits in, which is why it comes from discovery
    rather than from the record payloads.

    Args:
        out_dir: The experiment directory, used when no inventory is passed.
        inventory: The discovery the caller already ran over ``out_dir``.
            ``discover_rounds`` globs, parses every stage marker, counts
            manifest lines, and re-reads the experiment TOML, so a caller that
            already holds an inventory passes it instead of paying for an
            identical second pass -- identical because nothing rewrites the
            tree between the two. Omitted, it is discovered here.
    """
    resolved = inventory if inventory is not None else discover_rounds(out_dir)
    for record in resolved.rounds:
        if not record.has_ledger:
            continue
        records, decision = load_promotion_ledger(record.validation_round_path)
        yield record.round_index, records, decision


__all__ = [
    "MERGED_CATEGORY",
    "STAGE_EVALUATION",
    "STAGE_MINING",
    "STAGE_VALIDATION",
    "SURFACE_SOURCE_BACKFILLED",
    "SURFACE_SOURCE_LEDGER",
    "SURFACE_SOURCE_MERGED",
    "SURFACE_SOURCE_NONE",
    "EvalSetRecord",
    "ExperimentInventory",
    "RoundRecord",
    "RunCounts",
    "StageRepetitions",
    "SurfaceAttribution",
    "discover_rounds",
    "iter_promotion_rounds",
    "resolve_config",
    "resolve_surface",
    "stage_repetitions",
]
