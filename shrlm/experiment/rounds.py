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

Discovery answers two questions, and keeps them apart:

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
``evidence_complete````evidence_complete.json`` present and its recorded line
                     counts matching the persisted ``records.jsonl`` /
                     ``attributions.jsonl``. Never unknown. The re-check
                     matters because ``write_bundle`` publishes ``bundle.json``
                     before those two files, so a marker written over a
                     truncated audit file is exactly the crash window the
                     orchestrator's own marker payload guards.
``runs_complete``    ``n_present == n_expected`` when the expected count is
                     knowable; ``None`` (``count_unknown``) when it is not.
eval set ``complete``the set's outcome is ``completed`` with no skipped runs;
                     ``None`` when the set is absent from ``eval_summary.json``.
===================  ==========================================================

``unknown`` is always its own value and never collapses into false: an
unmeasurable round must never be presented as a failed one.

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
from collections.abc import Iterator
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
from shrlm.optimization.bundle import ATTRIBUTIONS_FILENAME, RECORDS_FILENAME, round_dir
from shrlm.optimization.costs import OUTCOME_COMPLETED
from shrlm.optimization.driver import INSTANCES_FILE, MANIFEST_FILE, load_manifest
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
    def complete(self) -> bool | None:
        """True / False / None per KTD9's evaluation row."""
        if not self.in_summary:
            return None
        return self.outcome == OUTCOME_COMPLETED and not self.skipped_run_ids


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


def _round_marker_agrees(payload: dict[str, Any] | None, round_index: int) -> bool:
    """Whether a round marker exists and records this round (KTD9 row 1)."""
    if payload is None:
        return False
    try:
        return int(payload["round"]) == round_index
    except (KeyError, TypeError, ValueError):
        return False


def _evidence_is_complete(payload: dict[str, Any] | None, mining_round_path: Path) -> bool:
    """Whether the evidence marker's counts match the persisted audit files.

    ``write_bundle`` publishes ``bundle.json`` before ``records.jsonl`` and
    ``attributions.jsonl``, so only the line-count re-check proves the triplet
    landed -- the same check ``orchestrator._evidence_marker_payload`` makes
    before it writes the marker, made again here because an analysis reads
    trees the current process did not write.
    """
    if payload is None:
        return False
    for file_name, key in (
        (RECORDS_FILENAME, "n_records"),
        (ATTRIBUTIONS_FILENAME, "n_attributions"),
    ):
        actual = _count_lines(mining_round_path / file_name)
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
        round_complete=_round_marker_agrees(round_marker, round_index),
        evidence_complete=_evidence_is_complete(evidence_marker, mining_round_path),
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


def iter_promotion_rounds(
    out_dir: Path | str,
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
    """
    for record in discover_rounds(out_dir).rounds:
        if not record.has_ledger:
            continue
        records, decision = load_promotion_ledger(record.validation_round_path)
        yield record.round_index, records, decision


__all__ = [
    "STAGE_EVALUATION",
    "STAGE_MINING",
    "STAGE_VALIDATION",
    "EvalSetRecord",
    "ExperimentInventory",
    "RoundRecord",
    "RunCounts",
    "StageRepetitions",
    "discover_rounds",
    "iter_promotion_rounds",
    "resolve_config",
    "stage_repetitions",
]
